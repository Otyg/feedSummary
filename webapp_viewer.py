import argparse
import asyncio
import datetime as dt
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
import traceback
from typing import Any, Dict, List, Optional, Tuple

import markdown as md
import yaml
from flask import Flask, Response, abort, redirect, render_template, request, url_for

from summarizer.main import run_pipeline
from summarizer.helpers import setup_logging
from uicommon import format_ts, get_store, load_config

setup_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates_viewer", static_folder="static")

APP_CONFIG_PATH: str = ""
APP_CFG: Dict[str, Any] = {}
APP_STORE = None

SCHEDULER = None

@app.errorhandler(Exception)
def handle_all_errors(e):
    tb = traceback.format_exc()
    return Response(tb, status=500, mimetype="text/plain")

def _resolve_path_from_cwd(p: str) -> str:
    pp = Path(os.path.expandvars(os.path.expanduser(p)))
    if not pp.is_absolute():
        pp = (Path.cwd() / pp).resolve()
    return str(pp)


def _resolve_config_path_cli(cli_path: Optional[str]) -> str:
    """
    Priority:
      1) CLI --config
      2) ENV FEEDSUMMARY_CONFIG
      3) cwd/config.yaml (fallback to cwd/config.yaml.dist)
    """
    if cli_path:
        return _resolve_path_from_cwd(cli_path)

    env = os.environ.get("FEEDSUMMARY_CONFIG", "").strip()
    if env:
        return _resolve_path_from_cwd(env)

    p = (Path.cwd() / "config.yaml").resolve()
    if p.exists():
        return str(p)

    pd = (Path.cwd() / "config.yaml.dist").resolve()
    return str(pd) if pd.exists() else str(p)


def init_app_state(config_path: str) -> None:
    """
    Load config + create store ONCE at startup.
    """
    global APP_CONFIG_PATH, APP_CFG, APP_STORE
    APP_CONFIG_PATH = config_path
    APP_CFG = load_config(config_path)
    APP_STORE = get_store(APP_CFG)
    logger.info("Viewer config loaded: %s", APP_CONFIG_PATH)


WEEKDAY = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tues": 1,
    "tuesday": 1,
    "wed": 2,
    "wednesday": 2,
    "thu": 3,
    "thurs": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}


@dataclass
class ScheduleEntry:
    name: str
    frequency: str  # "daily"|"weekly"
    time_hhmm: str  # "08:00"
    day: Optional[str]  # for weekly e.g. "sun"
    categories: List[str]  # treated as topics
    promptpackage: str


def _read_schedule_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        logger.warning("Schedule file not found: %s", p)
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _parse_time_hhmm(s: str) -> Tuple[int, int]:
    parts = (s or "").strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time format: {s!r} expected HH:MM")
    return int(parts[0]), int(parts[1])


def _next_run_for(entry: ScheduleEntry, now: dt.datetime) -> dt.datetime:
    hh, mm = _parse_time_hhmm(entry.time_hhmm)
    base = now.replace(hour=hh, minute=mm, second=0, microsecond=0)

    freq = entry.frequency.lower().strip()
    if freq == "daily":
        if base <= now:
            base = base + dt.timedelta(days=1)
        return base

    if freq == "weekly":
        if not entry.day:
            raise ValueError(f"Weekly schedule '{entry.name}' missing 'day'")
        wd = WEEKDAY.get(entry.day.lower().strip())
        if wd is None:
            raise ValueError(f"Invalid weekday for '{entry.name}': {entry.day}")

        days_ahead = (wd - now.weekday()) % 7
        candidate = base + dt.timedelta(days=days_ahead)
        if candidate <= now:
            candidate = candidate + dt.timedelta(days=7)
        return candidate

    raise ValueError(f"Unknown frequency: {entry.frequency!r}")


def _entry_to_overrides(entry: ScheduleEntry) -> Dict[str, Any]:
    """
    Converts schedule.yaml entry to pipeline overrides.
    - categories => topics
    - promptpackage => prompt_package
    - lookback derived from frequency
    """
    freq = entry.frequency.lower().strip()
    lookback = "1d" if freq == "daily" else "1w" if freq == "weekly" else ""

    overrides: Dict[str, Any] = {}
    if lookback:
        overrides["lookback"] = lookback
    if entry.categories:
        overrides["topics"] = list(entry.categories)
    if entry.promptpackage:
        overrides["prompt_package"] = entry.promptpackage
    return overrides


class PipelineScheduler(threading.Thread):
    """
    Very small scheduler:
    - loads schedules from YAML
    - computes next-run per job
    - wakes periodically, runs pipeline when due
    """

    def __init__(self, *, schedule_path: str, poll_seconds: int = 20):
        super().__init__(daemon=True)
        self.schedule_path = schedule_path
        self.poll_seconds = int(poll_seconds)
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self.last_reload_at: Optional[float] = None
        self.entries: List[ScheduleEntry] = []
        self.next_runs: Dict[str, dt.datetime] = {}
        self.last_runs: Dict[str, dt.datetime] = {}

        self.running_job: Optional[str] = None
        self.last_error: Optional[str] = None
        self.last_ok: Optional[str] = None

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "schedule_path": self.schedule_path,
                "entries": [e.__dict__ for e in self.entries],
                "next_runs": {k: v.isoformat() for k, v in self.next_runs.items()},
                "last_runs": {k: v.isoformat() for k, v in self.last_runs.items()},
                "running_job": self.running_job,
                "last_error": self.last_error,
                "last_ok": self.last_ok,
                "last_reload_at": self.last_reload_at,
                "poll_seconds": self.poll_seconds,
            }

    def _reload(self) -> None:
        raw = _read_schedule_yaml(self.schedule_path)
        entries: List[ScheduleEntry] = []

        for name, obj in raw.items():
            if not isinstance(obj, dict):
                continue
            frequency = str(obj.get("frequency") or "").strip()
            time_hhmm = str(obj.get("time") or "").strip()
            day = obj.get("day")
            day_s = str(day).strip() if day is not None else None
            categories = obj.get("categories") or []
            if not isinstance(categories, list):
                categories = []
            categories_s = [str(x).strip() for x in categories if str(x).strip()]
            promptpackage = str(obj.get("promptpackage") or "").strip()

            if not (frequency and time_hhmm and promptpackage):
                logger.warning("Skipping invalid schedule entry '%s': %s", name, obj)
                continue

            entries.append(
                ScheduleEntry(
                    name=name,
                    frequency=frequency,
                    time_hhmm=time_hhmm,
                    day=day_s,
                    categories=categories_s,
                    promptpackage=promptpackage,
                )
            )

        now = dt.datetime.now()
        next_runs: Dict[str, dt.datetime] = {}
        for e in entries:
            try:
                next_runs[e.name] = _next_run_for(e, now)
            except Exception as ex:
                logger.exception("Failed computing next run for %s: %s", e.name, ex)

        with self._lock:
            self.entries = entries
            self.next_runs.update(next_runs)
            self.last_reload_at = time.time()

        logger.info("Scheduler loaded %d entries from %s", len(entries), self.schedule_path)

    def _run_job(self, entry: ScheduleEntry) -> None:
        cfg_path = APP_CONFIG_PATH
        cfg = APP_CFG

        overrides = _entry_to_overrides(entry)
        logger.info("Running scheduled job '%s' overrides=%s", entry.name, overrides)

        with self._lock:
            self.running_job = entry.name
            self.last_error = None

        try:
            summary_id = asyncio.run(
                run_pipeline(
                    cfg_path,
                    job_id=None,
                    overrides=overrides,
                    config_dict=cfg,
                )
            )
            with self._lock:
                self.last_ok = f"{entry.name}: summary_id={summary_id}"
                self.last_runs[entry.name] = dt.datetime.now()
        except Exception as e:
            logger.exception("Scheduled job '%s' failed: %s", entry.name, e)
            with self._lock:
                self.last_error = f"{entry.name}: {e}"
        finally:
            with self._lock:
                self.running_job = None

    def run(self) -> None:
        self._reload()
        while not self._stop.is_set():
            try:
                if self.last_reload_at is None or (time.time() - self.last_reload_at) > 60:
                    self._reload()

                now = dt.datetime.now()
                due: List[ScheduleEntry] = []

                with self._lock:
                    entries = list(self.entries)
                    next_runs = dict(self.next_runs)

                for e in entries:
                    nr = next_runs.get(e.name)
                    if nr and nr <= now:
                        due.append(e)

                for e in due:
                    self._run_job(e)
                    try:
                        nxt = _next_run_for(e, dt.datetime.now())
                        with self._lock:
                            self.next_runs[e.name] = nxt
                    except Exception:
                        pass

                self._stop.wait(self.poll_seconds)

            except Exception as e:
                logger.exception("Scheduler loop error: %s", e)
                with self._lock:
                    self.last_error = f"scheduler_loop: {e}"
                self._stop.wait(self.poll_seconds)


def _scheduler_settings_from_sources(
    *,
    cfg: Dict[str, Any],
    cli_enabled: Optional[bool],
    cli_path: Optional[str],
    cli_poll_seconds: Optional[int],
) -> Tuple[bool, str, int]:
    """
    Resolve scheduler settings with priority:
      1) CLI args
      2) config.yaml (cfg["scheduler"])
      3) ENV FEEDSUMMARY_SCHEDULE / FEEDSUMMARY_SCHEDULE_PATH
      4) defaults (disabled, schedule.yaml, 20s)
    """
    enabled: bool = False
    path: str = "config/schedule.yaml"
    poll_seconds: int = 20

    sc = cfg.get("scheduler")
    if isinstance(sc, dict):
        if "enabled" in sc:
            enabled = bool(sc.get("enabled"))
        if "path" in sc and str(sc.get("path") or "").strip():
            path = str(sc.get("path")).strip()
        if "poll_seconds" in sc:
            try:
                poll_seconds = int(sc.get("poll_seconds") or poll_seconds)
            except Exception:
                pass

    env_enabled = os.environ.get("FEEDSUMMARY_SCHEDULE", "").strip()
    if env_enabled:
        enabled = env_enabled == "1" or env_enabled.lower() in ("true", "yes", "on")
    env_path = os.environ.get("FEEDSUMMARY_SCHEDULE_PATH", "").strip()
    if env_path:
        path = env_path
    env_poll = os.environ.get("FEEDSUMMARY_SCHEDULE_POLL", "").strip()
    if env_poll:
        try:
            poll_seconds = int(env_poll)
        except Exception:
            pass

    if cli_enabled is not None:
        enabled = bool(cli_enabled)
    if cli_path:
        path = cli_path
    if cli_poll_seconds is not None:
        poll_seconds = int(cli_poll_seconds)

    path = _resolve_path_from_cwd(path)

    return enabled, path, poll_seconds


def start_scheduler(
    *,
    cfg: Dict[str, Any],
    cli_enabled: Optional[bool],
    cli_path: Optional[str],
    cli_poll_seconds: Optional[int],
    debug: bool,
) -> None:
    """
    Start scheduler if enabled via CLI/config/env.
    """
    global SCHEDULER

    enabled, schedule_path, poll_seconds = _scheduler_settings_from_sources(
        cfg=cfg,
        cli_enabled=cli_enabled,
        cli_path=cli_path,
        cli_poll_seconds=cli_poll_seconds,
    )

    if not enabled:
        logger.info("Scheduler disabled.")
        return

    if debug and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        return

    SCHEDULER = PipelineScheduler(schedule_path=schedule_path, poll_seconds=poll_seconds)
    SCHEDULER.start()
    logger.info("Scheduler started: path=%s poll_seconds=%s", schedule_path, poll_seconds)


def _get_latest_summary(store) -> Optional[Dict[str, Any]]:
    # 1) Prefer store native helper
    fn = getattr(store, "get_latest_summary_doc", None)
    if callable(fn):
        try:
            d = fn()  # type: ignore
            if isinstance(d, dict) and (d.get("summary") or "").strip():
                return d
            if isinstance(d, dict) and d.get("id"):
                # fallback: try re-fetch by id
                d2 = store.get_summary_doc(str(d.get("id")))
                if isinstance(d2, dict) and (d2.get("summary") or "").strip():
                    return d2
                return d  # return something rather than None
        except Exception:
            pass

    # 2) Fallback: list + pick newest
    docs = store.list_summary_docs() or []
    if not docs:
        return None

    docs.sort(key=lambda x: int(x.get("created") or 0), reverse=True)

    # Try the first few in case newest is partially saved / missing summary
    for cand in docs[:10]:
        if not isinstance(cand, dict):
            continue

        # If list already contains summary text, accept it
        if (cand.get("summary") or "").strip():
            return cand

        sid = str(cand.get("id") or "").strip()
        if not sid:
            continue

        try:
            d = store.get_summary_doc(sid)
            if isinstance(d, dict) and ((d.get("summary") or "").strip() or d):
                return d
        except Exception:
            continue

    # Last resort: return the newest list entry even if it's missing summary
    return docs[0] if isinstance(docs[0], dict) else None


def _md_to_html(text: str) -> str:
    return md.markdown(text or "", extensions=["extra"])


@app.route("/")
def index():
    store = APP_STORE
    if store is None:
        abort(500)

    latest = _get_latest_summary(store)
    if not latest:
        return render_template(
            "index.html",
            summary=None,
            html="<p>Inga summaries ännu.</p>",
            summaries=[],
            default_selected=None,
        )

    sid = str(latest.get("id") or "")
    return redirect(url_for("view_summary", summary_id=sid))


@app.route("/summaries")
def list_summaries():
    store = APP_STORE
    if store is None:
        abort(500)

    docs = store.list_summary_docs() or []
    docs.sort(key=lambda d: int(d.get("created") or 0), reverse=True)
    return render_template("summaries.html", summaries=docs)


@app.route("/summary/<summary_id>")
def view_summary(summary_id: str):
    store = APP_STORE
    if store is None:
        abort(500)

    docs = store.list_summary_docs() or []
    docs = [d for d in docs if isinstance(d, dict)]
    docs.sort(key=lambda d: int(d.get("created") or 0), reverse=True)

    sid = str(summary_id).strip()

    # 1) Normal fetch by id
    sdoc = None
    try:
        sdoc = store.get_summary_doc(sid)
    except Exception:
        sdoc = None

    # 2) If not found, try to locate it in list_summary_docs() and use that object
    if not sdoc:
        for d in docs:
            if str(d.get("id") or "") == sid:
                sdoc = d
                break

    if not sdoc:
        abort(404)

    # Always produce non-empty body
    summary_text = ""
    if isinstance(sdoc, dict):
        summary_text = str(sdoc.get("summary") or "").strip()

    if not summary_text:
        # fallback diagnostic to avoid "blank page"
        keys = ", ".join(sorted(list(sdoc.keys()))) if isinstance(sdoc, dict) else ""
        summary_text = (
            "*(Ingen summary-text hittades i dokumentet.)*\n\n"
            f"- requested id: `{sid}`\n"
            f"- doc id: `{sdoc.get('id')}`\n"
            f"- created: `{sdoc.get('created')}`\n"
            f"- keys: `{keys}`\n"
        )

    html = _md_to_html(summary_text)

    return render_template(
        "index.html",
        summary=sdoc,
        html=html,
        summaries=docs,
        default_selected=sid,
        format_ts=format_ts,
    )


@app.route("/articles")
def list_articles():
    store = APP_STORE
    if store is None:
        abort(500)

    # limit param
    try:
        limit = int(request.args.get("limit", "300"))
    except Exception:
        limit = 300
    limit = max(1, min(limit, 5000))

    try:
        raw = store.list_articles(limit=limit) or []
    except Exception as e:
        # show a readable error instead of empty page
        return render_template(
            "articles.html",
            articles=[],
            format_ts=format_ts,
            error=f"Kunde inte läsa artiklar: {e}",
        ), 500

    # sanitize: keep only dicts, ensure id exists
    articles: List[Dict[str, Any]] = []
    for a in raw:
        if not isinstance(a, dict):
            continue
        if not a.get("id"):
            # skip invalid
            continue
        articles.append(a)

    def ts(a: Dict[str, Any]) -> int:
        p = a.get("published_ts")
        if isinstance(p, int) and p > 0:
            return p
        f = a.get("fetched_at")
        return int(f or 0)

    articles.sort(key=ts, reverse=True)

    return render_template(
        "articles.html",
        articles=articles,
        format_ts=format_ts,
        error=None,
    )

from flask import Response
from markupsafe import escape

@app.route("/article/<article_id>")
def view_article(article_id: str):
    store = APP_STORE
    if store is None:
        abort(500)

    try:
        a = store.get_article(str(article_id))
    except Exception as e:
        # Visa alltid något i UI istället för "tomt"
        return (
            f"get_article({article_id!r}) raised: {e}\n",
            500,
            {"Content-Type": "text/plain; charset=utf-8"},
        )

    if not isinstance(a, dict):
        return (
            f"Article not found for id={article_id!r}\n",
            404,
            {"Content-Type": "text/plain; charset=utf-8"},
        )

    # Gör template-säkert + debug om text saknas
    if a.get("title") is None:
        a["title"] = ""
    if a.get("source") is None:
        a["source"] = ""
    if a.get("url") is None:
        a["url"] = ""
    if a.get("text") is None:
        a["text"] = ""

    # Om text är tom: bygg fallback-text som visar vad som faktiskt finns i objektet
    if not str(a.get("text") or "").strip():
        keys = ", ".join(sorted(a.keys()))
        fallback = (
            "⚠️ Ingen artikeltext hittades i databasen för den här posten.\n\n"
            f"id: {a.get('id')}\n"
            f"source: {a.get('source')}\n"
            f"title: {a.get('title')}\n"
            f"url: {a.get('url')}\n"
            f"keys: {keys}\n"
        )
        a["text"] = fallback

    try:
        return render_template("article.html", a=a, format_ts=format_ts)
    except Exception as e:
        # Om templaten kraschar: returnera ren text så du ser felet direkt
        keys = ", ".join(sorted(a.keys()))
        return (
            f"render_template(article.html) failed: {e}\n\nkeys: {keys}\n",
            500,
            {"Content-Type": "text/plain; charset=utf-8"},
        )

@app.route("/status")
def status():
    base = {"config": APP_CONFIG_PATH}
    if SCHEDULER is None:
        base["scheduler"] = "disabled"
        return base
    snap = SCHEDULER.snapshot()
    snap.update(base)
    return snap


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="FeedSummary Viewer WebApp")
    parser.add_argument(
        "--config",
        dest="config",
        default=None,
        help="Path to config.yaml (default: FEEDSUMMARY_CONFIG or ./config.yaml)",
    )
    parser.add_argument(
        "--port",
        dest="port",
        type=int,
        default=int(os.environ.get("PORT", "5000")),
        help="Port to bind (default: 5000 or $PORT)",
    )
    parser.add_argument(
        "--host",
        dest="host",
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Enable Flask debug mode",
    )

    parser.add_argument(
        "--schedule",
        dest="schedule",
        action="store_true",
        help="Enable scheduler (overrides config/env).",
    )
    parser.add_argument(
        "--no-schedule",
        dest="no_schedule",
        action="store_true",
        help="Disable scheduler (overrides config/env).",
    )
    parser.add_argument(
        "--schedule-path",
        dest="schedule_path",
        default=None,
        help="Path to schedule.yaml (overrides config/env).",
    )
    parser.add_argument(
        "--schedule-poll",
        dest="schedule_poll",
        type=int,
        default=None,
        help="Scheduler poll seconds (overrides config/env).",
    )

    args = parser.parse_args()

    config_path = _resolve_config_path_cli(args.config)
    init_app_state(config_path)

    cli_enabled: Optional[bool] = None
    if args.schedule and args.no_schedule:
        raise SystemExit("Use only one of --schedule or --no-schedule")
    if args.schedule:
        cli_enabled = True
    if args.no_schedule:
        cli_enabled = False

    start_scheduler(
        cfg=APP_CFG,
        cli_enabled=cli_enabled,
        cli_path=args.schedule_path,
        cli_poll_seconds=args.schedule_poll,
        debug=bool(args.debug),
    )

    app.run(host=args.host, port=args.port, debug=bool(args.debug))