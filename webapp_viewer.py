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

        logger.info(
            "Scheduler loaded %d entries from %s", len(entries), self.schedule_path
        )

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
    logger.info(
        "Scheduler started: path=%s poll_seconds=%s", schedule_path, poll_seconds
    )


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


def _escape_html(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _topics_from_doc(d: Dict[str, Any]) -> List[str]:
    """
    'Kategorier' i viewer = selection.topics (om det finns).
    """
    sel = d.get("selection")
    if not isinstance(sel, dict):
        return []
    t = sel.get("topics")
    if isinstance(t, list):
        out = [str(x).strip() for x in t if str(x).strip()]
        return out
    if isinstance(t, str) and t.strip():
        return [t.strip()]
    return []


def _topics_label(d: Dict[str, Any]) -> str:
    topics = _topics_from_doc(d)
    if not topics:
        return ""
    return "Kategorier: " + ", ".join(topics)


def _enrich_docs_for_view(store, docs: List[Dict[str, Any]], *, limit: int = 200) -> List[Dict[str, Any]]:
    """
    list_summary_docs() kan ge "tunna" objekt beroende på store.
    Här försöker vi komplettera med title/selection via get_summary_doc()
    för de första N, och lägger till _viewer_topics_label.
    """
    out: List[Dict[str, Any]] = []
    for i, d in enumerate(docs):
        if not isinstance(d, dict):
            continue
        dd = dict(d)
        sid = str(dd.get("id") or "").strip()

        need = False
        if i < limit and sid:
            # om title/selection saknas i listobjektet, försök hämta full doc
            if not str(dd.get("title") or "").strip():
                need = True
            sel = dd.get("selection")
            if not isinstance(sel, dict) or not _topics_from_doc(dd):
                # urvalet kan vara tunt här, försök hämta
                need = True

        if need:
            try:
                full = store.get_summary_doc(sid)
                if isinstance(full, dict):
                    # Merge in missing keys (full wins)
                    dd.update(full)
            except Exception:
                pass

        dd["_viewer_topics_label"] = _topics_label(dd)
        out.append(dd)

    return out


def _render_summary_header_html(*, title: str, created_ts: int, doc_id: str, topics_label: str) -> str:
    """
    Render a small header block (Bootstrap-ish) that will be prepended to the summary HTML.
    Includes created + categories (if any).
    """
    created_str = format_ts(created_ts) if created_ts else str(created_ts)
    safe_title = (title or "").strip() or (doc_id or "").strip() or "Sammanfattning"
    safe_title = _escape_html(safe_title)

    extra_line = ""
    if topics_label:
        extra_line = f"<div class='text-muted small'>{_escape_html(topics_label)}</div>"

    return (
        f"<div class='d-flex justify-content-between align-items-start'>"
        f"  <div>"
        f"    <h4 class='mb-1'>{safe_title}</h4>"
        f"    <div class='text-muted small'>Skapad: {created_str}</div>"
        f"    {extra_line}"
        f"  </div>"
        f"</div>"
        f"<hr>"
    )


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
    docs = [d for d in docs if isinstance(d, dict)]
    docs.sort(key=lambda d: int(d.get("created") or 0), reverse=True)
    docs = _enrich_docs_for_view(store, docs, limit=300)

    return render_template("summaries.html", summaries=docs, format_ts=format_ts)


@app.route("/summary/<summary_id>")
def view_summary(summary_id: str):
    store = APP_STORE
    if store is None:
        abort(500)

    docs = store.list_summary_docs() or []
    docs = [d for d in docs if isinstance(d, dict)]
    docs.sort(key=lambda d: int(d.get("created") or 0), reverse=True)
    docs = _enrich_docs_for_view(store, docs, limit=300)

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

    if not sdoc or not isinstance(sdoc, dict):
        abort(404)

    # Always produce non-empty body
    summary_text = str(sdoc.get("summary") or "").strip()
    if not summary_text:
        keys = ", ".join(sorted(list(sdoc.keys())))
        summary_text = (
            "*(Ingen summary-text hittades i dokumentet.)*\n\n"
            f"- requested id: `{sid}`\n"
            f"- doc id: `{sdoc.get('id')}`\n"
            f"- created: `{sdoc.get('created')}`\n"
            f"- keys: `{keys}`\n"
        )

    doc_id = str(sdoc.get("id") or "").strip()
    created_ts = int(sdoc.get("created") or 0)

    display_title = str(sdoc.get("title") or "").strip() or doc_id or "Sammanfattning"
    topics_label = _topics_label(sdoc)

    body_html = _md_to_html(summary_text)
    header_html = _render_summary_header_html(
        title=display_title, created_ts=created_ts, doc_id=doc_id, topics_label=topics_label
    )
    html = header_html + body_html

    # Pass summary=None so templates_viewer/index.html does NOT render its own <h4>{{ summary.get('id') }}</h4>
    return render_template(
        "index.html",
        summary=None,
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
        return render_template(
            "articles.html",
            articles=[],
            format_ts=format_ts,
            error=f"Kunde inte läsa artiklar: {e}",
        ), 500

    articles: List[Dict[str, Any]] = []
    for a in raw:
        if not isinstance(a, dict):
            continue
        if not a.get("id"):
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

    if a.get("title") is None:
        a["title"] = ""
    if a.get("source") is None:
        a["source"] = ""
    if a.get("url") is None:
        a["url"] = ""
    if a.get("text") is None:
        a["text"] = ""

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