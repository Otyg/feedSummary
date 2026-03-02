# LICENSE HEADER MANAGED BY add-license-header
#
# BSD 3-Clause License
#
# Copyright (c) 2026, Martin Vesterlund
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import argparse
import asyncio
import datetime as dt
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys
from logging.handlers import RotatingFileHandler

import yaml
from feedsummary_core.summarizer.main import run_pipeline
from uicommon import load_config

log = logging.getLogger(__name__)


class _StreamToLogger:
    """
    File-like object that redirects writes to a logger.
    Captures prints and libraries writing to stdout/stderr.
    """

    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self._buf = ""

    def write(self, message: str) -> int:
        if not message:
            return 0
        self._buf += message
        # Flush on newline so we get sane log lines
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip()
            if line:
                self.logger.log(self.level, line)
        return len(message)

    def flush(self) -> None:
        line = (self._buf or "").strip()
        if line:
            self.logger.log(self.level, line)
        self._buf = ""


def _get_worker_log_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    config.yaml:
      worker:
        log_file: "./logs/feedsum_worker.log"
        log_level: "INFO"
        log_max_bytes: 10485760
        log_backup_count: 5
    """
    out = {
        "log_file": "./logs/feedsum_worker.log",
        "log_level": "INFO",
        "log_max_bytes": 10 * 1024 * 1024,
        "log_backup_count": 5,
    }
    w = cfg.get("worker")
    if isinstance(w, dict):
        if w.get("log_file"):
            out["log_file"] = str(w.get("log_file"))
        if w.get("log_level"):
            out["log_level"] = str(w.get("log_level"))
        if w.get("log_max_bytes") is not None:
            try:
                out["log_max_bytes"] = int(w.get("log_max_bytes"))
            except Exception:
                pass
        if w.get("log_backup_count") is not None:
            try:
                out["log_backup_count"] = int(w.get("log_backup_count"))
            except Exception:
                pass
    return out


def _setup_file_logging(
    *,
    log_file: str,
    log_level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """
    Configure root logger to write to rotating file.
    Redirect stdout/stderr into logging so prints also go to file.
    """
    lp = Path(os.path.expandvars(os.path.expanduser(log_file)))
    if not lp.is_absolute():
        lp = (Path.cwd() / lp).resolve()
    lp.parent.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, str(log_level).upper().strip(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any existing handlers to avoid duplicates
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh = RotatingFileHandler(
        str(lp),
        maxBytes=int(max_bytes),
        backupCount=int(backup_count),
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Capture warnings as logs
    logging.captureWarnings(True)

    # Redirect stdout/stderr into logging
    sys.stdout = _StreamToLogger(logging.getLogger("stdout"), logging.INFO)  # type: ignore
    sys.stderr = _StreamToLogger(logging.getLogger("stderr"), logging.ERROR)  # type: ignore

    logging.getLogger(__name__).propagate = True
    logging.getLogger(__name__).setLevel(level)

    logging.getLogger(__name__).info(
        "Logging initialized: file=%s level=%s", str(lp), logging.getLevelName(level)
    )


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


# ----------------------------
# Path + config helpers
# ----------------------------
def _resolve_path(base: Path, p: str) -> str:
    pp = Path(os.path.expandvars(os.path.expanduser(p)))
    if pp.is_absolute():
        return str(pp)
    return str((base / pp).resolve())


def _resolve_config_path(cli_path: Optional[str]) -> str:
    if cli_path:
        return str(Path(cli_path).expanduser().resolve())

    env = os.environ.get("FEEDSUMMARY_CONFIG", "").strip()
    if env:
        return str(Path(env).expanduser().resolve())

    p = (Path.cwd() / "config.yaml").resolve()
    if p.exists():
        return str(p)

    pd = (Path.cwd() / "config.yaml.dist").resolve()
    return str(pd) if pd.exists() else str(p)


def _abspath_cfg_paths(cfg: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    """
    Resolve selected cfg paths relative to the directory of config.yaml so
    running from a different CWD doesn't accidentally use a different DB / schedule.
    """
    base = Path(config_path).resolve().parent

    def abs_if_rel(p: str) -> str:
        pp = Path(os.path.expandvars(os.path.expanduser(p)))
        if pp.is_absolute():
            return str(pp)
        return str((base / pp).resolve())

    cfg2 = dict(cfg)

    st = cfg2.get("store")
    if isinstance(st, dict) and st.get("path"):
        st2 = dict(st)
        st2["path"] = abs_if_rel(str(st2["path"]))
        cfg2["store"] = st2

    sch = cfg2.get("scheduler")
    if isinstance(sch, dict) and sch.get("path"):
        sch2 = dict(sch)
        sch2["path"] = abs_if_rel(str(sch2["path"]))
        cfg2["scheduler"] = sch2

    return cfg2


def _read_schedule_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        log.warning("Schedule file not found: %s", p)
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _scheduler_settings(
    cfg: Dict[str, Any],
    cfg_path: str,
    cli_schedule_path: Optional[str],
    cli_poll: Optional[int],
) -> Tuple[bool, str, int]:
    enabled = False
    schedule_path = "schedule.yaml"
    poll_seconds = 20

    sc = cfg.get("scheduler")
    if isinstance(sc, dict):
        if "enabled" in sc:
            enabled = bool(sc.get("enabled"))
        if sc.get("path"):
            schedule_path = str(sc.get("path"))
        if sc.get("poll_seconds") is not None:
            try:
                poll_seconds = int(sc.get("poll_seconds"))  # type: ignore
            except Exception:
                pass

    env_enabled = os.environ.get("FEEDSUMMARY_SCHEDULE", "").strip()
    if env_enabled:
        enabled = env_enabled == "1" or env_enabled.lower() in ("true", "yes", "on")
    env_path = os.environ.get("FEEDSUMMARY_SCHEDULE_PATH", "").strip()
    if env_path:
        schedule_path = env_path
    env_poll = os.environ.get("FEEDSUMMARY_SCHEDULE_POLL", "").strip()
    if env_poll:
        try:
            poll_seconds = int(env_poll)
        except Exception:
            pass

    if cli_schedule_path:
        schedule_path = cli_schedule_path
    if cli_poll is not None:
        poll_seconds = int(cli_poll)

    base = Path(cfg_path).resolve().parent
    schedule_path_abs = _resolve_path(base, schedule_path)
    return enabled, schedule_path_abs, poll_seconds


# ----------------------------
# Cleanup/TTL settings
# ----------------------------
@dataclass(frozen=True)
class CleanupPolicy:
    enabled: bool = True
    run_every_minutes: int = 60

    articles_days: int = 180
    daily_summaries_days: int = 7
    weekly_summaries_days: int = 30
    temp_summaries_days: int = 30
    jobs_days: int = 90


def _cleanup_policy(cfg: Dict[str, Any]) -> CleanupPolicy:
    """
    Reads config.yaml:
      cleanup:
        enabled: true
        run_every_minutes: 60
        articles_days: 180
        daily_summaries_days: 7
        weekly_summaries_days: 30
        temp_summaries_days: 30
        jobs_days: 90
    Defaults match your rules.
    """
    d = cfg.get("cleanup")
    if not isinstance(d, dict):
        return CleanupPolicy()

    def geti(key: str, default: int) -> int:
        try:
            v = d.get(key, default)
            return int(v)
        except Exception:
            return default

    enabled = bool(d.get("enabled", True))
    return CleanupPolicy(
        enabled=enabled,
        run_every_minutes=geti("run_every_minutes", 60),
        articles_days=geti("articles_days", 180),
        daily_summaries_days=geti("daily_summaries_days", 7),
        weekly_summaries_days=geti("weekly_summaries_days", 30),
        temp_summaries_days=geti("temp_summaries_days", 30),
        jobs_days=geti("jobs_days", 90),
    )


def _classify_summary_doc(doc_json: str) -> str:
    """
    Returns: "daily" | "weekly" | "other"
    Uses selection.prompt_package when available.
    """
    try:
        doc = json.loads(doc_json) if doc_json else {}
    except Exception:
        doc = {}

    sel = doc.get("selection") if isinstance(doc, dict) else None
    pkg = ""
    if isinstance(sel, dict):
        pkg = str(sel.get("prompt_package") or "").lower().strip()

    # Heuristic: your packages include daily_* and weekly_*
    if "weekly" in pkg:
        return "weekly"
    if "daily" in pkg:
        return "daily"
    return "other"


def _run_cleanup_sqlite(db_path: str, pol: CleanupPolicy) -> Dict[str, int]:
    """
    Cleanup for SqliteStore schema (articles, summary_docs, temp_summaries, jobs).
    summary_docs: delete based on created + prompt_package classification.
    """
    now = int(time.time())
    cut_articles = now - pol.articles_days * 86400
    cut_daily = now - pol.daily_summaries_days * 86400
    cut_weekly = now - pol.weekly_summaries_days * 86400
    cut_temp = now - pol.temp_summaries_days * 86400
    cut_jobs = now - pol.jobs_days * 86400

    removed = {"articles": 0, "summary_docs": 0, "temp_summaries": 0, "jobs": 0}

    con = sqlite3.connect(db_path)
    try:
        con.row_factory = sqlite3.Row

        # Articles
        cur = con.execute(
            "DELETE FROM articles WHERE COALESCE(published_ts, fetched_at, 0) < ?",
            (cut_articles,),
        )
        removed["articles"] = cur.rowcount if cur.rowcount is not None else 0

        # Temp summaries
        cur = con.execute(
            "DELETE FROM temp_summaries WHERE COALESCE(created_at, 0) < ?",
            (cut_temp,),
        )
        removed["temp_summaries"] = cur.rowcount if cur.rowcount is not None else 0

        # Jobs: only delete old finished ones (never touch running/queued)
        cur = con.execute(
            """
            DELETE FROM jobs
            WHERE COALESCE(finished_at, created_at, 0) < ?
              AND COALESCE(status, '') IN ('done','failed')
            """,
            (cut_jobs,),
        )
        removed["jobs"] = cur.rowcount if cur.rowcount is not None else 0

        # Summary docs: fetch candidates up to max cutoff, classify in Python, delete by id
        max_cut = max(cut_daily, cut_weekly)
        rows = con.execute(
            "SELECT id, created, doc_json FROM summary_docs WHERE COALESCE(created,0) < ?",
            (max_cut,),
        ).fetchall()

        to_delete: List[str] = []
        for r in rows:
            sid = str(r["id"])
            created = int(r["created"] or 0)
            kind = _classify_summary_doc(str(r["doc_json"] or ""))

            if kind == "daily" and created < cut_daily:
                to_delete.append(sid)
            elif kind == "weekly" and created < cut_weekly:
                to_delete.append(sid)
            elif kind == "other" and created < cut_weekly:
                # unknown => keep like weekly by default
                to_delete.append(sid)

        if to_delete:
            # chunk deletes
            chunk = 200
            for i in range(0, len(to_delete), chunk):
                part = to_delete[i : i + chunk]
                placeholders = ",".join(["?"] * len(part))
                cur = con.execute(
                    f"DELETE FROM summary_docs WHERE id IN ({placeholders})",
                    tuple(part),
                )
                removed["summary_docs"] += (
                    cur.rowcount if cur.rowcount is not None else 0
                )

        con.commit()
        return removed
    finally:
        con.close()


def _run_cleanup_tinydb(db_path: str, pol: CleanupPolicy) -> Dict[str, int]:
    """
    Cleanup for TinyDB schema:
      tables: articles, summary_docs, temp_summaries, jobs
    """
    now = int(time.time())
    cut_articles = now - pol.articles_days * 86400
    cut_daily = now - pol.daily_summaries_days * 86400
    cut_weekly = now - pol.weekly_summaries_days * 86400
    cut_temp = now - pol.temp_summaries_days * 86400
    cut_jobs = now - pol.jobs_days * 86400

    removed = {"articles": 0, "summary_docs": 0, "temp_summaries": 0, "jobs": 0}

    # TinyDB import here to avoid dependency if user doesn't use it
    from tinydb import TinyDB

    db = TinyDB(db_path)
    try:
        # Articles
        at = db.table("articles")
        # remove uses a predicate for each row
        before = len(at)
        at.remove(
            lambda r: (
                int(r.get("published_ts") or r.get("fetched_at") or 0) < cut_articles
            )
        )
        removed["articles"] = max(0, before - len(at))

        # Temp summaries
        tt = db.table("temp_summaries")
        before = len(tt)
        tt.remove(lambda r: int(r.get("created_at") or 0) < cut_temp)
        removed["temp_summaries"] = max(0, before - len(tt))

        # Jobs (only done/failed)
        jt = db.table("jobs")
        before = len(jt)

        def job_old_finished(r: Dict[str, Any]) -> bool:
            ts = int(r.get("finished_at") or r.get("created_at") or 0)
            st = str(r.get("status") or "")
            return ts < cut_jobs and st in ("done", "failed")

        jt.remove(job_old_finished)
        removed["jobs"] = max(0, before - len(jt))

        # Summary docs
        sd = db.table("summary_docs")
        before = len(sd)

        def sum_should_remove(r: Dict[str, Any]) -> bool:
            created = int(r.get("created") or 0)
            # we need prompt_package; in tinydb it is stored inside the doc itself
            pkg = ""
            sel = r.get("selection")
            if isinstance(sel, dict):
                pkg = str(sel.get("prompt_package") or "").lower().strip()
            kind = "other"
            if "weekly" in pkg:
                kind = "weekly"
            elif "daily" in pkg:
                kind = "daily"

            if kind == "daily":
                return created < cut_daily
            if kind == "weekly":
                return created < cut_weekly
            return created < cut_weekly

        sd.remove(sum_should_remove)
        removed["summary_docs"] = max(0, before - len(sd))

        return removed
    finally:
        db.close()


def _run_cleanup(cfg: Dict[str, Any], cfg_path: str, pol: CleanupPolicy) -> None:
    if not pol.enabled:
        return

    store = cfg.get("store") or {}
    if not isinstance(store, dict):
        log.warning("cleanup: cfg.store is not a dict, skipping.")
        return

    provider = str(store.get("provider") or "tinydb").lower()
    db_path = str(store.get("path") or "").strip()
    if not db_path:
        log.warning("cleanup: no store.path configured, skipping.")
        return

    # store.path should already be made absolute by _abspath_cfg_paths; do it again defensively
    base = Path(cfg_path).resolve().parent
    db_path = _resolve_path(base, db_path)

    try:
        if provider in ("sqlite", "sqlite3"):
            removed = _run_cleanup_sqlite(db_path, pol)
        else:
            removed = _run_cleanup_tinydb(db_path, pol)

        log.info(
            "cleanup: removed articles=%s summaries=%s temp_summaries=%s jobs=%s",
            removed.get("articles", 0),
            removed.get("summary_docs", 0),
            removed.get("temp_summaries", 0),
            removed.get("jobs", 0),
        )
    except Exception as e:
        log.exception("cleanup: failed: %s", e)


# ----------------------------
# Scheduler logic (run jobs)
# ----------------------------
def _parse_hhmm(s: str) -> Tuple[int, int]:
    parts = (s or "").strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time format: {s!r} expected HH:MM")
    return int(parts[0]), int(parts[1])


def _is_due(entry: Dict[str, Any], now: dt.datetime) -> bool:
    freq = str(entry.get("frequency") or "").strip().lower()
    hhmm = str(entry.get("time") or "").strip()
    if not hhmm:
        return False
    try:
        hh, mm = _parse_hhmm(hhmm)
    except Exception:
        return False

    if now.hour != hh or now.minute != mm:
        return False

    if freq == "weekly":
        day = str(entry.get("day") or "").strip().lower()
        wd = WEEKDAY.get(day)
        if wd is None:
            return False
        if now.weekday() != wd:
            return False

    return freq in ("daily", "weekly")


def _entry_to_overrides(entry: Dict[str, Any]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    freq = str(entry.get("frequency") or "").strip().lower()
    if freq == "daily":
        overrides["lookback"] = "1d"
    elif freq == "weekly":
        overrides["lookback"] = "1w"

    cats = entry.get("categories") or []
    if isinstance(cats, list):
        topics = [str(x).strip() for x in cats if str(x).strip()]
        if topics:
            overrides["topics"] = topics

    pp = str(entry.get("promptpackage") or "").strip()
    if pp:
        overrides["prompt_package"] = pp

    return overrides


async def _run_one(
    config_path: str, cfg: Dict[str, Any], job_name: str, entry: Dict[str, Any]
) -> None:
    overrides = _entry_to_overrides(entry)
    log.info("Running job '%s' overrides=%s", job_name, overrides)
    summary_id = await run_pipeline(
        config_path,
        job_id=None,
        overrides=overrides,
        config_dict=cfg,
    )
    log.info("Job '%s' OK summary_id=%s", job_name, summary_id)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FeedSummary worker (scheduler + cleanup/TTL)"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml (default: FEEDSUMMARY_CONFIG or ./config.yaml)",
    )
    parser.add_argument(
        "--schedule-path", default=None, help="Override schedule.yaml path"
    )
    parser.add_argument(
        "--poll", type=int, default=None, help="Override scheduler poll seconds"
    )
    parser.add_argument(
        "--once", action="store_true", help="Run due jobs once and exit"
    )
    parser.add_argument(
        "--cleanup-once", action="store_true", help="Run cleanup once and exit"
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Write logs to this file (overrides config.yaml worker.log_file)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Log level DEBUG/INFO/WARNING/ERROR (overrides config.yaml worker.log_level)",
    )
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    cfg_raw = load_config(config_path)
    cfg = _abspath_cfg_paths(cfg_raw, config_path)
    # Setup file logging + capture stdout/stderr early
    wlog = _get_worker_log_cfg(cfg)
    if args.log_file:
        wlog["log_file"] = args.log_file
    if args.log_level:
        wlog["log_level"] = args.log_level

    _setup_file_logging(
        log_file=str(wlog["log_file"]),
        log_level=str(wlog["log_level"]),
        max_bytes=int(wlog["log_max_bytes"]),
        backup_count=int(wlog["log_backup_count"]),
    )
    enabled, schedule_path, poll_seconds = _scheduler_settings(
        cfg, config_path, args.schedule_path, args.poll
    )
    pol = _cleanup_policy(cfg)

    if args.cleanup_once:
        if not pol.enabled:
            log.info("cleanup disabled (cleanup.enabled=false).")
            return 0
        _run_cleanup(cfg, config_path, pol)
        return 0

    if not enabled:
        log.info(
            "Scheduler disabled. Set scheduler.enabled in config.yaml or FEEDSUMMARY_SCHEDULE=1."
        )
        # Still allow cleanup loop if enabled
        if not pol.enabled:
            return 0
        log.info("Scheduler disabled but cleanup enabled; running cleanup loop only.")

    log.info(
        "Worker started. config=%s schedule=%s poll=%ss cleanup_enabled=%s cleanup_every=%smin",
        config_path,
        schedule_path,
        poll_seconds,
        pol.enabled,
        pol.run_every_minutes,
    )

    last_run_minute: Dict[str, str] = {}
    last_cleanup_ts: float = 0.0

    while True:
        now = dt.datetime.now()
        stamp = now.strftime("%Y%m%d%H%M")

        # ---- cleanup loop ----
        if pol.enabled:
            every = max(1, int(pol.run_every_minutes)) * 60
            if (time.time() - last_cleanup_ts) >= every:
                _run_cleanup(cfg, config_path, pol)
                last_cleanup_ts = time.time()

        # ---- job scheduling loop ----
        if enabled:
            schedule = _read_schedule_yaml(schedule_path)
            for name, entry in schedule.items():
                if not isinstance(entry, dict):
                    continue
                if not _is_due(entry, now):
                    continue
                if last_run_minute.get(name) == stamp:
                    continue

                try:
                    asyncio.run(_run_one(config_path, cfg, str(name), entry))
                except Exception as e:
                    log.exception("Job '%s' FAILED: %s", name, e)

                last_run_minute[name] = stamp

        if args.once:
            log.info("--once: exiting after one scan.")
            return 0

        time.sleep(max(1, int(poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
