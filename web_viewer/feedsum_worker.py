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
import logging
import os
import time
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from feedsummary_core.summarizer.main import run_pipeline
from uicommon import load_config

from feedsummary_core.persistence import create_store

# CleanupPolicy lives in persistence; path may vary by version.
try:
    from feedsummary_core.persistence import CleanupPolicy  # type: ignore
except Exception:  # pragma: no cover
    try:
        from feedsummary_core.persistence.newsstore import CleanupPolicy  # type: ignore
    except Exception:
        from feedsummary_core.persistence.models import CleanupPolicy  # type: ignore

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

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
    fh = RotatingFileHandler(str(lp), maxBytes=int(max_bytes), backupCount=int(backup_count), encoding="utf-8")
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

    logging.getLogger(__name__).info("Logging initialized: file=%s level=%s", str(lp), logging.getLevelName(level))
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
# Path helpers
# ----------------------------
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
    Resolve store/scheduler paths relative to config.yaml directory
    to avoid CWD-dependent bugs.
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


# ----------------------------
# Scheduler settings + timezone
# ----------------------------
def _scheduler_settings(
    cfg: Dict[str, Any],
    cfg_path: str,
    cli_schedule_path: Optional[str],
    cli_poll: Optional[int],
) -> Tuple[bool, str, int, str]:
    enabled = False
    schedule_path = "schedule.yaml"
    poll_seconds = 20
    timezone = ""  # empty => local

    sc = cfg.get("scheduler")
    if isinstance(sc, dict):
        if "enabled" in sc:
            enabled = bool(sc.get("enabled"))
        if sc.get("path"):
            schedule_path = str(sc.get("path"))
        if sc.get("poll_seconds") is not None:
            try:
                poll_seconds = int(sc.get("poll_seconds"))
            except Exception:
                pass
        if sc.get("timezone"):
            timezone = str(sc.get("timezone")).strip()

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
    env_tz = os.environ.get("FEEDSUMMARY_SCHEDULE_TZ", "").strip()
    if env_tz:
        timezone = env_tz

    if cli_schedule_path:
        schedule_path = cli_schedule_path
    if cli_poll is not None:
        poll_seconds = int(cli_poll)

    base = Path(cfg_path).resolve().parent
    schedule_path_abs = (
        str((base / schedule_path).resolve())
        if not Path(schedule_path).is_absolute()
        else schedule_path
    )
    return enabled, schedule_path_abs, poll_seconds, timezone


def _get_tz(tz_name: str):
    if not tz_name:
        return None
    if ZoneInfo is None:
        log.warning("zoneinfo not available; ignoring scheduler.timezone=%s", tz_name)
        return None
    try:
        return ZoneInfo(tz_name)
    except Exception:
        log.warning("Invalid timezone %r; falling back to local time", tz_name)
        return None


def _now(tz) -> dt.datetime:
    return dt.datetime.now(tz) if tz else dt.datetime.now()


def _parse_hhmm(s: str) -> Tuple[int, int]:
    parts = (s or "").strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time format: {s!r} expected HH:MM")
    return int(parts[0]), int(parts[1])


def _grace_window_seconds(poll_seconds: int) -> int:
    # Generous grace to tolerate load/sleep/jitter
    return max(120, 3 * int(poll_seconds))


def _next_boundary(now: dt.datetime, boundary_hours: List[int]) -> dt.datetime:
    """
    Given fixed hour boundaries (e.g. [0,6,12,18]) return next boundary strictly after 'now'.
    Works with tz-aware or naive datetimes.
    """
    # Normalize to minute=0 boundary
    base = now.replace(minute=0, second=0, microsecond=0)
    for h in boundary_hours:
        cand = base.replace(hour=h)
        if cand > now:
            return cand
    # Next day at first boundary
    next_day = base + dt.timedelta(days=1)
    return next_day.replace(hour=boundary_hours[0])


def _next_run_dt(entry: Dict[str, Any], now: dt.datetime) -> Optional[dt.datetime]:
    """
    Compute next run datetime for an entry based on frequency.
    Supported:
      - daily: uses entry.time HH:MM
      - weekly: uses entry.day + entry.time
      - quarterday: fixed times 00:00,06:00,12:00,18:00 (ignores entry.time)
      - halfday: fixed times 00:00,12:00 (ignores entry.time)
    """
    freq = str(entry.get("frequency") or "").strip().lower()

    if freq == "quarterday":
        return _next_boundary(now, [0, 6, 12, 18])

    if freq == "halfday":
        return _next_boundary(now, [0, 12])

    hhmm = str(entry.get("time") or "").strip()
    if not hhmm:
        return None
    try:
        hh, mm = _parse_hhmm(hhmm)
    except Exception:
        return None

    candidate = now.replace(hour=hh, minute=mm, second=0, microsecond=0)

    if freq == "daily":
        if candidate <= now:
            candidate = candidate + dt.timedelta(days=1)
        return candidate

    if freq == "weekly":
        day = str(entry.get("day") or "").strip().lower()
        wd = WEEKDAY.get(day)
        if wd is None:
            return None
        days_ahead = (wd - now.weekday()) % 7
        candidate = candidate + dt.timedelta(days=days_ahead)
        if candidate <= now:
            candidate = candidate + dt.timedelta(days=7)
        return candidate

    return None


# ----------------------------
# Overrides mapping
# ----------------------------
def _entry_to_overrides(entry: Dict[str, Any]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    freq = str(entry.get("frequency") or "").strip().lower()
    # sensible lookback defaults for these schedules
    if freq == "daily":
        overrides["lookback"] = "1d"
    elif freq == "weekly":
        overrides["lookback"] = "1w"
    elif freq == "quarterday":
        overrides["lookback"] = "6h"
    elif freq == "halfday":
        overrides["lookback"] = "12h"

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


# ----------------------------
# Cleanup policy: call persistence run_cleanup
# ----------------------------
def _cleanup_policy(cfg: Dict[str, Any]) -> "CleanupPolicy":
    d = cfg.get("cleanup") if isinstance(cfg, dict) else None
    d = d if isinstance(d, dict) else {}

    enabled = bool(d.get("enabled", True))
    run_every_minutes = int(d.get("run_every_minutes", 60))

    kwargs = {
        "enabled": enabled,
        "run_every_minutes": run_every_minutes,
        "articles_days": int(d.get("articles_days", 180)),
        "daily_summaries_days": int(d.get("daily_summaries_days", 7)),
        "weekly_summaries_days": int(d.get("weekly_summaries_days", 30)),
        "temp_summaries_days": int(d.get("temp_summaries_days", 30)),
        "jobs_days": int(d.get("jobs_days", 90)),
    }

    try:
        return CleanupPolicy(**kwargs)  # type: ignore[arg-type]
    except TypeError:
        pol = CleanupPolicy()  # type: ignore[call-arg]
        for k, v in kwargs.items():
            if hasattr(pol, k):
                setattr(pol, k, v)
        return pol


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="FeedSummary worker (scheduler + cleanup)"
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--schedule-path", default=None, help="Override schedule.yaml path"
    )
    parser.add_argument("--poll", type=int, default=None, help="Override poll seconds")
    parser.add_argument(
        "--once", action="store_true", help="Run due jobs once and exit"
    )
    parser.add_argument(
        "--cleanup-once", action="store_true", help="Run cleanup once and exit"
    )
    parser.add_argument("--log-file", default=None, help="Write logs to this file (overrides config.yaml worker.log_file)")
    parser.add_argument("--log-level", default=None, help="Log level DEBUG/INFO/WARNING/ERROR (overrides config.yaml worker.log_level)")
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
    enabled, schedule_path, poll_seconds, tz_name = _scheduler_settings(
        cfg, config_path, args.schedule_path, args.poll
    )
    tz = _get_tz(tz_name)
    grace = _grace_window_seconds(poll_seconds)

    # Create store once; run_cleanup lives here.
    store_cfg = cfg.get("store") or {}
    store = create_store(store_cfg)

    # Cleanup policy
    pol = _cleanup_policy(cfg)

    if args.cleanup_once:
        if not getattr(pol, "enabled", True):
            log.info("cleanup disabled (cleanup.enabled=false).")
            return 0
        store.run_cleanup(pol)  # type: ignore[attr-defined]
        log.info("cleanup: run_cleanup executed (cleanup-once).")
        return 0

    if not enabled:
        log.info(
            "Scheduler disabled. Set scheduler.enabled in config.yaml or FEEDSUMMARY_SCHEDULE=1."
        )
        if not getattr(pol, "enabled", True):
            return 0
        log.info("Scheduler disabled but cleanup enabled; running cleanup loop only.")

    log.info(
        "Worker started. tz=%s schedule=%s poll=%ss grace=%ss cleanup_enabled=%s cleanup_every=%smin",
        tz_name or "local",
        schedule_path,
        poll_seconds,
        grace,
        getattr(pol, "enabled", True),
        int(getattr(pol, "run_every_minutes", 60)),
    )

    # next run schedule (robust)
    next_runs: Dict[str, dt.datetime] = {}
    last_run_stamp: Dict[str, str] = {}  # de-dupe per minute

    # cleanup scheduling
    last_cleanup_ts: float = 0.0

    # initial plan
    if enabled:
        sched0 = _read_schedule_yaml(schedule_path)
        now0 = _now(tz)
        for name, entry in sched0.items():
            if isinstance(entry, dict):
                nr = _next_run_dt(entry, now0)
                if nr:
                    next_runs[str(name)] = nr
        if next_runs:
            log.info(
                "Initial next runs: %s",
                {k: v.isoformat() for k, v in next_runs.items()},
            )

    while True:
        now = _now(tz)

        # ---- cleanup loop ----
        if getattr(pol, "enabled", True):
            every = max(1, int(getattr(pol, "run_every_minutes", 60))) * 60
            if (time.time() - last_cleanup_ts) >= every:
                try:
                    store.run_cleanup(pol)  # type: ignore[attr-defined]
                    log.info("cleanup: run_cleanup executed")
                except Exception as e:
                    log.exception("cleanup: run_cleanup failed: %s", e)
                last_cleanup_ts = time.time()

        # ---- scheduling loop ----
        if enabled:
            schedule = _read_schedule_yaml(schedule_path)

            # Ensure next_runs exists for all jobs (handles schedule.yaml edits)
            for name, entry in schedule.items():
                if not isinstance(entry, dict):
                    continue
                jn = str(name)
                if jn not in next_runs:
                    nr = _next_run_dt(entry, now)
                    if nr:
                        next_runs[jn] = nr

            # Run due jobs
            for name, entry in schedule.items():
                if not isinstance(entry, dict):
                    continue
                jn = str(name)

                nr = next_runs.get(jn)
                if not nr:
                    nr = _next_run_dt(entry, now)
                    if nr:
                        next_runs[jn] = nr
                    else:
                        continue

                if now < nr:
                    continue

                late_by = (now - nr).total_seconds()
                if late_by > grace:
                    log.warning(
                        "Job '%s' late by %.1fs (grace=%ss) - running anyway",
                        jn,
                        late_by,
                        grace,
                    )

                stamp = now.strftime("%Y%m%d%H%M")
                if last_run_stamp.get(jn) == stamp:
                    continue

                try:
                    asyncio.run(_run_one(config_path, cfg, jn, entry))
                except Exception as e:
                    log.exception("Job '%s' FAILED: %s", jn, e)

                last_run_stamp[jn] = stamp

                # schedule next run after execution
                nn = _next_run_dt(entry, now + dt.timedelta(seconds=1))
                if nn:
                    next_runs[jn] = nn

        if args.once:
            log.info("--once: exiting after one scan.")
            return 0

        time.sleep(max(1, int(poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
