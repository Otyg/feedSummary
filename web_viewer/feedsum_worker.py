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
from collections import deque
import datetime as dt
import inspect
import json
import logging
import os
import time
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from feedsummary_core.summarizer.main import (
    _build_composed_summary_text,
    _strip_sources_appendix_from_summary,
    run_pipeline,
    run_resume_job,
    compose_summary_docs,
)
from uicommon import load_config
from uicommon.proofread_rounds import enable_configurable_proofread_rounds

from feedsummary_core.persistence import create_store
from feedsummary_core.llm_client import create_llm_client

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
enable_configurable_proofread_rounds(logger=log)

TRIGGERS: Dict[str, Dict[str, Any]] = {}
TRIGGERS_LOCK = threading.Lock()
global RUNNING_JOB_ID
global RUNNING_JOB_LOCK
RUNNING_JOB_ID: Optional[int] = None
RUNNING_JOB_LOCK = threading.Lock()
WORKER_LOG_FILE: Optional[Path] = None


def _supports_composed_proofread() -> bool:
    try:
        sig = inspect.signature(compose_summary_docs)
    except (TypeError, ValueError):
        return False
    return "proofread_package" in sig.parameters


class _AsyncRunner:
    """Runs all coroutines on one dedicated event loop thread."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready.wait()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def run(self, coro):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop.close()


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
    global WORKER_LOG_FILE
    lp = Path(os.path.expandvars(os.path.expanduser(log_file)))
    if not lp.is_absolute():
        lp = (Path.cwd() / lp).resolve()
    lp.parent.mkdir(parents=True, exist_ok=True)
    WORKER_LOG_FILE = lp

    level = getattr(logging, str(log_level).upper().strip(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

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

    logging.captureWarnings(True)

    sys.stdout = _StreamToLogger(logging.getLogger("stdout"), logging.INFO)  # type: ignore
    sys.stderr = _StreamToLogger(logging.getLogger("stderr"), logging.ERROR)  # type: ignore

    logging.getLogger(__name__).propagate = True
    logging.getLogger(__name__).setLevel(level)

    logging.getLogger(__name__).info(
        "Logging initialized: file=%s level=%s", str(lp), logging.getLevelName(level)
    )


def _tail_worker_log_lines(limit: int = 5) -> List[str]:
    lp = WORKER_LOG_FILE
    if lp is None or limit <= 0:
        return []
    try:
        with lp.open("r", encoding="utf-8", errors="replace") as fh:
            return [
                line.rstrip("\r\n") for line in deque(fh, maxlen=limit) if line.strip()
            ]
    except FileNotFoundError:
        return []
    except Exception:
        log.exception("Failed to read worker log tail from %s", lp)
        return []


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
    cfg2 = dict(cfg)
    base_dir = os.path.dirname(os.path.abspath(config_path)) or "."

    def abs_if_rel(p: str) -> str:
        p2 = os.path.expanduser(os.path.expandvars(p))
        if os.path.isabs(p2):
            return p2
        return os.path.join(base_dir, p2)

    st = cfg2.get("store")
    if isinstance(st, dict) and st.get("path"):
        st2 = dict(st)
        st2["path"] = abs_if_rel(str(st2["path"]))
        cfg2["store"] = st2

    pr = cfg2.get("prompts")
    if isinstance(pr, dict) and pr.get("path"):
        pr2 = dict(pr)
        pr2["path"] = abs_if_rel(str(pr2["path"]))
        cfg2["prompts"] = pr2

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
) -> Tuple[bool, str, int, str]:
    enabled = False
    schedule_path = "schedule.yaml"
    poll_seconds = 20
    timezone = ""

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
    return max(120, 3 * int(poll_seconds))


def _next_boundary(now: dt.datetime, boundary_hours: List[int]) -> dt.datetime:
    base = now.replace(minute=0, second=0, microsecond=0)
    for h in boundary_hours:
        cand = base.replace(hour=h)
        if cand > now:
            return cand
    next_day = base + dt.timedelta(days=1)
    return next_day.replace(hour=boundary_hours[0])


def _next_run_dt(entry: Dict[str, Any], now: dt.datetime) -> Optional[dt.datetime]:
    freq = str(entry.get("frequency") or "").strip().lower()

    if freq == "triggered":
        return None

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


def _entry_to_overrides(entry: Dict[str, Any]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    lb = str(entry.get("lookback") or "").strip()
    if lb:
        overrides["lookback"] = lb
    else:
        freq = str(entry.get("frequency") or "").strip().lower()
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

    tags = entry.get("tags") or []
    if isinstance(tags, list):
        clean_tags = [str(x).strip() for x in tags if str(x).strip()]
        if clean_tags:
            overrides["tags"] = clean_tags

    pp = str(entry.get("promptpackage") or "").strip()
    if pp:
        overrides["prompt_package"] = pp

    return overrides


def _parse_contents_block(
    entry: Dict[str, Any],
) -> Tuple[List[Dict[str, str]], Optional[str], Optional[str], Optional[str]]:
    contents = entry.get("contents") or []
    if not isinstance(contents, list) or not contents:
        return [], None, None, None

    jobs: List[Dict[str, str]] = []
    proofread_pkg: Optional[str] = None
    title_pkg: Optional[str] = None
    ingress_pkg: Optional[str] = None

    for i, item in enumerate(contents):
        if not isinstance(item, dict):
            raise ValueError(f"contents[{i}] måste vara ett objekt")

        sched = str(item.get("schedule") or "").strip()
        proofread = str(item.get("proofread") or "").strip()
        title = str(item.get("title") or "").strip()
        ingress = str(item.get("ingress") or "").strip()

        if sched:
            jobs.append({"schedule": sched})
            continue

        if proofread:
            if proofread_pkg is not None:
                raise ValueError("Bara en proofread-post får finnas i contents")
            proofread_pkg = proofread
            continue

        if title:
            if title_pkg is not None:
                raise ValueError("Bara en title-post får finnas i contents")
            title_pkg = title
            continue

        if ingress:
            if ingress_pkg is not None:
                raise ValueError("Bara en ingress-post får finnas i contents")
            ingress_pkg = ingress
            continue

        raise ValueError(
            f"contents[{i}] måste innehålla schedule, proofread, title eller ingress"
        )

    if not jobs:
        raise ValueError("contents måste innehålla minst ett schedule-jobb")

    return jobs, proofread_pkg, title_pkg, ingress_pkg


def _store_composed_proofread_original(
    *,
    store,
    final_summary_id: str,
    job_name: str,
    proofread_package: str,
) -> None:
    """
    Persist pre-proofread composed text in the same summary_doc, so original vs
    published can be compared later.
    """
    sid = str(final_summary_id or "").strip()
    if not sid:
        return

    final_doc = store.get_summary_doc(sid)
    if not isinstance(final_doc, dict):
        return

    sections = final_doc.get("sections") or []
    if not isinstance(sections, list) or not sections:
        return

    loaded_sections: List[Dict[str, Any]] = []
    for s in sections:
        if not isinstance(s, dict):
            continue
        sec_id = str(s.get("summary_id") or "").strip()
        if not sec_id:
            continue
        sec_doc = store.get_summary_doc(sec_id)
        if not isinstance(sec_doc, dict):
            continue
        sec_summary = _strip_sources_appendix_from_summary(
            str(sec_doc.get("summary") or "")
        )
        heading = (
            str(s.get("tag") or "").strip()
            or str(s.get("schedule") or "").strip()
            or str(s.get("promptpackage") or "").strip()
        )
        loaded_sections.append({"tag": heading, "summary": sec_summary})

    if not loaded_sections:
        return

    original_composed = _build_composed_summary_text(
        sections=loaded_sections,
        ingress=None,
    )

    published_summary = str(final_doc.get("summary") or "")
    published_wo_sources = _strip_sources_appendix_from_summary(published_summary)
    now_ts = int(time.time())

    final_doc["proofread_original_summary"] = str(original_composed or "")
    final_doc["proofread_published_summary"] = published_summary
    final_doc["proofread_revised_summary"] = str(published_wo_sources or "")

    meta = final_doc.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    meta["proofread_original_summary"] = str(original_composed or "")
    final_doc["meta"] = meta

    audit_entry = {
        "created_at": now_ts,
        "job_name": str(job_name or ""),
        "proofread_package": str(proofread_package or ""),
        "original_summary": str(original_composed or ""),
        "revised_summary": str(published_wo_sources or ""),
        "published_summary": published_summary,
    }
    pa = final_doc.get("proofread_audit") or {}
    if not isinstance(pa, dict):
        pa = {}
    history = pa.get("history") or []
    if not isinstance(history, list):
        history = []
    history.append(audit_entry)
    history = history[-20:]
    final_doc["proofread_audit"] = {"latest": audit_entry, "history": history}

    store.save_summary_doc(final_doc)


async def _run_regular_entry(
    config_path: str,
    cfg: Dict[str, Any],
    store,
    job_name: str,
    entry: Dict[str, Any],
) -> str:
    global RUNNING_JOB_ID

    overrides = _entry_to_overrides(entry)
    job_id = None
    try:
        job_id = store.create_job()

        with RUNNING_JOB_LOCK:
            RUNNING_JOB_ID = int(job_id)

        log.info(
            "Running job '%s' (id: %s) overrides=%s", job_name, str(job_id), overrides
        )

        summary_id = await run_pipeline(
            config_path,
            job_id=job_id,
            overrides=overrides,
            config_dict=cfg,
        )

        log.info("Job '%s' OK summary_id=%s", job_name, summary_id)
        return str(summary_id)

    except Exception:
        log.exception("Job '%s' failed", job_name)
        raise

    finally:
        with RUNNING_JOB_LOCK:
            RUNNING_JOB_ID = None


async def _run_composed_entry(
    config_path: str,
    cfg: Dict[str, Any],
    store,
    schedule_path: str,
    job_name: str,
    entry: Dict[str, Any],
) -> str:
    global RUNNING_JOB_ID

    jobs, proofread_pkg, title_pkg, ingress_pkg = _parse_contents_block(entry)
    schedule = _read_schedule_yaml(schedule_path)

    parent_job_id = store.create_job()

    try:
        with RUNNING_JOB_LOCK:
            RUNNING_JOB_ID = int(parent_job_id)

        store.update_job(
            parent_job_id,
            status="running",
            started_at=int(time.time()),
            message=f"Startar composed-jobb '{job_name}'...",
        )

        section_results: List[Dict[str, str]] = []

        for idx, spec in enumerate(jobs, start=1):
            child_name = str(spec["schedule"]).strip()
            child_entry = schedule.get(child_name)
            if not isinstance(child_entry, dict):
                raise KeyError(f"Schedule '{child_name}' hittades inte")

            freq = str(child_entry.get("frequency") or "").strip().lower()
            if freq != "triggered":
                raise ValueError(
                    f"Schedule '{child_name}' måste ha frequency: triggered"
                )

            store.update_job(
                parent_job_id,
                message=f"Kör deljobb {idx}/{len(jobs)}: {child_name}...",
            )

            summary_id = await _run_regular_entry(
                config_path=config_path,
                cfg=cfg,
                store=store,
                job_name=f"{job_name}::{child_name}",
                entry=child_entry,
            )

            child_pp = str(child_entry.get("promptpackage") or "").strip()

            section_results.append(
                {
                    "schedule": child_name,
                    "promptpackage": child_pp,
                    "summary_id": str(summary_id),
                }
            )

        store.update_job(
            parent_job_id,
            message="Sammanfogar delresultat...",
        )

        if proofread_pkg and not _supports_composed_proofread():
            raise RuntimeError(
                "Installerad feedsummary_core saknar stöd för proofread_package i "
                "compose_summary_docs. Uppgradera till minst 1.10.0."
            )

        llm = create_llm_client(cfg)
        final_summary_id = await compose_summary_docs(
            config=cfg,
            store=store,
            llm=llm,
            job_id=parent_job_id,
            name=job_name,
            sections=section_results,
            proofread_package=proofread_pkg,
            ingress_package=ingress_pkg,
            title_package=title_pkg,
        )

        if proofread_pkg:
            try:
                _store_composed_proofread_original(
                    store=store,
                    final_summary_id=str(final_summary_id),
                    job_name=job_name,
                    proofread_package=proofread_pkg,
                )
                log.info(
                    "Stored original composed summary for proofread-composed doc: %s",
                    str(final_summary_id),
                )
            except Exception:
                log.exception(
                    "Failed to store original composed summary for summary_id=%s",
                    str(final_summary_id),
                )

        store.update_job(
            parent_job_id,
            status="done",
            finished_at=int(time.time()),
            message=f"Klart: composed-jobb '{job_name}' färdigt.",
            summary_id=str(final_summary_id),
        )
        return str(final_summary_id)

    except Exception as e:
        try:
            store.update_job(
                parent_job_id,
                status="failed",
                finished_at=int(time.time()),
                message=f"Fel: {e}",
            )
        except Exception:
            pass
        log.exception("Composed job '%s' failed", job_name)
        raise

    finally:
        with RUNNING_JOB_LOCK:
            RUNNING_JOB_ID = None


async def _run_one(
    config_path: str,
    cfg: Dict[str, Any],
    store,
    schedule_path: str,
    job_name: str,
    entry: Dict[str, Any],
) -> str:
    contents = entry.get("contents")
    if isinstance(contents, list) and contents:
        return await _run_composed_entry(
            config_path=config_path,
            cfg=cfg,
            store=store,
            schedule_path=schedule_path,
            job_name=job_name,
            entry=entry,
        )

    return await _run_regular_entry(
        config_path=config_path,
        cfg=cfg,
        store=store,
        job_name=job_name,
        entry=entry,
    )


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


def _worker_api_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    d = cfg.get("worker_api")
    if not isinstance(d, dict):
        d = {}
    return {
        "enabled": bool(d.get("enabled", True)),
        "host": str(d.get("host") or "127.0.0.1"),
        "port": int(d.get("port") or 8799),
    }


def _trigger_create(
    kind: str, name: str, overrides: Dict[str, Any], job_id: Optional[int] = None
) -> Dict[str, Any]:
    tid = f"{kind}_{uuid.uuid4().hex}"
    now = int(time.time())
    obj = {
        "id": tid,
        "kind": kind,
        "name": name,
        "job_id": job_id,
        "created_at": now,
        "started_at": None,
        "finished_at": None,
        "status": "queued",
        "overrides": overrides,
        "summary_id": None,
        "error": None,
    }
    with TRIGGERS_LOCK:
        TRIGGERS[tid] = obj
    return obj


def _trigger_update(tid: str, **fields: Any) -> None:
    with TRIGGERS_LOCK:
        if tid in TRIGGERS:
            TRIGGERS[tid].update(fields)


def _trigger_get(tid: str) -> Optional[Dict[str, Any]]:
    with TRIGGERS_LOCK:
        v = TRIGGERS.get(tid)
        return dict(v) if isinstance(v, dict) else None


class _WorkerControlHandler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._send_json(200, {"ok": True})
            return
        if self.path == "/status":
            with RUNNING_JOB_LOCK:
                rid = RUNNING_JOB_ID

            log_tail = _tail_worker_log_lines(5)

            if rid is None:
                self._send_json(
                    200,
                    {"running_job_id": None, "job": None, "last_log_lines": log_tail},
                )
                return

            try:
                store = self.server.store  # type: ignore[attr-defined]

                job = None
                get_job = getattr(store, "get_job", None)
                if callable(get_job):
                    job = get_job(int(rid))
                else:
                    list_jobs = getattr(store, "list_jobs", None)
                    if callable(list_jobs):
                        jobs = list_jobs(limit=200) or []
                        for j in jobs:
                            if isinstance(j, dict) and int(j.get("id") or -1) == int(
                                rid
                            ):
                                job = j
                                break

                self._send_json(
                    200,
                    {
                        "running_job_id": int(rid),
                        "job": job,
                        "last_log_lines": log_tail,
                    },
                )
                return
            except Exception as e:
                self._send_json(
                    500,
                    {
                        "error": "job_lookup_failed",
                        "running_job_id": int(rid),
                        "last_log_lines": log_tail,
                        "detail": str(e),
                    },
                )
                return
        if self.path.startswith("/trigger/"):
            tid = self.path.split("/trigger/", 1)[1].strip()
            item = _trigger_get(tid)
            if not item:
                self._send_json(404, {"error": "not_found", "trigger_id": tid})
                return
            self._send_json(200, {"item": item})
            return
        if self.path.startswith("/resume/"):
            rid = self.path.split("/resume/", 1)[1].strip()
            item = _trigger_get(rid)
            if not item:
                self._send_json(404, {"error": "not_found", "resume_id": rid})
                return
            self._send_json(200, {"item": item})
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self):  # noqa: N802
        if self.path == "/resume":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                body = json.loads(raw.decode("utf-8") or "{}")
            except Exception as e:
                self._send_json(400, {"error": "bad_json", "detail": str(e)})
                return

            jid = body.get("job_id")
            try:
                jid_i = int(jid)
            except Exception:
                self._send_json(400, {"error": "missing_or_invalid_job_id"})
                return

            try:
                res = self.server.resume_async(jid_i)  # type: ignore[attr-defined]
                self._send_json(202, res)
            except Exception as e:
                self._send_json(500, {"error": "resume_failed", "detail": str(e)})
            return

        if self.path != "/trigger":
            self._send_json(404, {"error": "not_found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            body = json.loads(raw.decode("utf-8") or "{}")
        except Exception as e:
            self._send_json(400, {"error": "bad_json", "detail": str(e)})
            return

        name = str(body.get("name") or "").strip()
        if not name:
            self._send_json(400, {"error": "missing_name"})
            return

        try:
            res = self.server.trigger_async(name)  # type: ignore[attr-defined]
            self._send_json(202, res)
        except KeyError:
            self._send_json(404, {"error": "schedule_not_found", "name": name})
        except Exception as e:
            self._send_json(500, {"error": "trigger_failed", "detail": str(e)})

    def log_message(self, format, *args):  # noqa: A003
        return


def _start_worker_control_server(
    cfg: Dict[str, Any], *, trigger_async, resume_async, store
) -> Optional[ThreadingHTTPServer]:
    s = _worker_api_settings(cfg)
    if not s["enabled"]:
        log.info("worker_api disabled")
        return None

    host = s["host"]
    port = s["port"]

    httpd = ThreadingHTTPServer((host, port), _WorkerControlHandler)
    httpd.trigger_async = trigger_async  # type: ignore[attr-defined]
    httpd.resume_async = resume_async  # type: ignore[attr-defined]
    httpd.store = store

    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()

    log.info("worker_api listening on http://%s:%s", host, port)
    return httpd


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
    parser.add_argument("--log-file", default=None, help="Write logs to this file")
    parser.add_argument(
        "--log-level", default=None, help="Log level DEBUG/INFO/WARNING/ERROR"
    )
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    cfg_raw = load_config(config_path)
    cfg = _abspath_cfg_paths(cfg_raw, config_path)

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

    store_cfg = cfg.get("store") or {}
    store = create_store(store_cfg)
    async_runner = _AsyncRunner()

    pol = _cleanup_policy(cfg)

    if args.cleanup_once:
        try:
            if not getattr(pol, "enabled", True):
                log.info("cleanup disabled (cleanup.enabled=false).")
                return 0
            store.run_cleanup(pol)  # type: ignore[attr-defined]
            log.info("cleanup: run_cleanup executed (cleanup-once).")
            return 0
        finally:
            async_runner.close()

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

    next_runs: Dict[str, dt.datetime] = {}
    last_run_stamp: Dict[str, str] = {}

    last_cleanup_ts: float = 0.0

    if enabled:
        sched0 = _read_schedule_yaml(schedule_path)
        now0 = _now(tz)
        for name, entry in sched0.items():
            if not isinstance(entry, dict):
                continue
            freq = str(entry.get("frequency") or "").strip().lower()
            if freq == "triggered":
                continue
            nr = _next_run_dt(entry, now0)
            if nr:
                next_runs[str(name)] = nr
        if next_runs:
            log.info(
                "Initial next runs: %s",
                {k: v.isoformat() for k, v in next_runs.items()},
            )

    def trigger_async(name: str) -> Dict[str, Any]:
        schedule = _read_schedule_yaml(schedule_path)
        if (
            not isinstance(schedule, dict)
            or name not in schedule
            or not isinstance(schedule[name], dict)
        ):
            raise KeyError(name)

        entry = schedule[name]
        overrides = _entry_to_overrides(entry)

        trig = _trigger_create("tr", name=name, overrides=overrides, job_id=None)
        tid = trig["id"]

        def _runner():
            _trigger_update(tid, status="running", started_at=int(time.time()))
            try:
                summary_id = async_runner.run(
                    _run_one(config_path, cfg, store, schedule_path, str(name), entry)
                )
                _trigger_update(
                    tid,
                    status="done",
                    finished_at=int(time.time()),
                    summary_id=str(summary_id),
                )
            except Exception as e:
                _trigger_update(
                    tid, status="failed", finished_at=int(time.time()), error=str(e)
                )

        threading.Thread(target=_runner, daemon=True).start()

        base = _worker_api_settings(cfg)
        status_url = f"http://{base['host']}:{base['port']}/trigger/{tid}"
        return {
            "accepted": True,
            "trigger_id": tid,
            "name": name,
            "status_url": status_url,
        }

    def resume_async(job_id: int) -> Dict[str, Any]:
        jid = int(job_id)
        overrides: Dict[str, Any] = {}
        trig = _trigger_create(
            "rs", name=f"resume_job_{jid}", overrides=overrides, job_id=jid
        )
        tid = trig["id"]

        def _runner():
            global RUNNING_JOB_ID
            _trigger_update(tid, status="running", started_at=int(time.time()))
            try:
                with RUNNING_JOB_LOCK:
                    RUNNING_JOB_ID = jid

                llm = create_llm_client(cfg)
                summary_id = async_runner.run(
                    run_resume_job(
                        config=cfg,
                        store=store,
                        llm=llm,
                        job_id=jid,
                    )
                )
                _trigger_update(
                    tid,
                    status="done",
                    finished_at=int(time.time()),
                    summary_id=str(summary_id),
                )
            except Exception as e:
                _trigger_update(
                    tid, status="failed", finished_at=int(time.time()), error=str(e)
                )
            finally:
                with RUNNING_JOB_LOCK:
                    RUNNING_JOB_ID = None

        threading.Thread(target=_runner, daemon=True).start()

        base = _worker_api_settings(cfg)
        status_url = f"http://{base['host']}:{base['port']}/resume/{tid}"
        return {
            "accepted": True,
            "resume_id": tid,
            "job_id": jid,
            "status_url": status_url,
        }

    _control = _start_worker_control_server(
        cfg,
        trigger_async=trigger_async,
        resume_async=resume_async,
        store=store,
    )

    try:
        while True:
            now = _now(tz)

            if getattr(pol, "enabled", True):
                every = max(1, int(getattr(pol, "run_every_minutes", 60))) * 60
                if (time.time() - last_cleanup_ts) >= every:
                    try:
                        store.run_cleanup(pol)  # type: ignore[attr-defined]
                        log.info("cleanup: run_cleanup executed")
                    except Exception as e:
                        log.exception("cleanup: run_cleanup failed: %s", e)
                    last_cleanup_ts = time.time()

            if enabled:
                schedule = _read_schedule_yaml(schedule_path)

                for name, entry in schedule.items():
                    if not isinstance(entry, dict):
                        continue
                    freq = str(entry.get("frequency") or "").strip().lower()
                    if freq == "triggered":
                        continue
                    jn = str(name)
                    if jn not in next_runs:
                        nr = _next_run_dt(entry, now)
                        if nr:
                            next_runs[jn] = nr

                for name, entry in schedule.items():
                    if not isinstance(entry, dict):
                        continue
                    freq = str(entry.get("frequency") or "").strip().lower()
                    if freq == "triggered":
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
                        async_runner.run(
                            _run_one(config_path, cfg, store, schedule_path, jn, entry)
                        )
                    except Exception as e:
                        log.exception("Job '%s' FAILED: %s", jn, e)

                    last_run_stamp[jn] = stamp

                    nn = _next_run_dt(entry, now + dt.timedelta(seconds=1))
                    if nn:
                        next_runs[jn] = nn

            if args.once:
                log.info("--once: exiting after one scan.")
                return 0

            time.sleep(max(1, int(poll_seconds)))
    finally:
        async_runner.close()


if __name__ == "__main__":
    raise SystemExit(main())
