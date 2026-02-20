from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List, Set, Tuple

import markdown as md
import yaml
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    stream_with_context,
    url_for,
)

from llmClient import create_llm_client
from persistence import NewsStore, create_store
from summarizer.helpers import setup_logging
from summarizer.main import run_pipeline
from summarizer.summarizer import run_resume_and_persist_summary

setup_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)
pipeline_lock = threading.Lock()

CONFIG_PATH = os.environ.get("FEEDSUMMARY_CONFIG", "config.yaml")
CONFIG_DIR = os.path.dirname(os.path.abspath(CONFIG_PATH)) or "."


def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    from summarizer.helpers import load_feeds_into_config

    cfg = load_feeds_into_config(cfg, base_config_path=path)
    return cfg


def get_store() -> NewsStore:
    cfg = load_config()
    return create_store(cfg.get("store", {}))


def format_ts(ts: Optional[int]) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _published_ts(a: Dict[str, Any]) -> int:
    ts = a.get("published_ts")
    if isinstance(ts, int) and ts > 0:
        return ts
    fa = a.get("fetched_at")
    if isinstance(fa, int) and fa > 0:
        return fa
    return 0


def _parse_ymd_to_range(date_str: str) -> Optional[Tuple[int, int]]:
    """
    Parse YYYY-MM-DD to (start_ts, end_ts) for that day in local time.
    """
    s = (date_str or "").strip()
    if not s:
        return None
    try:
        d = datetime.strptime(s, "%Y-%m-%d")
        start = int(d.timestamp())
        end = int((d + timedelta(days=1) - timedelta(seconds=1)).timestamp())
        return start, end
    except Exception:
        return None


def _get_config_sources(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = [
        cfg.get("sources"),
        cfg.get("feeds"),
        cfg.get("rss_sources"),
        (cfg.get("ingest") or {}).get("sources"),
        (cfg.get("ingest") or {}).get("feeds"),
    ]
    for c in candidates:
        if isinstance(c, list) and c and all(isinstance(x, dict) for x in c):
            return c  # type: ignore[return-value]
    return []


def _source_name(s: Dict[str, Any]) -> str:
    return str(s.get("name") or s.get("title") or s.get("label") or "").strip()


def _source_topics(s: Dict[str, Any]) -> List[str]:
    t = s.get("topics")
    if isinstance(t, list):
        return [str(x).strip() for x in t if str(x).strip()]
    if isinstance(t, str) and t.strip():
        return [t.strip()]
    t2 = s.get("topic")
    if isinstance(t2, str) and t2.strip():
        return [t2.strip()]
    return []


def _build_source_options(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in _get_config_sources(cfg):
        name = _source_name(s)
        if not name:
            continue
        out.append(
            {
                "name": name,
                "url": str(s.get("url") or s.get("href") or s.get("rss") or ""),
                "default_checked": bool(s.get("enabled", True)),
                "topics": _source_topics(s),
            }
        )
    out.sort(key=lambda x: x["name"].lower())
    return out


def _build_topic_options(cfg: Dict[str, Any]) -> List[str]:
    seen: Set[str] = set()
    topics: List[str] = []
    for s in _get_config_sources(cfg):
        for t in _source_topics(s):
            if t not in seen:
                seen.add(t)
                topics.append(t)
    topics.sort(key=lambda x: x.lower())
    return topics


def _resolve_path(p: str) -> str:
    """
    Expand env + ~ and resolve relative paths relative to config.yaml location.
    """
    p2 = os.path.expanduser(os.path.expandvars(p))
    if not os.path.isabs(p2):
        p2 = os.path.join(CONFIG_DIR, p2)
    return p2


def _load_prompt_packages(cfg: Dict[str, Any]) -> List[str]:
    """
    Reads config.prompts.path (default: config/prompts.yaml) and returns package names.
    Logs warnings if file is missing/invalid so the UI doesn't fail silently.
    """
    p_cfg = cfg.get("prompts") or {}
    if not isinstance(p_cfg, dict):
        return []

    raw_path = str(p_cfg.get("path") or "config/prompts.yaml")
    path = _resolve_path(raw_path)

    try:
        with open(path, "r", encoding="utf-8") as f:
            all_pkgs = yaml.safe_load(f) or {}
        if isinstance(all_pkgs, dict) and all_pkgs:
            logger.info("prompts.yaml loaded: %s", path)
            return sorted([str(k) for k in all_pkgs.keys()])
        logger.warning("prompts.yaml loaded but empty or not a dict: %s", path)
        return []
    except FileNotFoundError as e:
        logger.error(
            "prompts.yaml not found: %s (from prompts.path=%s)", path, raw_path
        )
        raise e
    except Exception as e:
        logger.error("failed to read prompts.yaml: %s -> %s", path, e)
        raise e


@app.get("/")
def index():
    store = get_store()
    cfg = load_config()
    prompt_packages = _load_prompt_packages(cfg)
    p_cfg = cfg.get("prompts") or {}
    default_prompt_pkg = ""
    if isinstance(p_cfg, dict):
        default_prompt_pkg = str(
            p_cfg.get("selected") or p_cfg.get("default_package") or ""
        ).strip()
    if not default_prompt_pkg and prompt_packages:
        default_prompt_pkg = prompt_packages[0]

    all_docs = store.list_summary_docs() or []
    sidebar_items = [
        {
            "id": str(d.get("id")),
            "time": format_ts(int(d.get("created") or 0)),
            "n_articles": len(d.get("sources", []) or []),
        }
        for d in all_docs
    ]

    selected_id = request.args.get("summary_id")
    selected: Optional[Dict[str, Any]] = None
    if selected_id:
        selected = store.get_summary_doc(str(selected_id))
    if not selected:
        selected = store.get_latest_summary_doc()

    ingest = cfg.get("ingest") or {}
    default_lookback = str(ingest.get("lookback") or "24h")
    lb_val = "".join([c for c in default_lookback if c.isdigit()]) or "24"
    lb_unit = "".join([c for c in default_lookback if not c.isdigit()]).strip() or "h"

    source_options = _build_source_options(cfg)
    topic_options = _build_topic_options(cfg)

    if not selected:
        return render_template(
            "index.html",
            summary=None,
            summary_list=sidebar_items,
            selected_id=selected_id,
            source_options=source_options,
            topic_options=topic_options,
            default_lb_value=lb_val,
            default_lb_unit=lb_unit,
            prompt_packages=prompt_packages,
            default_prompt_package=default_prompt_pkg,
        )

    summary_text = selected.get("summary", "")
    created_at = int(selected.get("created") or 0)
    article_ids = selected.get("sources", []) or []
    selected_id = str(selected.get("id"))

    summary_html = md.markdown(summary_text, extensions=["extra"])

    return render_template(
        "index.html",
        summary=summary_text,
        summary_html=summary_html,
        summary_time=format_ts(created_at),
        n_articles=len(article_ids),
        summary_list=sidebar_items,
        selected_id=selected_id,
        source_options=source_options,
        topic_options=topic_options,
        default_lb_value=lb_val,
        default_lb_unit=lb_unit,
        prompt_packages=prompt_packages,
        default_prompt_package=default_prompt_pkg,
    )


@app.post("/refresh")
def refresh():
    store = get_store()
    cfg = load_config()

    job_id = store.create_job()

    lookback_value = (request.form.get("lookback_value") or "").strip()
    lookback_unit = (request.form.get("lookback_unit") or "").strip().lower()
    selected_sources = request.form.getlist("sources") or []
    selected_topics = request.form.getlist("topics") or []
    prompt_package = (request.form.get("prompt_package") or "").strip()

    overrides: Dict[str, Any] = {}
    if prompt_package:
        overrides["prompt_package"] = prompt_package
    if lookback_value and lookback_unit:
        overrides["lookback"] = f"{lookback_value}{lookback_unit}"
    if selected_sources:
        overrides["sources"] = selected_sources
    elif selected_topics:
        overrides["topics"] = selected_topics

    store.update_job(
        job_id,
        status="running",
        started_at=int(time.time()),
        message=(
            "Startar refresh… ("
            f"lookback={overrides.get('lookback', 'default')}, "
            f"sources={len(selected_sources) or 'default'}, "
            f"topics={len(selected_topics) or 'default'}, "
            f"prompts={prompt_package or 'default'})"
        ),
    )

    def worker(jid: int):
        if not pipeline_lock.acquire(blocking=False):
            store.update_job(
                jid,
                status="error",
                finished_at=int(time.time()),
                message="En refresh kör redan. Försök igen om en stund.",
            )
            return
        try:
            asyncio.run(
                run_pipeline(
                    CONFIG_PATH, job_id=jid, overrides=overrides, config_dict=cfg
                )
            )
        except Exception as e:
            store.update_job(
                jid,
                status="error",
                finished_at=int(time.time()),
                message=f"Refresh misslyckades: {e}",
            )
            logger.error("%s Refresh misslyckades: %s", jid, e)
        finally:
            pipeline_lock.release()

    threading.Thread(target=worker, args=(job_id,), daemon=True).start()
    return redirect(url_for("index", job=job_id))


@app.get("/resume")
def resume():
    store = get_store()
    cfg = load_config()

    job_id = request.args.get("job", type=int)
    if not job_id:
        return jsonify(
            {"status": "error", "message": "Saknar job. Använd /resume?job=<id>"}
        ), 400

    store.update_job(
        job_id,
        status="running",
        started_at=int(time.time()),
        message="Återupptar från checkpoint...",
    )

    llm = create_llm_client(cfg)

    def worker(jid: int):
        if not pipeline_lock.acquire(blocking=False):
            store.update_job(
                jid,
                status="error",
                finished_at=int(time.time()),
                message="En körning pågår redan. Försök igen om en stund.",
            )
            return

        try:
            asyncio.run(run_resume_and_persist_summary(cfg, store, llm, jid))
            store.update_job(
                jid,
                status="done",
                finished_at=int(time.time()),
                message="Resume klart.",
            )
        except Exception as e:
            store.update_job(
                jid,
                status="error",
                finished_at=int(time.time()),
                message=f"Resume misslyckades: {e}",
            )
            logger.error("Resume misslyckades (job=%s): %s", jid, e)
        finally:
            pipeline_lock.release()

    threading.Thread(target=worker, args=(job_id,), daemon=True).start()
    return redirect(url_for("index", job=job_id))


@app.get("/api/status/stream/<int:job_id>")
def api_status_stream(job_id: int):
    store = get_store()

    def event(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    @stream_with_context
    def generate():
        last_payload = None
        last_emit = 0.0

        job = store.get_job(job_id)
        if not job:
            yield event({"status": "error", "message": "Jobb hittades inte."})
            return

        while True:
            job = store.get_job(job_id)
            if not job:
                yield event({"status": "error", "message": "Jobb hittades inte."})
                return

            payload = {
                "status": job.get("status"),
                "message": job.get("message", ""),
                "created_at": job.get("created_at"),
                "started_at": job.get("started_at"),
                "finished_at": job.get("finished_at"),
                "summary_id": job.get("summary_id"),
            }

            now = time.time()
            if payload != last_payload or (now - last_emit) > 10:
                yield event(payload)
                last_payload = payload
                last_emit = now

            if payload["status"] in ("done", "error"):
                return

            time.sleep(1.0)

    return Response(
        generate(),  # type: ignore
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/articles")
def list_articles():
    """
    Articles list with optional filters:

    - sources: repeated query param (e.g. ?sources=SVT&sources=DN)
    - topics: repeated query param (e.g. ?topics=Cyber&topics=Europa)
    - from: YYYY-MM-DD (inclusive)
    - to:   YYYY-MM-DD (inclusive)
    """
    store = get_store()
    cfg = load_config()

    articles = store.list_articles()
    all_sources_in_db: List[str] = sorted(
        {
            str(a.get("source") or "").strip()
            for a in articles
            if str(a.get("source") or "").strip()
        },
        key=lambda s: s.lower(),
    )
    source_options_all = _build_source_options(cfg)
    source_options = [
        s for s in source_options_all if s.get("name") in set(all_sources_in_db)
    ]
    topic_options = _build_topic_options(cfg)

    source_to_topics: Dict[str, List[str]] = {
        str(s.get("name")): list(s.get("topics") or []) for s in source_options
    }

    selected_sources = request.args.getlist("sources") or []
    selected_sources = [s.strip() for s in selected_sources if s.strip()]

    selected_topics = request.args.getlist("topics") or []
    selected_topics = [t.strip() for t in selected_topics if t.strip()]
    selected_topics_set: Set[str] = set(selected_topics)

    from_str = (request.args.get("from") or "").strip()
    to_str = (request.args.get("to") or "").strip()

    from_range = _parse_ymd_to_range(from_str) if from_str else None
    to_range = _parse_ymd_to_range(to_str) if to_str else None

    from_ts: Optional[int] = from_range[0] if from_range else None
    to_ts: Optional[int] = to_range[1] if to_range else None

    allowed_sources: Optional[Set[str]] = None
    if selected_sources:
        allowed_sources = set(selected_sources)
    elif selected_topics:
        allowed: Set[str] = set()
        for src, ts in source_to_topics.items():
            if set(ts).intersection(selected_topics_set):
                allowed.add(src)
        allowed_sources = allowed if allowed else set()

    def keep(a: Dict[str, Any]) -> bool:
        src = str(a.get("source") or "").strip()

        if allowed_sources is not None:
            if src not in allowed_sources:
                return False

        ts = _published_ts(a)
        if from_ts is not None and ts and ts < from_ts:
            return False
        if to_ts is not None and ts and ts > to_ts:
            return False
        if ts == 0 and (from_ts is not None or to_ts is not None):
            return False
        return True

    filtered = [a for a in articles if keep(a)]

    # Show newest first in articles view
    filtered.sort(key=_published_ts, reverse=True)

    view_articles = [
        {
            "title": a.get("title", ""),
            "url": a.get("url", ""),
            "source": a.get("source", ""),
            "published": a.get("published", ""),
            "published_ts": _published_ts(a),
            "text": (a.get("text", "") or "")[:500],
        }
        for a in filtered
    ]

    return render_template(
        "articles.html",
        articles=view_articles,
        source_options=source_options,
        topic_options=topic_options,
        sources=all_sources_in_db,
        selected_sources=selected_sources,
        selected_topics=selected_topics,
        from_date=from_str,
        to_date=to_str,
        total_count=len(articles),
        filtered_count=len(filtered),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
