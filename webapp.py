from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional, List

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
from summarizer.summarizer import run_resume_and_persist_summary, run_resume_from_checkpoint

setup_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)
pipeline_lock = threading.Lock()

# Allow overriding config path (helps when running from other cwd)
CONFIG_PATH = os.environ.get("FEEDSUMMARY_CONFIG", "config.yaml")
CONFIG_DIR = os.path.dirname(os.path.abspath(CONFIG_PATH)) or "."


def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_store() -> NewsStore:
    cfg = load_config()
    return create_store(cfg.get("store", {}))


def format_ts(ts: Optional[int]) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------
# Config sources (for UI)
# ----------------------------
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
            }
        )
    out.sort(key=lambda x: x["name"].lower())
    return out


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
    except FileNotFoundError:
        logger.warning("prompts.yaml not found: %s (from prompts.path=%s)", path, raw_path)
        return []
    except Exception as e:
        logger.warning("failed to read prompts.yaml: %s -> %s", path, e)
        return []


# ----------------------------
# Summary compatibility layer (legacy summaries + new summary_docs)
# ----------------------------
def _summary_doc_to_legacy_like(doc: Dict[str, Any]) -> Dict[str, Any]:
    created = doc.get("created") or doc.get("created_at") or 0
    sources = doc.get("sources") or doc.get("article_ids") or []
    return {
        "id": doc.get("id"),
        "created_at": int(created) if created else 0,
        "summary": doc.get("summary", ""),
        "article_ids": sources,
        "_kind": doc.get("kind", "summary"),
        "_raw": doc,
    }


def _get_latest_summary_compat(store: NewsStore) -> Optional[Dict[str, Any]]:
    fn = getattr(store, "get_latest_summary_doc", None)
    if callable(fn):
        try:
            doc = fn()
            if doc:
                return _summary_doc_to_legacy_like(doc)
        except Exception:
            pass

    try:
        latest = store.get_latest_summary()
        if latest:
            return latest
    except Exception:
        pass

    return None


def _list_summaries_compat(store: NewsStore) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    fn_list_docs = getattr(store, "list_summary_docs", None)
    if callable(fn_list_docs):
        try:
            docs = fn_list_docs() or []
            for d in docs:
                if isinstance(d, dict):
                    items.append(_summary_doc_to_legacy_like(d))
        except Exception:
            pass

    try:
        legacy = store.list_summaries() or []
        for s in legacy:
            if isinstance(s, dict):
                items.append(s)
    except Exception:
        pass

    seen = set()
    out: List[Dict[str, Any]] = []
    for it in items:
        sid = it.get("id")
        if sid in seen:
            continue
        seen.add(sid)
        out.append(it)

    out.sort(key=lambda r: int(r.get("created_at") or 0), reverse=True)
    return out


def _get_summary_compat(store: NewsStore, summary_id: str) -> Optional[Dict[str, Any]]:
    if summary_id.isdigit():
        try:
            s = store.get_summary(int(summary_id))
            if s:
                return s
        except Exception:
            pass

    fn_get_doc = getattr(store, "get_summary_doc", None)
    if callable(fn_get_doc):
        try:
            d = fn_get_doc(summary_id)
            if d:
                return _summary_doc_to_legacy_like(d)
        except Exception:
            pass

    fn_get_doc2 = getattr(store, "get_summary_document", None)
    if callable(fn_get_doc2):
        try:
            d = fn_get_doc2(summary_id)
            if d:
                return _summary_doc_to_legacy_like(d)
        except Exception:
            pass

    return None


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def index():
    store = get_store()
    cfg = load_config()

    # always compute prompt packages (even if there is no summary yet)
    prompt_packages = _load_prompt_packages(cfg)
    p_cfg = cfg.get("prompts") or {}
    default_prompt_pkg = ""
    if isinstance(p_cfg, dict):
        default_prompt_pkg = str(p_cfg.get("selected") or p_cfg.get("default_package") or "").strip()
    if not default_prompt_pkg and prompt_packages:
        default_prompt_pkg = prompt_packages[0]

    all_summaries = _list_summaries_compat(store)
    sidebar_items = [
        {
            "id": str(s.get("id")),
            "time": format_ts(int(s.get("created_at") or 0)),
            "n_articles": len(s.get("article_ids", []) or []),
        }
        for s in all_summaries
    ]

    selected_id = request.args.get("summary_id")
    selected: Optional[Dict[str, Any]] = None
    if selected_id:
        selected = _get_summary_compat(store, str(selected_id))
    if not selected:
        selected = _get_latest_summary_compat(store)

    ingest = cfg.get("ingest") or {}
    default_lookback = str(ingest.get("lookback") or "24h")
    lb_val = "".join([c for c in default_lookback if c.isdigit()]) or "24"
    lb_unit = "".join([c for c in default_lookback if not c.isdigit()]).strip() or "h"

    source_options = _build_source_options(cfg)

    if not selected:
        return render_template(
            "index.html",
            summary=None,
            summary_list=sidebar_items,
            selected_id=selected_id,
            source_options=source_options,
            default_lb_value=lb_val,
            default_lb_unit=lb_unit,
            prompt_packages=prompt_packages,
            default_prompt_package=default_prompt_pkg,
        )

    summary_text = selected.get("summary", "")
    created_at = int(selected.get("created_at") or 0)
    article_ids = selected.get("article_ids", []) or []
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
    prompt_package = (request.form.get("prompt_package") or "").strip()

    overrides: Dict[str, Any] = {}
    if prompt_package:
        overrides["prompt_package"] = prompt_package
    if lookback_value and lookback_unit:
        overrides["lookback"] = f"{lookback_value}{lookback_unit}"
    if selected_sources:
        overrides["sources"] = selected_sources

    store.update_job(
        job_id,
        status="running",
        started_at=int(time.time()),
        message=f"Startar refresh… (lookback={overrides.get('lookback','default')}, sources={len(selected_sources) or 'default'}, prompts={prompt_package or 'default'})",
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
            asyncio.run(run_pipeline(CONFIG_PATH, job_id=jid, overrides=overrides, config_dict=cfg))
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
        return jsonify({"status": "error", "message": "Saknar job. Använd /resume?job=<id>"}), 400

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
    store = get_store()
    articles = store.list_articles()

    view_articles = [
        {
            "title": a.get("title", ""),
            "url": a.get("url", ""),
            "source": a.get("source", ""),
            "published": a.get("published", ""),
            "text": a.get("text", "")[:500],
        }
        for a in articles
    ]

    return render_template(
        "articles.html",
        articles=view_articles,
    )
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
