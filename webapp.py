from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional, List

import markdown as md
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
from summarizer.main import run_pipeline
from summarizer.summarizer import run_resume_and_persist_summary
from summarizer.helpers import setup_logging

from uicommon import (
    load_config,
    get_store,
    format_ts,
    published_ts,
    filter_articles,
    ArticleFilters,
    get_ui_options,
    build_refresh_overrides,
)

setup_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)
pipeline_lock = threading.Lock()

CONFIG_PATH = os.environ.get("FEEDSUMMARY_CONFIG", "config.yaml")


def _safe_int(s: str, default: int) -> int:
    try:
        v = int(str(s).strip())
        return v if v > 0 else default
    except Exception:
        return default


@app.get("/")
def index():
    cfg = load_config(CONFIG_PATH)
    store = get_store(cfg)
    opts = get_ui_options(cfg, config_path=CONFIG_PATH)

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

    if not selected:
        return render_template(
            "index.html",
            summary=None,
            summary_list=sidebar_items,
            selected_id=selected_id,
            source_options=opts.source_options,
            topic_options=opts.topic_options,
            default_lb_value=str(opts.default_lookback_value),
            default_lb_unit=opts.default_lookback_unit,
            prompt_packages=opts.prompt_packages,
            default_prompt_package=opts.default_prompt_package,
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
        source_options=opts.source_options,
        topic_options=opts.topic_options,
        default_lb_value=str(opts.default_lookback_value),
        default_lb_unit=opts.default_lookback_unit,
        prompt_packages=opts.prompt_packages,
        default_prompt_package=opts.default_prompt_package,
    )


@app.post("/refresh")
def refresh():
    cfg = load_config(CONFIG_PATH)
    store = get_store(cfg)
    opts = get_ui_options(cfg, config_path=CONFIG_PATH)

    job_id = store.create_job()

    lookback_value_raw = (request.form.get("lookback_value") or "").strip()
    lookback_unit = (request.form.get("lookback_unit") or "").strip().lower()
    prompt_package = (request.form.get("prompt_package") or "").strip()

    selected_sources = request.form.getlist("sources") or []
    selected_topics = request.form.getlist("topics") or []

    overrides = build_refresh_overrides(
        lookback_value=_safe_int(lookback_value_raw, int(opts.default_lookback_value)),
        lookback_unit=lookback_unit or opts.default_lookback_unit,
        prompt_package=prompt_package,
        selected_sources=selected_sources,
        selected_topics=selected_topics,
    )

    store.update_job(
        job_id,
        status="running",
        started_at=int(time.time()),
        message=f"Startar refresh… (overrides: {overrides})",
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
                    CONFIG_PATH,
                    job_id=jid,
                    overrides=overrides,
                    config_dict=cfg,
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
    cfg = load_config(CONFIG_PATH)
    store = get_store(cfg)

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
    cfg = load_config(CONFIG_PATH)
    store = get_store(cfg)

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
    cfg = load_config(CONFIG_PATH)
    store = get_store(cfg)
    opts = get_ui_options(cfg, config_path=CONFIG_PATH)

    articles = store.list_articles() or []

    selected_sources = [
        s.strip() for s in (request.args.getlist("sources") or []) if s.strip()
    ]
    selected_topics = [
        t.strip() for t in (request.args.getlist("topics") or []) if t.strip()
    ]
    from_str = (request.args.get("from") or "").strip()
    to_str = (request.args.get("to") or "").strip()

    filtered = filter_articles(
        articles,
        cfg=cfg,
        filters=ArticleFilters(
            sources=selected_sources,
            topics=selected_topics,
            from_ymd=from_str,
            to_ymd=to_str,
        ),
    )

    view_articles = [
        {
            "title": a.get("title", ""),
            "url": a.get("url", ""),
            "source": a.get("source", ""),
            "published": a.get("published", ""),
            "published_ts": published_ts(a),
            "text": (a.get("text", "") or "")[:500],
        }
        for a in filtered
    ]

    all_sources: List[str] = sorted(
        {
            str(a.get("source") or "").strip()
            for a in articles
            if str(a.get("source") or "").strip()
        },
        key=lambda s: s.lower(),
    )

    return render_template(
        "articles.html",
        articles=view_articles,
        sources=all_sources,
        source_options=opts.source_options,
        topic_options=opts.topic_options,
        selected_sources=selected_sources,
        selected_topics=selected_topics,
        from_date=from_str,
        to_date=to_str,
        total_count=len(articles),
        filtered_count=len(filtered),
    )


if __name__ == "__main__":
    port = int(os.environ.get("FEEDSUMMARY_PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)
