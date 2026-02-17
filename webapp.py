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

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

import markdown as md
import yaml
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request as flask_request,
    url_for,
    stream_with_context,
    Response,
)

from summarizer.helpers import setup_logging
from summarizer.main import run_pipeline
from summarizer.prompt_lab import run_promptlab_summarization
from persistence import NewsStore, create_store
from llmClient import create_llm_client

setup_logging()
logger = logging.getLogger(__name__)
app = Flask(__name__)
pipeline_lock = threading.Lock()


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_store() -> NewsStore:
    cfg = load_config()
    return create_store(cfg.get("store", {}))


def format_ts(ts: Optional[int]) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _get_promptlab_prompts(cfg: dict, form=None) -> dict:
    p_cfg = cfg.get("prompts") or {}
    keys = ["batch_system", "batch_user_template", "meta_system", "meta_user_template"]
    out = {k: str(p_cfg.get(k, "")) for k in keys}
    if form is not None:
        for k in keys:
            v = form.get(k)
            # Viktigt: behåll användarens inmatning även om den är tom sträng (om de tömmer med flit)
            if v is not None:
                out[k] = str(v)
    return out


@app.get("/")
def index():
    store = get_store()
    latest = store.get_latest_summary()

    if not latest:
        return render_template("index.html", summary=None)

    summary_text = latest.get("summary", "")
    created_at = latest.get("created_at", 0)
    article_ids = latest.get("article_ids", []) or []

    summary_html = md.markdown(summary_text, extensions=["extra"])
    articles = store.get_articles_by_ids(article_ids)

    view_articles = [
        {
            "title": a.get("title", ""),
            "url": a.get("url", ""),
            "source": a.get("source", ""),
            "published": a.get("published", ""),
        }
        for a in articles
    ]

    return render_template(
        "index.html",
        summary=summary_text,
        summary_html=summary_html,
        summary_time=format_ts(created_at),
        n_articles=len(article_ids),
        articles=view_articles,
    )


@app.get("/history")
def history():
    store = get_store()
    summaries = store.list_summaries()
    items = [
        {
            "id": s.get("id"),
            "time": format_ts(s.get("created_at")),
            "n_articles": len(s.get("article_ids", []) or []),
        }
        for s in summaries
    ]
    return render_template("history.html", items=items)


@app.get("/summary/<int:summary_id>")
def view_summary(summary_id: int):
    store = get_store()
    s = store.get_summary(summary_id)
    if not s:
        return redirect(url_for("history"))

    summary_text = s.get("summary", "")
    created_at = s.get("created_at", 0)
    article_ids = s.get("article_ids", []) or []

    summary_html = md.markdown(summary_text, extensions=["extra"])
    articles = store.get_articles_by_ids(article_ids)

    view_articles = [
        {
            "title": a.get("title", ""),
            "url": a.get("url", ""),
            "source": a.get("source", ""),
            "published": a.get("published", ""),
        }
        for a in articles
    ]

    return render_template(
        "summary.html",
        summary_html=summary_html,
        summary_time=format_ts(created_at),
        n_articles=len(article_ids),
        articles=view_articles,
    )


@app.post("/refresh")
def refresh():
    store = get_store()
    job_id = store.create_job()

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
            asyncio.run(run_pipeline("config.yaml", job_id=jid))
        except Exception as e:
            store.update_job(
                jid,
                status="error",
                finished_at=int(time.time()),
                message=f"Refresh misslyckades: {e}",
            )
            logger.error(f"{jid} Refresh misslyckades: {e}")
        finally:
            pipeline_lock.release()

    threading.Thread(target=worker, args=(job_id,), daemon=True).start()
    return redirect(url_for("index", job=job_id))


@app.get("/api/status/<int:job_id>")
def api_status(job_id: int):
    store = get_store()
    job = store.get_job(job_id)
    if not job:
        return jsonify({"status": "error", "message": "Jobb hittades inte."}), 404

    return jsonify(
        {
            "status": job.get("status"),
            "message": job.get("message", ""),
            "created_at": job.get("created_at"),
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
            "summary_id": job.get("summary_id"),
        }
    )


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


@app.get("/prompt-lab")
def prompt_lab():
    cfg = load_config()
    store = get_store()

    summaries = store.list_summaries()
    items = [
        {
            "id": s["id"],
            "time": format_ts(s.get("created_at")),
            "n": len(s.get("article_ids", []) or []),
        }
        for s in summaries
    ]

    job = flask_request.args.get("job", type=int)
    result = flask_request.args.get("result", type=int)
    selected_summary_id = flask_request.args.get("summary_id", type=int)

    job_or_result = job or result

    prompts = _get_promptlab_prompts(cfg, form=None)
    temp = None
    temp_html = None

    if job_or_result:
        temp = store.get_temp_summary(job_or_result)

        if temp and isinstance(temp, dict):
            meta = temp.get("meta") or {}
            tp = meta.get("prompts")
            if isinstance(tp, dict):
                for k in list(prompts.keys()):
                    if k in tp and isinstance(tp[k], str):
                        prompts[k] = tp[k]

            if temp.get("summary"):
                temp_html = md.markdown(str(temp["summary"]), extensions=["extra"])

    return render_template(
        "prompt_lab.html",
        prompts=prompts,
        summaries=items,
        selected_summary_id=selected_summary_id,
        job_id=job,
        result_id=result,
        temp=temp,
        temp_html=temp_html,
    )


@app.post("/prompt-lab/run")
def prompt_lab_run():
    cfg = load_config()
    store = get_store()

    summary_id = flask_request.form.get("summary_id", type=int)
    prompts = _get_promptlab_prompts(cfg, form=flask_request.form)

    job_id = store.create_job()
    store.update_job(
        job_id,
        status="running",
        started_at=int(time.time()),
        message="Prompt-lab: startar...",
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
            if summary_id is None:
                s = store.get_latest_summary()
                if not s:
                    raise RuntimeError(
                        "Ingen sparad summary finns att använda i prompt-lab."
                    )
            else:
                s = store.get_summary(summary_id)
                if not s:
                    raise RuntimeError(f"Kunde inte hitta summary_id={summary_id}")

            source_summary_id = int(s.get("id"))  # type: ignore
            article_ids = s.get("article_ids", []) or []
            articles = store.get_articles_by_ids(article_ids)

            store.update_job(
                jid,
                message=f"Prompt-lab: använder summary {source_summary_id} ({len(articles)} artiklar)",
            )

            asyncio.run(
                run_promptlab_summarization(
                    config=cfg,
                    prompts=prompts,
                    store=store,
                    llm=llm,
                    job_id=jid,
                    source_summary_id=source_summary_id,
                    articles=articles,
                )
            )

            store.update_job(
                jid,
                status="done",
                finished_at=int(time.time()),
                message="Prompt-lab: klart.",
            )
        except Exception as e:
            store.update_job(
                jid,
                status="error",
                finished_at=int(time.time()),
                message=f"Prompt-lab misslyckades: {e}",
            )
        finally:
            pipeline_lock.release()

    threading.Thread(target=worker, args=(job_id,), daemon=True).start()
    return redirect(url_for("prompt_lab", job=job_id, summary_id=summary_id))


@app.post("/prompt-lab/apply")
def prompt_lab_apply():
    cfg = load_config()
    cfg.setdefault("prompts", {})
    for k in [
        "batch_system",
        "batch_user_template",
        "meta_system",
        "meta_user_template",
    ]:
        cfg["prompts"][k] = flask_request.form.get(k, "")  # type: ignore

    with open("config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    return redirect(url_for("prompt_lab"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
