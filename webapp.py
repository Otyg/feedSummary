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
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

import markdown as md
import yaml
from flask import Flask, jsonify, redirect, render_template, url_for

from summarizer.main import run_pipeline
from persistence import NewsStore, create_store

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


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
