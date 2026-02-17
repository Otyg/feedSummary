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

import logging
import time
from typing import Any, Dict, List, Optional

from tinydb import Query, TinyDB

logger = logging.getLogger(__name__)


class TinyDBStore:
    """
    TinyDB-backed store (JSON file). Implements NewsStore.
    Uses TinyDB doc_id as the integer ID for summaries/jobs.
    """

    def __init__(self, path: str = "news_docs.json"):
        self.path = path

    def _db(self) -> TinyDB:
        return TinyDB(self.path)

    # ---- Articles
    def get_article(self, article_id: str) -> Optional[Dict[str, Any]]:
        db = self._db()
        A = Query()
        res = db.table("articles").search(A.id == article_id)
        db.close()
        return res[0] if res else None

    def upsert_article(self, article_doc: Dict[str, Any]) -> None:
        db = self._db()
        A = Query()
        db.table("articles").upsert(article_doc, A.id == article_doc["id"])
        db.close()

    def list_unsummarized_articles(self, limit: int = 200) -> List[Dict[str, Any]]:
        db = self._db()
        A = Query()
        res = db.table("articles").search((A.summarized != True))  # noqa: E712
        db.close()
        return res[:limit]  # pyright: ignore[reportReturnType]

    def mark_articles_summarized(self, article_ids: List[str]) -> None:
        db = self._db()
        A = Query()
        ts = int(time.time())
        for aid in article_ids:
            db.table("articles").update(
                {"summarized": True, "summarized_at": ts}, A.id == aid
            )
        db.close()

    # ---- Summaries
    def save_summary(self, summary_text: str, article_ids: List[str]) -> int:
        db = self._db()
        doc_id = db.table("summaries").insert(
            {
                "created_at": int(time.time()),
                "summary": summary_text,
                "article_ids": article_ids,
            }
        )
        db.close()
        return doc_id

    def get_latest_summary(self) -> Optional[Dict[str, Any]]:
        db = self._db()
        docs = list(db.table("summaries"))
        db.close()
        if not docs:
            return None
        latest = sorted(docs, key=lambda d: d.get("created_at", 0), reverse=True)[0]
        return {"id": latest.doc_id, **dict(latest)}

    def list_summaries(self) -> List[Dict[str, Any]]:
        db = self._db()
        docs = list(db.table("summaries"))
        db.close()
        # include id
        out = [{"id": d.doc_id, **dict(d)} for d in docs]
        out.sort(key=lambda r: r.get("created_at", 0), reverse=True)
        return out

    def get_summary(self, summary_id: int) -> Optional[Dict[str, Any]]:
        db = self._db()
        doc = db.table("summaries").get(doc_id=summary_id)
        db.close()
        if not doc:
            return None
        return {"id": summary_id, **dict(doc)}  # pyright: ignore[reportCallIssue, reportArgumentType, reportReturnType]

    # ---- Jobs
    def create_job(self) -> int:
        db = self._db()
        jid = db.table("jobs").insert(
            {
                "created_at": int(time.time()),
                "started_at": None,
                "finished_at": None,
                "status": "queued",  # queued|running|done|error
                "message": "",
                "summary_id": None,
            }
        )
        db.close()
        logger.info(f"Job {jid} created")
        return jid

    def update_job(self, job_id: int, **fields) -> None:
        db = self._db()
        db.table("jobs").update(fields, doc_ids=[job_id])
        logger.info(f"Job {job_id} updated")
        db.close()

    def get_job(self, job_id: int) -> Optional[Dict[str, Any]]:
        db = self._db()
        doc = db.table("jobs").get(doc_id=job_id)
        db.close()
        if not doc:
            return None
        return {"id": job_id, **dict(doc)}  # pyright: ignore[reportCallIssue, reportArgumentType, reportReturnType]

    # ---- Utility
    def get_articles_by_ids(self, article_ids: List[str]) -> List[Dict[str, Any]]:
        # keep input order
        db = self._db()
        at = db.table("articles")
        out: List[Dict[str, Any]] = []
        for aid in article_ids:
            rows = at.search(lambda r: r.get("id") == aid)
            if rows:
                out.append(rows[0])
        db.close()
        return out

    # ---- Temp summaries
    def save_temp_summary(
        self, job_id: int, summary_text: str, meta: Dict[str, Any]
    ) -> None:
        db = self._db()
        t = db.table("temp_summaries")
        # upsert på job_id
        T = Query()
        t.upsert(
            {
                "job_id": job_id,
                "created_at": int(time.time()),
                "summary": summary_text,
                "meta": meta or {},
            },
            T.job_id == job_id,
        )
        db.close()

    def get_temp_summary(self, job_id: int) -> Optional[Dict[str, Any]]:
        db = self._db()
        t = db.table("temp_summaries")
        T = Query()
        rows = t.search(T.job_id == job_id)
        db.close()
        return rows[0] if rows else None
