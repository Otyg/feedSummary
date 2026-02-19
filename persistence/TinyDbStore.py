# LICENSE HEADER MANAGED BY add-license-header
#
# BSD 3-Clause License
# ... (oförändrad header)

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Set

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

    def list_articles(self, limit: int = 2000) -> List[Dict[str, Any]]:
        """
        Returnera artiklar utan att använda 'summarized'-flagga.
        OBS: här returnerar vi själva dokumenten (dvs id = artikelns id).
        """
        db = self._db()
        docs = list(db.table("articles"))
        db.close()
        out = [dict(d) for d in docs]
        # sort oldest-first på published_ts för stabil batching
        out.sort(key=lambda r: int(r.get("published_ts") or r.get("fetched_at") or 0))
        return out[:limit]

    def list_articles_by_filter(
        self,
        *,
        sources: List[str],
        since_ts: int,
        until_ts: Optional[int] = None,
        limit: int = 2000,
    ) -> List[Dict[str, Any]]:
        """
        Filtrera artiklar baserat på:
          - source ∈ sources
          - published_ts >= since_ts
          - och om until_ts: published_ts <= until_ts
        """
        srcset: Set[str] = {str(s) for s in (sources or []) if str(s).strip()}
        db = self._db()
        at = db.table("articles")

        def match(row: Dict[str, Any]) -> bool:
            if srcset and row.get("source") not in srcset:
                return False
            ts = row.get("published_ts")
            if not isinstance(ts, int) or ts <= 0:
                # om published_ts saknas: fall back fetched_at
                ts = row.get("fetched_at")
                if not isinstance(ts, int) or ts <= 0:
                    return False
            if ts < since_ts:
                return False
            if until_ts is not None and ts > until_ts:
                return False
            return True

        rows = at.search(match)
        db.close()

        rows_sorted = sorted(rows, key=lambda r: int(r.get("published_ts") or r.get("fetched_at") or 0))
        return [dict(r) for r in rows_sorted[:limit]]

    # ---- Legacy (bakåtkomp; används ej för urval längre)
    def list_unsummarized_articles(self, limit: int = 200) -> List[Dict[str, Any]]:
        db = self._db()
        A = Query()
        res = db.table("articles").search((A.summarized != True))  # noqa: E712
        db.close()
        return res[:limit]  # pyright: ignore[reportReturnType]

    def mark_articles_summarized(self, article_ids: List[str]) -> None:
        """
        Legacy: Behålls för bakåtkomp, men pipeline använder den inte längre.
        """
        db = self._db()
        A = Query()
        ts = int(time.time())
        for aid in article_ids:
            db.table("articles").update({"summarized": True, "summarized_at": ts}, A.id == aid)
        db.close()

    # ---- Summaries (legacy)
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

    # ---- Summary documents (new)
    def save_summary_doc(self, summary_doc: Dict[str, Any]) -> Any:
        db = self._db()
        t = db.table("summary_docs")
        Q = Query()

        doc = dict(summary_doc or {})
        if "created" not in doc:
            doc["created"] = int(time.time())
        if "kind" not in doc:
            doc["kind"] = "summary"

        if doc.get("id"):
            sid = str(doc["id"])
            t.upsert(doc, Q.id == sid)
            db.close()
            return sid

        doc_id = t.insert(doc)
        try:
            t.update({"id": f"summary_doc_{doc_id}"}, doc_ids=[doc_id])
        except Exception:
            pass
        db.close()
        return doc_id

    def save_summary_document(self, summary_doc: Dict[str, Any]) -> Any:
        return self.save_summary_doc(summary_doc)

    def put_summary_doc(self, summary_doc: Dict[str, Any]) -> Any:
        return self.save_summary_doc(summary_doc)

    def insert_summary_doc(self, summary_doc: Dict[str, Any]) -> Any:
        return self.save_summary_doc(summary_doc)

    def get_summary_doc(self, summary_doc_id: str) -> Optional[Dict[str, Any]]:
        db = self._db()
        t = db.table("summary_docs")
        Q = Query()
        rows = t.search(Q.id == str(summary_doc_id))
        db.close()
        return rows[0] if rows else None

    def get_summary_document(self, summary_doc_id: str) -> Optional[Dict[str, Any]]:
        return self.get_summary_doc(summary_doc_id)

    def list_summary_docs(self) -> List[Dict[str, Any]]:
        db = self._db()
        docs = list(db.table("summary_docs"))
        db.close()
        out = [dict(d) for d in docs]
        out.sort(key=lambda r: r.get("created", 0), reverse=True)
        return out

    def get_latest_summary_doc(self) -> Optional[Dict[str, Any]]:
        docs = self.list_summary_docs()
        return docs[0] if docs else None

    # ---- Jobs
    def create_job(self) -> int:
        db = self._db()
        jid = db.table("jobs").insert(
            {
                "created_at": int(time.time()),
                "started_at": None,
                "finished_at": None,
                "status": "queued",
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
        logger.info(f"Job {job_id} updated: {fields}")
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
    def put_temp_summary(self, job_id: int, payload: Dict[str, Any]) -> None:
        db = self._db()
        t = db.table("temp_summaries")
        T = Query()
        doc = dict(payload or {})
        doc["job_id"] = job_id
        if "created_at" not in doc:
            doc["created_at"] = int(time.time())
        t.upsert(doc, T.job_id == job_id)
        db.close()

    def save_temp_summary(self, job_id: int, summary_text: str, meta: Dict[str, Any]) -> None:
        self.put_temp_summary(job_id, {"summary": summary_text, "meta": meta or {}})

    def get_temp_summary(self, job_id: int) -> Optional[Dict[str, Any]]:
        db = self._db()
        t = db.table("temp_summaries")
        T = Query()
        rows = t.search(T.job_id == job_id)
        db.close()
        return rows[0] if rows else None
