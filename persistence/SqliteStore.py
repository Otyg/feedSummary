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

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _now_ts() -> int:
    return int(time.time())


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        if isinstance(v, bool):
            return int(v)
        return int(v)
    except Exception:
        return default


def _published_ts(doc: Dict[str, Any]) -> int:
    ts = doc.get("published_ts")
    if isinstance(ts, int) and ts > 0:
        return ts
    fa = doc.get("fetched_at")
    if isinstance(fa, int) and fa > 0:
        return fa
    return 0


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _json_loads(s: Optional[str]) -> Any:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


class SqliteStore:
    """
    SQLite-backed store.

    Design choice:
    - Store the full document as JSON (doc_json) for flexibility.
    - Also store a few indexed columns for filtering/sorting (source, published_ts, fetched_at, etc.).
    """

    def __init__(
        self,
        path: str = "news_docs.sqlite",
        *,
        pragmas: Optional[Dict[str, str]] = None,
    ):
        self.path = str(Path(path))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._pragmas = pragmas or {
            "journal_mode": "WAL",
            "synchronous": "NORMAL",
            "temp_store": "MEMORY",
            "foreign_keys": "ON",
        }
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        for k, v in self._pragmas.items():
            try:
                con.execute(f"PRAGMA {k}={v}")
            except Exception:
                pass
        return con

    def _init_db(self) -> None:
        con = self._connect()
        try:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    id            TEXT PRIMARY KEY,
                    url           TEXT,
                    source        TEXT,
                    title         TEXT,
                    published     TEXT,
                    published_ts  INTEGER,
                    fetched_at    INTEGER,
                    content_hash  TEXT,
                    summarized    INTEGER DEFAULT 0,
                    summarized_at INTEGER,
                    doc_json      TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_articles_source_ts
                  ON articles(source, published_ts);

                CREATE INDEX IF NOT EXISTS idx_articles_published_ts
                  ON articles(published_ts);

                CREATE INDEX IF NOT EXISTS idx_articles_fetched_at
                  ON articles(fetched_at);

                CREATE TABLE IF NOT EXISTS summary_docs (
                    id       TEXT PRIMARY KEY,
                    created  INTEGER,
                    kind     TEXT,
                    doc_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_summary_docs_created
                  ON summary_docs(created);

                CREATE TABLE IF NOT EXISTS jobs (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at  INTEGER,
                    started_at  INTEGER,
                    finished_at INTEGER,
                    status      TEXT,
                    message     TEXT,
                    summary_id  TEXT,
                    fields_json TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_created_at
                  ON jobs(created_at);

                CREATE TABLE IF NOT EXISTS temp_summaries (
                    job_id     INTEGER PRIMARY KEY,
                    created_at INTEGER,
                    summary    TEXT,
                    meta_json  TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS ux_articles_url
                    ON articles(url)
                    WHERE url IS NOT NULL AND url != '';
                """
            )
            con.commit()
        finally:
            con.close()

    def get_article(self, article_id: str) -> Optional[Dict[str, Any]]:
        con = self._connect()
        try:
            row = con.execute(
                "SELECT doc_json FROM articles WHERE id = ?",
                (str(article_id),),
            ).fetchone()
            if not row:
                return None
            doc = _json_loads(row["doc_json"]) or {}
            if "id" not in doc:
                doc["id"] = str(article_id)
            return doc
        finally:
            con.close()

    def upsert_article(self, article_doc: Dict[str, Any]) -> None:
        if not isinstance(article_doc, dict):
            raise ValueError("article_doc must be a dict")
        if not article_doc.get("id"):
            raise ValueError("article_doc must contain 'id'")

        doc = dict(article_doc)
        aid = str(doc["id"])

        url = (doc.get("url") or "").strip() or None
        source = (doc.get("source") or "").strip() or None
        title = (doc.get("title") or "").strip() or None
        published = (doc.get("published") or "").strip() or None
        published_ts = _safe_int(doc.get("published_ts"), 0) or None
        fetched_at = _safe_int(doc.get("fetched_at"), 0) or None
        content_hash = (doc.get("content_hash") or "").strip() or None

        summarized = 1 if bool(doc.get("summarized")) else 0
        summarized_at = _safe_int(doc.get("summarized_at"), 0) or None

        doc_json = _json_dumps(doc)

        con = self._connect()
        try:
            con.execute(
                """
                INSERT INTO articles (
                    id, url, source, title, published, published_ts, fetched_at,
                    content_hash, summarized, summarized_at, doc_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    url=excluded.url,
                    source=excluded.source,
                    title=excluded.title,
                    published=excluded.published,
                    published_ts=excluded.published_ts,
                    fetched_at=excluded.fetched_at,
                    content_hash=excluded.content_hash,
                    summarized=excluded.summarized,
                    summarized_at=excluded.summarized_at,
                    doc_json=excluded.doc_json
                """,
                (
                    aid,
                    url,
                    source,
                    title,
                    published,
                    published_ts,
                    fetched_at,
                    content_hash,
                    summarized,
                    summarized_at,
                    doc_json,
                ),
            )
            con.commit()
        finally:
            con.close()

    def list_articles(self, limit: int = 2000) -> List[Dict[str, Any]]:
        con = self._connect()
        try:
            rows = con.execute(
                """
                SELECT doc_json
                FROM articles
                ORDER BY COALESCE(published_ts, fetched_at, 0) ASC
                LIMIT ?
                """,
                (_safe_int(limit, 2000),),
            ).fetchall()

            out: List[Dict[str, Any]] = []
            for r in rows:
                doc = _json_loads(r["doc_json"]) or {}
                if isinstance(doc, dict):
                    out.append(doc)
            return out
        finally:
            con.close()

    def list_articles_by_filter(
        self,
        *,
        sources: List[str],
        since_ts: int,
        until_ts: Optional[int] = None,
        limit: int = 2000,
    ) -> List[Dict[str, Any]]:
        srcs = [str(s).strip() for s in (sources or []) if str(s).strip()]
        if not srcs:
            return []

        since_i = _safe_int(since_ts, 0)
        until_i = _safe_int(until_ts, 0) if until_ts is not None else None
        lim = _safe_int(limit, 2000)

        placeholders = ",".join(["?"] * len(srcs))
        params: List[Any] = []
        params.extend(srcs)
        params.append(since_i)

        where = (
            f"source IN ({placeholders}) AND COALESCE(published_ts, fetched_at, 0) >= ?"
        )
        if until_i is not None:
            where += " AND COALESCE(published_ts, fetched_at, 0) <= ?"
            params.append(until_i)

        params.append(lim)

        con = self._connect()
        try:
            rows = con.execute(
                f"""
                SELECT doc_json
                FROM articles
                WHERE {where}
                ORDER BY COALESCE(published_ts, fetched_at, 0) ASC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()

            out: List[Dict[str, Any]] = []
            for r in rows:
                doc = _json_loads(r["doc_json"]) or {}
                if isinstance(doc, dict):
                    out.append(doc)
            return out
        finally:
            con.close()

    def list_unsummarized_articles(self, limit: int = 200) -> List[Dict[str, Any]]:
        con = self._connect()
        try:
            rows = con.execute(
                """
                SELECT doc_json
                FROM articles
                WHERE summarized != 1
                ORDER BY COALESCE(published_ts, fetched_at, 0) ASC
                LIMIT ?
                """,
                (_safe_int(limit, 200),),
            ).fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                doc = _json_loads(r["doc_json"]) or {}
                if isinstance(doc, dict):
                    out.append(doc)
            return out
        finally:
            con.close()

    def mark_articles_summarized(self, article_ids: List[str]) -> None:
        ids = [str(x) for x in (article_ids or []) if str(x).strip()]
        if not ids:
            return

        ts = _now_ts()
        con = self._connect()
        try:
            con.execute("BEGIN")
            for aid in ids:
                row = con.execute(
                    "SELECT doc_json FROM articles WHERE id = ?",
                    (aid,),
                ).fetchone()
                if not row:
                    continue
                doc = _json_loads(row["doc_json"]) or {}
                if isinstance(doc, dict):
                    doc["summarized"] = True
                    doc["summarized_at"] = ts
                    con.execute(
                        """
                        UPDATE articles
                        SET summarized = 1, summarized_at = ?, doc_json = ?
                        WHERE id = ?
                        """,
                        (ts, _json_dumps(doc), aid),
                    )
                else:
                    con.execute(
                        "UPDATE articles SET summarized = 1, summarized_at = ? WHERE id = ?",
                        (ts, aid),
                    )
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def save_summary_doc(self, summary_doc: Dict[str, Any]) -> Any:
        if not isinstance(summary_doc, dict):
            raise ValueError("summary_doc must be a dict")

        doc = dict(summary_doc)
        created = _safe_int(doc.get("created"), 0) or _now_ts()
        kind = str(doc.get("kind") or "summary")
        sid = str(doc.get("id") or "").strip()

        if not sid:
            sid = f"summary_{created}"

        doc["id"] = sid
        doc["created"] = created
        doc["kind"] = kind

        con = self._connect()
        try:
            con.execute(
                """
                INSERT INTO summary_docs (id, created, kind, doc_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    created=excluded.created,
                    kind=excluded.kind,
                    doc_json=excluded.doc_json
                """,
                (sid, created, kind, _json_dumps(doc)),
            )
            con.commit()
            return sid
        finally:
            con.close()

    def get_summary_doc(self, summary_doc_id: str) -> Optional[Dict[str, Any]]:
        sid = str(summary_doc_id)
        con = self._connect()
        try:
            row = con.execute(
                "SELECT doc_json FROM summary_docs WHERE id = ?",
                (sid,),
            ).fetchone()
            if not row:
                return None
            doc = _json_loads(row["doc_json"]) or {}
            if isinstance(doc, dict) and "id" not in doc:
                doc["id"] = sid
            return doc if isinstance(doc, dict) else None
        finally:
            con.close()

    def list_summary_docs(self) -> List[Dict[str, Any]]:
        con = self._connect()
        try:
            rows = con.execute(
                """
                SELECT doc_json
                FROM summary_docs
                ORDER BY COALESCE(created, 0) DESC
                """
            ).fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                doc = _json_loads(r["doc_json"]) or {}
                if isinstance(doc, dict):
                    out.append(doc)
            return out
        finally:
            con.close()

    def get_latest_summary_doc(self) -> Optional[Dict[str, Any]]:
        con = self._connect()
        try:
            row = con.execute(
                """
                SELECT doc_json
                FROM summary_docs
                ORDER BY COALESCE(created, 0) DESC
                LIMIT 1
                """
            ).fetchone()
            if not row:
                return None
            doc = _json_loads(row["doc_json"]) or {}
            return doc if isinstance(doc, dict) else None
        finally:
            con.close()

    def create_job(self) -> int:
        con = self._connect()
        try:
            cur = con.execute(
                """
                INSERT INTO jobs (created_at, started_at, finished_at, status, message, summary_id, fields_json)
                VALUES (?, NULL, NULL, ?, ?, NULL, ?)
                """,
                (_now_ts(), "queued", "", _json_dumps({})),
            )
            con.commit()
            jid = int(cur.lastrowid)  # type: ignore
            logger.info("Job %s created", jid)
            return jid
        finally:
            con.close()

    def update_job(self, job_id: int, **fields) -> None:
        jid = _safe_int(job_id, 0)
        if jid <= 0:
            raise ValueError("job_id must be a positive int")

        known_cols = {
            "created_at",
            "started_at",
            "finished_at",
            "status",
            "message",
            "summary_id",
        }

        con = self._connect()
        try:
            row = con.execute(
                "SELECT fields_json FROM jobs WHERE id = ?", (jid,)
            ).fetchone()
            extra = _json_loads(row["fields_json"]) if row else {}
            if not isinstance(extra, dict):
                extra = {}

            set_parts: List[str] = []
            params: List[Any] = []

            for k in list(fields.keys()):
                if k in known_cols:
                    set_parts.append(f"{k} = ?")
                    params.append(fields[k])

            for k, v in fields.items():
                if k not in known_cols:
                    extra[k] = v

            set_parts.append("fields_json = ?")
            params.append(_json_dumps(extra))

            params.append(jid)

            con.execute(
                f"UPDATE jobs SET {', '.join(set_parts)} WHERE id = ?",
                tuple(params),
            )
            con.commit()
            logger.info("Job %s updated: %s", jid, fields)
        finally:
            con.close()

    def get_job(self, job_id: int) -> Optional[Dict[str, Any]]:
        jid = _safe_int(job_id, 0)
        con = self._connect()
        try:
            row = con.execute(
                """
                SELECT id, created_at, started_at, finished_at, status, message, summary_id, fields_json
                FROM jobs
                WHERE id = ?
                """,
                (jid,),
            ).fetchone()
            if not row:
                return None

            out: Dict[str, Any] = {
                "id": int(row["id"]),
                "created_at": row["created_at"],
                "started_at": row["started_at"],
                "finished_at": row["finished_at"],
                "status": row["status"],
                "message": row["message"],
                "summary_id": row["summary_id"],
            }
            extra = _json_loads(row["fields_json"])
            if isinstance(extra, dict):
                out.update(extra)
            return out
        finally:
            con.close()

    def get_articles_by_ids(self, article_ids: List[str]) -> List[Dict[str, Any]]:
        ids = [str(x) for x in (article_ids or []) if str(x).strip()]
        if not ids:
            return []

        placeholders = ",".join(["?"] * len(ids))
        con = self._connect()
        try:
            rows = con.execute(
                f"SELECT doc_json FROM articles WHERE id IN ({placeholders})",
                tuple(ids),
            ).fetchall()

            by_id: Dict[str, Dict[str, Any]] = {}
            for r in rows:
                doc = _json_loads(r["doc_json"]) or {}
                if isinstance(doc, dict) and doc.get("id"):
                    by_id[str(doc["id"])] = doc

            ordered: List[Dict[str, Any]] = []
            for aid in ids:
                if aid in by_id:
                    ordered.append(by_id[aid])
            return ordered
        finally:
            con.close()

    def save_temp_summary(
        self, job_id: int, summary_text: str, meta: Dict[str, Any]
    ) -> None:
        jid = _safe_int(job_id, 0)
        con = self._connect()
        try:
            con.execute(
                """
                INSERT INTO temp_summaries (job_id, created_at, summary, meta_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    created_at=excluded.created_at,
                    summary=excluded.summary,
                    meta_json=excluded.meta_json
                """,
                (jid, _now_ts(), summary_text or "", _json_dumps(meta or {})),
            )
            con.commit()
        finally:
            con.close()

    def get_temp_summary(self, job_id: int) -> Optional[Dict[str, Any]]:
        jid = _safe_int(job_id, 0)
        con = self._connect()
        try:
            row = con.execute(
                "SELECT job_id, created_at, summary, meta_json FROM temp_summaries WHERE job_id = ?",
                (jid,),
            ).fetchone()
            if not row:
                return None
            return {
                "job_id": int(row["job_id"]),
                "created_at": row["created_at"],
                "summary": row["summary"],
                "meta": _json_loads(row["meta_json"]) or {},
            }
        finally:
            con.close()
