#!/usr/bin/env python3
"""Migrate a FeedSummary CouchDB store to MongoDB.

The command is read-only unless ``--apply`` is supplied.  Documents are copied
collection-for-collection so ids, embeddings, tag relations, jobs and summaries
are preserved.  Existing documents with the same id are replaced, making a
normal migration safe to resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from feedsummary_core.persistence import CouchDBStore, MongoDBStore

try:
    from pymongo import ReplaceOne
except ImportError:  # pragma: no cover - dependency is checked by MongoDBStore
    ReplaceOne = None  # type: ignore[assignment]


COLLECTIONS: Sequence[str] = (
    "articles",
    "article_tags",
    "counters",
    "jobs",
    "summary_docs",
    "tag_categories",
    "tag_relations",
    "tags",
    "temp_summaries",
)


@dataclass
class CollectionReport:
    source: int = 0
    target_before: int = 0
    written: int = 0
    verified: int = 0


def _chunks(items: List[Dict[str, Any]], size: int) -> Iterator[List[Dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _canonical(doc: Dict[str, Any]) -> bytes:
    """Return a stable representation for cross-database verification."""
    return json.dumps(
        doc,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _digest(documents: Iterable[Dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    rows = sorted((_canonical(dict(doc)) for doc in documents))
    for row in rows:
        digest.update(len(row).to_bytes(8, "big"))
        digest.update(row)
    return digest.hexdigest()


def _write_batch(collection: Any, documents: List[Dict[str, Any]]) -> int:
    if not documents:
        return 0
    if ReplaceOne is not None:
        operations = [ReplaceOne({"_id": doc["_id"]}, doc, upsert=True) for doc in documents]
        try:
            collection.bulk_write(operations, ordered=True)
            return len(documents)
        except TypeError:
            # Some Mongo-compatible servers lag PyMongo's bulk API.
            pass
    for doc in documents:
        collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)
    return len(documents)


def _verify_collection(
    source_collection: Any,
    target_collection: Any,
    *,
    batch_size: int,
) -> int:
    """Verify every source document against the corresponding target document."""
    verified = 0
    batch: List[Dict[str, Any]] = []
    for source_doc in source_collection.find({}):
        batch.append(dict(source_doc))
        if len(batch) < batch_size:
            continue
        verified += _verify_batch(batch, target_collection)
        batch.clear()
    if batch:
        verified += _verify_batch(batch, target_collection)
    return verified


def _verify_batch(source_docs: List[Dict[str, Any]], target_collection: Any) -> int:
    ids = [doc["_id"] for doc in source_docs]
    targets = {doc["_id"]: dict(doc) for doc in target_collection.find({"_id": {"$in": ids}})}
    for source_doc in source_docs:
        target_doc = targets.get(source_doc["_id"])
        if target_doc is None:
            raise RuntimeError(f"Måldokument saknas: {source_doc['_id']!r}")
        if _digest([source_doc]) != _digest([target_doc]):
            raise RuntimeError(f"Verifieringen misslyckades för dokument {source_doc['_id']!r}")
    return len(source_docs)


def migrate(
    *,
    couch_url: str,
    couch_database: str,
    mongo_uri: str,
    mongo_database: str,
    couch_username: Optional[str] = None,
    couch_password: Optional[str] = None,
    batch_size: int = 500,
    apply: bool = False,
    replace_target: bool = False,
    verify: bool = True,
    couch_client: Any = None,
    mongo_client: Any = None,
) -> Dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size måste vara positivt")
    if replace_target and not apply:
        raise ValueError("--replace-target kräver --apply")

    source = CouchDBStore(
        url=couch_url,
        database=couch_database,
        username=couch_username,
        password=couch_password,
        client=couch_client,
        initialize_schema=False,
    )
    target = MongoDBStore(
        uri=mongo_uri,
        database=mongo_database,
        client=mongo_client,
        initialize_schema=False,
    )
    report: Dict[str, Any] = {
        "mode": "apply" if apply else "preview",
        "source": {"provider": "couchdb", "database": couch_database},
        "target": {"provider": "mongodb", "database": mongo_database},
        "replace_target": replace_target,
        "collections": {},
    }
    try:
        # Force an early, clear connectivity failure before any target mutation.
        couch_status, _ = source.client.request("GET", "/")
        if couch_status != 200:
            raise RuntimeError(f"CouchDB svarade med HTTP {couch_status}")
        target.client.admin.command("ping")

        for name in COLLECTIONS:
            report["collections"][name] = CollectionReport(
                source=source.db[name].count_documents({}),
                target_before=target.db[name].count_documents({}),
            ).__dict__

        if not apply:
            return report

        if replace_target:
            target.client.drop_database(mongo_database)
        target._init_db()

        for name in COLLECTIONS:
            pending: List[Dict[str, Any]] = []
            for raw_doc in source.db[name].find({}):
                pending.append(dict(raw_doc))
                if len(pending) >= batch_size:
                    report["collections"][name]["written"] += _write_batch(
                        target.db[name], pending
                    )
                    pending = []
            if pending:
                report["collections"][name]["written"] += _write_batch(
                    target.db[name], pending
                )

        if verify:
            for name in COLLECTIONS:
                report["collections"][name]["verified"] = _verify_collection(
                    source.db[name], target.db[name], batch_size=batch_size
                )

        # Do this after exact verification: older sources can have stale
        # counters, and MongoDB must never allocate an already occupied id.
        for name in ("jobs", "tags", "tag_categories"):
            target._seed_counter(name, target.db[name])
        return report
    finally:
        source.close()
        target.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migrera hela FeedSummary från CouchDB till MongoDB."
    )
    parser.add_argument(
        "--couch-url",
        default=os.environ.get("FEEDSUMMARY_COUCHDB_URL", "http://localhost:5984"),
    )
    parser.add_argument(
        "--couch-database",
        default=os.environ.get("FEEDSUMMARY_COUCHDB_DATABASE", "feedsummary"),
    )
    parser.add_argument(
        "--couch-username", default=os.environ.get("FEEDSUMMARY_COUCHDB_USER")
    )
    parser.add_argument(
        "--couch-password", default=os.environ.get("FEEDSUMMARY_COUCHDB_PASSWORD")
    )
    parser.add_argument(
        "--mongo-uri",
        default=os.environ.get("FEEDSUMMARY_MONGODB_URI", "mongodb://localhost:27017"),
    )
    parser.add_argument(
        "--mongo-database",
        default=os.environ.get("FEEDSUMMARY_MONGODB_DATABASE", "feedsummary"),
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--apply", action="store_true", help="Skriv till MongoDB (annars endast förhandsgranskning)"
    )
    parser.add_argument(
        "--replace-target",
        action="store_true",
        help="Radera den namngivna MongoDB-databasen före migrering (kräver --apply)",
    )
    parser.add_argument("--no-verify", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = migrate(
            couch_url=args.couch_url,
            couch_database=args.couch_database,
            couch_username=args.couch_username,
            couch_password=args.couch_password,
            mongo_uri=args.mongo_uri,
            mongo_database=args.mongo_database,
            batch_size=args.batch_size,
            apply=args.apply,
            replace_target=args.replace_target,
            verify=not args.no_verify,
        )
    except Exception as exc:
        print(f"Migreringen misslyckades: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not args.apply:
        print("\nFörhandsgranskning klar. Kör igen med --apply för att skriva.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
