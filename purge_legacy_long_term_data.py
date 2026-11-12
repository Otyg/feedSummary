#!/usr/bin/env python3
"""Remove legacy long-term-analysis data and schema without deleting articles."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml

from feedsummary_core.persistence import CouchDBStore, MongoDBStore, SqliteStore, TinyDBStore
from feedsummary_core.persistence import create_store


LEGACY_NAMES = (
    "long_term_state",
    "threat_cluster_memberships",
    "threat_clusters",
    "long_term_cluster_reconciliations",
    "long_term_reconciliation_journal",
    "long_term_cluster_membership_edits",
    "long_term_membership_edit_journal",
    "long_term_assignment_journal",
    "long_term_article_quarantine",
    "threat_cluster_snapshots",
    "long_term_snapshot_journal",
    "threat_landscape_reports",
    "long_term_runs",
)


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise TypeError("Konfigurationen måste vara ett YAML-objekt")
    return config


def _store_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    raw = config.get("store")
    if not isinstance(raw, dict):
        raise TypeError("Konfigurationen saknar ett giltigt 'store'-objekt")
    result = dict(raw)
    result["initialize_schema"] = False
    if result.get("path"):
        path = Path(os.path.expandvars(os.path.expanduser(str(result["path"]))))
        if not path.is_absolute():
            path = config_path.parent / path
        result["path"] = str(path.resolve())
    return result


def _legacy_summary(document: dict[str, Any]) -> bool:
    meta = document.get("meta") if isinstance(document.get("meta"), dict) else {}
    return (
        str(document.get("kind") or "") == "threat_landscape"
        or str(document.get("id") or "").startswith("feedsummary:landscape-report:")
        or str(meta.get("authoritative_collection") or "")
        == "threat_landscape_reports"
    )


def _purge_document_store(store: Any, *, apply: bool) -> dict[str, int]:
    summary_collection = store.db.summary_docs
    summary_ids = [
        row["_id"]
        for row in summary_collection.find({})
        if isinstance(row, dict) and row.get("_id") is not None and _legacy_summary(row)
    ]
    counts = {"summary_docs:threat_landscape": len(summary_ids)}
    if summary_ids and apply:
        summary_collection.delete_many({"_id": {"$in": summary_ids}})

    if isinstance(store, CouchDBStore):
        status, databases = store.client.request("GET", "/_all_dbs")
        if status != 200 or not isinstance(databases, list):
            raise RuntimeError(f"Kunde inte lista CouchDB-databaser: HTTP {status}")
        existing = {str(name) for name in databases}
        for name in LEGACY_NAMES:
            database = f"{store.database_name}_{name}".lower()
            if database not in existing:
                counts[name] = 0
                continue
            status, details = store.client.request("GET", "/" + quote(database, safe=""))
            if status != 200 or not isinstance(details, dict):
                raise RuntimeError(f"Kunde inte läsa CouchDB-databasen {database}")
            counts[name] = int(details.get("doc_count") or 0)
            if apply:
                status, _ = store.client.request(
                    "DELETE", "/" + quote(database, safe="")
                )
                if status not in {200, 202, 404}:
                    raise RuntimeError(
                        f"Kunde inte ta bort CouchDB-databasen {database}: HTTP {status}"
                    )
        return counts

    existing = set(store.db.list_collection_names())
    for name in LEGACY_NAMES:
        counts[name] = (
            int(store.db[name].count_documents({})) if name in existing else 0
        )
        if apply and name in existing:
            store.db.drop_collection(name)
    return counts


def _purge_sqlite(store: SqliteStore, *, apply: bool) -> dict[str, int]:
    connection = store._connect()
    try:
        existing = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        counts = {
            name: (
                int(connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
                if name in existing
                else 0
            )
            for name in LEGACY_NAMES
        }
        summary_ids = []
        if "summary_docs" in existing:
            for row in connection.execute("SELECT id, doc_json FROM summary_docs"):
                try:
                    document = json.loads(row[1] or "{}")
                except (TypeError, json.JSONDecodeError):
                    continue
                if isinstance(document, dict) and _legacy_summary(document):
                    summary_ids.append(str(row[0]))
        counts["summary_docs:threat_landscape"] = len(summary_ids)
        if apply:
            for name in LEGACY_NAMES:
                connection.execute(f'DROP TABLE IF EXISTS "{name}"')
            if summary_ids:
                placeholders = ",".join("?" for _ in summary_ids)
                connection.execute(
                    f"DELETE FROM summary_docs WHERE id IN ({placeholders})",
                    summary_ids,
                )
            connection.commit()
        return counts
    finally:
        connection.close()


def _purge_tinydb(store: TinyDBStore, *, apply: bool) -> dict[str, int]:
    database = store._db()
    try:
        existing = database.tables()
        counts = {
            name: len(database.table(name)) if name in existing else 0
            for name in LEGACY_NAMES
        }
        summaries = database.table("summary_docs")
        summary_ids = [row.doc_id for row in summaries if _legacy_summary(dict(row))]
        counts["summary_docs:threat_landscape"] = len(summary_ids)
        if apply:
            for name in LEGACY_NAMES:
                if name in existing:
                    database.drop_table(name)
            if summary_ids:
                summaries.remove(doc_ids=summary_ids)
        return counts
    finally:
        database.close()


def purge(config_path: Path, *, apply: bool) -> dict[str, Any]:
    config = _load_config(config_path)
    store = create_store(_store_config(config, config_path))
    try:
        if isinstance(store, CouchDBStore):
            counts = _purge_document_store(store, apply=apply)
        elif isinstance(store, MongoDBStore):
            counts = _purge_document_store(store, apply=apply)
        elif isinstance(store, SqliteStore):
            counts = _purge_sqlite(store, apply=apply)
        elif isinstance(store, TinyDBStore):
            counts = _purge_tinydb(store, apply=apply)
        else:
            raise TypeError(f"Lagringsbackend stöds inte: {type(store).__name__}")
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()
    return {
        "dry_run": not apply,
        "deleted_or_found": counts,
        "preserved": ["articles", "tags", "article_tags", "embeddings"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Ta bort data och schema; utan flaggan görs endast en dry-run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = purge(Path(args.config).expanduser().resolve(), apply=args.yes)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as error:  # noqa: BLE001 - command boundary
        print(f"Rensning av gammal långtidsdata misslyckades: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
