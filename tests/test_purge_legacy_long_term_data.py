import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from feedsummary_core.persistence import CouchDBStore
from purge_legacy_long_term_data import _purge_document_store, purge


class _SummaryCollection:
    def find(self, query):
        return []

    def delete_many(self, query):
        raise AssertionError("Inga speglade rapporter ska tas bort i detta test")


class _CouchDatabase:
    summary_docs = _SummaryCollection()


class _CouchClient:
    def __init__(self):
        self.deleted = []

    def request(self, method, path, payload=None):
        if method == "GET" and path == "/_all_dbs":
            return 200, ["feedsummary_threat_clusters"]
        if method == "GET" and path == "/feedsummary_threat_clusters":
            return 200, {"doc_count": 3}
        if method == "DELETE":
            self.deleted.append(path)
            return 200, {"ok": True}
        raise AssertionError((method, path, payload))


class PurgeLegacyLongTermDataTests(unittest.TestCase):
    def test_couchdb_legacy_database_is_counted_and_deleted(self):
        store = CouchDBStore.__new__(CouchDBStore)
        store.db = _CouchDatabase()
        store.client = _CouchClient()
        store.database_name = "feedsummary"

        preview = _purge_document_store(store, apply=False)
        self.assertEqual(3, preview["threat_clusters"])
        self.assertEqual([], store.client.deleted)

        result = _purge_document_store(store, apply=True)
        self.assertEqual(3, result["threat_clusters"])
        self.assertEqual(["/feedsummary_threat_clusters"], store.client.deleted)

    def test_sqlite_dry_run_then_drop_preserves_normal_data(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "news.sqlite"
            config = root / "config.yaml"
            config.write_text(
                "store:\n  provider: sqlite\n  path: news.sqlite\n",
                encoding="utf-8",
            )
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE articles (id TEXT PRIMARY KEY);
                INSERT INTO articles VALUES ('article-1');
                CREATE TABLE summary_docs (id TEXT PRIMARY KEY, doc_json TEXT);
                CREATE TABLE threat_clusters (id TEXT PRIMARY KEY);
                INSERT INTO threat_clusters VALUES ('cluster-1');
                """
            )
            connection.execute(
                "INSERT INTO summary_docs VALUES (?, ?)",
                ("normal", json.dumps({"id": "normal", "kind": "daily"})),
            )
            connection.execute(
                "INSERT INTO summary_docs VALUES (?, ?)",
                (
                    "legacy",
                    json.dumps({"id": "legacy", "kind": "threat_landscape"}),
                ),
            )
            connection.commit()
            connection.close()

            preview = purge(config, apply=False)
            self.assertTrue(preview["dry_run"])
            self.assertEqual(1, preview["deleted_or_found"]["threat_clusters"])

            connection = sqlite3.connect(database)
            self.assertEqual(
                1, connection.execute("SELECT COUNT(*) FROM threat_clusters").fetchone()[0]
            )
            connection.close()

            result = purge(config, apply=True)
            self.assertFalse(result["dry_run"])
            connection = sqlite3.connect(database)
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            self.assertNotIn("threat_clusters", tables)
            self.assertEqual(
                [("normal",)],
                connection.execute("SELECT id FROM summary_docs").fetchall(),
            )
            self.assertEqual(
                [("article-1",)], connection.execute("SELECT id FROM articles").fetchall()
            )
            connection.close()


if __name__ == "__main__":
    unittest.main()
