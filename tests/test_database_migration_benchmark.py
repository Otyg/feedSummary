import unittest

from benchmark_databases import _measure, _percentile, _temporary_articles
from migrate_couchdb_to_mongodb import _digest, _verify_batch


class _Collection:
    def __init__(self, documents):
        self.documents = {document["_id"]: document for document in documents}

    def find(self, query):
        return [
            self.documents[value]
            for value in query["_id"]["$in"]
            if value in self.documents
        ]


class DatabaseMigrationBenchmarkTests(unittest.TestCase):
    def test_digest_is_independent_of_document_and_key_order(self):
        left = [{"_id": "b", "value": 2}, {"value": 1, "_id": "a"}]
        right = [{"_id": "a", "value": 1}, {"_id": "b", "value": 2}]
        self.assertEqual(_digest(left), _digest(right))

    def test_verify_batch_detects_changed_content(self):
        source = [{"_id": "a", "value": 1}]
        self.assertEqual(1, _verify_batch(source, _Collection(source)))
        with self.assertRaisesRegex(RuntimeError, "Verifieringen"):
            _verify_batch(source, _Collection([{"_id": "a", "value": 2}]))

    def test_write_documents_are_unique_for_every_iteration(self):
        batches = _temporary_articles("prefix_", 3, 2)
        ids = [document["id"] for batch in batches for document in batch]
        self.assertEqual(2, len(batches))
        self.assertEqual(6, len(ids))
        self.assertEqual(6, len(set(ids)))

    def test_couchdb_benchmark_ids_do_not_use_reserved_prefix(self):
        batches = _temporary_articles("fs-benchmark-run-", 1, 1)
        self.assertFalse(batches[0][0]["id"].startswith("_"))

    def test_measure_separates_warmups_and_iterations(self):
        calls = []

        def operation():
            calls.append(1)
            return [1, 2]

        report = _measure(operation, warmups=2, iterations=3, units=2)
        self.assertEqual(5, len(calls))
        self.assertEqual(3, len(report["samples_ms"]))
        self.assertEqual(2, report["result_size"])

    def test_percentile_uses_nearest_rank(self):
        self.assertEqual(5, _percentile([1, 2, 3, 4, 5], 0.95))


if __name__ == "__main__":
    unittest.main()
