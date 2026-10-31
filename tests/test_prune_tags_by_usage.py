import unittest

from prune_tags_by_usage import delete_candidates, find_prune_candidates


class FakeStore:
    def __init__(self):
        self.tags = {
            1: {"id": 1, "name": "unused-general", "category": "GENERAL"},
            2: {"id": 2, "name": "once-general", "category": "general"},
            3: {"id": 3, "name": "often-general", "category": "GENERAL"},
            4: {"id": 4, "name": "once-threat", "category": "THREAT"},
            5: {"id": 5, "name": "often-threat", "category": "THREAT"},
            6: {"id": 6, "name": "CVE-2026-1234", "category": "GENERAL"},
            7: {"id": 7, "name": "cve-2026-5678", "category": "GENERAL"},
            8: {"id": 8, "name": "CWE-79", "category": "GENERAL"},
            9: {"id": 9, "name": "cwe-89", "category": "GENERAL"},
        }
        self.article_tags = {
            "a1": {2, 3, 4, 5},
            "a2": {3, 5},
        }
        self.deleted = []

    def get_all_tags(self):
        return [tag.copy() for tag in self.tags.values()]

    def get_articles_by_tags(self, tag_names, match_mode="any"):
        requested = {name.casefold() for name in tag_names}
        return [
            {"id": article_id}
            for article_id, tag_ids in self.article_tags.items()
            if any(
                self.tags[tag_id]["name"].casefold() in requested
                for tag_id in tag_ids
            )
        ]

    def get_article_tags(self, article_id):
        return [self.tags[tag_id].copy() for tag_id in self.article_tags[article_id]]

    def delete_tag(self, tag_id):
        if tag_id not in self.tags:
            return False
        del self.tags[tag_id]
        for tag_ids in self.article_tags.values():
            tag_ids.discard(tag_id)
        self.deleted.append(tag_id)
        return True


class PruneTagsByUsageTests(unittest.TestCase):
    def test_filters_category_case_insensitively_and_uses_strict_threshold(self):
        candidates = find_prune_candidates(
            FakeStore(),
            category="GENERAL",
            min_uses=2,
        )

        self.assertEqual(
            [("unused-general", 0), ("once-general", 1)],
            [(candidate["tag"], candidate["uses"]) for candidate in candidates],
        )

    def test_supports_another_category(self):
        candidates = find_prune_candidates(
            FakeStore(),
            category="threat",
            min_uses=2,
        )

        self.assertEqual(
            [("once-threat", 1)],
            [(candidate["tag"], candidate["uses"]) for candidate in candidates],
        )

    def test_always_ignores_cve_and_cwe_prefixes_case_insensitively(self):
        store = FakeStore()

        candidates = find_prune_candidates(
            store,
            category="GENERAL",
            min_uses=100,
        )
        candidate_names = {candidate["tag"] for candidate in candidates}

        self.assertTrue(
            {"CVE-2026-1234", "cve-2026-5678", "CWE-79", "cwe-89"}
            .isdisjoint(candidate_names)
        )

    def test_zero_minimum_never_deletes_a_tag(self):
        candidates = find_prune_candidates(
            FakeStore(),
            category="GENERAL",
            min_uses=0,
        )

        self.assertEqual([], candidates)

    def test_deletes_candidates_and_article_relations(self):
        store = FakeStore()
        candidates = find_prune_candidates(
            store,
            category="GENERAL",
            min_uses=2,
        )

        results = delete_candidates(
            store,
            candidates,
            category="GENERAL",
            min_uses=2,
        )

        self.assertEqual([1, 2], store.deleted)
        self.assertTrue(all(result["deleted"] for result in results))
        self.assertNotIn(2, store.article_tags["a1"])
        self.assertIn(4, store.tags)

    def test_rechecks_usage_before_each_delete(self):
        store = FakeStore()
        candidates = find_prune_candidates(
            store,
            category="GENERAL",
            min_uses=2,
        )
        store.article_tags["a2"].add(2)

        results = delete_candidates(
            store,
            candidates,
            category="GENERAL",
            min_uses=2,
        )

        once_result = next(
            result for result in results if result["tag"] == "once-general"
        )
        self.assertFalse(once_result["deleted"])
        self.assertEqual(2, once_result["uses"])
        self.assertIn(2, store.tags)

    def test_rechecks_category_before_each_delete(self):
        store = FakeStore()
        candidates = find_prune_candidates(
            store,
            category="GENERAL",
            min_uses=2,
        )
        store.tags[2]["category"] = "THREAT"

        results = delete_candidates(
            store,
            candidates,
            category="GENERAL",
            min_uses=2,
        )

        once_result = next(
            result for result in results if result["tag"] == "once-general"
        )
        self.assertFalse(once_result["deleted"])
        self.assertIn("inte längre kategorin GENERAL", once_result["error"])
        self.assertIn(2, store.tags)

    def test_rejects_negative_threshold(self):
        with self.assertRaisesRegex(ValueError, "0 eller större"):
            find_prune_candidates(
                FakeStore(),
                category="GENERAL",
                min_uses=-1,
            )


if __name__ == "__main__":
    unittest.main()
