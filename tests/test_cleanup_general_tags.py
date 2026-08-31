import unittest

from cleanup_general_tags import delete_candidates, find_cleanup_candidates


class FakeStore:
    def __init__(self):
        self.tags = {
            1: {"id": 1, "name": "unused", "category": "GENERAL"},
            2: {"id": 2, "name": "once", "category": "general"},
            3: {"id": 3, "name": "often", "category": "GENERAL"},
            4: {"id": 4, "name": "other", "category": "THREAT"},
        }
        self.article_tags = {
            "a1": {2, 3, 4},
            "a2": {3},
        }
        self.deleted = []

    def get_all_tags(self):
        return [tag.copy() for tag in self.tags.values()]

    def get_articles_by_tags(self, tag_names, match_mode="any"):
        requested = {name.casefold() for name in tag_names}
        return [
            {"id": article_id}
            for article_id, tag_ids in self.article_tags.items()
            if any(self.tags[tag_id]["name"].casefold() in requested for tag_id in tag_ids)
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


class CleanupGeneralTagsTests(unittest.TestCase):
    def test_finds_only_general_tags_used_at_most_once(self):
        store = FakeStore()

        candidates = find_cleanup_candidates(store)

        self.assertEqual(
            [("unused", 0), ("once", 1)],
            [(candidate["tag"], candidate["uses"]) for candidate in candidates],
        )

    def test_deletes_candidates_and_their_article_relations(self):
        store = FakeStore()
        candidates = find_cleanup_candidates(store)

        results = delete_candidates(store, candidates)

        self.assertEqual([1, 2], store.deleted)
        self.assertTrue(all(result["deleted"] for result in results))
        self.assertNotIn(2, store.article_tags["a1"])
        self.assertIn(4, store.tags)

    def test_rechecks_usage_immediately_before_deleting(self):
        store = FakeStore()
        candidates = find_cleanup_candidates(store)
        store.article_tags["a2"].add(2)

        results = delete_candidates(store, candidates)

        once_result = next(result for result in results if result["tag"] == "once")
        self.assertFalse(once_result["deleted"])
        self.assertEqual(2, once_result["uses"])
        self.assertIn(2, store.tags)

    def test_rechecks_category_immediately_before_deleting(self):
        store = FakeStore()
        candidates = find_cleanup_candidates(store)
        store.tags[2]["category"] = "THREAT"

        results = delete_candidates(store, candidates)

        once_result = next(result for result in results if result["tag"] == "once")
        self.assertFalse(once_result["deleted"])
        self.assertIn("inte längre kategorin GENERAL", once_result["error"])
        self.assertIn(2, store.tags)


if __name__ == "__main__":
    unittest.main()
