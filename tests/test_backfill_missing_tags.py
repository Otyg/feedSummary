import unittest

from backfill_missing_tags import _find_untagged_article_ids


class CouchLikeStore:
    """Minimal store exposing CouchDB's public article/tag iteration shape."""

    def __init__(self):
        self.articles = [
            {"id": "oldest-untagged"},
            {"id": "tagged"},
            {"id": "dangling-relation"},
            {"id": "newest-untagged"},
        ]

    def iter_articles(self):
        yield from self.articles

    def iter_articles_with_tags(self):
        yield {"article": {"id": "tagged"}, "tags": [{"id": 1, "name": "security"}]}
        # A relation whose tag document is missing does not count as a tag.
        yield {"article": {"id": "dangling-relation"}, "tags": []}

    def get_article_tags(self, article_id):
        raise AssertionError("CouchDB must not be queried once per article")


class FindUntaggedArticleIdsTests(unittest.TestCase):
    def test_finds_couchdb_articles_without_per_article_queries(self):
        self.assertEqual(
            ["oldest-untagged", "dangling-relation", "newest-untagged"],
            _find_untagged_article_ids(CouchLikeStore(), limit=None),
        )

    def test_limit_keeps_oldest_untagged_articles(self):
        self.assertEqual(
            ["oldest-untagged", "dangling-relation"],
            _find_untagged_article_ids(CouchLikeStore(), limit=2),
        )

    def test_falls_back_for_stores_without_bulk_tag_iteration(self):
        class LegacyStore:
            def iter_articles(self):
                yield {"id": "untagged"}
                yield {"id": "tagged"}

            def get_article_tags(self, article_id):
                return [{"id": 1}] if article_id == "tagged" else []

        self.assertEqual(
            ["untagged"],
            _find_untagged_article_ids(LegacyStore(), limit=None),
        )


if __name__ == "__main__":
    unittest.main()
