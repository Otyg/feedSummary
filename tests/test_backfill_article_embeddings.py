import unittest

from backfill_article_embeddings import _article_selected


class BackfillArticleEmbeddingTests(unittest.TestCase):
    def test_article_selection_applies_profile_source_and_time_window(self):
        article = {"source": "Security feed", "published_ts": 200}
        self.assertTrue(
            _article_selected(
                article,
                sources={"Security feed"},
                since_ts=100,
                until_ts=300,
            )
        )
        self.assertFalse(
            _article_selected(
                article,
                sources={"Other feed"},
                since_ts=100,
                until_ts=300,
            )
        )
        self.assertFalse(
            _article_selected(
                article,
                sources=None,
                since_ts=201,
                until_ts=None,
            )
        )


if __name__ == "__main__":
    unittest.main()
