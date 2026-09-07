import unittest

from calibrate_long_term_clustering import (
    EmbeddedArticle,
    calculate_similarity_report,
    collect_compatible_articles,
)


class LongTermCalibrationTests(unittest.TestCase):
    def test_collects_the_most_common_valid_signature(self):
        articles = [
            self.article("a", [1, 0], model="main"),
            self.article("b", [0, 1], model="main"),
            self.article("c", [1, 0, 0], model="other"),
            {"id": "missing", "similarity_embedding_model": "main"},
        ]

        selected, metadata = collect_compatible_articles(articles)

        self.assertEqual(["a", "b"], [item.article_id for item in selected])
        self.assertEqual(2, metadata["articles_selected"])
        self.assertEqual(2, metadata["embedding_signature"]["dimensions"])
        self.assertEqual(1, metadata["skipped_by_reason"]["missing_embedding"])
        self.assertEqual(1, metadata["skipped_by_reason"]["incompatible_signature"])

    def test_similarity_report_counts_pairs_and_label_metrics(self):
        articles = [
            self.embedded("a", (1.0, 0.0)),
            self.embedded("b", (0.99, 0.01)),
            self.embedded("c", (0.0, 1.0)),
        ]

        report = calculate_similarity_report(
            articles,
            thresholds=(0.8,),
            labels={"a": "event-1", "b": "event-1", "c": "event-2"},
            block_size=2,
        )

        threshold = report["thresholds"][0]
        self.assertEqual(1, threshold["pairs_at_or_above"])
        self.assertEqual(2, threshold["articles_with_neighbour_at_or_above"])
        self.assertEqual(1, threshold["labelled_pair_metrics"]["true_positive"])
        self.assertEqual(0, threshold["labelled_pair_metrics"]["false_positive"])
        self.assertEqual(1.0, threshold["labelled_pair_metrics"]["precision"])
        self.assertEqual(1.0, threshold["labelled_pair_metrics"]["recall"])

    @staticmethod
    def article(article_id, vector, model):
        return {
            "id": article_id,
            "title": article_id,
            "source": "source",
            "published_ts": 1,
            "similarity_embedding_vector": vector,
            "similarity_embedding_model": model,
            "similarity_embedding_instruction": "Represent the event",
        }

    @staticmethod
    def embedded(article_id, vector):
        return EmbeddedArticle(
            article_id=article_id,
            title=article_id,
            source="source",
            published_ts=1,
            vector=vector,
            signature=("model", len(vector), "instruction"),
        )


if __name__ == "__main__":
    unittest.main()
