import unittest

from benchmark_tagging_ml import (
    ArticleExample,
    _category_scopes,
    _embedding_subset,
    _candidate_factories,
    _fit_features,
    _format_default_names,
    _merge_categories,
    _select_common_embedding_examples,
    _transform_features,
    chronological_split,
    eligible_labels,
    select_recommendation,
)


def _example(index, *labels):
    return ArticleExample(str(index), f"article {index}", index, tuple(labels))


class BenchmarkDatasetTests(unittest.TestCase):
    def test_single_default_category_is_not_split_into_characters(self):
        self.assertEqual("DOMAIN_ENTITY", _format_default_names("DOMAIN_ENTITY"))
        self.assertEqual(
            "DOMAIN_ENTITY", _format_default_names(("DOMAIN_ENTITY",))
        )

    def test_manual_categories_are_appended_and_deduplicated_in_order(self):
        categories = _merge_categories(
            ("DOMAIN_ENTITY", "GENERAL"), ("general", "LOCATION")
        )

        self.assertEqual(["DOMAIN_ENTITY", "GENERAL", "LOCATION"], categories)
        self.assertEqual(
            [
                ("Alla kategorier", ["DOMAIN_ENTITY", "GENERAL", "LOCATION"]),
                ("Kategori DOMAIN_ENTITY", ["DOMAIN_ENTITY"]),
                ("Kategori GENERAL", ["GENERAL"]),
                ("Kategori LOCATION", ["LOCATION"]),
            ],
            _category_scopes(categories),
        )

    def test_embedding_subset_uses_most_common_compatible_signature(self):
        examples = [
            ArticleExample("1", "one", 1, (), (1.0, 2.0), "model-a"),
            ArticleExample("2", "two", 2, (), (3.0, 4.0), "model-a"),
            ArticleExample("3", "three", 3, (), (5.0,), "model-b"),
            ArticleExample("4", "four", 4, ()),
        ]

        selected, metadata = _embedding_subset(examples)

        self.assertEqual(["1", "2"], [item.article_id for item in selected])
        self.assertEqual("model-a", metadata["embedding_model"])
        self.assertEqual(2, metadata["embedding_dimension"])
        self.assertEqual(0.5, metadata["embedding_coverage"])

    def test_all_representations_receive_the_same_embedding_covered_corpus(self):
        examples = [
            ArticleExample("1", "one", 1, ("tag",), (1.0, 2.0), "model"),
            ArticleExample("2", "two", 2, (), None, ""),
            ArticleExample("3", "three", 3, ("tag",), (3.0, 4.0), "model"),
        ]

        selected, dataset, embedding = _select_common_embedding_examples(
            examples, {"target_relations": 99}
        )

        self.assertEqual(["1", "3"], [item.article_id for item in selected])
        self.assertEqual(3, dataset["source_articles"])
        self.assertEqual(2, dataset["articles"])
        self.assertEqual(2, dataset["target_relations"])
        self.assertEqual(2, embedding["embedding_articles"])

    def test_hybrid_features_are_nonnegative_for_naive_bayes(self):
        train = [
            ArticleExample("1", "alpha security", 1, (), (-1.0, 2.0), "model"),
            ArticleExample("2", "beta security", 2, (), (1.0, -2.0), "model"),
        ]
        validation = [
            ArticleExample("3", "alpha beta", 3, (), (0.0, 4.0), "model")
        ]
        candidate = _candidate_factories()["multinomial_nb"]

        x_train, artifact = _fit_features(
            "hybrid", candidate, train, max_features=128, embedding_weight=1.0
        )
        x_validation = _transform_features(
            "hybrid", artifact, validation, embedding_weight=1.0
        )

        self.assertGreater(x_train.shape[1], 2)
        self.assertGreaterEqual(x_train.min(), 0)
        self.assertGreaterEqual(x_validation.min(), 0)

    def test_chronological_split_preserves_order_and_partitions_all_rows(self):
        examples = [_example(index) for index in range(10)]

        train, validation, test = chronological_split(examples, 0.2, 0.2)

        self.assertEqual(list(range(6)), [item.timestamp for item in train])
        self.assertEqual([6, 7], [item.timestamp for item in validation])
        self.assertEqual([8, 9], [item.timestamp for item in test])

    def test_eligible_labels_excludes_unseen_and_low_support_labels(self):
        train = [_example(1, "stable"), _example(2), _example(3, "train-only")]
        validation = [_example(4, "stable", "future")]
        test = [_example(5, "stable", "future")]

        included, excluded = eligible_labels(train, validation, test, min_support=2)

        self.assertEqual(["stable"], included)
        self.assertIn("future", excluded)
        self.assertIn("train-only", excluded)

    def test_recommendation_prefers_recall_after_quality_gate(self):
        def result(name, recall, seconds):
            return {
                "algorithm": name,
                "vectorizer": "tfidf",
                "status": "ok",
                "threshold": {"qualified": True},
                "metrics": {
                    "micro_precision": 0.95,
                    "micro_recall": recall,
                    "article_hit_rate": recall,
                },
                "performance": {"train_seconds": seconds, "artifact_size_mb": 1.0},
            }

        recommendation = select_recommendation(
            [result("fast", 0.4, 0.1), result("broad", 0.7, 2.0)],
            min_precision=0.90,
            max_train_seconds=10.0,
        )

        self.assertEqual("broad", recommendation["algorithm"])
        self.assertTrue(recommendation["meets_quality_gate"])


if __name__ == "__main__":
    unittest.main()
