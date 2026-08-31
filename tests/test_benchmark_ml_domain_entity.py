import json
import unittest
from tempfile import TemporaryDirectory
from typing import ClassVar

from benchmark_ml_domain_entity import audit_predictions, render_markdown, write_reports


class FakeSettings:
    algorithm = "logistic_regression"
    categories = ("DOMAIN_ENTITY",)
    threshold = 0.8
    max_tags_per_article = 5


class FakeTagger:
    ready = True
    settings = FakeSettings()
    label_names = ("acme", "globex")
    model_metadata: ClassVar[dict] = {
        "classifier": "logistic_regression",
        "categories": ["DOMAIN_ENTITY"],
        "threshold": 0.8,
        "training_articles": 40,
        "label_count": 2,
        "trained_at": "2026-08-29T00:00:00+00:00",
    }

    def __init__(self):
        self.scores = {
            "article-1": [("acme", 0.95), ("globex", 0.85)],
            "article-2": [("globex", 0.90), ("acme", 0.10)],
            "article-3": [("acme", 0.20), ("globex", 0.10)],
        }

    def embedding_incompatibility_reason(self, article):
        return "missing_embedding" if article["id"] == "article-4" else None

    def score_names(self, article):
        return self.scores.get(article["id"], [])

    def predict_tags(self, article, store, *, scores=None):
        predictions = []
        for name, probability in scores or []:
            if probability < self.settings.threshold:
                continue
            tag = store.get_tag_by_name(name)
            predictions.append({**tag, "ml_probability": probability})
        return predictions[: self.settings.max_tags_per_article]


class FakeStore:
    def __init__(self):
        self.articles = [
            {"id": "article-1", "title": "Acme incident", "embedding_vector": [1.0]},
            {"id": "article-2", "title": "Previously untagged", "embedding_vector": [2.0]},
            {"id": "article-3", "title": "Rare entity", "embedding_vector": [3.0]},
            {"id": "article-4", "title": "No embedding"},
        ]
        self.tags = {
            "acme": {"id": 1, "name": "acme", "category": "DOMAIN_ENTITY"},
            "globex": {"id": 2, "name": "globex", "category": "DOMAIN_ENTITY"},
            "rare": {"id": 3, "name": "rare", "category": "DOMAIN_ENTITY"},
            "general": {"id": 4, "name": "general", "category": "GENERAL"},
        }
        self.assignments = {
            "article-1": [self.tags["acme"], self.tags["general"]],
            "article-2": [],
            "article-3": [self.tags["rare"]],
            "article-4": [self.tags["acme"]],
        }

    def iter_articles(self, limit=None):
        yield from self.articles[:limit]

    def get_article_tags(self, article_id):
        return [dict(tag) for tag in self.assignments[article_id]]

    def get_tag_by_name(self, name):
        return dict(self.tags[name])


class DomainEntityAuditTests(unittest.TestCase):
    def test_audit_compares_predictions_and_actual_tags_without_writes(self):
        report = audit_predictions(FakeStore(), FakeTagger())

        summary = report["summary"]
        self.assertEqual(4, summary["articles"])
        self.assertEqual(3, summary["evaluated_articles"])
        self.assertEqual(1, summary["skipped_articles"])
        self.assertEqual(2, summary["articles_with_suggested_additions"])
        self.assertEqual(1, summary["untagged_articles_with_suggestions"])
        self.assertEqual({"missing_embedding": 1}, summary["skipped_by_reason"])

        eligible = summary["eligible_vocabulary_metrics"]
        self.assertEqual(1, eligible["true_positive"])
        self.assertEqual(2, eligible["false_positive"])
        self.assertEqual(0, eligible["false_negative"])
        self.assertAlmostEqual(1 / 3, eligible["precision"])
        self.assertEqual(1.0, eligible["recall"])

        all_actual = summary["all_actual_tags_metrics"]
        self.assertEqual(1, all_actual["false_negative"])
        self.assertEqual(0.5, all_actual["recall"])

        by_id = {row["article_id"]: row for row in report["articles"]}
        self.assertEqual(
            [{"name": "globex", "probability": 0.85}],
            by_id["article-1"]["suggested_additions"],
        )
        self.assertEqual("untagged_with_suggestions", by_id["article-2"]["status"])
        self.assertEqual(["rare"], by_id["article-3"]["actual_outside_model_vocabulary"])
        self.assertEqual("not_evaluated", by_id["article-4"]["status"])
        self.assertTrue(report["scope"]["read_only"])

    def test_limit_is_applied_to_all_article_iterator(self):
        report = audit_predictions(FakeStore(), FakeTagger(), limit=2)

        self.assertEqual(2, report["summary"]["articles"])
        self.assertEqual(["article-1", "article-2"], [row["article_id"] for row in report["articles"]])

    def test_writes_json_and_human_readable_markdown(self):
        report = audit_predictions(FakeStore(), FakeTagger())

        with TemporaryDirectory() as directory:
            json_path, markdown_path = write_reports(report, directory)
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(4, loaded["summary"]["articles"])
        self.assertIn("# ML-audit: DOMAIN_ENTITY", markdown)
        self.assertIn("Föreslagna tillägg att granska", markdown)
        self.assertIn("globex (0.850)", markdown)
        self.assertIn("Previously untagged", markdown)

    def test_markdown_marks_in_sample_limitations(self):
        markdown = render_markdown(audit_predictions(FakeStore(), FakeTagger()))

        self.assertIn("in-sample-audit", markdown)
        self.assertIn("inte automatiska", markdown)


if __name__ == "__main__":
    unittest.main()
