import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from calibrate_long_term_clustering import (
    EmbeddedArticle,
    LabelledPair,
    build_pair_review_packet,
    build_portable_dataset,
    calculate_similarity_report,
    collect_compatible_articles,
    evaluate_online_clustering,
    evaluate_online_scenarios,
    load_pair_labels,
    load_portable_dataset,
    resolve_profile_sources,
    select_competing_background_articles,
    select_review_pairs,
    write_pair_review_packet,
    write_portable_dataset,
    write_review_template,
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

    def test_pair_labels_report_same_different_and_ambiguous_separately(self):
        articles = [
            self.embedded("a", (1.0, 0.0)),
            self.embedded("b", (0.99, 0.01)),
            self.embedded("c", (0.0, 1.0)),
        ]

        report = calculate_similarity_report(
            articles,
            thresholds=(0.8,),
            pair_labels=(
                LabelledPair("a", "b", "same"),
                LabelledPair("a", "c", "different"),
                LabelledPair("b", "c", "ambiguous"),
            ),
        )

        metrics = report["thresholds"][0]["labelled_pair_metrics"]
        self.assertEqual(1, metrics["true_positive"])
        self.assertEqual(1, metrics["true_negative"])
        self.assertEqual(1, metrics["ambiguous"])
        self.assertEqual(2, metrics["evaluated_pairs"])
        self.assertEqual(1.0, metrics["precision"])
        self.assertEqual(1.0, metrics["recall"])

    def test_review_template_round_trips_after_manual_labels_are_added(self):
        articles = [
            self.embedded("a", (1.0, 0.0)),
            self.embedded("b", (0.99, 0.01)),
            self.embedded("c", (0.8, 0.6)),
            self.embedded("d", (0.7, 0.714)),
        ]
        rows = select_review_pairs(
            articles,
            threshold=0.8,
            ambiguity_margin=0.03,
            pairs_per_band=1,
        )

        self.assertEqual(
            ["high_similarity", "threshold_boundary", "near_negative"],
            [row["review_band"] for row in rows],
        )
        rows[0]["judgement"] = "same"
        with TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            write_review_template(path, rows[:1])
            labels = load_pair_labels(path)

        self.assertEqual("same", labels[0].judgement)

    def test_review_selection_prefers_cross_source_pairs(self):
        articles = [
            self.embedded("a", (1.0, 0.0), source="one"),
            self.embedded("b", (1.0, 0.0), source="one"),
            self.embedded("c", (0.99, 0.01), source="two"),
        ]

        rows = select_review_pairs(articles, pairs_per_band=1)

        high = next(row for row in rows if row["review_band"] == "high_similarity")
        self.assertNotEqual(high["left_source"], high["right_source"])

    def test_review_selection_can_enforce_candidate_time_window(self):
        base_ts = 1_700_000_000
        articles = [
            self.embedded("old", (1.0, 0.0), published_ts=base_ts),
            self.embedded(
                "too-late", (1.0, 0.0), published_ts=base_ts + 61 * 86400
            ),
            self.embedded(
                "in-window", (0.99, 0.01), published_ts=base_ts + 59 * 86400
            ),
        ]

        rows = select_review_pairs(
            articles,
            pairs_per_band=3,
            max_time_distance_days=60,
        )
        pairs = {
            frozenset((row["left_article_id"], row["right_article_id"]))
            for row in rows
        }

        self.assertNotIn(frozenset(("old", "too-late")), pairs)
        self.assertIn(frozenset(("old", "in-window")), pairs)

    def test_pair_review_packet_is_blind_deduplicated_and_insert_only(self):
        articles = [
            self.embedded("raw-a", (1.0, 0.0), source="one"),
            self.embedded("raw-b", (0.9, 0.1), source="two"),
            self.embedded("raw-c", (0.0, 1.0), source="three"),
        ]
        pairs = (
            LabelledPair("raw-a", "raw-b", "same", "secret answer"),
            LabelledPair("raw-a", "raw-c", "different", "secret answer"),
        )
        packet = build_pair_review_packet(
            articles,
            pairs,
            {
                "raw-a": {"text": "alpha full text", "url": "https://a.test"},
                "raw-b": {"text": "bravo full text", "url": "https://b.test"},
                "raw-c": {"text": "charlie full text", "url": "https://c.test"},
            },
        )

        self.assertTrue(packet["blind_review"])
        self.assertEqual(3, packet["article_count"])
        self.assertEqual(2, packet["pair_count"])
        self.assertEqual(
            ["article-001", "article-002", "article-003"],
            [article["article_id"] for article in packet["articles"]],
        )
        self.assertEqual("", packet["pairs"][0]["reviewer_judgement"])
        self.assertNotIn("judgement", packet["pairs"][0])
        self.assertNotIn(
            "raw-a", {article["article_id"] for article in packet["articles"]}
        )
        self.assertNotIn(
            "raw-a",
            {
                article_id
                for pair in packet["pairs"]
                for article_id in (
                    pair["left_article_id"],
                    pair["right_article_id"],
                )
            },
        )
        self.assertNotIn("secret answer", str(packet))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "blind-review.json"
            write_pair_review_packet(path, packet)
            with self.assertRaises(FileExistsError):
                write_pair_review_packet(path, packet)

    def test_rejects_invalid_programmatic_pair_label(self):
        articles = [
            self.embedded("a", (1.0, 0.0)),
            self.embedded("b", (0.0, 1.0)),
        ]

        with self.assertRaisesRegex(ValueError, "Ogiltig paretikett"):
            calculate_similarity_report(
                articles,
                pair_labels=(LabelledPair("a", "b", "unknown"),),
            )

    def test_resolves_profile_categories_to_feed_sources(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "feeds.yaml").write_text(
                "- name: Security feed\n"
                "  topics: [Cybersecurity]\n"
                "- name: Other feed\n"
                "  topics: [Sport]\n",
                encoding="utf-8",
            )
            config_path = root / "config.yaml"
            config_path.write_text(
                "feeds:\n"
                "  path: feeds.yaml\n"
                "long_term_analysis:\n"
                "  profiles:\n"
                "    target:\n"
                "      categories: [Cybersecurity]\n"
                "      sources: [Manual source]\n",
                encoding="utf-8",
            )

            sources, metadata = resolve_profile_sources(config_path, "target")

        self.assertEqual({"Manual source", "Security feed"}, sources)
        self.assertEqual("target", metadata["profile_id"])

    def test_repository_golden_pair_draft_has_all_review_classes(self):
        path = Path(__file__).parents[1] / "config" / "long_term_golden_pairs_v1.csv"

        labels = load_pair_labels(path)

        self.assertEqual(33, len(labels))
        self.assertEqual(
            {"same": 13, "different": 15, "ambiguous": 5},
            dict(Counter(pair.judgement for pair in labels)),
        )

    def test_background_selection_uses_nearest_earlier_article_in_window(self):
        base_ts = 1_700_000_000
        articles = [
            self.embedded("gold-a", (1.0, 0.0), published_ts=base_ts + 10),
            self.embedded("gold-b", (0.99, 0.01), published_ts=base_ts + 20),
            self.embedded("background", (0.98, 0.02), published_ts=base_ts + 5),
            self.embedded("future", (1.0, 0.0), published_ts=base_ts + 30),
            self.embedded("distant", (0.0, 1.0), published_ts=base_ts + 4),
        ]

        selected = select_competing_background_articles(
            articles,
            (LabelledPair("gold-a", "gold-b", "same"),),
            neighbors_per_article=1,
            min_similarity=0.75,
        )

        self.assertEqual(["background"], [article.article_id for article in selected])

    def test_portable_dataset_round_trip_and_online_cve_guard(self):
        articles = [
            self.embedded("original-a", (1.0, 0.0), indicators=("cve:first",)),
            self.embedded("original-b", (0.99, 0.01), indicators=("cve:first",)),
            self.embedded("original-c", (1.0, 0.0), indicators=("cve:second",)),
            self.embedded("original-d", (0.99, 0.01), indicators=("cve:second",)),
        ]
        pairs = (
            LabelledPair("original-a", "original-b", "same"),
            LabelledPair("original-c", "original-d", "same"),
            LabelledPair("original-a", "original-c", "different"),
        )
        dataset = build_portable_dataset(articles, pairs, profile_id="profile")

        with TemporaryDirectory() as directory:
            path = Path(directory) / "portable.json"
            write_portable_dataset(path, dataset)
            loaded_articles, loaded_pairs, metadata = load_portable_dataset(path)

        self.assertEqual(4, len(loaded_articles))
        self.assertEqual(3, len(loaded_pairs))
        self.assertEqual("profile", metadata["profile_id"])
        self.assertEqual(2, metadata["online_scenario_count"])
        self.assertTrue(
            all(article.article_id.startswith("article-") for article in loaded_articles)
        )
        self.assertNotIn("original-a", str(dataset))

        report = evaluate_online_clustering(
            loaded_articles,
            loaded_pairs,
            thresholds=(0.8,),
        )
        metrics = report["thresholds"][0]["labelled_pair_metrics"]
        self.assertEqual(2, metrics["true_positive"])
        self.assertEqual(1, metrics["true_negative"])
        self.assertEqual(0, metrics["false_positive"])
        self.assertEqual(1.0, metrics["precision"])
        self.assertEqual(1.0, metrics["recall"])
        scenarios = evaluate_online_scenarios(metadata["online_scenarios"])
        self.assertTrue(scenarios["passed"])
        self.assertEqual(4, scenarios["checks"])

    def test_repository_portable_fixture_reproduces_online_baseline(self):
        path = (
            Path(__file__).parents[1]
            / "config"
            / "long_term_golden_fixture_v1.json"
        )

        articles, pairs, metadata = load_portable_dataset(path)
        report = evaluate_online_clustering(
            articles,
            pairs,
            thresholds=(0.8, 0.82),
            ambiguity_margin=0.03,
            candidate_window_days=60,
        )

        self.assertEqual(61, metadata["article_count"])
        self.assertEqual(33, metadata["pair_count"])
        self.assertEqual(2, metadata["online_scenario_count"])
        rows = {row["threshold"]: row for row in report["thresholds"]}
        self.assertAlmostEqual(
            13 / 14, rows[0.8]["labelled_pair_metrics"]["precision"]
        )
        self.assertEqual(1.0, rows[0.82]["labelled_pair_metrics"]["precision"])
        for row in rows.values():
            self.assertEqual(1.0, row["labelled_pair_metrics"]["recall"])
            self.assertEqual(0, row["needs_review_cluster_count"])
        scenarios = evaluate_online_scenarios(metadata["online_scenarios"])
        self.assertTrue(scenarios["passed"])

        with_background = evaluate_online_clustering(
            articles,
            pairs,
            thresholds=(0.8, 0.82),
            ambiguity_margin=0.03,
            candidate_window_days=60,
            background_articles=metadata["background_articles"],
        )
        rows = {row["threshold"]: row for row in with_background["thresholds"]}
        self.assertEqual(67, with_background["background_article_count"])
        self.assertEqual(1, rows[0.8]["labelled_pair_metrics"]["false_positive"])
        self.assertEqual(0, rows[0.82]["labelled_pair_metrics"]["false_positive"])
        self.assertEqual(0, rows[0.8]["labelled_pair_metrics"]["false_negative"])
        self.assertEqual(0, rows[0.82]["labelled_pair_metrics"]["false_negative"])

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
    def embedded(
        article_id,
        vector,
        *,
        source="source",
        indicators=(),
        published_ts=1,
    ):
        return EmbeddedArticle(
            article_id=article_id,
            title=article_id,
            source=source,
            published_ts=published_ts,
            vector=vector,
            signature=("model", len(vector), "instruction"),
            strong_indicators=indicators,
        )


if __name__ == "__main__":
    unittest.main()
