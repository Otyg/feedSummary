import argparse
import unittest
from pathlib import Path

import run_long_term_reconciliation as cli


class ReconciliationCliTests(unittest.TestCase):
    def test_settings_use_reconciliation_policy(self):
        args = argparse.Namespace(
            auto_threshold=None,
            review_threshold=None,
            window_days=None,
            max_review_pairs=None,
        )
        settings = cli._settings(
            {
                "clustering": {"similarity_threshold": 0.80, "candidate_window_days": 30},
                "reconciliation": {
                    "auto_merge_similarity": 0.91,
                    "review_similarity": 0.84,
                    "candidate_window_days": 45,
                    "max_review_pairs": 25,
                },
            },
            args,
        )
        self.assertEqual(0.91, settings.auto_merge_similarity)
        self.assertEqual(0.84, settings.review_similarity)
        self.assertEqual(45, settings.candidate_window_days)
        self.assertEqual(25, settings.max_review_pairs)

    def test_summary_reports_group_sizes(self):
        summary = cli._summary(
            {
                "mode": "proposal_only",
                "profile_id": "profile",
                "cluster_count": 10,
                "compared_pair_count": 20,
                "auto_merge_edges": [{}, {}],
                "review_edges": [{}],
                "blocked_strict_cve_pair_count": 3,
                "groups": [
                    {"cluster_ids": ["a", "b", "c"]},
                    {"cluster_ids": ["d", "e"]},
                ],
            }
        )
        self.assertEqual(2, summary["merge_group_count"])
        self.assertEqual(5, summary["clusters_in_merge_groups"])
        self.assertEqual(3, summary["largest_merge_group"])

    def test_operation_id_changes_with_membership_revision(self):
        first = [{"id": "a", "membership_revision": 1}]
        second = [{"id": "a", "membership_revision": 2}]
        self.assertNotEqual(
            cli._stable_operation_id("profile", "a", first),
            cli._stable_operation_id("profile", "a", second),
        )

    def test_apply_requires_exact_operation_id_confirmation(self):
        operation = {
            "id": "reconciliation-1",
            "profile_id": "profile",
            "primary_cluster_id": "primary",
            "source_cluster_ids": ["primary", "secondary"],
            "expected_membership_revisions": {"primary": 1, "secondary": 1},
            "membership_revision_by_article_id": {"a": 2, "b": 3},
            "merged_cluster": {
                "id": "primary",
                "profile_id": "profile",
                "member_count": 2,
                "membership_revision": 3,
                "summarized_revision": 1,
                "latest_snapshot_id": None,
            },
            "superseded_clusters": [
                {
                    "id": "secondary",
                    "profile_id": "profile",
                    "status": "closed",
                    "member_count": 0,
                    "membership_revision": 2,
                    "superseded_by_cluster_id": "primary",
                    "reconciliation_id": "reconciliation-1",
                }
            ],
            "reconciled_at": 100,
            "snapshot_policy": "preserve_history_rebuild_primary",
        }
        args = argparse.Namespace(
            config=None,
            profile="profile",
            apply_operation=Path("unused.json"),
            confirm_apply="wrong-id",
        )
        config = {
            "long_term_analysis": {
                "enabled": True,
                "profiles": {"profile": {"enabled": True}},
            }
        }
        with self.assertRaisesRegex(ValueError, "exakt motsvara"):
            cli.apply_prepared_reconciliation(
                args,
                operation=operation,
                config_path=Path("/tmp/config.yaml"),
                config=config,
                store=object(),
            )


if __name__ == "__main__":
    unittest.main()
