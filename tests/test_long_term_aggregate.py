import copy
import unittest

from long_term_aggregate import (
    AGGREGATE_UPDATE_WARNING,
    AggregateReportStore,
    aggregate_profile_ids,
    normalize_profile_failures,
)


class _Store:
    def __init__(self):
        self.clusters = {
            "child_a": [self._cluster("cluster-a", "child_a")],
            "child_b": [self._cluster("cluster-b", "child_b")],
        }
        self.memberships = {
            "cluster-a": [
                self._membership("cluster-a", "article-1"),
                self._membership("cluster-a", "article-2"),
            ],
            "cluster-b": [
                self._membership("cluster-b", "article-1"),
                self._membership("cluster-b", "article-2"),
            ],
        }
        self.snapshots = {
            "child_a": [self._snapshot("snapshot-a", "cluster-a", "child_a", 150)],
            "child_b": [self._snapshot("snapshot-b", "cluster-b", "child_b", 180)],
        }
        self.saved_report = None

    @staticmethod
    def _cluster(cluster_id, profile_id):
        return {
            "id": cluster_id,
            "profile_id": profile_id,
            "status": "active",
            "tag_scope": "malware:test",
            "first_seen_ts": 100,
            "last_seen_ts": 190,
            "embedding_model": "embedding-model",
            "embedding_dimension": 2,
            "embedding_instruction": "event",
            "member_count": 2,
            "membership_revision": 2,
            "summarized_revision": 2,
        }

    @staticmethod
    def _membership(cluster_id, article_id):
        return {
            "profile_id": "ignored",
            "cluster_id": cluster_id,
            "article_id": article_id,
            "tag_scope": "malware:test",
            "article_ts": 120,
            "evidence": {"source": "source"},
        }

    @staticmethod
    def _snapshot(snapshot_id, cluster_id, profile_id, created_at):
        return {
            "id": snapshot_id,
            "profile_id": profile_id,
            "cluster_id": cluster_id,
            "membership_revision": 2,
            "created_at": created_at,
            "payload": {
                "schema_version": 1,
                "profile_id": profile_id,
                "cluster_id": cluster_id,
                "membership_revision": 2,
                "title": "Event",
                "summary": "Summary",
                "confidence": "medium",
                "insufficient_evidence": False,
                "facts": [],
                "timeline": [],
                "mitre_techniques": [],
                "uncertainties": [],
            },
        }

    def list_threat_clusters(self, profile_id, **_kwargs):
        return copy.deepcopy(self.clusters.get(profile_id, []))

    def list_cluster_memberships(self, cluster_id, *, limit=10000):
        return copy.deepcopy(self.memberships.get(cluster_id, [])[:limit])

    def list_cluster_snapshots(self, profile_id, *, cluster_id=None, limit=1000):
        rows = self.snapshots.get(profile_id, [])
        if cluster_id:
            rows = [row for row in rows if row["cluster_id"] == cluster_id]
        return copy.deepcopy(rows[:limit])

    def save_threat_landscape_report(self, report):
        self.saved_report = copy.deepcopy(report)
        return True


class AggregateProfileTests(unittest.TestCase):
    def test_child_profiles_are_validated_and_deduplicated(self):
        long_term = {
            "profiles": {
                "all": {
                    "enabled": True,
                    "aggregate_profiles": ["child_a", "child_a", "child_b"],
                },
                "child_a": {"enabled": True},
                "child_b": {"enabled": True},
            }
        }

        self.assertEqual(
            ("child_a", "child_b"), aggregate_profile_ids(long_term, "all")
        )
        self.assertEqual((), aggregate_profile_ids(long_term, "child_a"))

    def test_projection_reuses_newest_child_snapshot_and_deduplicates_event(self):
        store = _Store()
        projected = AggregateReportStore(
            store,
            aggregate_profile_id="all",
            child_profile_ids=("child_a", "child_b"),
            period_end_ts=200,
            max_memberships_per_cluster=100,
            max_snapshot_records=100,
        )

        clusters = projected.list_threat_clusters("all", limit=100)
        snapshots = projected.list_cluster_snapshots("all", limit=100)

        self.assertEqual(1, len(clusters))
        self.assertEqual("cluster-b", clusters[0]["id"])
        self.assertEqual("all", clusters[0]["profile_id"])
        self.assertEqual(
            ["child_a", "child_b"], clusters[0]["aggregate_source_profiles"]
        )
        self.assertEqual(1, len(snapshots))
        self.assertEqual("snapshot-b", snapshots[0]["id"])
        self.assertEqual("all", snapshots[0]["profile_id"])
        self.assertEqual("all", snapshots[0]["payload"]["profile_id"])
        self.assertEqual("child_b", store.snapshots["child_b"][0]["profile_id"])
        self.assertEqual(1, projected.report_quality()["aggregate_deduplicated_cluster_count"])

    def test_failed_child_update_marks_persisted_report_partial(self):
        store = _Store()
        failure = {
            "profile_id": "child_a",
            "stage": "map",
            "error_type": "RuntimeError",
            "error": "provider unavailable",
            "observed_at": 200,
        }
        projected = AggregateReportStore(
            store,
            aggregate_profile_id="all",
            child_profile_ids=("child_a", "child_b"),
            period_end_ts=200,
            max_memberships_per_cluster=100,
            max_snapshot_records=100,
            profile_update_failures=[failure, failure],
        )
        report = {"quality": {"warning_codes": [AGGREGATE_UPDATE_WARNING]}}

        self.assertTrue(projected.save_threat_landscape_report(report))

        self.assertEqual("partial", store.saved_report["quality"]["completeness"])
        self.assertEqual(
            ["child_a", "child_b"],
            store.saved_report["quality"]["aggregate_profile_ids"],
        )
        self.assertEqual(
            1,
            len(store.saved_report["quality"]["aggregate_profile_update_failures"]),
        )

    def test_failure_contract_is_bounded(self):
        failures = normalize_profile_failures(
            [
                {
                    "profile_id": " child ",
                    "stage": "map",
                    "error": "x" * 2000,
                }
            ]
        )

        self.assertEqual("child", failures[0]["profile_id"])
        self.assertEqual(1000, len(failures[0]["error"]))


if __name__ == "__main__":
    unittest.main()
