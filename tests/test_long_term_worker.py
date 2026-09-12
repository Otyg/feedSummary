import copy
import unittest
from unittest.mock import AsyncMock, patch

from web_viewer import feedsum_worker


class _Store:
    def __init__(self):
        self.runs = {}
        self.jobs = {}

    def create_long_term_run(self, document):
        run_id = str(document["id"])
        if run_id in self.runs:
            return False
        self.runs[run_id] = copy.deepcopy(document)
        return True

    def update_long_term_run(self, run_id, *, expected_status, fields):
        row = self.runs.get(run_id)
        if row is None or row.get("status") != expected_status:
            return False
        row.update(copy.deepcopy(fields))
        return True

    def create_job(self):
        self.jobs[17] = {"status": "queued"}
        return 17

    def update_job(self, job_id, **fields):
        self.jobs[job_id].update(copy.deepcopy(fields))


class LongTermWorkerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = {
            "long_term_analysis": {
                "enabled": True,
                "profiles": {"healthcare_europe": {"enabled": True}},
            }
        }
        self.entry = {
            "type": "long_term_pipeline",
            "profile": "healthcare_europe",
            "limit": 100,
            "map_limit": 10,
        }

    async def test_pipeline_records_stage_results_and_report_id(self):
        store = _Store()
        clustering = {
            "profile_id": "healthcare_europe",
            "run_id": "cluster-run",
            "blocked_reason": None,
            "assigned": 2,
            "assignments": [{"article_id": "a"}],
        }
        map_result = {
            "profile_id": "healthcare_europe",
            "dry_run": False,
            "updates": [{"snapshot_id": "snapshot-1"}],
        }
        report_result = {
            "profile_id": "healthcare_europe",
            "dry_run": False,
            "report_id": "report-1",
        }
        with (
            patch.object(
                feedsum_worker,
                "execute_long_term_clustering",
                return_value=clustering,
            ) as run_clustering,
            patch.object(
                feedsum_worker,
                "execute_long_term_map",
                AsyncMock(return_value=map_result),
            ) as run_map,
            patch.object(
                feedsum_worker,
                "execute_long_term_report",
                AsyncMock(return_value=report_result),
            ) as run_report,
        ):
            outcome = await feedsum_worker._run_long_term_entry(
                config_path="config.yaml",
                cfg=self.config,
                store=store,
                job_id=17,
                job_name="weekly-landscape",
                entry=self.entry,
            )

        self.assertEqual("report-1", outcome["report_id"])
        run = store.runs[outcome["run_id"]]
        self.assertEqual("done", run["status"])
        self.assertEqual("done", run["stage"])
        self.assertEqual("publish", run["mode"])
        self.assertEqual("publish", outcome["mode"])
        self.assertEqual(64, len(run["config_hash"]))
        self.assertNotIn("assignments", run["results"]["clustering"])
        self.assertEqual(1, run["results"]["map"]["update_count"])
        self.assertEqual(1, run_clustering.call_count)
        self.assertFalse(run_clustering.call_args.args[0].all_sources)
        self.assertEqual(1, run_map.await_count)
        self.assertEqual(1, run_report.await_count)

    async def test_stage_failure_is_recorded_without_running_later_stages(self):
        store = _Store()
        with (
            patch.object(
                feedsum_worker,
                "execute_long_term_clustering",
                return_value={"blocked_reason": None, "assignments": []},
            ),
            patch.object(
                feedsum_worker,
                "execute_long_term_map",
                AsyncMock(side_effect=RuntimeError("map unavailable")),
            ),
            patch.object(
                feedsum_worker,
                "execute_long_term_report",
                AsyncMock(),
            ) as run_report,
            self.assertRaisesRegex(RuntimeError, "map unavailable"),
        ):
            await feedsum_worker._run_long_term_entry(
                config_path="config.yaml",
                cfg=self.config,
                store=store,
                job_id=17,
                job_name="weekly-landscape",
                entry=self.entry,
            )

        run = next(iter(store.runs.values()))
        self.assertEqual("failed", run["status"])
        self.assertEqual("map", run["stage"])
        self.assertEqual("RuntimeError", run["error_type"])
        self.assertEqual(0, run_report.await_count)

    async def test_blocked_clustering_marks_pipeline_blocked(self):
        store = _Store()
        with (
            patch.object(
                feedsum_worker,
                "execute_long_term_clustering",
                return_value={
                    "blocked_reason": "article identity missing",
                    "assignments": [],
                },
            ),
            patch.object(
                feedsum_worker, "execute_long_term_map", AsyncMock()
            ) as run_map,
            self.assertRaisesRegex(
                feedsum_worker.LongTermWorkerBlocked, "article identity missing"
            ),
        ):
            await feedsum_worker._run_long_term_entry(
                config_path="config.yaml",
                cfg=self.config,
                store=store,
                job_id=17,
                job_name="weekly-landscape",
                entry=self.entry,
            )

        run = next(iter(store.runs.values()))
        self.assertEqual("blocked", run["status"])
        self.assertEqual("clustering", run["stage"])
        self.assertEqual(0, run_map.await_count)

    async def test_blocked_pipeline_can_restart_as_new_successful_run(self):
        store = _Store()
        clustering = [
            {
                "blocked_reason": "article identity missing",
                "assignments": [],
            },
            {
                "blocked_reason": None,
                "assigned": 1,
                "assignments": [{"article_id": "article-1"}],
            },
        ]
        with (
            patch.object(
                feedsum_worker,
                "execute_long_term_clustering",
                side_effect=clustering,
            ),
            patch.object(
                feedsum_worker,
                "execute_long_term_map",
                AsyncMock(return_value={"updates": []}),
            ),
            patch.object(
                feedsum_worker,
                "execute_long_term_report",
                AsyncMock(return_value={"report_id": "report-1"}),
            ),
        ):
            with self.assertRaises(feedsum_worker.LongTermWorkerBlocked):
                await feedsum_worker._run_long_term_entry(
                    config_path="config.yaml",
                    cfg=self.config,
                    store=store,
                    job_id=17,
                    job_name="weekly-landscape",
                    entry=self.entry,
                )
            resumed = await feedsum_worker._run_long_term_entry(
                config_path="config.yaml",
                cfg=self.config,
                store=store,
                job_id=18,
                job_name="weekly-landscape-retry",
                entry=self.entry,
            )

        self.assertEqual("report-1", resumed["report_id"])
        self.assertEqual(
            ["blocked", "done"],
            [run["status"] for run in store.runs.values()],
        )

    async def test_failed_pipeline_can_restart_without_overwriting_failed_run(self):
        store = _Store()
        map_runner = AsyncMock(
            side_effect=[
                RuntimeError("temporary map failure"),
                {"updates": [{"snapshot_id": "snapshot-1"}]},
            ]
        )
        with (
            patch.object(
                feedsum_worker,
                "execute_long_term_clustering",
                return_value={"blocked_reason": None, "assignments": []},
            ),
            patch.object(feedsum_worker, "execute_long_term_map", map_runner),
            patch.object(
                feedsum_worker,
                "execute_long_term_report",
                AsyncMock(return_value={"report_id": "report-1"}),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "temporary map failure"):
                await feedsum_worker._run_long_term_entry(
                    config_path="config.yaml",
                    cfg=self.config,
                    store=store,
                    job_id=17,
                    job_name="weekly-landscape",
                    entry=self.entry,
                )
            resumed = await feedsum_worker._run_long_term_entry(
                config_path="config.yaml",
                cfg=self.config,
                store=store,
                job_id=18,
                job_name="weekly-landscape-retry",
                entry=self.entry,
            )

        self.assertEqual("report-1", resumed["report_id"])
        runs = list(store.runs.values())
        self.assertEqual(["failed", "done"], [run["status"] for run in runs])
        self.assertEqual("map", runs[0]["stage"])
        self.assertEqual(2, map_runner.await_count)

    async def test_disabled_feature_and_missing_profile_are_rejected(self):
        store = _Store()
        disabled = copy.deepcopy(self.config)
        disabled["long_term_analysis"]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "enabled är false"):
            await feedsum_worker._run_long_term_entry(
                config_path="config.yaml",
                cfg=disabled,
                store=store,
                job_id=17,
                job_name="disabled",
                entry=self.entry,
            )
        with self.assertRaisesRegex(ValueError, "kräver fältet 'profile'"):
            await feedsum_worker._run_long_term_entry(
                config_path="config.yaml",
                cfg=self.config,
                store=store,
                job_id=17,
                job_name="missing-profile",
                entry={"type": "long_term_report"},
            )
        self.assertEqual({}, store.runs)

        invalid_mode = copy.deepcopy(self.config)
        invalid_mode["long_term_analysis"]["mode"] = "preview"
        with self.assertRaisesRegex(ValueError, "shadow eller publish"):
            await feedsum_worker._run_long_term_entry(
                config_path="config.yaml",
                cfg=invalid_mode,
                store=store,
                job_id=17,
                job_name="invalid-mode",
                entry=self.entry,
            )

    def test_manual_trigger_allows_only_bounded_long_term_overrides(self):
        merged = feedsum_worker._merge_trigger_overrides(
            self.entry,
            {
                "profile": "another-profile",
                "map_limit": 5,
                "force": True,
                "all_sources": True,
            },
        )

        self.assertEqual("another-profile", merged["profile"])
        self.assertEqual(5, merged["map_limit"])
        self.assertTrue(merged["force"])
        self.assertTrue(merged["all_sources"])
        with self.assertRaisesRegex(ValueError, "Otillåtna"):
            feedsum_worker._merge_trigger_overrides(
                self.entry, {"type": "tag_based_summary"}
            )
        with self.assertRaisesRegex(ValueError, "endast för långtidsjobb"):
            feedsum_worker._merge_trigger_overrides(
                {"type": "tag_based_summary"}, {"profile": "profile"}
            )

    def test_cleanup_policy_uses_long_term_retention(self):
        policy = feedsum_worker._cleanup_policy(
            {
                "cleanup": {"articles_days": 90},
                "long_term_analysis": {"retention_days": 540},
            }
        )

        self.assertEqual(90, policy.articles_days)
        self.assertEqual(540, policy.long_term_days)

        class LegacyCleanupPolicy:
            def __init__(self, **kwargs):
                if "long_term_days" in kwargs:
                    raise TypeError("legacy policy")
                self.__dict__.update(kwargs)

        with patch.object(
            feedsum_worker, "CleanupPolicy", LegacyCleanupPolicy
        ):
            legacy = feedsum_worker._cleanup_policy(
                {"long_term_analysis": {"retention_days": 540}}
            )
        self.assertFalse(hasattr(legacy, "long_term_days"))

    async def test_regular_worker_job_persists_outer_status(self):
        store = _Store()
        result = {
            "run_id": "worker-run",
            "report_id": "report-1",
            "profile_id": "healthcare_europe",
        }
        with patch.object(
            feedsum_worker,
            "_run_long_term_entry",
            AsyncMock(return_value=result),
        ):
            result_id = await feedsum_worker._run_regular_entry(
                "config.yaml",
                self.config,
                store,
                "weekly-landscape",
                self.entry,
            )

        self.assertEqual("report-1", result_id)
        self.assertEqual("done", store.jobs[17]["status"])
        self.assertEqual("report-1", store.jobs[17]["summary_id"])
        self.assertEqual("worker-run", store.jobs[17]["long_term_run_id"])


if __name__ == "__main__":
    unittest.main()
