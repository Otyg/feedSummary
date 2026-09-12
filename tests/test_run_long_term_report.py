import argparse
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from feedsummary_core.long_term import ReduceResult

from run_long_term_report import (
    _long_term_mode,
    _model_label,
    _reduce_llm_config,
    _reduce_settings,
    _report_cluster_ids,
    execute_long_term_report,
)


class LongTermReportCliTests(unittest.TestCase):
    def test_report_cluster_ids_separate_atomic_observations(self):
        events, eligible = _report_cluster_ids(
            {
                "windows": [
                    {
                        "event_cluster_ids": ["atomic", "mapped"],
                        "snapshot_eligible_cluster_ids": ["mapped"],
                    }
                ]
            }
        )

        self.assertEqual({"atomic", "mapped"}, events)
        self.assertEqual({"mapped"}, eligible)

    def test_shadow_mode_is_explicit_and_invalid_mode_is_rejected(self):
        self.assertEqual("shadow", _long_term_mode({"mode": "shadow"}))
        self.assertEqual("publish", _long_term_mode({}))
        with self.assertRaisesRegex(ValueError, "shadow eller publish"):
            _long_term_mode({"mode": "preview"})

    def test_local_reduce_provider_selects_only_local_llm(self):
        config = {
            "llm": [
                {"provider": "ollama_cloud", "model": "cloud"},
                {"provider": "ollama_local", "model": "local"},
            ],
            "llm_fallback": {"provider": "ollama_cloud", "model": "fallback"},
        }
        selected = _reduce_llm_config(config, "local")

        self.assertEqual("ollama_local", selected["llm"]["provider"])
        self.assertNotIn("llm_fallback", selected)

    def test_cloud_reduce_provider_excludes_local_fallbacks(self):
        config = {
            "llm": [
                {
                    "provider": "ollama_cloud",
                    "model": "cloud-primary",
                    "context_window_tokens": 24000,
                },
                {
                    "provider": "ollama_local",
                    "model": "local",
                    "context_window_tokens": 8000,
                },
            ],
            "llm_fallback": {
                "provider": "ollama_cloud",
                "model": "cloud-fallback",
                "context_window_tokens": 16000,
            },
        }

        selected = _reduce_llm_config(config, "cloud")

        self.assertEqual(
            ["cloud-primary", "cloud-fallback"],
            [entry["model"] for entry in selected["llm"]],
        )
        self.assertNotIn("llm_fallback", selected)

    def test_reduce_budget_uses_safest_limits_across_provider_chain(self):
        llm_config = {
            "llm": [
                {
                    "provider": "ollama_cloud",
                    "model": "large",
                    "context_window_tokens": 16000,
                    "prompt_safety_margin": 400,
                },
                {
                    "provider": "ollama_local",
                    "model": "small",
                    "context_window_tokens": 8000,
                    "prompt_safety_margin": 600,
                },
            ]
        }
        settings = _reduce_settings(
            {
                "llm": {"reduce_max_output_tokens": 1800},
                "reports": {
                    "max_snapshots_per_segment": 8,
                    "max_snapshot_records": 500,
                },
            },
            llm_config,
        )

        self.assertEqual(8000, settings.max_context_tokens)
        self.assertEqual(600, settings.safety_margin_tokens)
        self.assertEqual(1800, settings.max_output_tokens)
        self.assertEqual(8, settings.max_snapshots_per_segment)
        self.assertEqual(
            "ollama_cloud:large -> ollama_local:small",
            _model_label(llm_config),
        )


class LongTermReportExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_renews_and_passes_lease_guard_before_report_write(self):
        class Store:
            def __init__(self):
                self.renewals = 0
                self.releases = 0

            def claim_long_term_lease(self, *_args, **_kwargs):
                return True

            def renew_long_term_lease(self, *_args, **_kwargs):
                self.renewals += 1
                return True

            def release_long_term_lease(self, *_args, **_kwargs):
                self.releases += 1
                return True

        class Client:
            async def aclose(self):
                return None

        store = Store()
        config = {
            "llm": {"provider": "ollama_local", "model": "test"},
            "long_term_analysis": {
                "enabled": True,
                "report_lease_seconds": 60,
                "profiles": {"profile": {"enabled": True}},
            },
        }
        args = argparse.Namespace(
            config=None,
            profile="profile",
            at=None,
            dry_run=False,
            owner_id="worker",
        )

        async def reduce(*_args, lease_guard, **_kwargs):
            await lease_guard.ensure_owned()
            return ReduceResult(
                action="saved",
                report_id="report-1",
                input_snapshot_ids=("snapshot-1",),
                segment_count=0,
                llm_call_count=1,
                repair_attempted=False,
                estimated_final_prompt_tokens=10,
                mirror_action="disabled",
            )

        with (
            patch("run_long_term_report.build_landscape_metrics", return_value={}),
            patch("run_long_term_report.select_report_snapshots", return_value=[]),
            patch("run_long_term_report.create_llm_client", return_value=Client()),
            patch("run_long_term_report._prompt_package", return_value={}),
            patch(
                "run_long_term_report.run_landscape_reduce",
                AsyncMock(side_effect=reduce),
            ) as run_reduce,
        ):
            result = await execute_long_term_report(
                args,
                config_path=Path("/tmp/config.yaml"),
                config=config,
                store=store,
            )

        self.assertEqual("report-1", result["report_id"])
        self.assertEqual("publish", result["mode"])
        self.assertTrue(run_reduce.await_args.kwargs["mirror_to_summary_docs"])
        self.assertIsNotNone(run_reduce.await_args.kwargs["lease_guard"])
        self.assertEqual(1, store.renewals)
        self.assertEqual(1, store.releases)

    async def test_shadow_mode_forces_summary_mirror_off(self):
        config = {
            "llm": {"provider": "ollama_local", "model": "test"},
            "long_term_analysis": {
                "enabled": True,
                "mode": "shadow",
                "profiles": {"profile": {"enabled": True}},
                "reports": {"mirror_to_summary_docs": True},
            },
        }
        args = argparse.Namespace(
            config=None,
            profile="profile",
            at=None,
            dry_run=False,
            owner_id="worker",
        )

        class Store:
            def claim_long_term_lease(self, *_args, **_kwargs):
                return True

            def renew_long_term_lease(self, *_args, **_kwargs):
                return True

            def release_long_term_lease(self, *_args, **_kwargs):
                return True

        class Client:
            async def aclose(self):
                return None

        async def reduce(*_args, **kwargs):
            self.assertFalse(kwargs["mirror_to_summary_docs"])
            return ReduceResult(
                action="saved",
                report_id="report-shadow",
                input_snapshot_ids=(),
                segment_count=0,
                llm_call_count=1,
                repair_attempted=False,
                estimated_final_prompt_tokens=10,
                mirror_action="disabled",
            )

        with (
            patch("run_long_term_report.build_landscape_metrics", return_value={}),
            patch("run_long_term_report.select_report_snapshots", return_value=[]),
            patch("run_long_term_report.create_llm_client", return_value=Client()),
            patch("run_long_term_report._prompt_package", return_value={}),
            patch(
                "run_long_term_report.run_landscape_reduce",
                AsyncMock(side_effect=reduce),
            ),
        ):
            result = await execute_long_term_report(
                args,
                config_path=Path("/tmp/config.yaml"),
                config=config,
                store=Store(),
            )

        self.assertEqual("shadow", result["mode"])


if __name__ == "__main__":
    unittest.main()
