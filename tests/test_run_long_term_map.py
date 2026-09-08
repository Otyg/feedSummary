import argparse
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from feedsummary_core.long_term import MapUpdateResult

from run_long_term_map import _map_llm_config, _map_settings, execute_long_term_map


class LongTermMapCliTests(unittest.TestCase):
    def test_local_map_provider_selects_only_local_llm(self):
        config = {
            "llm": [
                {"provider": "ollama_cloud", "model": "cloud"},
                {
                    "provider": "ollama_local",
                    "model": "local",
                    "context_window_tokens": 4096,
                    "prompt_safety_margin": 400,
                },
            ],
            "llm_fallback": {"provider": "ollama_cloud"},
        }
        selected = _map_llm_config(config, "local")
        self.assertEqual("ollama_local", selected["llm"]["provider"])
        self.assertNotIn("llm_fallback", selected)

    def test_map_settings_combine_long_term_and_provider_limits(self):
        settings = _map_settings(
            {
                "llm": {"map_max_output_tokens": 900},
                "cluster_updates": {
                    "min_pending_articles": 2,
                    "max_update_delay_hours": 12,
                    "max_articles_per_call": 8,
                    "article_clip_chars": 1500,
                },
            },
            {
                "llm": {
                    "provider": "ollama_local",
                    "context_window_tokens": 4096,
                    "prompt_safety_margin": 400,
                }
            },
        )
        self.assertEqual(2, settings.min_pending_articles)
        self.assertEqual(4096, settings.max_context_tokens)
        self.assertEqual(900, settings.max_output_tokens)
        self.assertEqual(400, settings.safety_margin_tokens)


class LongTermMapExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_renews_and_passes_lease_guard_before_snapshot_write(self):
        class Store:
            def __init__(self):
                self.renewals = 0
                self.releases = 0

            def list_threat_clusters(self, *_args, **_kwargs):
                return [{"id": "cluster-1"}]

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
            "llm": {"provider": "ollama_local"},
            "long_term_analysis": {
                "enabled": True,
                "map_lease_seconds": 60,
                "profiles": {"profile": {"enabled": True}},
            },
        }
        args = argparse.Namespace(
            config=None,
            profile="profile",
            cluster=None,
            limit=1,
            force=True,
            dry_run=False,
            owner_id="worker",
        )

        async def update(*_args, lease_guard, **_kwargs):
            await lease_guard.ensure_owned()
            return MapUpdateResult(
                cluster_id="cluster-1",
                action="saved",
                previous_revision=0,
                summarized_revision=1,
                snapshot_id="snapshot-1",
                input_article_ids=("article-1",),
                repair_attempted=False,
                estimated_prompt_tokens=10,
            )

        with (
            patch(
                "run_long_term_map.ThreatCluster.from_document",
                return_value=SimpleNamespace(
                    id="cluster-1", membership_revision=1, summarized_revision=0
                ),
            ),
            patch("run_long_term_map.create_llm_client", return_value=Client()),
            patch("run_long_term_map._prompt_package", return_value={}),
            patch(
                "run_long_term_map.update_cluster_map_snapshot",
                AsyncMock(side_effect=update),
            ) as run_update,
        ):
            result = await execute_long_term_map(
                args,
                config_path=Path("/tmp/config.yaml"),
                config=config,
                store=store,
            )

        self.assertEqual("snapshot-1", result["updates"][0]["snapshot_id"])
        self.assertIsNotNone(run_update.await_args.kwargs["lease_guard"])
        self.assertEqual(1, store.renewals)
        self.assertEqual(1, store.releases)


if __name__ == "__main__":
    unittest.main()
