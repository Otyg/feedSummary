import argparse
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml
from feedsummary_core.long_term import (
    ClusteringSettings,
    EmbeddingSignature,
    IncrementalBatchResult,
    assign_article,
    create_cluster,
)

import run_long_term_clustering as cli
from feedsummary_core.long_term.processor import _tag_scopes


class LongTermClusteringCliTests(unittest.TestCase):
    def test_tag_scope_is_the_article_tag_union_and_ignores_vulnerabilities(self):
        scopes = _tag_scopes(
            [
                {"category": "MALWARE", "name": "Alpha"},
                {"category": "SECTOR", "name": "Healthcare"},
                {"category": "LOCATION", "name": "Europe"},
                {"category": "VULNERABILITY", "name": "CVE-2026-1"},
            ],
            frozenset({"MALWARE", "SECTOR", "LOCATION", "VULNERABILITY"}),
        )

        self.assertEqual(
            (
                "LOCATION:europe|MALWARE:alpha|SECTOR:healthcare",
            ),
            scopes,
        )

    def test_embedding_candidates_can_reuse_a_larger_tag_union(self):
        signature = EmbeddingSignature("model", 2, "instruction")
        cluster = create_cluster(
            profile_id="profile",
            article_id="old",
            article_ts=100,
            embedding=[1.0, 0.0],
            signature=signature,
            tag_scope="MALWARE:alpha|SECTOR:healthcare|LOCATION:europe",
        )

        decision = assign_article(
            profile_id="profile",
            article_ts=110,
            embedding=[1.0, 0.0],
            signature=signature,
            candidates=[cluster],
            tag_scope="MALWARE:alpha",
            settings=ClusteringSettings(similarity_threshold=0.8, min_shared_tags=1),
        )

        self.assertEqual("match", decision.action.value)
        self.assertEqual(cluster.id, decision.cluster_id)

    def test_all_sources_bypasses_profile_source_resolution(self):
        result = IncrementalBatchResult(
            run_id="run",
            profile_id="profile",
            dry_run=False,
            fetched=0,
            assigned=0,
            existing=0,
            created=0,
            matched=0,
            needs_review=0,
            quarantined=0,
            quarantine_retried=0,
            filtered=0,
            lifecycle_updated=0,
            cursor_fetched_at=0,
            cursor_article_id="",
            blocked_article_id=None,
            blocked_reason=None,
            assignments=(),
        )
        args = argparse.Namespace(
            config=None,
            profile="profile",
            limit=10,
            since=None,
            until=None,
            all_sources=True,
            dry_run=False,
        )
        config = {
            "long_term_analysis": {
                "enabled": True,
                "profiles": {
                    "profile": {"enabled": True, "categories": ["missing"]}
                },
            }
        }

        with patch.object(
            cli, "run_incremental_clustering", return_value=result
        ) as run:
            cli.execute_long_term_clustering(
                args,
                config_path=Path("/tmp/config.yaml"),
                config=config,
                store=object(),
            )

        self.assertIsNone(run.call_args.kwargs["sources"])
        self.assertEqual(
            (
                "DOMAIN_ENTITY",
                "MALWARE",
                "THREAT_ACTOR",
                "SECTOR",
                "LOCATION",
                "ORGANIZATION",
            ),
            run.call_args.kwargs["settings"].tag_scope_categories,
        )

    def test_resolves_profile_categories_to_feed_sources(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "feeds.yaml").write_text(
                yaml.safe_dump(
                    [
                        {"name": "Security feed", "topics": ["Cybersecurity"]},
                        {"name": "Other feed", "topics": ["Technology"]},
                    ]
                ),
                encoding="utf-8",
            )
            config = {"feeds": {"path": "feeds.yaml"}}
            sources = cli._resolve_sources(
                config,
                root / "config.yaml",
                {"categories": ["Cybersecurity"], "sources": ["Manual feed"]},
            )
        self.assertEqual(["Manual feed", "Security feed"], sources)

    def test_aggregate_profile_uses_union_of_member_sources(self):
        config = {
            "feeds": [
                {"name": "Security feed", "topics": ["Cybersecurity"]},
                {"name": "Health feed", "topics": ["Healthcare"]},
            ],
            "long_term_analysis": {
                "profiles": {
                    "security": {"enabled": True, "categories": ["Cybersecurity"]},
                    "health": {
                        "enabled": True,
                        "categories": ["Healthcare"],
                        "sources": ["Security feed"],
                    },
                    "all_profiles": {
                        "enabled": True,
                        "aggregate_profiles": ["security", "health"],
                    },
                }
            },
        }
        profile = config["long_term_analysis"]["profiles"]["all_profiles"]

        self.assertEqual(
            ["Health feed", "Security feed"],
            cli._resolve_sources(config, Path("config.yaml"), profile),
        )
        config["long_term_analysis"]["profiles"]["health"]["required_tags"] = ["health"]
        with self.assertRaisesRegex(ValueError, "required_tags"):
            cli._resolve_sources(config, Path("config.yaml"), profile)

    def test_timestamp_supports_unix_and_iso_8601(self):
        self.assertEqual(100, cli._timestamp("100"))
        self.assertEqual(0, cli._timestamp("1970-01-01T00:00:00Z"))

    def test_apply_requires_global_feature_flag(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "store": {"provider": "sqlite", "path": "store.sqlite"},
                        "long_term_analysis": {
                            "enabled": False,
                            "profiles": {
                                "profile": {
                                    "enabled": True,
                                    "include_all_sources": True,
                                }
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            exit_code = cli.main(["--config", str(config_path), "--profile", "profile"])
        self.assertEqual(2, exit_code)


if __name__ == "__main__":
    unittest.main()
