import argparse
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml
from feedsummary_core.long_term import IncrementalBatchResult

import run_long_term_clustering as cli


class LongTermClusteringCliTests(unittest.TestCase):
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
