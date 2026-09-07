import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

import run_long_term_clustering as cli


class LongTermClusteringCliTests(unittest.TestCase):
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
