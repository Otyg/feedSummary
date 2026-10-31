import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import rebuild_long_term_clusters as rebuild
import run_long_term_clustering as clustering_cli
import yaml

from feedsummary_core.long_term import IncrementalSettings, run_incremental_clustering
from feedsummary_core.persistence import SqliteStore


class LongTermRebuildTests(unittest.TestCase):
    def test_archives_old_clusters_and_rebuilds_tags_with_resumable_cursor(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = SqliteStore(str(root / "articles.sqlite"))
            now = int(time.time())
            malware = store.create_tag("Malware A", "MALWARE")
            sector = store.create_tag("Healthcare", "SECTOR")
            for article_id, fetched_at, tags in (
                ("first", now - 200, (malware, sector)),
                ("second", now - 100, (malware, sector)),
            ):
                store.upsert_article(
                    {
                        "id": article_id,
                        "title": article_id,
                        "source": "feed",
                        "fetched_at": fetched_at,
                        "published_ts": fetched_at,
                        "similarity_embedding_vector": [1.0, 0.0],
                        "similarity_embedding_model": "model",
                        "similarity_embedding_instruction": "Represent event",
                    }
                )
                for tag in tags:
                    store.add_tag_to_article(article_id, tag["id"])
            old = run_incremental_clustering(
                store,
                profile_id="healthcare",
                owner_id="old",
                until_fetched_at=now,
                now_ts=now,
                settings=IncrementalSettings(batch_size=10),
            )
            self.assertEqual(1, old.created)
            config = {
                "store": {"provider": "sqlite", "path": str(root / "articles.sqlite")},
                "long_term_analysis": {
                    "enabled": True,
                    "batch_size": 1,
                    "clustering": {
                        "tag_scope_categories": ["MALWARE", "SECTOR"],
                    },
                    "profiles": {
                        "healthcare": {"enabled": True, "include_all_sources": True},
                    },
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            output_path = root / "tagged.yaml"
            arguments = [
                "--config",
                str(config_path),
                "--profile",
                "healthcare",
                "--batch-size",
                "1",
                "--until",
                str(now),
                "--output-config",
                str(output_path),
            ]
            preview = rebuild.execute_rebuild(
                rebuild._parser().parse_args(arguments),
                store=store,
            )
            self.assertTrue(preview["dry_run"])
            self.assertFalse(output_path.exists())
            self.assertEqual("active", store.list_threat_clusters("healthcare")[0]["status"])

            self.assertTrue(
                store.claim_long_term_lease(
                    "healthcare", "other-map-job", now_ts=now, lease_seconds=300
                )
            )
            with self.assertRaisesRegex(RuntimeError, "ägare=other-map-job"):
                rebuild.execute_rebuild(
                    rebuild._parser().parse_args([*arguments, "--apply"]),
                    store=store,
                )
            self.assertFalse(output_path.exists())
            self.assertEqual("active", store.list_threat_clusters("healthcare")[0]["status"])
            self.assertTrue(store.release_long_term_lease("healthcare", "other-map-job"))

            result = rebuild.execute_rebuild(
                rebuild._parser().parse_args([*arguments, "--apply"]),
                store=store,
            )
            self.assertEqual(1, result["profiles"][0]["closed"])
            self.assertEqual(2, result["profiles"][0]["rebuilt_assignments"])
            old_cluster = store.list_threat_clusters("healthcare")[0]
            self.assertEqual("closed", old_cluster["status"])
            self.assertEqual([2.0, 0.0], old_cluster["vector_sum"])
            self.assertIsNotNone(store.get_cluster_membership("healthcare", "first"))
            target = "healthcare_tagged_v2"
            new_clusters = store.list_threat_clusters(target)
            self.assertEqual([2], sorted(row["member_count"] for row in new_clusters))
            memberships = store.list_article_cluster_memberships(target, "first")
            self.assertEqual(
                {"MALWARE:malware a|SECTOR:healthcare"},
                {row["tag_scope"] for row in memberships},
            )
            prepared = yaml.safe_load(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                ["MALWARE", "SECTOR"],
                prepared["long_term_analysis"]["clustering"]["tag_scope_categories"],
            )
            profiles = prepared["long_term_analysis"]["profiles"]
            self.assertFalse(profiles["healthcare"]["clustering_enabled"])
            self.assertEqual("healthcare", profiles[target]["rebuild_source_profile_id"])
            with self.assertRaisesRegex(ValueError, "arkivprofilen"):
                clustering_cli.execute_long_term_clustering(
                    clustering_cli._parser().parse_args(
                        [
                            "--config",
                            str(output_path),
                            "--profile",
                            "healthcare",
                        ]
                    ),
                    store=store,
                )

            resumed = rebuild.execute_rebuild(
                rebuild._parser().parse_args(
                    [
                        "--config",
                        str(output_path),
                        "--profile",
                        "healthcare",
                        "--batch-size",
                        "1",
                        "--apply",
                    ]
                ),
                store=store,
            )
            self.assertEqual(0, resumed["profiles"][0]["closed"])
            self.assertEqual(0, resumed["profiles"][0]["rebuilt_assignments"])
            self.assertEqual(1, len(store.list_threat_clusters(target)))


if __name__ == "__main__":
    unittest.main()
