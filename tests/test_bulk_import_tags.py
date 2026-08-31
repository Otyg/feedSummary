import json
import tempfile
import unittest
from pathlib import Path

from bulk_import_tags import import_tags, load_tag_import


class FakeStore:
    def __init__(self):
        self.tags = {
            1: {
                "id": 1,
                "name": "cybersecurity",
                "category": "DOMAIN_ENTITY",
                "synonyms": ["information security"],
            },
            2: {
                "id": 2,
                "name": "infosec",
                "category": "GENERAL",
                "synonyms": [],
            },
        }
        self.article_tags = {"article-1": {2}}
        self.next_id = 3

    def get_all_tags(self):
        return [tag.copy() for tag in self.tags.values()]

    def create_tag(self, name, category="GENERAL", description="", synonyms=None):
        if any(tag["name"].casefold() == name.casefold() for tag in self.tags.values()):
            return None
        tag = {
            "id": self.next_id,
            "name": name,
            "category": category,
            "description": description,
            "synonyms": list(synonyms or []),
        }
        self.tags[self.next_id] = tag
        self.next_id += 1
        return tag.copy()

    def update_tag(
        self,
        tag_id,
        name=None,
        category=None,
        description=None,
        synonyms=None,
    ):
        tag = self.tags.get(tag_id)
        if tag is None:
            return None
        if synonyms is not None:
            tag["synonyms"] = list(synonyms)
        return tag.copy()

    def migrate_synonym_to_main_tag(self, main_tag_id, synonym_tag_ids):
        migrated = 0
        deleted = 0
        for synonym_tag_id in synonym_tag_ids:
            for tag_ids in self.article_tags.values():
                if synonym_tag_id in tag_ids:
                    tag_ids.discard(synonym_tag_id)
                    tag_ids.add(main_tag_id)
                    migrated += 1
            if self.tags.pop(synonym_tag_id, None) is not None:
                deleted += 1
        return migrated, deleted


class LoadTagImportTests(unittest.TestCase):
    def _load(self, payload):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tags.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return load_tag_import(path)

    def test_accepts_a_single_object_and_removes_duplicate_aliases(self):
        entries = self._load(
            {
                "name": "Cybersecurity",
                "category": "DOMAIN_ENTITY",
                "aliases": ["Infosec", "infosec", "Cybersecurity"],
            }
        )

        self.assertEqual(
            [
                {
                    "name": "Cybersecurity",
                    "category": "DOMAIN_ENTITY",
                    "aliases": ["Infosec"],
                }
            ],
            entries,
        )

    def test_merges_a_tag_that_is_an_alias_in_the_same_file(self):
        entries = self._load(
            [
                {"name": "malware", "category": "THREAT", "aliases": ["virus"]},
                {"name": "virus", "category": "THREAT", "aliases": []},
            ]
        )

        self.assertEqual(
            [{"name": "malware", "category": "THREAT", "aliases": ["virus"]}],
            entries,
        )

    def test_merges_objects_that_share_an_alias(self):
        entries = self._load(
            [
                {
                    "name": "Ember Bear",
                    "category": "THREAT_ACTOR",
                    "aliases": ["UNC2589", "UAC-0056"],
                },
                {
                    "name": "Saint Bear",
                    "category": "THREAT_ACTOR",
                    "aliases": ["UAC-0056", "Storm Bear"],
                },
            ]
        )

        self.assertEqual(
            [
                {
                    "name": "Ember Bear",
                    "category": "THREAT_ACTOR",
                    "aliases": [
                        "UNC2589",
                        "UAC-0056",
                        "Saint Bear",
                        "Storm Bear",
                    ],
                }
            ],
            entries,
        )

    def test_shared_alias_merges_transitively(self):
        entries = self._load(
            [
                {"name": "A", "category": "MALWARE", "aliases": ["shared-1"]},
                {
                    "name": "B",
                    "category": "MALWARE",
                    "aliases": ["shared-1", "shared-2"],
                },
                {"name": "C", "category": "MALWARE", "aliases": ["shared-2"]},
            ]
        )

        self.assertEqual(1, len(entries))
        self.assertEqual("A", entries[0]["name"])
        self.assertEqual(
            ["shared-1", "B", "shared-2", "C"],
            entries[0]["aliases"],
        )

    def test_rejects_shared_alias_across_different_categories(self):
        with self.assertRaisesRegex(ValueError, "olika kategorier"):
            self._load(
                [
                    {"name": "A", "category": "MALWARE", "aliases": ["shared"]},
                    {
                        "name": "B",
                        "category": "THREAT_ACTOR",
                        "aliases": ["shared"],
                    },
                ]
            )


class BulkImportTagTests(unittest.TestCase):
    def test_preview_does_not_change_the_store(self):
        store = FakeStore()

        result = import_tags(
            store,
            [
                {
                    "name": "cybersecurity",
                    "category": "GENERAL",
                    "aliases": ["infosec"],
                },
                {"name": "malware", "category": "THREAT", "aliases": ["virus"]},
            ],
        )

        self.assertEqual(1, result["created"])
        self.assertEqual(1, result["updated"])
        self.assertEqual({1, 2}, set(store.tags))
        self.assertEqual({2}, store.article_tags["article-1"])

    def test_existing_tag_gets_only_new_aliases_and_migrates_matching_tag(self):
        store = FakeStore()

        result = import_tags(
            store,
            [
                {
                    "name": "CyberSecurity",
                    "category": "SHOULD_NOT_REPLACE_EXISTING",
                    "aliases": ["INFORMATION SECURITY", "Infosec"],
                }
            ],
            apply=True,
        )

        self.assertEqual(1, result["updated"])
        self.assertEqual(1, result["articles_migrated"])
        self.assertEqual(1, result["synonym_tags_deleted"])
        self.assertEqual(
            ["information security", "Infosec"], store.tags[1]["synonyms"]
        )
        self.assertEqual("DOMAIN_ENTITY", store.tags[1]["category"])
        self.assertNotIn(2, store.tags)
        self.assertEqual({1}, store.article_tags["article-1"])

    def test_new_tag_aliases_use_the_same_migration_flow(self):
        store = FakeStore()

        result = import_tags(
            store,
            [
                {
                    "name": "security",
                    "category": "DOMAIN_ENTITY",
                    "aliases": ["infosec"],
                }
            ],
            apply=True,
        )

        self.assertEqual(1, result["created"])
        self.assertEqual(1, result["articles_migrated"])
        self.assertEqual(["infosec"], store.tags[3]["synonyms"])
        self.assertNotIn(2, store.tags)
        self.assertEqual({3}, store.article_tags["article-1"])

    def test_existing_alias_comparison_is_case_insensitive(self):
        store = FakeStore()

        result = import_tags(
            store,
            [
                {
                    "name": "cybersecurity",
                    "category": "DOMAIN_ENTITY",
                    "aliases": ["Information Security"],
                }
            ],
            apply=True,
        )

        self.assertEqual(1, result["unchanged"])
        self.assertEqual(["information security"], store.tags[1]["synonyms"])


if __name__ == "__main__":
    unittest.main()
