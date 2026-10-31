import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from audit_tag_synonym_candidates import apply_choice, audit_candidates, find_candidate_groups
from feedsummary_core.persistence.TinyDbStore import TinyDBStore
from tests.test_audit_location_synonyms import _Store


class _CandidateStore(_Store):
    def __init__(self):
        super().__init__()
        self.tags = {
            1: {"id": 1, "name": "NetScaler", "category": "PRODUCT", "synonyms": ["Citrix ADC"]},
            2: {"id": 2, "name": "Citrix NetScaler", "category": "PRODUCT", "synonyms": []},
            3: {"id": 3, "name": "NetScaler ADC", "category": "PRODUCT", "synonyms": ["ADC appliance"]},
            4: {"id": 4, "name": "Unrelated product", "category": "PRODUCT", "synonyms": []},
            5: {"id": 5, "name": "NetScaler", "category": "OTHER", "synonyms": []},
        }
        self.relations = set()
        self.article_tags = {"a1": {2}, "a2": {3}, "a3": {1, 3}}

    def migrate_synonym_to_main_tag(self, main_id, synonym_ids):
        migrated = 0
        deleted = 0
        for synonym_id in synonym_ids:
            for tag_ids in self.article_tags.values():
                if synonym_id in tag_ids:
                    tag_ids.remove(synonym_id)
                    tag_ids.add(main_id)
                    migrated += 1
            if self.tags.pop(synonym_id, None):
                deleted += 1
        return migrated, deleted


class CandidateAuditTests(unittest.TestCase):
    def test_discovers_complete_group_before_review(self):
        store = _CandidateStore()
        groups = find_candidate_groups(store, "PRODUCT")

        self.assertEqual(1, len(groups))
        self.assertEqual(
            ["Citrix NetScaler", "NetScaler", "NetScaler ADC"],
            [tag["name"] for tag in groups[0]["tags"]],
        )
        self.assertEqual(3, len(groups[0]["links"]))

    def test_finds_related_names_without_short_anchor_tag(self):
        store = _CandidateStore()
        store.tags.pop(1)

        groups = find_candidate_groups(store, "PRODUCT")

        self.assertEqual(1, len(groups))
        self.assertEqual([2, 3], [tag["id"] for tag in groups[0]["tags"]])

    def test_unrelated_endpoints_do_not_join_through_name_word_chain(self):
        store = _CandidateStore()
        names = [
            "389 directory server",
            "directory server",
            "active directory",
            "abandoned cart pro for woocommerce",
            "woocommerce",
            "zscaler client connector",
            "client connector",
        ]
        store.tags = {
            index: {"id": index, "name": name, "category": "PRODUCT", "synonyms": []}
            for index, name in enumerate(names, 1)
        }

        groups = find_candidate_groups(store, "PRODUCT")
        grouped_names = [set(tag["name"] for tag in group["tags"]) for group in groups]

        self.assertTrue(all(len(group) <= 2 for group in grouped_names))
        self.assertIn({"389 directory server", "directory server"}, grouped_names)
        self.assertNotIn({"389 directory server", "active directory"}, grouped_names)
        self.assertTrue(all("abandoned cart pro for woocommerce" not in group for group in grouped_names))

    def test_shared_vendor_or_product_type_does_not_make_a_group(self):
        store = _CandidateStore()
        names = ["Microsoft", *[f"Microsoft Product {number}" for number in range(13)],
                 "Huly Platform", "Logix Platform"]
        store.tags = {
            index: {"id": index, "name": name, "category": "PRODUCT", "synonyms": []}
            for index, name in enumerate(names, 1)
        }

        self.assertEqual([], find_candidate_groups(store, "PRODUCT"))

    def test_selects_main_and_applies_each_relation(self):
        store = _CandidateStore()
        answers = iter(["2", "b", "y"])

        report = audit_candidates(store, category="product", input_fn=lambda _: next(answers))

        self.assertEqual("PRODUCT", report["category"])
        self.assertEqual({(1, 2)}, store.relations)
        self.assertNotIn(3, store.tags)
        self.assertEqual({1}, store.article_tags["a2"])
        self.assertEqual({1}, store.article_tags["a3"])
        self.assertEqual(
            {"citrix adc", "netscaler adc", "adc appliance"},
            {alias.casefold() for alias in store.tags[1]["synonyms"]},
        )
        self.assertEqual(1, report["totals"]["children"])
        self.assertEqual(1, report["totals"]["synonyms"])
        self.assertEqual(2, report["totals"]["articles_migrated"])

    def test_dry_run_records_choices_without_writing(self):
        store = _CandidateStore()
        answers = iter(["2", "b", "y"])

        report = audit_candidates(
            store, category="PRODUCT", dry_run=True,
            input_fn=lambda _: next(answers),
        )

        self.assertEqual({"would_child", "would_synonym"}, {item["status"] for item in report["decisions"]})
        self.assertEqual(set(), store.relations)
        self.assertIn(3, store.tags)
        self.assertEqual([], store.tags[1]["synonyms"][1:])

    def test_preserves_other_parent_and_rejects_deleting_related_tag(self):
        store = _CandidateStore()
        store.relations.add((4, 2))
        store.tags[1]["synonyms"].append("Citrix NetScaler")

        apply_choice(store, "PRODUCT", 1, 2, "child")
        self.assertEqual({(4, 2), (1, 2)}, store.relations)
        self.assertEqual(["Citrix ADC"], store.tags[1]["synonyms"])
        with self.assertRaisesRegex(ValueError, "har relationer"):
            apply_choice(store, "PRODUCT", 1, 2, "synonym")
        self.assertIn(2, store.tags)

    def test_tinydb_migrates_article_tags_and_aliases(self):
        with TemporaryDirectory() as directory:
            store = TinyDBStore(str(Path(directory) / "tags.json"))
            main = store.create_tag("NetScaler", "PRODUCT", synonyms=["Citrix ADC"])
            other = store.create_tag("NetScaler ADC", "PRODUCT", synonyms=["ADC appliance"])
            store.upsert_article({"id": "a1", "title": "ADC incident"})
            store.add_tag_to_article("a1", other["id"])

            result = apply_choice(store, "PRODUCT", main["id"], other["id"], "synonym")

            self.assertEqual("synonym", result["status"])
            self.assertEqual(1, result["articles_migrated"])
            self.assertEqual([main["id"]], [tag["id"] for tag in store.get_article_tags("a1")])
            self.assertEqual(["citrix adc", "netscaler adc", "adc appliance"],
                             next(tag for tag in store.get_all_tags() if tag["id"] == main["id"])["synonyms"])


if __name__ == "__main__":
    unittest.main()
