import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from audit_product_organizations import (
    audit_articles,
    discover_pairs,
    load_reviewed_pairs,
    run_audit,
)
from feedsummary_core.persistence.TinyDbStore import TinyDBStore


def tag(tag_id, name, category, synonyms=None):
    return {"id": tag_id, "name": name, "category": category,
            "synonyms": list(synonyms or [])}


class _Store:
    def __init__(self):
        self.tags = [
            tag(1, "Cisco AsyncOS", "PRODUCT"),
            tag(2, "Cisco IOS", "PRODUCT"),
            tag(3, "Microsoft Active Directory", "PRODUCT"),
            tag(4, "Cisco", "ORGANIZATION"),
            tag(5, "Microsoft", "ORGANIZATION"),
        ]
        self.articles = [
            {"id": "a1", "title": "Cisco products"},
            {"id": "a2", "title": "Microsoft product"},
            {"id": "a3", "title": "Already tagged"},
        ]
        self.article_tags = {"a1": {1, 2}, "a2": {3}, "a3": {1, 4}}
        self.added = []

    def get_all_tags(self):
        return list(self.tags)

    def get_articles_by_tags(self, names, match_mode="any"):
        wanted = {tag["id"] for tag in self.tags if tag["name"] in names}
        return [article for article in self.articles
                if self.article_tags[article["id"]] & wanted]

    def get_article_tags(self, article_id):
        return [tag for tag in self.tags if tag["id"] in self.article_tags[article_id]]

    def add_tag_to_article(self, article_id, tag_id):
        self.added.append((article_id, tag_id))
        self.article_tags[article_id].add(tag_id)
        return True


class ProductOrganizationAuditTests(unittest.TestCase):
    def test_discovers_prefix_and_alias_pairs_without_substring_matches(self):
        tags = [
            tag(1, "Cisco AsyncOS", "PRODUCT"),
            tag(2, "Microsoft Active Directory", "PRODUCT"),
            tag(3, "HPE ProLiant", "PRODUCT"),
            tag(4, "AsyncOS", "PRODUCT", ["Cisco AsyncOS"]),
            tag(5, "NotCisco Tool", "PRODUCT"),
            tag(6, "Cisco Systems IOS", "PRODUCT"),
            tag(10, "Cisco", "ORGANIZATION"),
            tag(11, "Microsoft", "ORGANIZATION"),
            tag(12, "Hewlett Packard Enterprise", "ORGANIZATION", ["HPE"]),
            tag(13, "Cisco Systems", "ORGANIZATION"),
        ]

        result = discover_pairs(tags)
        mapping = {pair["product"]: pair["organization"] for pair in result["pairs"]}

        self.assertEqual("Cisco", mapping["Cisco AsyncOS"])
        self.assertEqual("Microsoft", mapping["Microsoft Active Directory"])
        self.assertEqual("Hewlett Packard Enterprise", mapping["HPE ProLiant"])
        self.assertEqual("Cisco", mapping["AsyncOS"])
        self.assertEqual("Cisco Systems", mapping["Cisco Systems IOS"])
        self.assertNotIn("NotCisco Tool", mapping)

    def test_checks_each_article_and_adds_each_missing_organization_once(self):
        store = _Store()
        pairs = discover_pairs(store.get_all_tags())["pairs"]

        preview = audit_articles(store, pairs)
        self.assertEqual(3, preview["articles_checked"])
        self.assertEqual(2, preview["organization_tags_missing"])
        self.assertEqual(1, preview["organization_tags_present"])
        self.assertEqual([], store.added)
        self.assertEqual(["Cisco AsyncOS", "Cisco IOS"], preview["missing"][0]["products"])

        applied = audit_articles(store, pairs, apply=True)
        self.assertEqual(2, applied["organization_tags_added"])
        self.assertEqual([("a1", 4), ("a2", 5)], store.added)
        self.assertEqual(0, audit_articles(store, pairs)["organization_tags_missing"])

    def test_ambiguous_organization_alias_is_not_paired(self):
        tags = [
            tag(1, "HPE ProLiant", "PRODUCT"),
            tag(2, "Hewlett Packard Enterprise", "ORGANIZATION", ["HPE"]),
            tag(3, "Another HPE Organization", "ORGANIZATION", ["HPE"]),
        ]

        result = discover_pairs(tags)

        self.assertEqual([], result["pairs"])
        self.assertEqual(["HPE ProLiant"], [item["product"] for item in result["ambiguous"]])

    def test_reviewed_pairs_require_current_ids_names_and_categories(self):
        store = _Store()
        pair = discover_pairs(store.tags)["pairs"][0]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.json"
            path.write_text(json.dumps({"pairs": [pair]}), encoding="utf-8")
            self.assertEqual([pair["product_id"]],
                             [item["product_id"] for item in load_reviewed_pairs(path, store.tags)])
            store.tags[0]["name"] = "Renamed AsyncOS"
            with self.assertRaisesRegex(ValueError, "stämmer inte längre"):
                load_reviewed_pairs(path, store.tags)

    def test_tinydb_audit_reports_and_then_adds_missing_tag(self):
        with TemporaryDirectory() as directory:
            store = TinyDBStore(str(Path(directory) / "tags.json"))
            product = store.create_tag("Cisco AsyncOS", "PRODUCT")
            organization = store.create_tag("Cisco", "ORGANIZATION")
            store.upsert_article({"id": "a1", "title": "AsyncOS flaw"})
            store.add_tag_to_article("a1", product["id"])

            preview = run_audit(store)
            self.assertEqual(1, preview["totals"]["organization_tags_missing"])
            self.assertEqual([product["id"]],
                             [tag["id"] for tag in store.get_article_tags("a1")])

            applied = run_audit(store, apply=True)
            self.assertEqual(1, applied["totals"]["organization_tags_added"])
            self.assertEqual({product["id"], organization["id"]},
                             {tag["id"] for tag in store.get_article_tags("a1")})


if __name__ == "__main__":
    unittest.main()
