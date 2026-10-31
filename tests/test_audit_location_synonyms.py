import unittest

from audit_location_synonyms import (
    audit_tag_synonyms,
    audit_location_synonyms,
    create_child_from_synonym,
    find_tags_with_synonyms,
    find_location_tags_with_synonyms,
)


class _Store:
    def __init__(self):
        self.tags = {
            1: {
                "id": 1,
                "name": "Europe",
                "category": "LOCATION",
                "description": "",
                "synonyms": ["Sweden", "sweden", "France"],
            },
            2: {
                "id": 2,
                "name": "Ransomware",
                "category": "THREAT",
                "description": "",
                "synonyms": ["crypto malware"],
            },
            3: {
                "id": 3,
                "name": "Norway",
                "category": "LOCATION",
                "description": "",
                "synonyms": [],
            },
        }
        self.relations = set()
        self.articles = {
            "a1": {"id": "a1", "title": "Incident in Sweden", "content": "Sweden"},
            "a2": {"id": "a2", "title": "Incident in Norway", "content": "Norway"},
            "a3": {
                "id": "a3",
                "title": "Crypto malware campaign",
                "content": "Crypto malware encrypted the systems.",
            },
        }
        self.article_tags = {"a1": {1}, "a2": {1}, "a3": {2}}

    def get_all_tags(self):
        return [dict(tag) for tag in self.tags.values()]

    def create_tag(self, name, category="GENERAL", description="", synonyms=None):
        if any(tag["name"].casefold() == name.casefold() for tag in self.tags.values()):
            return None
        tag_id = max(self.tags) + 1
        self.tags[tag_id] = {
            "id": tag_id,
            "name": name,
            "category": category,
            "description": description,
            "synonyms": list(synonyms or []),
        }
        return dict(self.tags[tag_id])

    def update_tag(self, tag_id, **changes):
        if tag_id not in self.tags:
            return None
        self.tags[tag_id].update(
            {key: value for key, value in changes.items() if value is not None}
        )
        return dict(self.tags[tag_id])

    def delete_tag(self, tag_id):
        return self.tags.pop(tag_id, None) is not None

    def get_tag_relations(self, tag_id):
        return {
            "parents": [
                dict(self.tags[parent])
                for parent, child in self.relations
                if child == tag_id
            ],
            "children": [
                dict(self.tags[child])
                for parent, child in self.relations
                if parent == tag_id
            ],
        }

    def set_tag_relations(self, tag_id, *, parent_ids=None, child_ids=None):
        if parent_ids is not None:
            self.relations = {edge for edge in self.relations if edge[1] != tag_id}
            self.relations.update((parent_id, tag_id) for parent_id in parent_ids)
        if child_ids is not None:
            self.relations = {edge for edge in self.relations if edge[0] != tag_id}
            self.relations.update((tag_id, child_id) for child_id in child_ids)
        return self.get_tag_relations(tag_id)

    def get_articles_by_tags(self, tag_names, match_mode="any"):
        wanted_ids = {
            tag["id"]
            for tag in self.tags.values()
            if tag["name"].casefold() in {name.casefold() for name in tag_names}
        }
        return [
            dict(article)
            for article_id, article in self.articles.items()
            if self.article_tags.get(article_id, set()) & wanted_ids
        ]

    def get_article_tags(self, article_id):
        return [dict(self.tags[tag_id]) for tag_id in self.article_tags[article_id]]

    def add_tag_to_article(self, article_id, tag_id):
        if tag_id in self.article_tags[article_id]:
            return False
        self.article_tags[article_id].add(tag_id)
        return True

    def remove_article_tag(self, article_id, tag_id):
        if tag_id not in self.article_tags[article_id]:
            return False
        self.article_tags[article_id].remove(tag_id)
        return True


class _LLM:
    async def chat(self, messages, temperature=0.3):
        prompt = messages[0]["content"]
        self.last_prompt = prompt
        if "Incident in Sweden" in prompt:
            return '{"suggested_tags": [{"tag": "Sweden", "reasoning": "Nämns"}]}'
        if "Crypto malware campaign" in prompt:
            return '{"suggested_tags": [{"tag": "crypto malware", "reasoning": "Nämns"}]}'
        return '{"suggested_tags": []}'


class LocationSynonymAuditTests(unittest.IsolatedAsyncioTestCase):
    def test_finds_only_location_tags_and_deduplicates_synonyms(self):
        store = _Store()

        tags = find_location_tags_with_synonyms(store)

        self.assertEqual(["Europe"], [tag["name"] for tag in tags])
        self.assertEqual(["Sweden", "France"], tags[0]["synonyms"])

    def test_creates_child_relation_and_removes_synonym(self):
        store = _Store()

        child, created = create_child_from_synonym(store, store.tags[1], "Sweden")

        self.assertTrue(created)
        self.assertEqual("LOCATION", child["category"])
        self.assertEqual({(1, child["id"])}, store.relations)
        self.assertEqual(["France"], store.tags[1]["synonyms"])

    async def test_interactive_conversion_retags_only_matching_articles(self):
        store = _Store()
        llm = _LLM()
        answers = iter(["n", "b"])

        report = await audit_location_synonyms(
            store,
            llm,
            input_fn=lambda _prompt: next(answers),
        )

        child_id = next(
            tag_id for tag_id, tag in store.tags.items() if tag["name"] == "Sweden"
        )
        self.assertEqual({child_id}, store.article_tags["a1"])
        self.assertEqual({1}, store.article_tags["a2"])
        self.assertEqual(1, report["totals"]["articles_replaced"])
        suggestions = report["decisions"][0]["retagging"]["articles"][0][
            "suggested_tags"
        ]
        self.assertEqual(["Sweden"], suggestions)
        self.assertNotIn("Ransomware", suggestions)
        self.assertNotIn("Ransomware", llm.last_prompt)
        self.assertEqual(1, report["totals"]["kept"])

    async def test_can_audit_an_arbitrary_category(self):
        store = _Store()
        llm = _LLM()

        tags = find_tags_with_synonyms(store, "THREAT")
        report = await audit_tag_synonyms(
            store,
            llm,
            category="threat",
            input_fn=lambda _prompt: "n",
        )

        self.assertEqual(["Ransomware"], [tag["name"] for tag in tags])
        child_id = next(
            tag_id
            for tag_id, tag in store.tags.items()
            if tag["name"] == "crypto malware"
        )
        self.assertEqual("THREAT", store.tags[child_id]["category"])
        self.assertEqual({child_id}, store.article_tags["a3"])
        self.assertEqual("THREAT", report["category"])
        self.assertEqual(1, report["totals"]["articles_replaced"])
        self.assertNotIn("Europe", llm.last_prompt)

    async def test_dry_run_does_not_change_store_or_call_llm(self):
        store = _Store()
        original_tags = {tag_id: dict(tag) for tag_id, tag in store.tags.items()}

        report = await audit_location_synonyms(
            store,
            None,
            dry_run=True,
            input_fn=lambda _prompt: "n",
        )

        self.assertEqual(original_tags, store.tags)
        self.assertEqual(set(), store.relations)
        self.assertTrue(
            all(
                item["status"] == "would_create_child"
                for item in report["decisions"]
            )
        )


if __name__ == "__main__":
    unittest.main()
