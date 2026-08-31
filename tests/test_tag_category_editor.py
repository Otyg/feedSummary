import unittest

from web_viewer import webapp_viewer


class FakeTagStore:
    def __init__(self):
        self.categories = [
            {"id": 1, "name": "GENERAL", "label": "Allmän"},
            {"id": 2, "name": "THREAT", "label": "Hot"},
        ]
        self.tags = [
            {
                "id": 10,
                "name": "ransomware",
                "category": "THREAT",
                "description": "Utpressningsprogram",
                "synonyms": ["crypto-malware"],
            },
            {
                "id": 11,
                "name": "misc",
                "category": "GENERAL",
                "description": "",
                "synonyms": [],
            },
        ]
        self.update_calls = []
        self.relation_edges = set()

    def get_all_categories(self):
        return [category.copy() for category in self.categories]

    def get_all_tags(self):
        return [tag.copy() for tag in self.tags]

    def list_articles(self, limit=10000):
        return []

    def get_article_tags(self, article_id):
        return []

    def get_tag_relations(self, tag_id):
        tags_by_id = {tag["id"]: tag for tag in self.tags}
        return {
            "parents": [
                tags_by_id[parent].copy()
                for parent, child in sorted(self.relation_edges)
                if child == tag_id
            ],
            "children": [
                tags_by_id[child].copy()
                for parent, child in sorted(self.relation_edges)
                if parent == tag_id
            ],
        }

    def set_tag_relations(self, tag_id, *, parent_ids=None, child_ids=None):
        tags_by_id = {tag["id"]: tag for tag in self.tags}
        if tag_id not in tags_by_id:
            raise webapp_viewer.TagRelationError(f"tag not found: {tag_id}")
        parent_ids = None if parent_ids is None else {int(value) for value in parent_ids}
        child_ids = None if child_ids is None else {int(value) for value in child_ids}
        related_ids = (parent_ids or set()) | (child_ids or set())
        if any(
            related_id not in tags_by_id
            or tags_by_id[related_id]["category"] != tags_by_id[tag_id]["category"]
            for related_id in related_ids
        ):
            raise webapp_viewer.TagRelationError(
                "parent-child relations cannot cross categories"
            )
        if parent_ids is not None:
            self.relation_edges = {
                edge for edge in self.relation_edges if edge[1] != tag_id
            }
            self.relation_edges.update((parent_id, tag_id) for parent_id in parent_ids)
        if child_ids is not None:
            self.relation_edges = {
                edge for edge in self.relation_edges if edge[0] != tag_id
            }
            self.relation_edges.update((tag_id, child_id) for child_id in child_ids)
        return self.get_tag_relations(tag_id)

    def update_tag(
        self,
        tag_id,
        name=None,
        category=None,
        description=None,
        synonyms=None,
    ):
        tag = next((tag for tag in self.tags if tag["id"] == tag_id), None)
        if tag is None:
            return None
        self.update_calls.append(
            {
                "tag_id": tag_id,
                "name": name,
                "category": category,
                "description": description,
                "synonyms": synonyms,
            }
        )
        if name is not None:
            tag["name"] = name
        if category is not None:
            if category != tag["category"]:
                self.relation_edges = {
                    edge for edge in self.relation_edges if tag_id not in edge
                }
            tag["category"] = category
        if description is not None:
            tag["description"] = description
        if synonyms is not None:
            tag["synonyms"] = synonyms
        return tag.copy()


class TagCategoryEditorTests(unittest.TestCase):
    def setUp(self):
        self.original_store = webapp_viewer.APP_STORE
        self.store = FakeTagStore()
        webapp_viewer.APP_STORE = self.store
        webapp_viewer.app.config.update(TESTING=True)
        self.client = webapp_viewer.app.test_client()

    def tearDown(self):
        webapp_viewer.APP_STORE = self.original_store

    def test_page_lists_only_tags_in_selected_category(self):
        response = self.client.get("/tag-categories?category=THREAT")

        self.assertEqual(200, response.status_code)
        html = response.get_data(as_text=True)
        self.assertIn("ransomware", html)
        self.assertNotIn(">misc<", html)
        self.assertIn('data-original-category="THREAT"', html)

    def test_tag_list_can_be_filtered_by_category(self):
        response = self.client.get("/tags?category=THREAT")

        self.assertEqual(200, response.status_code)
        html = response.get_data(as_text=True)
        self.assertIn('id="category-filter"', html)
        self.assertIn('value="THREAT" selected', html)
        self.assertIn("ransomware", html)
        self.assertNotIn("<strong>misc</strong>", html)
        self.assertIn("Visar 1 av 2 taggar", html)

    def test_tag_list_shows_all_tags_without_category_filter(self):
        response = self.client.get("/tags")

        self.assertEqual(200, response.status_code)
        html = response.get_data(as_text=True)
        self.assertIn("ransomware", html)
        self.assertIn("<strong>misc</strong>", html)
        self.assertIn("Alla kategorier (2)", html)
        self.assertIn("Relationer", html)
        self.assertIn("Föräldrar", html)

    def test_batch_update_changes_all_requested_categories(self):
        response = self.client.put(
            "/api/v1/tags/categories",
            json={
                "changes": [
                    {"tag_id": 10, "category": "GENERAL"},
                    {"tag_id": 11, "category": "THREAT"},
                ]
            },
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual(2, response.get_json()["updated_count"])
        self.assertEqual("GENERAL", self.store.tags[0]["category"])
        self.assertEqual("THREAT", self.store.tags[1]["category"])

    def test_batch_update_validates_every_change_before_writing(self):
        response = self.client.put(
            "/api/v1/tags/categories",
            json={
                "changes": [
                    {"tag_id": 10, "category": "GENERAL"},
                    {"tag_id": 11, "category": "UNKNOWN"},
                ]
            },
        )

        self.assertEqual(400, response.status_code)
        self.assertEqual([], self.store.update_calls)
        self.assertEqual("THREAT", self.store.tags[0]["category"])

    def test_partial_tag_update_does_not_clear_synonyms(self):
        response = self.client.put(
            "/api/v1/tags/10",
            json={"category": "GENERAL"},
        )

        self.assertEqual(200, response.status_code)
        self.assertIsNone(self.store.update_calls[0]["synonyms"])
        self.assertEqual(["crypto-malware"], self.store.tags[0]["synonyms"])

    def test_parent_child_relation_is_visible_from_both_tags(self):
        self.store.tags.append(
            {
                "id": 12,
                "name": "malware",
                "category": "THREAT",
                "description": "",
                "synonyms": [],
            }
        )

        response = self.client.put(
            "/api/v1/tags/12/relations", json={"child_ids": [10]}
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            ["ransomware"],
            [tag["name"] for tag in response.get_json()["relations"]["children"]],
        )
        inverse = self.client.get("/api/v1/tags/10/relations").get_json()
        self.assertEqual(
            ["malware"],
            [tag["name"] for tag in inverse["relations"]["parents"]],
        )

    def test_parent_child_relation_cannot_cross_categories(self):
        response = self.client.put(
            "/api/v1/tags/10/relations", json={"child_ids": [11]}
        )

        self.assertEqual(400, response.status_code)
        self.assertIn("cross categories", response.get_json()["error"])
        self.assertEqual(set(), self.store.relation_edges)


if __name__ == "__main__":
    unittest.main()
