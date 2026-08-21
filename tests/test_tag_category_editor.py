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

    def get_all_categories(self):
        return [category.copy() for category in self.categories]

    def get_all_tags(self):
        return [tag.copy() for tag in self.tags]

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


if __name__ == "__main__":
    unittest.main()
