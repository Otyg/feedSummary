import sqlite3
import tempfile
import unittest
from pathlib import Path

from web_viewer import webapp_viewer


class _SqliteTagStore:
    def __init__(self, path: Path):
        self.path = path

    def _connect(self):
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return con

    def get_article(self, article_id):
        if article_id != "article-1":
            return None
        return {
            "id": article_id,
            "title": "Testartikel",
            "source": "Testkälla",
            "url": "https://example.test/article",
            "text": "Artikeltext",
            "published_ts": 1,
        }

    def get_article_tags(self, article_id):
        return [
            {"id": 7, "name": "ransomware", "category": "THREAT"},
            {"id": 8, "name": "manuell", "category": "GENERAL"},
        ]


class _MongoCollection:
    def find(self, query):
        if query == {"article_id": "article-1"}:
            return [
                {
                    "article_id": "article-1",
                    "tag_id": 7,
                    "motivering": "Hämtad från MongoDB-relationen.",
                }
            ]
        return []


class _MongoDatabase:
    article_tags = _MongoCollection()


class _MongoTagStore:
    db = _MongoDatabase()

    def get_article_tags(self, article_id):
        return [{"id": 7, "name": "ransomware", "category": "THREAT"}]


class ArticleTagMotiveringarTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        path = Path(self.temp_dir.name) / "tags.sqlite"
        self.store = _SqliteTagStore(path)
        con = self.store._connect()
        try:
            con.execute(
                "CREATE TABLE article_tags "
                "(article_id TEXT, tag_id INTEGER, motivering TEXT)"
            )
            con.executemany(
                "INSERT INTO article_tags VALUES (?, ?, ?)",
                [
                    ("article-1", 7, "Ransomware är artikelns huvudämne."),
                    ("article-1", 8, None),
                ],
            )
            con.commit()
        finally:
            con.close()

        self.original_store = webapp_viewer.APP_STORE
        webapp_viewer.APP_STORE = self.store
        webapp_viewer.app.config.update(TESTING=True)
        self.client = webapp_viewer.app.test_client()

    def tearDown(self):
        webapp_viewer.APP_STORE = self.original_store
        self.temp_dir.cleanup()

    def test_enriches_tags_from_article_tag_relation(self):
        tags = webapp_viewer._article_tags_with_motiveringar(self.store, "article-1")

        self.assertEqual("Ransomware är artikelns huvudämne.", tags[0]["motivering"])
        self.assertNotIn("motivering", tags[1])

    def test_enriches_tags_from_mongodb_article_tag_relation(self):
        tags = webapp_viewer._article_tags_with_motiveringar(
            _MongoTagStore(), "article-1"
        )

        self.assertEqual("Hämtad från MongoDB-relationen.", tags[0]["motivering"])

    def test_article_view_only_lists_existing_motiveringar(self):
        response = self.client.get("/article/article-1")

        self.assertEqual(200, response.status_code)
        html = response.get_data(as_text=True)
        self.assertIn("Visa motiveringar", html)
        self.assertIn("Ransomware är artikelns huvudämne.", html)
        self.assertIn('id="tag-motiveringar-count">1</span>', html)
        self.assertNotIn("<strong>manuell:</strong>", html)

    def test_article_api_includes_motivering(self):
        response = self.client.get("/api/v1/article/article-1")

        self.assertEqual(200, response.status_code)
        tags = response.get_json()["item"]["tags"]
        self.assertEqual("Ransomware är artikelns huvudämne.", tags[0]["motivering"])
        self.assertEqual("", tags[1]["motivering"])


if __name__ == "__main__":
    unittest.main()
