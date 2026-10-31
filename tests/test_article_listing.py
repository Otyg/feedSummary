import os
import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

_previous_auto_init = os.environ.get("FEEDSUMMARY_SKIP_VIEWER_AUTO_INIT")
os.environ["FEEDSUMMARY_SKIP_VIEWER_AUTO_INIT"] = "1"
try:
    from web_viewer import webapp_viewer
    from web_viewer.article_listing import list_article_page
finally:
    if _previous_auto_init is None:
        os.environ.pop("FEEDSUMMARY_SKIP_VIEWER_AUTO_INIT", None)
    else:
        os.environ["FEEDSUMMARY_SKIP_VIEWER_AUTO_INIT"] = _previous_auto_init


class _ArticleStore:
    def __init__(self, path):
        self.path = path

    def _connect(self):
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return con

    def get_all_categories(self):
        return [
            {
                "name": "LOCATION", "label": "Plats",
                "bg_color": "#118855", "text_color": "#ffffff",
            },
            {
                "name": "THREAT", "label": "Hot",
                "bg_color": "bg-danger", "text_color": "text-white",
            },
        ]

    def get_tag_relations(self, tag_id):
        return {"children": ([{"id": 2}] if tag_id == 1 else [])}

    def list_articles(self, *args, **kwargs):
        raise AssertionError("The article page must not load all article documents")

    def get_all_tags(self):
        raise AssertionError("The article page must not load all tags")


class ArticleListingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = str(Path(self.tmp.name) / "articles.sqlite")
        self.store = _ArticleStore(path)
        con = self.store._connect()
        con.executescript("""
            CREATE TABLE articles (
                id TEXT PRIMARY KEY, title TEXT, source TEXT, url TEXT,
                published_ts INTEGER, fetched_at INTEGER, doc_json TEXT
            );
            CREATE TABLE tags (id INTEGER PRIMARY KEY, name TEXT, category TEXT);
            CREATE TABLE article_tags (article_id TEXT, tag_id INTEGER);
        """)
        con.executemany(
            "INSERT INTO tags VALUES (?, ?, ?)",
            [(1, "Europe", "LOCATION"), (2, "Sweden", "LOCATION"), (3, "Malware", "THREAT")],
        )
        now = int(datetime.now().timestamp())
        for index in range(27):
            article_id = f"recent-{index:02d}"
            con.execute(
                "INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?, ?)",
                (article_id, f"Recent article {index}",
                 "Source A" if index % 2 == 0 else "Source B",
                 f"https://example.com/{index}", now - 3600 - index, now, "{}"),
            )
            con.execute(
                "INSERT INTO article_tags VALUES (?, ?)",
                (article_id, 2 if index % 2 == 0 else 3),
            )
        con.execute(
            "INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("old", "Old article", "Source A", "https://example.com/old", now - 172800, now, "{}"),
        )
        con.commit()
        con.close()
        self.original_store = webapp_viewer.APP_STORE
        self.original_config = webapp_viewer.APP_CFG
        webapp_viewer.APP_STORE = self.store
        webapp_viewer.APP_CFG = {
            "feeds": [
                {"name": "Source A", "url": "https://example.com/a.xml"},
                {"name": "Source B", "url": "https://example.com/b.xml"},
                {
                    "name": "Configured without recent articles",
                    "url": "https://example.com/empty.xml",
                },
            ]
        }
        self.addCleanup(setattr, webapp_viewer, "APP_STORE", self.original_store)
        self.addCleanup(setattr, webapp_viewer, "APP_CFG", self.original_config)
        webapp_viewer.app.config.update(TESTING=True)
        self.client = webapp_viewer.app.test_client()

    def test_default_shows_only_last_day_and_paginates_in_database(self):
        response = self.client.get("/articles")
        self.assertEqual(200, response.status_code)
        html = response.get_data(as_text=True)
        self.assertEqual(25, html.count('class="article-row"'))
        self.assertIn("Visar 1–25 av 27 artiklar", html)
        self.assertNotIn("Old article", html)
        self.assertIn('class="nav-link active" href="/articles">Alla</a>', html)
        self.assertNotIn("nav-tabs mb-3 flex-nowrap", html)

        response = self.client.get("/articles?range=custom&per_page=25&page=2")
        html = response.get_data(as_text=True)
        self.assertEqual(3, html.count('class="article-row"'))
        self.assertIn("Visar 26–28 av 28 artiklar", html)
        self.assertIn("Old article", html)

    def test_article_badges_use_their_category_colors(self):
        response = self.client.get("/articles")
        html = response.get_data(as_text=True)

        self.assertIn(
            'style="background-color: #118855; color: #ffffff;">Sweden</span>', html
        )
        self.assertIn(
            'style="background-color: #dc3545; color: #fff;">Malware</span>', html
        )

    def test_source_filter_uses_all_configured_feeds(self):
        response = self.client.get("/articles")
        html = response.get_data(as_text=True)

        self.assertIn('value="Source A"', html)
        self.assertIn('value="Source B"', html)
        self.assertIn('value="Configured without recent articles"', html)

    def test_date_source_category_and_parent_tag_filters_apply_before_paging(self):
        response = self.client.get(
            "/articles?range=custom&start=2020-01-01T00:00&end=2030-01-01T00:00"
            "&source=Source+A&category=LOCATION&tag=1&per_page=25"
        )
        self.assertEqual(200, response.status_code)
        html = response.get_data(as_text=True)
        self.assertIn("Visar 1–14 av 14 artiklar", html)
        self.assertIn("Recent article 0", html)
        self.assertNotIn('>Recent article 1</div>', html)
        self.assertNotIn("Old article", html)

    def test_tag_options_require_category_and_only_include_selected_categories(self):
        empty = self.client.get("/api/v1/article-filter-tags")
        self.assertEqual([], empty.json["tags"])
        selected = self.client.get("/api/v1/article-filter-tags?category=LOCATION")
        self.assertEqual(["Europe", "Sweden"], [tag["name"] for tag in selected.json["tags"]])
        self.assertNotIn("Malware", selected.get_data(as_text=True))

    def test_per_page_dropdown_and_page_links_preserve_filters(self):
        response = self.client.get(
            "/articles?range=custom&source=Source+A&source=Source+B&per_page=25&page=1"
        )
        html = response.get_data(as_text=True)
        self.assertIn('id="per-page"', html)
        self.assertIn('name="start"', html)
        self.assertIn('name="end"', html)
        self.assertIn("Source+A", html)
        self.assertIn("Source+B", html)
        self.assertIn("page=2", html)

    def test_invalid_tag_outside_selected_category_is_ignored(self):
        response = self.client.get(
            "/articles?range=custom&category=THREAT&tag=1"
        )
        self.assertEqual(200, response.status_code)
        self.assertIn("av 13 artiklar", response.get_data(as_text=True))


class _MongoCursor:
    def __init__(self, rows):
        self.rows = rows
        self.offset = 0
        self.maximum = None

    def sort(self, *args):
        return self

    def skip(self, offset):
        self.offset = offset
        return self

    def limit(self, maximum):
        self.maximum = maximum
        return self

    def __iter__(self):
        end = None if self.maximum is None else self.offset + self.maximum
        return iter(self.rows[self.offset:end])


class MongoArticlePageTests(unittest.TestCase):
    def test_mongo_page_projects_only_visible_rows_and_bulk_loads_their_tags(self):
        rows = [
            {"_id": f"article-{index}", "title": f"Article {index}",
             "source": "Source", "published_ts": 100 - index}
            for index in range(30)
        ]

        class Articles:
            def count_documents(self, query):
                self.count_query = query
                return len(rows)

            def find(self, query, projection):
                self.find_query = query
                self.projection = projection
                self.cursor = _MongoCursor(rows)
                return self.cursor

        class ArticleTags:
            def find(self, query, projection):
                self.query = query
                return []

        articles = Articles()
        article_tags = ArticleTags()
        store = SimpleNamespace(db=SimpleNamespace(
            articles=articles, article_tags=article_tags, tags=object()
        ))

        result = list_article_page(
            store, start_ts=10, end_ts=100, sources=[], categories=[],
            selected_tags=[], page=2, per_page=25,
        )

        self.assertEqual(["article-25", "article-26", "article-27", "article-28", "article-29"],
                         [article["id"] for article in result["articles"]])
        self.assertEqual({"_mongo_sort_ts": {"$gte": 10, "$lte": 100}}, articles.find_query)
        self.assertNotIn("doc_json", articles.projection)
        self.assertEqual(25, articles.cursor.offset)
        self.assertEqual(25, articles.cursor.maximum)
        self.assertEqual(
            [f"article-{index}" for index in range(25, 30)],
            article_tags.query["article_id"]["$in"],
        )

    def test_mongo_category_and_tag_filters_run_before_page_limit(self):
        class Articles:
            def __init__(self):
                self.pipelines = []

            def aggregate(self, pipeline):
                self.pipelines.append(pipeline)
                if "$count" in pipeline[-1]:
                    return [{"total": 1}]
                return [{"_id": "article-1", "title": "Filtered", "published_ts": 50}]

        class Tags:
            def find(self, query, projection):
                return _MongoCursor([
                    {"_id": 7, "name": "Selected", "category": "LOCATION"}
                ])

        class ArticleTags:
            def find(self, query, projection):
                return [{"article_id": "article-1", "tag_id": 7}]

        articles = Articles()
        store = SimpleNamespace(db=SimpleNamespace(
            articles=articles, article_tags=ArticleTags(), tags=Tags()
        ))
        result = list_article_page(
            store, start_ts=10, end_ts=100, sources=["Source"],
            categories=["LOCATION"], selected_tags=[7], page=1, per_page=25,
        )

        self.assertEqual(1, result["total"])
        self.assertEqual("Selected", result["articles"][0]["tags"][0]["name"])
        page_pipeline = articles.pipelines[1]
        self.assertEqual({"_mongo_sort_ts": {"$gte": 10, "$lte": 100},
                          "source": {"$in": ["Source"]}}, page_pipeline[0]["$match"])
        self.assertEqual([7], page_pipeline[1]["$lookup"]["pipeline"][0]["$match"]["tag_id"]["$in"])
        self.assertEqual({"$limit": 25}, page_pipeline[-2])


if __name__ == "__main__":
    unittest.main()
