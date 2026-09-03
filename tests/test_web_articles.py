import unittest
from datetime import datetime

from web_viewer import webapp_viewer


class _FilteredArticleStore:
    def __init__(self, articles):
        self.articles = articles
        self.requested_sources = None

    def list_articles_by_filter(self, *, sources, since_ts, until_ts, limit):
        self.requested_sources = sources
        rows = [
            article
            for article in self.articles
            if since_ts <= article["published_ts"] <= until_ts
            and (not sources or article["source"] in sources)
        ]
        return rows[:limit]

    def list_articles(self, limit=2000):
        return self.articles[:limit]


class _HierarchicalTagStore:
    def __init__(self):
        self.parents = {
            3: [{"id": 2, "name": "Sweden", "category": "LOCATION"}],
            2: [{"id": 1, "name": "Northern Europe", "category": "LOCATION"}],
        }

    def get_tag_relations(self, tag_id):
        return {"parents": self.parents.get(tag_id, []), "children": []}


class ArticleDateTabTests(unittest.TestCase):
    def setUp(self):
        self.original_cfg = webapp_viewer.APP_CFG
        self.original_store = webapp_viewer.APP_STORE

    def tearDown(self):
        webapp_viewer.APP_CFG = self.original_cfg
        webapp_viewer.APP_STORE = self.original_store

    def test_day_list_is_not_limited_to_currently_configured_sources(self):
        published_ts = int(datetime(2026, 8, 30, 12).timestamp())
        articles = [
            {
                "id": f"configured-{index}",
                "title": "Configured source",
                "source": "Configured",
                "published_ts": published_ts,
            }
            for index in range(28)
        ] + [
            {
                "id": f"removed-{index}",
                "title": "Removed source",
                "source": "Removed",
                "published_ts": published_ts,
            }
            for index in range(45)
        ]
        store = _FilteredArticleStore(articles)
        webapp_viewer.APP_CFG = {"store": {"provider": "mongodb"}}

        rows = webapp_viewer._list_articles_for_day_fast(
            store, date_ymd="2026-08-30", limit=2000
        )

        self.assertEqual([], store.requested_sources)
        self.assertEqual(73, len(rows))

    def test_selected_day_uses_its_tab_count_as_total(self):
        published_ts = int(datetime(2026, 8, 30, 12).timestamp())
        store = _FilteredArticleStore(
            [
                {
                    "id": f"article-{index}",
                    "title": "Article",
                    "source": "Source",
                    "published_ts": published_ts,
                }
                for index in range(73)
            ]
        )
        store.get_article_tags = lambda _article_id: []
        webapp_viewer.APP_CFG = {"store": {"provider": "mongodb"}}
        webapp_viewer.APP_STORE = store
        webapp_viewer.app.config.update(TESTING=True)

        response = webapp_viewer.app.test_client().get(
            "/articles?date=2026-08-30&limit=28"
        )

        self.assertEqual(200, response.status_code)
        html = response.get_data(as_text=True)
        self.assertEqual(28, html.count('class="article-row"'))
        self.assertIn('Artiklar: <span id="article-count">28</span>', html)
        self.assertIn("av 73", html)

        all_response = webapp_viewer.app.test_client().get("/articles?limit=28")
        all_html = all_response.get_data(as_text=True)
        self.assertIn("Alla", all_html)
        self.assertIn(
            '<span class="badge text-bg-light ms-1">73</span>', all_html
        )

    def test_article_page_exposes_source_filter(self):
        published_ts = int(datetime(2026, 8, 30, 12).timestamp())
        store = _FilteredArticleStore(
            [
                {
                    "id": "article-a",
                    "title": "Article A",
                    "source": "Source A",
                    "published_ts": published_ts,
                },
                {
                    "id": "article-b",
                    "title": "Article B",
                    "source": "Source B",
                    "published_ts": published_ts,
                },
            ]
        )
        store.get_article_tags = lambda _article_id: []
        webapp_viewer.APP_CFG = {"store": {"provider": "mongodb"}}
        webapp_viewer.APP_STORE = store
        webapp_viewer.app.config.update(TESTING=True)

        response = webapp_viewer.app.test_client().get(
            "/articles?date=2026-08-30"
        )

        self.assertEqual(200, response.status_code)
        html = response.get_data(as_text=True)
        self.assertIn('id="source-filter-container"', html)
        self.assertIn('id="selected-sources-count"', html)
        self.assertIn('id="clear-source-filters"', html)
        self.assertIn('source: "Source A"', html)
        self.assertIn('source: "Source B"', html)
        self.assertIn("selectedSources.has(article.source)", html)

    def test_tag_filter_tree_includes_all_ancestors_and_descendants(self):
        articles = [
            {
                "id": "stockholm-article",
                "tags": [
                    {"id": 3, "name": "Stockholm", "category": "LOCATION"}
                ],
            }
        ]

        tags, descendants = webapp_viewer._article_tag_filter_tree(
            _HierarchicalTagStore(), articles
        )

        self.assertEqual(
            ["Northern Europe", "Stockholm", "Sweden"],
            [tag["name"] for tag in tags],
        )
        self.assertEqual(["1", "2", "3"], descendants["1"])
        self.assertEqual(["2", "3"], descendants["2"])
        self.assertEqual(["3"], descendants["3"])

    def test_article_page_uses_descendant_ids_for_tag_filtering(self):
        published_ts = int(datetime(2026, 8, 30, 12).timestamp())
        store = _FilteredArticleStore(
            [
                {
                    "id": "sweden-article",
                    "title": "Sweden",
                    "source": "Source",
                    "published_ts": published_ts,
                }
            ]
        )
        store.get_article_tags = lambda _article_id: [
            {"id": 2, "name": "Sweden", "category": "LOCATION"}
        ]
        store.get_tag_relations = lambda tag_id: {
            "parents": (
                [{"id": 1, "name": "Northern Europe", "category": "LOCATION"}]
                if tag_id == 2
                else []
            ),
            "children": [],
        }
        webapp_viewer.APP_CFG = {"store": {"provider": "mongodb"}}
        webapp_viewer.APP_STORE = store
        webapp_viewer.app.config.update(TESTING=True)

        response = webapp_viewer.app.test_client().get("/articles")

        self.assertEqual(200, response.status_code)
        html = response.get_data(as_text=True)
        self.assertIn("Northern Europe", html)
        self.assertIn('"1": ["1", "2"]', html)
        self.assertIn("tagDescendants[tagId] || [tagId]", html)
        self.assertIn("expandedSelectedTagIds.has(String(tag.id))", html)


if __name__ == "__main__":
    unittest.main()
