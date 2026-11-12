# LICENSE HEADER MANAGED BY add-license-header
#
# BSD 3-Clause License
#
# Copyright (c) 2026, Martin Vesterlund
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import unittest
from xml.etree import ElementTree

from web_viewer import webapp_viewer as viewer
from web_viewer import webapp_viewer_readonly as readonly


class _ReadStore:
    def __init__(self):
        self.article = {
            "id": "article-1",
            "title": "Read-only article",
            "source": "Example",
            "url": "https://example.invalid/article",
            "published_ts": 1,
            "text": "Article body",
        }

    def get_article(self, article_id):
        return dict(self.article) if article_id == "article-1" else None

    def get_article_tags(self, article_id):
        if article_id != "article-1":
            return []
        return [{"id": 7, "name": "Example tag", "category": "GENERAL"}]

    def get_all_tags(self):
        return [
            {
                "id": 7,
                "name": "Example tag",
                "category": "GENERAL",
                "description": "A tag",
            }
        ]

    def get_tag_relations(self, tag_id):
        return {"parents": [], "children": []}

    def get_all_categories(self):
        return [
            {
                "id": 3,
                "name": "GENERAL",
                "label": "Allmän",
                "bg_color": "bg-secondary",
                "text_color": "text-light",
            }
        ]

    def list_articles(self, limit=2000):
        return [dict(self.article)]

    def list_summary_docs(self):
        return [
            {
                "id": "summary-1",
                "title": "Daily summary",
                "created": 1_700_000_000,
                "summary": "**Important** summary.",
            }
        ]

    def get_summary_doc(self, summary_id):
        rows = self.list_summary_docs()
        return dict(rows[0]) if summary_id == "summary-1" else None









class ReadOnlyViewerTests(unittest.TestCase):
    def setUp(self):
        self.original_store = viewer.APP_STORE
        self.original_cfg = viewer.APP_CFG
        viewer.APP_STORE = _ReadStore()
        viewer.APP_CFG = {}
        readonly.app.config.update(TESTING=True)
        self.client = readonly.app.test_client()

    def tearDown(self):
        viewer.APP_STORE = self.original_store
        viewer.APP_CFG = self.original_cfg

    def test_all_registered_write_routes_return_401(self):
        paths = {
            "/api/v1/tags/categories": "PUT",
            "/api/v1/tags": "POST",
            "/api/v1/article/article-1/tags": "POST",
            "/api/v1/article/article-1/tags/7": "DELETE",
            "/api/v1/article/article-1/reclassify": "POST",
            "/api/v1/tags/7": "PUT",
            "/api/v1/tags/7?delete=1": "DELETE",
            "/api/v1/tags/7/relations": "PUT",
            "/api/v1/categories": "POST",
            "/api/v1/categories/7": "PUT",
            "/api/v1/categories/7?delete=1": "DELETE",
        }

        for path, method in paths.items():
            with self.subTest(path=path, method=method):
                response = self.client.open(path, method=method, json={})
                self.assertEqual(401, response.status_code)
                self.assertEqual("read_only", response.get_json()["error"])

    def test_unknown_unsafe_method_is_also_rejected(self):
        response = self.client.patch("/api/v1/future-write", json={})

        self.assertEqual(401, response.status_code)

    def test_article_filter_tags_is_available_read_only(self):
        response = self.client.get("/api/v1/article-filter-tags?category=GENERAL")

        self.assertEqual(200, response.status_code)
        self.assertEqual(["Example tag"], [tag["name"] for tag in response.json["tags"]])

    def test_worker_routes_and_status_are_absent(self):
        for path, method in (
            ("/status", "GET"),
            ("/api/v1/job/resume", "POST"),
            ("/api/v1/job/resume/resume-1", "GET"),
            ("/api/v1/schedule/trigger", "POST"),
            ("/api/v1/schedule/trigger/trigger-1", "GET"),
        ):
            with self.subTest(path=path, method=method):
                response = self.client.open(path, method=method, json={})
                self.assertEqual(404, response.status_code)

        endpoints = {rule.endpoint for rule in readonly.app.url_map.iter_rules()}
        self.assertNotIn("status", endpoints)
        self.assertNotIn("api_job_resume", endpoints)
        self.assertNotIn("api_job_resume_status", endpoints)
        self.assertNotIn("api_schedule_trigger", endpoints)
        self.assertNotIn("api_schedule_trigger_status", endpoints)

    def test_article_is_readable_without_write_controls(self):
        response = self.client.get("/article/article-1")

        self.assertEqual(200, response.status_code)
        html = response.get_data(as_text=True)
        self.assertIn("Read-only article", html)
        self.assertIn("Example tag", html)
        self.assertIn("Läsbehörighet", html)
        self.assertNotIn("Lägg till befintlig tagg", html)
        self.assertNotIn("Föreslå taggar från befintliga", html)
        self.assertNotIn("Ta bort tagg", html)

    def test_management_pages_are_rendered_as_read_only_lists(self):
        tags_html = self.client.get("/tags").get_data(as_text=True)
        categories_html = self.client.get("/categories").get_data(as_text=True)
        editor_html = self.client.get("/tag-categories").get_data(as_text=True)

        self.assertIn("Example tag", tags_html)
        self.assertNotIn("Ny tagg", tags_html)
        self.assertNotIn(">Redigera<", tags_html)
        self.assertNotIn(">Radera<", tags_html)

        self.assertIn("Allmän", categories_html)
        self.assertNotIn("Ny kategori", categories_html)
        self.assertNotIn(">Redigera<", categories_html)
        self.assertNotIn(">Radera<", categories_html)

        self.assertIn("Example tag", editor_html)
        self.assertIn("disabled", editor_html)
        self.assertNotIn("Spara ändringar", editor_html)

    def test_summary_and_article_rss_feeds_are_readable(self):
        summary_response = self.client.get("/rss/summaries.xml")
        article_response = self.client.get("/rss/articles.xml")

        self.assertEqual(200, summary_response.status_code)
        self.assertEqual(200, article_response.status_code)
        self.assertTrue(
            summary_response.content_type.startswith("application/rss+xml")
        )
        self.assertTrue(article_response.content_type.startswith("application/rss+xml"))

        summary_root = ElementTree.fromstring(summary_response.data)
        article_root = ElementTree.fromstring(article_response.data)
        self.assertEqual("Daily summary", summary_root.findtext("channel/item/title"))
        self.assertIn(
            "<strong>Important</strong>",
            summary_root.findtext("channel/item/description"),
        )
        self.assertEqual(
            "Read-only article", article_root.findtext("channel/item/title")
        )
        self.assertEqual(
            "https://example.invalid/article",
            article_root.findtext("channel/item/link"),
        )

        for endpoint in ("rss_summaries", "rss_articles"):
            with self.subTest(endpoint=endpoint):
                self.assertIn(endpoint, viewer.app.view_functions)
                self.assertIn(endpoint, readonly.app.view_functions)






if __name__ == "__main__":
    unittest.main()
