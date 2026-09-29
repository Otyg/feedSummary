import os
import unittest
from unittest.mock import AsyncMock, patch

_previous_auto_init = os.environ.get("FEEDSUMMARY_SKIP_VIEWER_AUTO_INIT")
os.environ["FEEDSUMMARY_SKIP_VIEWER_AUTO_INIT"] = "1"
try:
    from web_viewer import webapp_viewer
finally:
    if _previous_auto_init is None:
        os.environ.pop("FEEDSUMMARY_SKIP_VIEWER_AUTO_INIT", None)
    else:
        os.environ["FEEDSUMMARY_SKIP_VIEWER_AUTO_INIT"] = _previous_auto_init


class ArticleReclassificationLimitTests(unittest.TestCase):
    def test_reclassification_accepts_twenty_and_caps_larger_requests(self):
        previous_store = webapp_viewer.APP_STORE
        previous_config = webapp_viewer.APP_CFG
        store = type("ArticleStore", (), {
            "get_article": lambda _self, _id: {"id": "article-1", "title": "Article"},
            "get_article_tags": lambda _self, _id: [],
        })()
        webapp_viewer.APP_STORE = store
        webapp_viewer.APP_CFG = {"store": {"provider": "sqlite"}}
        webapp_viewer.app.config.update(TESTING=True)
        try:
            with (
                patch.object(webapp_viewer, "create_llm_client", return_value=object()),
                patch.object(
                    webapp_viewer.TagManager,
                    "reclassify_article_with_existing_tags",
                    new_callable=AsyncMock,
                    return_value=[],
                ) as classify,
            ):
                client = webapp_viewer.app.test_client()
                for requested in (20, 25):
                    response = client.post(
                        "/api/v1/article/article-1/reclassify",
                        json={"max_tags": requested},
                    )
                    self.assertEqual(200, response.status_code)
                    self.assertEqual(20, classify.await_args.kwargs["max_tags"])
        finally:
            webapp_viewer.APP_STORE = previous_store
            webapp_viewer.APP_CFG = previous_config


if __name__ == "__main__":
    unittest.main()
