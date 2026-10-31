import os
import threading
import unittest
from unittest.mock import patch

from uicommon.tagging_ml_retrain import CoalescingRetrainer

_previous_auto_init = os.environ.get("FEEDSUMMARY_SKIP_VIEWER_AUTO_INIT")
os.environ["FEEDSUMMARY_SKIP_VIEWER_AUTO_INIT"] = "1"
try:
    from web_viewer import webapp_viewer
finally:
    if _previous_auto_init is None:
        os.environ.pop("FEEDSUMMARY_SKIP_VIEWER_AUTO_INIT", None)
    else:
        os.environ["FEEDSUMMARY_SKIP_VIEWER_AUTO_INIT"] = _previous_auto_init


class CoalescingRetrainerTests(unittest.TestCase):
    def test_requests_during_training_produce_one_follow_up_pass(self):
        first_started = threading.Event()
        release_first = threading.Event()
        second_finished = threading.Event()
        calls = []

        def retrain():
            calls.append(len(calls) + 1)
            if len(calls) == 1:
                first_started.set()
                release_first.wait(2)
            else:
                second_finished.set()

        runner = CoalescingRetrainer(retrain, debounce_seconds=0)
        self.assertEqual("started", runner.request("first edit"))
        self.assertTrue(first_started.wait(2))
        self.assertEqual("queued", runner.request("second edit"))
        self.assertEqual("queued", runner.request("third edit"))
        release_first.set()

        self.assertTrue(second_finished.wait(2))
        self.assertEqual([1, 2], calls)


class FakeArticleTagStore:
    def __init__(self):
        self.tags = set()

    def add_tag_to_article(self, article_id, tag_id):
        key = (article_id, tag_id)
        if key in self.tags:
            return False
        self.tags.add(key)
        return True

    def remove_article_tag(self, article_id, tag_id):
        key = (article_id, tag_id)
        if key not in self.tags:
            return False
        self.tags.remove(key)
        return True


class ManualTagRetrainingApiTests(unittest.TestCase):
    def setUp(self):
        self.previous_store = webapp_viewer.APP_STORE
        self.previous_config = webapp_viewer.APP_CFG
        webapp_viewer.APP_STORE = FakeArticleTagStore()
        webapp_viewer.APP_CFG = {
            "tagging": {
                "ml": {
                    "enabled": True,
                    "retrain_on_manual_tag_change": True,
                }
            }
        }
        webapp_viewer.app.config.update(TESTING=True)
        self.client = webapp_viewer.app.test_client()

    def tearDown(self):
        webapp_viewer.APP_STORE = self.previous_store
        webapp_viewer.APP_CFG = self.previous_config

    def test_successful_add_and_remove_schedule_retraining(self):
        with patch.object(
            webapp_viewer._TAGGING_ML_RETRAINER,
            "request",
            side_effect=["started", "queued"],
        ) as request_retrain:
            added = self.client.post(
                "/api/v1/article/article-1/tags",
                json={"tag_id": 17},
            )
            removed = self.client.delete("/api/v1/article/article-1/tags/17")

        self.assertEqual(200, added.status_code)
        self.assertEqual("started", added.get_json()["ml_retrain"])
        self.assertEqual(200, removed.status_code)
        self.assertEqual("queued", removed.get_json()["ml_retrain"])
        self.assertEqual(2, request_retrain.call_count)

    def test_failed_mutations_do_not_schedule_retraining(self):
        with patch.object(
            webapp_viewer._TAGGING_ML_RETRAINER,
            "request",
            return_value="started",
        ) as request_retrain:
            missing = self.client.delete("/api/v1/article/article-1/tags/17")
            self.client.post(
                "/api/v1/article/article-1/tags",
                json={"tag_id": 17},
            )
            duplicate = self.client.post(
                "/api/v1/article/article-1/tags",
                json={"tag_id": 17},
            )

        self.assertEqual(404, missing.status_code)
        self.assertEqual(409, duplicate.status_code)
        self.assertEqual(1, request_retrain.call_count)

    def test_feature_flag_can_disable_manual_retraining(self):
        webapp_viewer.APP_CFG["tagging"]["ml"][
            "retrain_on_manual_tag_change"
        ] = False
        with patch.object(
            webapp_viewer._TAGGING_ML_RETRAINER,
            "request",
        ) as request_retrain:
            response = self.client.post(
                "/api/v1/article/article-1/tags",
                json={"tag_id": 17},
            )

        self.assertEqual(200, response.status_code)
        self.assertEqual("disabled", response.get_json()["ml_retrain"])
        request_retrain.assert_not_called()

    def test_queue_failure_does_not_turn_successful_tag_change_into_http_error(self):
        with patch.object(
            webapp_viewer._TAGGING_ML_RETRAINER,
            "request",
            side_effect=RuntimeError("thread unavailable"),
        ):
            response = self.client.post(
                "/api/v1/article/article-1/tags",
                json={"tag_id": 17},
            )

        self.assertEqual(200, response.status_code)
        self.assertEqual("queue_failed", response.get_json()["ml_retrain"])


if __name__ == "__main__":
    unittest.main()
