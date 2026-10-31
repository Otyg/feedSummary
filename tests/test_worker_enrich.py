import unittest
from unittest.mock import AsyncMock, patch

from web_viewer import feedsum_worker


class _Store:
    def create_job(self):
        return 17


class WorkerEnrichTests(unittest.IsolatedAsyncioTestCase):
    def test_entry_to_overrides_preserves_boolean_enrich(self):
        overrides = feedsum_worker._entry_to_overrides(
            {"lookback": "1d", "enrich": True}
        )

        self.assertTrue(overrides["enrich"])

    async def test_tag_based_summary_receives_enrich(self):
        run_summary = AsyncMock(return_value="summary-1")
        with (
            patch.object(feedsum_worker, "run_tag_based_summary", run_summary),
            patch.object(feedsum_worker, "_annotate_summary_with_schedule_name"),
            patch.object(feedsum_worker, "_tag_created_summary", AsyncMock()),
        ):
            result = await feedsum_worker._run_regular_entry(
                "config.yaml",
                {},
                _Store(),
                "daily",
                {
                    "type": "tag_based_summary",
                    "tags": ["incident"],
                    "lookback": "1d",
                    "enrich": True,
                },
            )

        self.assertEqual("summary-1", result)
        self.assertTrue(run_summary.await_args.kwargs["enrich"])


if __name__ == "__main__":
    unittest.main()
