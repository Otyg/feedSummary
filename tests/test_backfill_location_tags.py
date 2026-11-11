import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backfill_location_tags import _parse_response, apply_location_report, backfill_location_tags, main


class FakeStore:
    def __init__(self):
        self.articles = [
            {"id": "cve", "title": "CVE-2026-12345 affects Sweden", "content": "Sweden"},
            {"id": "tagged", "title": "Stockholm news", "content": "Stockholm"},
            {"id": "new", "title": "Incident in Stockholm", "content": "The incident affected Stockholm."},
            {"id": "none", "title": "Software update", "content": "New release available."},
        ]
        self.tags = [{"id": 1, "name": "sweden", "category": "LOCATION"}]
        self.assigned = {"tagged": [self.tags[0]]}
        self.added = []

    def list_articles(self, limit=2000):
        return self.articles[:limit]

    def get_article(self, article_id):
        return next((article.copy() for article in self.articles if article["id"] == article_id), None)

    def get_all_tags(self):
        return self.tags[:]

    def get_article_tags(self, article_id):
        return self.assigned.get(article_id, [])[:]

    def get_tag_by_name(self, name):
        return next((t for t in self.tags if t["name"].casefold() == name.casefold()), None)

    def add_tag(self, name, category="GENERAL"):
        tag = {"id": len(self.tags) + 1, "name": name.lower(), "category": category}
        self.tags.append(tag)
        return tag["id"]

    def add_tag_to_article(self, article_id, tag_id):
        self.assigned.setdefault(article_id, []).append(next(t for t in self.tags if t["id"] == tag_id))
        self.added.append((article_id, tag_id))
        return True


class FakeLLM:
    def __init__(self):
        self.calls = 0

    async def chat(self, messages, *, temperature=0.2):
        self.calls += 1
        if "Incident in Stockholm" in messages[0]["content"]:
            return json.dumps({"locations": [{"name": "Sweden", "evidence": "Stockholm", "confidence": "high"}]})
        return '{"locations": []}'


class LocationBackfillTests(unittest.IsolatedAsyncioTestCase):
    async def test_preview_filters_cves_and_tagged_articles_without_writing(self):
        store, llm = FakeStore(), FakeLLM()
        report = await backfill_location_tags(store, llm)
        self.assertEqual(["new", "none"], [row["article_id"] for row in report["articles"]])
        self.assertEqual(2, llm.calls)
        self.assertEqual(0, report["added"])
        self.assertEqual([], store.added)
        self.assertEqual("Sweden", report["articles"][0]["locations"][0]["name"])

    async def test_apply_reuses_existing_location_tag(self):
        store = FakeStore()
        report = await backfill_location_tags(store, FakeLLM(), apply=True)
        self.assertEqual([("new", 1)], store.added)
        self.assertEqual(1, report["added"])
        self.assertEqual(0, report["created_tags"])

    async def test_apply_creates_missing_location_tag(self):
        store = FakeStore()

        class RegionLLM:
            async def chat(self, messages, *, temperature=0.2):
                return '{"locations":[{"name":"Scandinavia","evidence":"Stockholm","confidence":"medium"}]}'

        report = await backfill_location_tags(store, RegionLLM(), apply=True, limit=1)
        self.assertEqual(2, report["eligible"])
        self.assertEqual(1, report["assessed"])
        self.assertEqual(1, report["created_tags"])
        self.assertEqual("LOCATION", store.get_tag_by_name("Scandinavia")["category"])
        self.assertEqual([("new", 2)], store.added)

    async def test_saved_preview_applies_without_another_llm_call(self):
        store, llm = FakeStore(), FakeLLM()
        preview = await backfill_location_tags(store, llm)
        result = apply_location_report(store, preview)
        self.assertEqual("apply_report", result["mode"])
        self.assertEqual(1, result["added"])
        self.assertEqual([("new", 1)], store.added)
        self.assertEqual(2, llm.calls)
        self.assertEqual("preview", preview["mode"])
        self.assertNotIn("added", preview["articles"][0]["locations"][0])

    async def test_saved_preview_skips_changed_or_already_tagged_article(self):
        store = FakeStore()
        preview = await backfill_location_tags(store, FakeLLM(), limit=1)
        store.articles[2]["content"] = "Different article text"
        result = apply_location_report(store, preview)
        self.assertEqual(0, result["added"])
        self.assertIn("ändrats", result["articles"][0]["skipped"])

        store.articles[2]["content"] = "The incident affected Stockholm."
        store.assigned["new"] = [store.tags[0]]
        result = apply_location_report(store, preview)
        self.assertEqual(0, result["added"])
        self.assertIn("redan", result["articles"][0]["skipped"])

    async def test_older_preview_without_hash_can_be_applied(self):
        store = FakeStore()
        preview = await backfill_location_tags(store, FakeLLM(), limit=1)
        preview.pop("max_chars")
        preview["articles"][0].pop("article_sha256")
        result = apply_location_report(store, preview)
        self.assertEqual(1, result["added"])

    async def test_malformed_report_rejected_before_writes(self):
        store = FakeStore()
        preview = await backfill_location_tags(store, FakeLLM(), limit=1)
        preview["articles"].append(dict(preview["articles"][0]))
        with self.assertRaises(ValueError):
            apply_location_report(store, preview)
        self.assertEqual([], store.added)

    async def test_cli_input_does_not_create_llm_client(self):
        store = FakeStore()
        preview = await backfill_location_tags(store, FakeLLM(), limit=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "preview.json"
            output_path = Path(temp_dir) / "applied.json"
            input_path.write_text(json.dumps(preview), encoding="utf-8")
            with (
                patch("backfill_location_tags._load_config", return_value={"store": {}}),
                patch("backfill_location_tags.create_store", return_value=store),
                patch("backfill_location_tags.create_llm_client", side_effect=AssertionError("LLM skapades")),
            ):
                exit_code = main(["--apply", "--input", str(input_path), "--output", str(output_path)])
            self.assertEqual(0, exit_code)
            self.assertEqual(1, json.loads(output_path.read_text(encoding="utf-8"))["added"])

    def test_accepts_inferred_country_without_verbatim_place_name(self):
        locations = _parse_response(
            '{"locations":[{"name":"Sweden","evidence":"Myndighetens nationella ansvar pekar på Sverige.","confidence":"low"}]}'
        )
        self.assertEqual("Sweden", locations[0]["name"])
        self.assertEqual("low", locations[0]["confidence"])

    def test_rejects_unexplained_guess(self):
        with self.assertRaises(ValueError):
            _parse_response('{"locations":[{"name":"Sweden","evidence":"","confidence":"low"}]}')


if __name__ == "__main__":
    unittest.main()
