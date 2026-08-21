import json
import unittest

from audit_article_tags import (
    _articles_for_tag_names,
    _parse_assessments,
    apply_audit_report,
    audit_tags,
)


class FakeStore:
    def __init__(self):
        self.tags = {
            "ray": {"id": 1, "name": "ray", "category": "GENERAL"},
            "comfast": {"id": 2, "name": "comfast", "category": "DOMAIN_ENTITY"},
            "orphan": {"id": 3, "name": "orphan", "category": "GENERAL"},
        }
        self.articles = [
            {
                "id": "a1",
                "title": "Password spraying campaign",
                "content": "Attackers sprayed passwords.",
            },
            {
                "id": "a2",
                "title": "Network hardware",
                "content": "Ray investigated Comfast devices.",
            },
        ]
        self.article_tags = {
            "a1": [self.tags["ray"]],
            "a2": [self.tags["ray"], self.tags["comfast"]],
        }
        self.removed = []
        self.deleted_tags = []
        self.article_queries = []

    def get_tag_by_name(self, name):
        tag = self.tags.get(name.strip().lower())
        return tag.copy() if tag else None

    def get_all_tags(self):
        return [tag.copy() for tag in self.tags.values()]

    def get_articles_by_tags(self, tag_names, match_mode="any"):
        self.article_queries.append(list(tag_names))
        requested = {name.lower() for name in tag_names}
        return [
            article.copy()
            for article in self.articles
            if any(
                tag["name"].lower() in requested
                for tag in self.article_tags[article["id"]]
            )
        ]

    def get_article_tags(self, article_id):
        return [tag.copy() for tag in self.article_tags[article_id]]

    def remove_article_tag(self, article_id, tag_id):
        before = len(self.article_tags[article_id])
        self.article_tags[article_id] = [
            tag for tag in self.article_tags[article_id] if tag["id"] != tag_id
        ]
        removed = len(self.article_tags[article_id]) < before
        if removed:
            self.removed.append((article_id, tag_id))
        return removed

    def delete_tag(self, tag_id):
        match = next(
            (name for name, tag in self.tags.items() if tag["id"] == tag_id),
            None,
        )
        if match is None:
            return False
        del self.tags[match]
        self.deleted_tags.append(tag_id)
        return True


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def chat(self, messages, *, temperature=0.2):
        self.calls.append((messages, temperature))
        return self.responses.pop(0)


def response(*assessments):
    return json.dumps({"assessments": list(assessments)})


class AssessmentParsingTests(unittest.TestCase):
    def test_parses_fenced_json_and_preserves_expected_order(self):
        tags = [
            {"id": 1, "name": "ray", "category": "GENERAL"},
            {"id": 2, "name": "comfast", "category": "DOMAIN_ENTITY"},
        ]
        raw = "```json\n" + response(
            {"tag": "comfast", "relevant": True, "confidence": "high", "reasoning": "Nämns."},
            {"tag": "ray", "relevant": False, "confidence": "medium", "reasoning": "Delsträng."},
        ) + "\n```"

        parsed = _parse_assessments(raw, tags)

        self.assertEqual(["ray", "comfast"], [item["tag"] for item in parsed])
        self.assertFalse(parsed[0]["relevant"])
        self.assertTrue(parsed[1]["relevant"])

    def test_rejects_missing_tag_assessment(self):
        tags = [
            {"id": 1, "name": "ray", "category": "GENERAL"},
            {"id": 2, "name": "comfast", "category": "DOMAIN_ENTITY"},
        ]

        with self.assertRaisesRegex(ValueError, "saknar bedömning"):
            _parse_assessments(
                response(
                    {"tag": "ray", "relevant": False, "confidence": "high", "reasoning": "Fel."}
                ),
                tags,
            )

    def test_article_queries_are_chunked_for_large_tag_sets(self):
        store = FakeStore()

        articles = _articles_for_tag_names(
            store,
            ["ray", "comfast", *[f"tag-{index}" for index in range(399)]],
        )

        self.assertEqual(2, len(store.article_queries))
        self.assertEqual([400, 1], [len(query) for query in store.article_queries])
        self.assertEqual(["a1", "a2"], sorted(article["id"] for article in articles))


class TagAuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_audit_includes_synonyms_and_records_synonym_match(self):
        store = FakeStore()
        store.tags["europe"] = {
            "id": 4,
            "name": "Europe",
            "category": "LOCATION",
            "synonyms": ["france", "sweden"],
        }
        store.articles.append(
            {
                "id": "a3",
                "title": "French ministry breached",
                "content": "The incident affected France's education ministry.",
            }
        )
        store.article_tags["a3"] = [store.tags["europe"]]
        client = FakeLLMClient(
            [
                response(
                    {
                        "tag": "Europe",
                        "relevant": True,
                        "confidence": "high",
                        "match_type": "synonym",
                        "matched_term": "France",
                        "reasoning": "France är en angiven synonym och ligger i Europa.",
                    }
                )
            ]
        )

        report = await audit_tags(store, client, ["Europe"])

        prompt = client.calls[0][0][0]["content"]
        self.assertIn('"synonyms": [', prompt)
        self.assertIn('"france"', prompt)
        self.assertIn("överordnad geografisk tagg", prompt)
        self.assertEqual("synonym", report["assessments"][0]["match_type"])
        self.assertEqual("France", report["assessments"][0]["matched_term"])

    async def test_audit_without_tag_filter_checks_all_associations_read_only(self):
        store = FakeStore()
        client = FakeLLMClient(
            [
                response(
                    {
                        "tag": "ray",
                        "relevant": False,
                        "confidence": "high",
                        "reasoning": "Delsträng.",
                    }
                ),
                response(
                    {"tag": "ray", "relevant": True, "confidence": "high", "reasoning": "Nämns."},
                    {
                        "tag": "comfast",
                        "relevant": True,
                        "confidence": "high",
                        "reasoning": "Nämns.",
                    },
                ),
            ]
        )

        report = await audit_tags(store, client, [])

        self.assertEqual(
            {
                "relevant": 2,
                "irrelevant": 1,
                "errors": 0,
                "removed": 0,
                "unused_tags": 1,
                "deleted_unused_tags": 0,
            },
            report["totals"],
        )
        self.assertEqual([], store.removed)
        self.assertEqual([], store.deleted_tags)
        self.assertEqual("all", report["scope"])
        self.assertEqual(["comfast", "orphan", "ray"], report["requested_tags"])
        self.assertEqual("orphan", report["unused_tags"][0]["tag"])
        self.assertTrue(all(call[1] == 0.0 for call in client.calls))

    async def test_remove_invalid_uses_store_api(self):
        store = FakeStore()
        client = FakeLLMClient(
            [
                response(
                    {
                        "tag": "ray",
                        "relevant": False,
                        "confidence": "high",
                        "reasoning": "Delsträng.",
                    }
                ),
                response(
                    {"tag": "ray", "relevant": True, "confidence": "high", "reasoning": "Nämns."},
                    {
                        "tag": "comfast",
                        "relevant": False,
                        "confidence": "medium",
                        "reasoning": "Perifert.",
                    },
                ),
            ]
        )

        report = await audit_tags(
            store,
            client,
            [],
            remove_invalid=True,
        )

        self.assertEqual([("a1", 1), ("a2", 2)], store.removed)
        self.assertEqual(2, report["totals"]["removed"])
        self.assertEqual([2, 3], store.deleted_tags)
        self.assertEqual(2, report["totals"]["deleted_unused_tags"])
        self.assertIn("ray", store.tags)

    async def test_existing_report_can_be_applied_without_another_llm_audit(self):
        store = FakeStore()
        client = FakeLLMClient(
            [
                response(
                    {
                        "tag": "ray",
                        "relevant": False,
                        "confidence": "high",
                        "reasoning": "Delsträng.",
                    }
                ),
                response(
                    {"tag": "ray", "relevant": True, "confidence": "high", "reasoning": "Nämns."},
                    {
                        "tag": "comfast",
                        "relevant": False,
                        "confidence": "medium",
                        "reasoning": "Perifert.",
                    },
                ),
            ]
        )
        source_report = await audit_tags(store, client, [])
        llm_call_count = len(client.calls)

        applied_report = apply_audit_report(store, source_report)

        self.assertEqual(llm_call_count, len(client.calls))
        self.assertEqual("apply_report", applied_report["mode"])
        self.assertEqual([("a1", 1), ("a2", 2)], store.removed)
        self.assertEqual([2, 3], store.deleted_tags)
        self.assertEqual(2, applied_report["totals"]["removed"])

    async def test_report_tag_name_must_match_current_database(self):
        store = FakeStore()
        source_report = {
            "assessments": [
                {
                    "article_id": "a1",
                    "tag_id": 1,
                    "tag": "not-ray",
                    "relevant": False,
                    "confidence": "high",
                    "reasoning": "Fel taggnamn i rapporten.",
                }
            ]
        }

        applied_report = apply_audit_report(store, source_report)

        self.assertEqual([], store.removed)
        self.assertEqual("tag_mismatch", applied_report["assessments"][0]["removal_status"])
        self.assertEqual(1, applied_report["totals"]["errors"])

    async def test_legacy_report_cannot_remove_tag_with_synonyms(self):
        store = FakeStore()
        store.tags["europe"] = {
            "id": 4,
            "name": "Europe",
            "category": "LOCATION",
            "synonyms": ["france", "sweden"],
        }
        store.articles.append(
            {
                "id": "a3",
                "title": "French ministry breached",
                "content": "The incident affected France's education ministry.",
            }
        )
        store.article_tags["a3"] = [store.tags["europe"]]
        legacy_report = {
            "assessments": [
                {
                    "article_id": "a3",
                    "tag_id": 4,
                    "tag": "Europe",
                    "relevant": False,
                    "confidence": "high",
                    "reasoning": "Äldre bedömning utan synonymunderlag.",
                }
            ]
        }

        applied_report = apply_audit_report(store, legacy_report)

        self.assertNotIn(("a3", 4), store.removed)
        self.assertEqual(
            "legacy_report_with_synonyms",
            applied_report["assessments"][0]["removal_status"],
        )
        self.assertEqual(1, applied_report["totals"]["errors"])

    async def test_invalid_response_is_retried(self):
        store = FakeStore()
        client = FakeLLMClient(
            [
                "not json",
                response(
                    {
                        "tag": "ray",
                        "relevant": False,
                        "confidence": "high",
                        "reasoning": "Delsträng.",
                    }
                ),
            ]
        )

        report = await audit_tags(store, client, ["ray"], limit=1, attempts=2)

        self.assertEqual(2, len(client.calls))
        self.assertEqual(1, report["totals"]["irrelevant"])
        self.assertEqual(0, report["totals"]["errors"])


if __name__ == "__main__":
    unittest.main()
