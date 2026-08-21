import unittest
from unittest.mock import patch

from uicommon.summary_tags import tag_summary_doc
from web_viewer.webapp_viewer import app


class _Store:
    def __init__(self):
        self.doc = {
            "id": "summary-1",
            "title": "Veckans hotbild",
            "summary": "En rapport om ransomware och en ny sårbarhet.",
            "meta": {"composed": False},
        }
        self.saved = []

    def get_summary_doc(self, summary_id):
        return dict(self.doc) if summary_id == self.doc["id"] else None

    def save_summary_doc(self, doc):
        self.doc = dict(doc)
        self.saved.append(dict(doc))
        return doc["id"]


class _TagManager:
    calls = []

    def __init__(self, store, llm_client=None):
        self.store = store

    async def generate_tags_for_article(self, **kwargs):
        self.calls.append(kwargs)
        return [
            {
                "id": 7,
                "name": "ransomware",
                "category": "THREAT",
                "reasoning": "Centralt ämne",
            }
        ]

    async def select_tags_for_article_async(self, **kwargs):
        self.calls.append(kwargs)
        return [
            {
                "id": 100 + index,
                "name": candidate["name"].lower(),
                "category": "VULNERABILITY",
                "reasoning": candidate.get("reasoning", ""),
            }
            for index, candidate in enumerate(kwargs["candidate_tags"])
        ]


class SummaryTaggingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        _TagManager.calls.clear()

    @patch("uicommon.summary_tags.TagManager", _TagManager)
    async def test_tags_summary_with_article_tagger_and_persists_result(self):
        store = _Store()

        tags = await tag_summary_doc(
            store=store,
            llm_client=object(),
            config={},
            summary_id="summary-1",
        )

        self.assertEqual("ransomware", tags[0]["name"])
        self.assertEqual(tags, store.doc["tags"])
        self.assertIn("summary_tagged_at", store.doc["meta"])
        self.assertEqual("summary-1", _TagManager.calls[0]["article"]["id"])
        self.assertEqual(20, _TagManager.calls[0]["max_tags"])
        self.assertEqual(
            store.doc["summary"], _TagManager.calls[0]["article"]["content"]
        )

    @patch("uicommon.summary_tags.TagManager", _TagManager)
    async def test_adds_every_mentioned_cve_outside_normal_tag_limit(self):
        store = _Store()
        store.doc["summary"] = (
            "CVE-2026-1234 nämns först. "
            + ("lång text " * 400)
            + "CVE-2026-99999 nämns sist. cve-2026-1234 nämns igen."
        )

        tags = await tag_summary_doc(
            store=store,
            llm_client=object(),
            config={"tagging": {"summary_max_tags": 15}},
            summary_id="summary-1",
        )

        self.assertEqual(15, _TagManager.calls[0]["max_tags"])
        cve_names = {
            tag["name"].upper()
            for tag in tags
            if tag.get("category") == "VULNERABILITY"
        }
        self.assertEqual({"CVE-2026-1234", "CVE-2026-99999"}, cve_names)
        self.assertEqual(2, len(_TagManager.calls[1]["candidate_tags"]))

    @patch("uicommon.summary_tags.TagManager", _TagManager)
    async def test_does_not_tag_an_already_tagged_summary_again(self):
        store = _Store()
        store.doc["tags"] = [{"id": 3, "name": "privacy", "category": "GENERAL"}]

        tags = await tag_summary_doc(
            store=store,
            llm_client=object(),
            config={},
            summary_id="summary-1",
        )

        self.assertEqual("privacy", tags[0]["name"])
        self.assertEqual([], _TagManager.calls)
        self.assertEqual([], store.saved)


class SummaryTagRenderingTests(unittest.TestCase):
    def test_summary_tags_render_below_title(self):
        context = {
            "summary": {
                "id": "summary-1",
                "title": "Veckans hotbild",
                "_viewer_tags": [
                    {
                        "name": "ransomware",
                        "category": "THREAT",
                        "bg_color": "bg-danger",
                        "text_color": "text-white",
                    }
                ],
            },
            "html": "<p>Summarytext</p>",
            "has_proofread_audit": False,
            "summaries": [],
            "default_selected": "summary-1",
            "available_topics": [],
            "active_topics": [],
            "format_ts": lambda value: str(value),
        }

        with app.test_request_context("/summary/summary-1"):
            rendered = app.jinja_env.get_template("index.html").render(**context)

        self.assertIn("bg-danger text-white", rendered)
        self.assertLess(rendered.index("Veckans hotbild"), rendered.index("ransomware"))
        self.assertLess(rendered.index("ransomware"), rendered.index("Summarytext"))
