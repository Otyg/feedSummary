import asyncio
import json

from feedsummary_core.summarizer import summarizer

from uicommon.proofread_rounds import (
    _strip_proofread_feedback_from_summary,
    enable_configurable_proofread_rounds,
)


class _ProofreadLLM:
    def __init__(self) -> None:
        self.proofread_calls = 0
        self.revise_calls = 0

    async def chat(self, messages, **_kwargs):
        system = str(messages[0]["content"])
        if system.startswith("PROOFREAD"):
            self.proofread_calls += 1
            status = "PASS" if self.proofread_calls == 3 else "REVISE"
            issues = []
            if status == "REVISE":
                issues = [
                    {
                        "issue_id": f"issue-{self.proofread_calls}",
                        "type": "test",
                        "target_quote": "",
                        "action": "test",
                        "preserve_requirement": "",
                    }
                ]
            return json.dumps({"status": status, "issues": issues})

        self.revise_calls += 1
        return '{"operations":[]}'


def test_configured_rounds_end_with_audit_only_proofread():
    enable_configurable_proofread_rounds()
    llm = _ProofreadLLM()

    async def run():
        return await summarizer._proofread_and_revise_meta_with_stats(
            config={
                "llm": [{"context_window_tokens": 32768, "max_output_tokens": 700}],
                "batching": {"proofread_max_rounds": 2},
            },
            llm=llm,
            store=object(),
            job_id=None,
            prompts={
                "proofread_system": "PROOFREAD",
                "proofread_user_template": (
                    "{lookback} {draft_summary} {desk_underlag} {feedback}"
                ),
                "revise_system": "REVISE",
                "revise_user_template": (
                    "{lookback} {draft_summary} {desk_underlag} {feedback}"
                ),
            },
            lookback="test",
            meta_text="Published summary",
            batch_summaries=[],
            sources_text="",
            max_rounds=4,
        )

    revised, stats = asyncio.run(run())

    assert revised == "Published summary"
    assert llm.proofread_calls == 3
    assert llm.revise_calls == 2
    assert stats["proofread_rounds"] == 3
    assert json.loads(stats["proofread_output"]) == {"status": "PASS", "issues": []}
    assert [(row["step"], row["round"]) for row in stats["proofread_trace"]] == [
        ("proofread", 1),
        ("revise", 1),
        ("proofread", 2),
        ("revise", 2),
        ("proofread", 3),
    ]


def test_structured_proofread_report_is_removed_from_summary():
    report = '{"status":"PASS","issues":[]}'
    summary = f"Ett aktuellt stycke.\n\n{report}\n\n## Källor\n- Källa"

    cleaned = _strip_proofread_feedback_from_summary(
        summary, {"proofread_output": report}
    )

    assert report not in cleaned
    assert cleaned == "Ett aktuellt stycke.\n\n\n## Källor\n- Källa"
