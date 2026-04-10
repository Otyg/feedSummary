from __future__ import annotations

import contextvars
import logging
import re
import time
from typing import Any, Dict, List, Optional
from uicommon.proofread_merge import stabilize_revise_output_from_messages

_PATCHED = False
_LAST_LOGGED_EFFECTIVE: Optional[int] = None
_PROOFREAD_SNAPSHOT: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar("proofread_snapshot", default=None)
)


def _clip(s: Any, max_len: int = 12000) -> str:
    t = str(s or "")
    return t if len(t) <= max_len else (t[: max_len - 3] + "...")


def _strip_proofread_feedback_from_summary(
    summary_text: str, proofread_stats: Optional[Dict[str, Any]]
) -> str:
    text = str(summary_text or "")
    stats = proofread_stats if isinstance(proofread_stats, dict) else {}

    # Remove exact injected feedback blobs when present.
    for key in ("proofread_last_feedback", "proofread_output"):
        blob = str(stats.get(key) or "").strip()
        if len(blob) < 40:
            continue
        if blob in text:
            text = text.replace(blob, "").strip()

    # Remove orphan PASS/FAIL lines right before sources appendix.
    text = re.sub(
        r"\n{2,}(?:PASS|FAIL)\s*\n{2,}(?=##\s*Källor\b)",
        "\n\n",
        text,
        flags=re.IGNORECASE,
    )
    # Remove injected PASS/FAIL + optional ISSUES block before sources appendix.
    text = re.sub(
        r"\n{2,}(?:PASS|FAIL)\s*(?:\n+[\s\S]*?)?(?=\n##\s*Källor\b)",
        "\n\n",
        text,
        flags=re.IGNORECASE,
    )
    # Normalize excessive blank lines left after stripping.
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def _pick_effective_max_rounds(config: Dict[str, Any], fallback: int) -> int:
    try:
        batching = config.get("batching") or {}
        if not isinstance(batching, dict):
            return int(fallback)
        raw = batching.get("proofread_max_rounds")
        if raw is None or str(raw).strip() == "":
            return int(fallback)
        parsed = int(raw)
        if parsed <= 0:
            return int(fallback)
        return parsed
    except Exception:
        return int(fallback)


def _should_override_rounds(callsite_max_rounds: int) -> bool:
    """
    Only override regular pipeline rounds.
    feedsummary_core regular flow calls with 4; composed flow calls with 1.
    """
    try:
        return int(callsite_max_rounds) >= 2
    except Exception:
        return False


def enable_configurable_proofread_rounds(
    *, logger: Optional[logging.Logger] = None
) -> None:
    """
    Allow regular pipeline runs to override proofread/revise rounds via:
      batching.proofread_max_rounds

    feedsummary_core currently hardcodes max_rounds in call sites; this wraps the
    shared helper and replaces the passed-in value with config when set.
    """

    global _PATCHED
    global _LAST_LOGGED_EFFECTIVE

    if _PATCHED:
        return

    try:
        from feedsummary_core.summarizer import main as main_mod
        from feedsummary_core.summarizer import summarizer as summarizer_mod
    except Exception:
        return

    original = getattr(summarizer_mod, "_proofread_and_revise_meta_with_stats", None)
    if original is None:
        return

    if getattr(original, "__name__", "") == "_proofread_and_revise_meta_with_config":
        _PATCHED = True
        return

    original_persist = getattr(main_mod, "_persist_summary_doc", None)
    if not callable(original_persist):
        return

    async def _proofread_and_revise_meta_with_config(
        *,
        config: Dict[str, Any],
        llm: Any,
        store: Any,
        job_id: Any,
        prompts: Dict[str, Any],
        lookback: str,
        meta_text: str,
        batch_summaries: Any,
        sources_text: str,
        max_rounds: int = 1,
    ):
        callsite_rounds = int(max_rounds)
        effective = (
            _pick_effective_max_rounds(config, callsite_rounds)
            if _should_override_rounds(callsite_rounds)
            else callsite_rounds
        )
        original_meta = str(meta_text or "")

        global _LAST_LOGGED_EFFECTIVE
        if logger is not None and _LAST_LOGGED_EFFECTIVE != effective:
            logger.info(
                "Proofread rounds configured: requested=%s effective=%s (batching.proofread_max_rounds)",
                int(max_rounds),
                effective,
            )
            _LAST_LOGGED_EFFECTIVE = effective

        proof_sys = str(prompts.get("proofread_system") or "").strip()
        revise_sys = str(prompts.get("revise_system") or "").strip()

        proofread_trace: List[Dict[str, Any]] = []
        proofread_round = 0
        revise_round = 0

        class _LLMRecorder:
            def __init__(self, inner: Any):
                self._inner = inner

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

            async def chat(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
                nonlocal proofread_round, revise_round
                reply = await self._inner.chat(messages, *args, **kwargs)

                role = "other"
                sys_msg = ""
                try:
                    if isinstance(messages, list) and messages:
                        first = messages[0] if isinstance(messages[0], dict) else {}
                        sys_msg = str(first.get("content") or "").strip()
                except Exception:
                    sys_msg = ""

                if proof_sys and sys_msg == proof_sys:
                    role = "proofread"
                    proofread_round += 1
                    round_no = proofread_round
                elif revise_sys and sys_msg == revise_sys:
                    role = "revise"
                    revise_round += 1
                    round_no = revise_round
                    if isinstance(messages, list):
                        reply = stabilize_revise_output_from_messages(
                            messages=messages, raw_reply=str(reply or "")
                        )
                else:
                    round_no = 0

                if role in {"proofread", "revise"}:
                    if logger is not None:
                        preview = _clip(reply, 200).replace("\n", " ")
                        logger.info(
                            "Proofread trace step=%s round=%s preview=%s",
                            role,
                            int(round_no),
                            preview,
                        )
                    proofread_trace.append(
                        {
                            "round": int(round_no),
                            "step": role,
                            "at": int(time.time()),
                            "output": _clip(reply, 16000),
                        }
                    )
                return reply

        llm_rec = _LLMRecorder(llm)

        revised_text, stats = await original(
            config=config,
            llm=llm_rec,
            store=store,
            job_id=job_id,
            prompts=prompts,
            lookback=lookback,
            meta_text=meta_text,
            batch_summaries=batch_summaries,
            sources_text=sources_text,
            max_rounds=effective,
        )
        stats_out = dict(stats or {})
        stats_out["proofread_trace"] = list(proofread_trace)
        if logger is not None:
            logger.info(
                "Proofread flow done: enabled=%s rounds=%s output=%s last_feedback_len=%s",
                int(stats_out.get("proofread_enabled") or 0),
                int(stats_out.get("proofread_rounds") or 0),
                _clip(str(stats_out.get("proofread_output") or ""), 120),
                len(str(stats_out.get("proofread_last_feedback") or "")),
            )
        _PROOFREAD_SNAPSHOT.set(
            {
                "original_summary": original_meta,
                "revised_summary": str(revised_text or ""),
                "proofread_stats": stats_out,
            }
        )
        return revised_text, stats_out

    def _persist_summary_doc_with_proofread_snapshot(store: Any, doc: Dict[str, Any]) -> Any:
        snapshot = _PROOFREAD_SNAPSHOT.get()
        _PROOFREAD_SNAPSHOT.set(None)

        out = dict(doc or {})
        try:
            if isinstance(snapshot, dict):
                original_summary = str(snapshot.get("original_summary") or "").strip()
                revised_summary = str(snapshot.get("revised_summary") or "").strip()
                if original_summary and revised_summary:
                    existing_original = str(
                        out.get("proofread_original_summary") or ""
                    ).strip()
                    existing_revised = str(
                        out.get("proofread_revised_summary") or ""
                    ).strip()

                    if not existing_original:
                        out["proofread_original_summary"] = original_summary
                    if not existing_revised:
                        out["proofread_revised_summary"] = revised_summary

                    audit_entry = {
                        "created_at": int(time.time()),
                        "prompt_package": str(
                            (out.get("prompts") or {}).get("_package")
                            or (out.get("selection") or {}).get("prompt_package")
                            or ""
                        ),
                        "original_summary": str(
                            out.get("proofread_original_summary") or original_summary
                        ),
                        "revised_summary": str(
                            out.get("proofread_revised_summary") or revised_summary
                        ),
                        "proofread_output": str(
                            (snapshot.get("proofread_stats") or {}).get(
                                "proofread_output"
                            )
                            or ""
                        ),
                        "proofread_last_feedback": str(
                            (snapshot.get("proofread_stats") or {}).get(
                                "proofread_last_feedback"
                            )
                            or ""
                        ),
                        "proofread_rounds": int(
                            (snapshot.get("proofread_stats") or {}).get(
                                "proofread_rounds"
                            )
                            or 0
                        ),
                        "proofread_trace": (
                            (snapshot.get("proofread_stats") or {}).get(
                                "proofread_trace"
                            )
                            or []
                        ),
                    }
                    pa = out.get("proofread_audit") or {}
                    if not isinstance(pa, dict):
                        pa = {}
                    history = pa.get("history") or []
                    if not isinstance(history, list):
                        history = []
                    history.append(audit_entry)
                    out["proofread_audit"] = {"latest": audit_entry, "history": history[-20:]}

                if str(out.get("summary") or "").strip():
                    out["summary"] = _strip_proofread_feedback_from_summary(
                        str(out.get("summary") or ""),
                        snapshot.get("proofread_stats")
                        if isinstance(snapshot.get("proofread_stats"), dict)
                        else {},
                    )
        except Exception:
            out = dict(doc or {})

        return original_persist(store, out)

    summarizer_mod._proofread_and_revise_meta_with_stats = (
        _proofread_and_revise_meta_with_config
    )

    if hasattr(main_mod, "_proofread_and_revise_meta_with_stats"):
        main_mod._proofread_and_revise_meta_with_stats = (
            _proofread_and_revise_meta_with_config
        )
    main_mod._persist_summary_doc = _persist_summary_doc_with_proofread_snapshot

    _PATCHED = True
