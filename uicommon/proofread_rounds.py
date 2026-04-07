from __future__ import annotations

import contextvars
import logging
import time
from typing import Any, Dict, Optional

_PATCHED = False
_LAST_LOGGED_EFFECTIVE: Optional[int] = None
_PROOFREAD_SNAPSHOT: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar("proofread_snapshot", default=None)
)


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
        effective = _pick_effective_max_rounds(config, int(max_rounds))
        original_meta = str(meta_text or "")

        global _LAST_LOGGED_EFFECTIVE
        if logger is not None and _LAST_LOGGED_EFFECTIVE != effective:
            logger.info(
                "Proofread rounds configured: requested=%s effective=%s (batching.proofread_max_rounds)",
                int(max_rounds),
                effective,
            )
            _LAST_LOGGED_EFFECTIVE = effective

        revised_text, stats = await original(
            config=config,
            llm=llm,
            store=store,
            job_id=job_id,
            prompts=prompts,
            lookback=lookback,
            meta_text=meta_text,
            batch_summaries=batch_summaries,
            sources_text=sources_text,
            max_rounds=effective,
        )
        _PROOFREAD_SNAPSHOT.set(
            {
                "original_summary": original_meta,
                "revised_summary": str(revised_text or ""),
                "proofread_stats": dict(stats or {}),
            }
        )
        return revised_text, stats

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
                    }
                    pa = out.get("proofread_audit") or {}
                    if not isinstance(pa, dict):
                        pa = {}
                    history = pa.get("history") or []
                    if not isinstance(history, list):
                        history = []
                    history.append(audit_entry)
                    out["proofread_audit"] = {"latest": audit_entry, "history": history[-20:]}
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
