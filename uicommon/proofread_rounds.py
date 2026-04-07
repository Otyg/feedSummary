from __future__ import annotations

import logging
from typing import Any, Dict, Optional

_PATCHED = False
_LAST_LOGGED_EFFECTIVE: Optional[int] = None


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

        global _LAST_LOGGED_EFFECTIVE
        if logger is not None and _LAST_LOGGED_EFFECTIVE != effective:
            logger.info(
                "Proofread rounds configured: requested=%s effective=%s (batching.proofread_max_rounds)",
                int(max_rounds),
                effective,
            )
            _LAST_LOGGED_EFFECTIVE = effective

        return await original(
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

    summarizer_mod._proofread_and_revise_meta_with_stats = (
        _proofread_and_revise_meta_with_config
    )

    if hasattr(main_mod, "_proofread_and_revise_meta_with_stats"):
        main_mod._proofread_and_revise_meta_with_stats = (
            _proofread_and_revise_meta_with_config
        )

    _PATCHED = True

