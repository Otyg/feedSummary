#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Proofread + revise an EXISTING summary_doc using the articles that summary_doc references.
This is useful for iteratively improving a summary_doc without losing the original summary text as reference.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import yaml

from feedsummary_core.llm_client import create_llm_client, LLMClient
from feedsummary_core.persistence import create_store, NewsStore
from feedsummary_core.summarizer.helpers import (
    clip_text,
    interleave_by_source_oldest_first,
    load_prompts,
    lookback_label_from_range,
    set_job,
)
from feedsummary_core.summarizer.batching import (
    PromptTooLongStructural,
    _choose_trim_action,
    _estimate_article_chars,
    _move_article_to_tail_batch,
    batch_articles,
    build_messages_for_batch,
    trim_text_tail_by_words,
)
from feedsummary_core.summarizer.chat import chat_guarded
from feedsummary_core.summarizer.summarizer import _proofread_and_revise_meta_with_stats

from uicommon.bootstrap_ui import _setup_logging_if_needed
log = logging.getLogger(__name__)
_setup_logging_if_needed()
def _load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _get_store_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # develop: create_store(cfg_dict) expects dict with provider/path
    for k in ("store", "db", "database", "persistence"):
        v = cfg.get(k)
        if isinstance(v, dict):
            return v
    return {
        "provider": cfg.get("store_provider", "tinydb"),
        "path": cfg.get("store_path", "news_docs.json"),
    }


def _derive_lookback_label(cfg: Dict[str, Any], summary_doc: Dict[str, Any]) -> str:
    ingest = cfg.get("ingest") or {}
    lookback_raw = str(ingest.get("lookback") or "").strip()

    from_ts = int(summary_doc.get("from") or 0)
    to_ts = int(summary_doc.get("to") or 0)

    if from_ts and to_ts:
        return lookback_label_from_range(lookback_raw, from_ts, to_ts)

    # fallback: if no timestamps in doc, just use configured lookback string
    return lookback_raw


async def _build_batch_summaries_from_articles(
    *,
    cfg: Dict[str, Any],
    prompts: Dict[str, str],
    llm: LLMClient,
    articles: List[dict],
    store: NewsStore,
) -> List[Tuple[int, str]]:
    """
    Re-run the batch step on the exact same article set,
    so proofread/revise has desk-underlag based on current batch prompts.
    Mirrors summarizer.py's guarded retry logic for prompt-too-long.
    """
    batching = cfg.get("batching", {}) or {}
    max_chars = int(batching.get("max_chars_per_batch", 18000))
    max_n = int(batching.get("max_articles_per_batch", 10))
    article_clip_chars = int(batching.get("article_clip_chars", 6000))

    llm_cfg = cfg.get("llm") or {}
    max_ctx = int(llm_cfg.get("context_window_tokens", 32768))
    max_out = int(llm_cfg.get("max_output_tokens", 700))
    margin = int(llm_cfg.get("prompt_safety_margin", 1024))
    chars_per_token = float(llm_cfg.get("token_chars_per_token", 2.4))
    max_attempts = int(llm_cfg.get("prompt_too_long_max_attempts", 6))
    structural_threshold = int(llm_cfg.get("prompt_too_long_structural_threshold_tokens", 1200))

    ordered = interleave_by_source_oldest_first(articles)
    batches = batch_articles(ordered, max_chars, max_n, article_clip_chars=article_clip_chars)

    out: List[Tuple[int, str]] = []
    idx = 1
    while idx <= len(batches):
        batch = batches[idx - 1]
        set_job(f"[offline] Summerar batch {idx}/{len(batches)}...", job_id=None, store=store)
        log.info(f"Building batch summary {idx}/{len(batches)} with {len(batch)} articles...")
        while True:
            try:
                summary = await chat_guarded(
                    llm=llm,
                    messages=build_messages_for_batch(
                        prompts=prompts,
                        batch_index=idx,
                        batch_total=len(batches),
                        batch_items=batch,
                    ),
                    temperature=0.2,
                    max_ctx=max_ctx,
                    max_out=max_out,
                    margin=margin,
                    chars_per_token=chars_per_token,
                    max_attempts=max_attempts,
                    structural_threshold=structural_threshold,
                )
                break
            except PromptTooLongStructural as e:
                overflow = int(getattr(e, "overflow_tokens", 0) or 0)
                action = _choose_trim_action(overflow, structural_threshold)
                log.warning(f"Batch {idx}/{len(batches)} prompt too long by {overflow} tokens. Action: {action}")
                # Single-article batch: trim text instead of moving
                if len(batch) <= 1:
                    a0 = batch[0]
                    remove_tokens = (overflow + 2048) if overflow else 4096
                    a0["text"] = trim_text_tail_by_words(
                        a0.get("text", "") or "",
                        remove_tokens,
                        chars_per_token=chars_per_token,
                    )
                    continue

                # Move articles to tail until prompt fits
                target_remove_tokens = overflow + 1024
                target_remove_chars = int(target_remove_tokens * chars_per_token)

                removed_chars = 0
                if action == "drop_one_article":
                    a = batch.pop()
                    removed_chars = _estimate_article_chars(a)
                    _move_article_to_tail_batch(
                        batches,
                        a,
                        max_chars_per_batch=max_chars,
                        max_articles_per_batch=max_n,
                        avoid_batch=batch,
                    )
                else:
                    while len(batch) > 1 and removed_chars < target_remove_chars:
                        a = batch.pop()
                        removed_chars += _estimate_article_chars(a)
                        _move_article_to_tail_batch(
                            batches,
                            a,
                            max_chars_per_batch=max_chars,
                            max_articles_per_batch=max_n,
                            avoid_batch=batch,
                        )
                continue

        out.append((idx, summary))
        idx += 1

    return out


async def _run(args: argparse.Namespace) -> int:
    cfg = _load_yaml(args.config)

    # Optional: override prompt package selection in config (same mechanism as app)
    if args.prompt_package:
        p = cfg.get("prompts")
        if not isinstance(p, dict):
            p = {}
            cfg["prompts"] = p
        p["selected"] = str(args.prompt_package)

    store_cfg = _get_store_cfg(cfg)
    store: NewsStore = create_store(store_cfg)
    llm = create_llm_client(cfg)
    prompts = load_prompts(cfg)

    # Load summary_doc
    if args.latest:
        doc = store.get_latest_summary_doc()
        if not doc:
            log.error("No summary_docs found in DB.", file=sys.stderr)
            return 2
    else:
        if not args.summary_id:
            log.error("Need --summary-id or --latest", file=sys.stderr)
            return 2
        doc = store.get_summary_doc(args.summary_id)
        if not doc:
            log.error(f"summary_doc not found: {args.summary_id}", file=sys.stderr)
            return 2

    summary_id = str(doc.get("id") or args.summary_id or "latest")
    summary_text = str(doc.get("summary") or "").strip()
    if not summary_text:
        log.error(f"summary_doc {summary_id} has empty 'summary'", file=sys.stderr)
        return 3

    sources: List[str] = [str(x) for x in (doc.get("sources") or []) if str(x).strip()]
    if not sources:
        log.error(f"summary_doc {summary_id} has no 'sources' list", file=sys.stderr)
        return 3

    articles = store.get_articles_by_ids(sources)
    if not articles:
        log.error("Could not load any articles for summary_doc sources.", file=sys.stderr)
        return 4

    # Rebuild desk-underlag from those articles (batch step)
    batch_summaries = await _build_batch_summaries_from_articles(
        cfg=cfg,
        prompts=prompts,
        llm=llm,
        articles=articles,
        store=store,
    )

    # Build sources_text (same style as summarizer does; meta prompt uses it)
    batching = cfg.get("batching", {}) or {}
    meta_sources_clip_chars = int(batching.get("meta_sources_clip_chars", 140))
    sources_text = "\n".join(
        [
            f"- {clip_text(a.get('title', ''), meta_sources_clip_chars)} — {(a.get('url') or '').strip()}"
            for a in articles
        ]
    ).strip()

    lookback_label = _derive_lookback_label(cfg, doc)

    log.info(f"Proofreading summary_doc {summary_id} with lookback label: {lookback_label}")
    revised, pr_stats = await _proofread_and_revise_meta_with_stats(
        config=cfg,
        llm=llm,
        store=store,
        job_id=None,
        prompts=prompts,
        lookback=lookback_label,
        meta_text=summary_text,
        batch_summaries=batch_summaries,
        sources_text=sources_text,
        max_rounds=int(args.max_rounds),
    )

    proof_out = (pr_stats.get("proofread_output") or pr_stats.get("proofread_last_feedback") or "").strip()

    log.info("\n" + "=" * 90)
    log.info(f"SUMMARY_DOC: {summary_id}")
    log.info(f"SOURCES: {len(articles)} articles")
    log.info(f"LOOKBACK LABEL: {lookback_label}")
    log.info("=" * 90 + "\n")

    log.info("ORIGINAL SUMMARY (clip 2000):\n")
    log.info(clip_text(summary_text, 2000))
    log.info("\n" + "-" * 90 + "\n")

    log.info("PROOFREAD OUTPUT:\n")
    log.info(proof_out if proof_out else "(no proofread output)")
    log.info("\n" + "-" * 90 + "\n")

    log.info("REVISED SUMMARY:\n")
    log.info(revised.strip())
    log.info("\n" + "=" * 90 + "\n")

    if args.out:
        Path(args.out).write_text(revised.strip() + "\n", encoding="utf-8")
        log.info(f"Wrote revised summary to: {args.out}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    ap.add_argument("--summary-id", default="", help="summary_doc id to proofread/revise")
    ap.add_argument("--latest", action="store_true", help="Use latest summary_doc instead of --summary-id")
    ap.add_argument("--prompt-package", default="", help="Override prompts.selected for this run")
    ap.add_argument("--max-rounds", type=int, default=4, help="Max proofread critique rounds")
    ap.add_argument("--out", default="", help="Write revised summary to file")
    args = ap.parse_args()

    if not args.latest and not args.summary_id:
        ap.error("Provide --summary-id or --latest")

    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())