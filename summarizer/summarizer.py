# ----------------------------
# Summarization (LLM + stable checkpoint/resume + budgeted meta)
# ----------------------------
from llmClient import LLMClient
from persistence import NewsStore
from summarizer.batching import (
    PromptTooLongStructural,
    _batch_article_ids_map,
    _budgeted_meta_user,
    _build_batches_from_checkpoint,
    _choose_trim_action,
    _done_batches_payload,
    _done_map_from_done_batches,
    _est_user_tokens,
    _estimate_article_chars,
    _move_article_to_tail_batch,
    batch_articles,
    build_messages_for_batch,
    trim_text_tail_by_words,
)
from summarizer.chat import chat_guarded
from summarizer.helpers import (
    _atomic_write_json,
    _checkpoint_key,
    _checkpoint_path,
    _extract_overflow_tokens,
    _load_checkpoint,
    _meta_ckpt_path,
    clip_text,
    interleave_by_source_oldest_first,
    load_prompts,
    set_job,
)
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


async def summarize_batches_then_meta(
    config: Dict[str, Any],
    articles: List[dict],
    llm: LLMClient,
    store: NewsStore,
    job_id: Optional[int] = None,
) -> str:
    """
    - checkpoint efter varje batch (inkl. batch_article_ids + done_batches)
    - HELT stabil resume: återskapar batches från checkpointens batch_article_ids
    - robust prompt-too-long: flytta artiklar (undvik tail-loop) och trimma single-article batch vid ordgräns
    - meta byggs budgeterat för att hålla context
    """
    prompts = load_prompts(config)

    batching = config.get("batching", {}) or {}
    max_chars = int(batching.get("max_chars_per_batch", 18000))
    max_n = int(batching.get("max_articles_per_batch", 10))
    article_clip_chars = int(batching.get("article_clip_chars", 6000))
    meta_sources_clip_chars = int(batching.get("meta_sources_clip_chars", 140))

    llm_cfg = config.get("llm") or {}
    max_ctx = int(llm_cfg.get("context_window_tokens", 32768))
    max_out = int(llm_cfg.get("max_output_tokens", 700))
    margin = int(llm_cfg.get("prompt_safety_margin", 1024))
    chars_per_token = float(llm_cfg.get("token_chars_per_token", 2.4))
    max_attempts = int(llm_cfg.get("prompt_too_long_max_attempts", 6))
    structural_threshold = int(
        llm_cfg.get("prompt_too_long_structural_threshold_tokens", 1200)
    )

    # ---- checkpoint setup ----
    cp_cfg = config.get("checkpointing") or {}
    cp_enabled = bool(cp_cfg.get("enabled", True))
    cp_key = _checkpoint_key(job_id, articles)
    cp_path: Optional[Path] = _checkpoint_path(config, cp_key) if cp_enabled else None
    meta_path: Optional[Path] = _meta_ckpt_path(config, cp_key) if cp_enabled else None

    articles_ordered = interleave_by_source_oldest_first(articles)
    batches = batch_articles(
        articles_ordered, max_chars, max_n, article_clip_chars=article_clip_chars
    )

    # meta resume (om redan klar)
    if cp_enabled and meta_path is not None:
        meta_cp = _load_checkpoint(meta_path)
        if meta_cp and meta_cp.get("kind") == "meta_result":
            cached = (meta_cp.get("meta") or "").strip()
            if cached:
                set_job("Återupptar: meta redan klar (från checkpoint).", job_id, store)
                return cached

    # batch resume
    done_map: Dict[int, str] = {}
    cp = _load_checkpoint(cp_path) if (cp_enabled and cp_path is not None) else None

    # HELT stabil resume: återskapa batches från checkpointens batch_article_ids
    if cp and cp.get("kind") == "batch_summaries":
        cp_batch_article_ids = cp.get("batch_article_ids") or {}
        cp_done_batches = cp.get("done_batches") or {}

        if isinstance(cp_batch_article_ids, dict) and cp_batch_article_ids:
            try:
                batches = _build_batches_from_checkpoint(
                    cp_batch_article_ids, articles, clip_chars=article_clip_chars
                )
                done_map = _done_map_from_done_batches(cp_done_batches)
                set_job(
                    f"Återupptar stabilt från checkpoint: {len(done_map)}/{len(batches)} batcher klara.",
                    job_id,
                    store,
                )
            except Exception as e:
                logger.warning(
                    "Resume: kunde inte återskapa batches från checkpoint (%s). Faller tillbaka.",
                    e,
                )
                done_map = {}

        if not done_map:
            # fallback för äldre checkpointformat (index-match)
            done = cp.get("done") or {}
            if isinstance(done, dict) and cp.get("batch_total") == len(batches):
                try:
                    for k, v in done.items():
                        done_map[int(k)] = str(v)
                    if done_map:
                        set_job(
                            f"Återupptar från checkpoint (index): {len(done_map)}/{len(batches)} batcher klara.",
                            job_id,
                            store,
                        )
                except Exception:
                    done_map = {}

    batch_summaries: List[Tuple[int, str]] = [
        (i, done_map[i]) for i in sorted(done_map.keys())
    ]

    # --- kör batches (med structural trim + tail-loop-skydd) ---
    idx = 1
    while idx <= len(batches):
        if idx in done_map:
            idx += 1
            continue

        batch = batches[idx - 1]
        set_job(f"Summerar batch {idx}/{len(batches)}...", job_id, store)

        # retry-loop för samma batch vid PromptTooLongStructural
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

                # Single-article batch: trimma artikeln, inte flytta
                if len(batch) <= 1:
                    a0 = batch[0]
                    remove_tokens = (overflow + 2048) if overflow else 4096
                    before_len = len(a0.get("text", "") or "")
                    a0["text"] = trim_text_tail_by_words(
                        a0.get("text", "") or "",
                        remove_tokens,
                        chars_per_token=chars_per_token,
                    )
                    after_len = len(a0["text"])
                    logger.warning(
                        "Single-article batch %s too long (overflow=%s). Trim by words: %s -> %s chars",
                        idx,
                        overflow,
                        before_len,
                        after_len,
                    )
                    continue

                target_remove_tokens = overflow + 1024
                target_remove_chars = int(target_remove_tokens * chars_per_token)

                removed_count = 0
                removed_chars = 0

                if action == "drop_one_article":
                    a = batch.pop()
                    removed_count = 1
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
                        removed_count += 1
                        removed_chars += _estimate_article_chars(a)
                        _move_article_to_tail_batch(
                            batches,
                            a,
                            max_chars_per_batch=max_chars,
                            max_articles_per_batch=max_n,
                            avoid_batch=batch,
                        )

                logger.warning(
                    "Prompt too long structural: overflow=%s action=%s removed=%s (chars~%s) from batch=%s. "
                    "Moved to tail. Retrying same batch.",
                    overflow,
                    action,
                    removed_count,
                    removed_chars,
                    idx,
                )
                continue

        done_map[idx] = summary
        batch_summaries.append((idx, summary))

        # checkpoint efter varje batch
        if cp_enabled and cp_path is not None:
            payload = {
                "kind": "batch_summaries",
                "created_at": int(time.time()),
                "job_id": job_id,
                "checkpoint_key": cp_key,
                "batch_total": len(batches),
                "done": {str(k): v for k, v in sorted(done_map.items())},  # bakåtkomp
                "done_batches": _done_batches_payload(done_map, batches),
                "batch_article_ids": _batch_article_ids_map(batches),
                "article_ids": [a.get("id", "") for a in articles],
            }
            _atomic_write_json(cp_path, payload)

        idx += 1

    # --- META (adaptivt budgeterad) ---
    set_job("Skapar metasammanfattning...", job_id, store)

    sources_list: List[str] = []
    for a in articles:
        title = clip_text(a.get("title", ""), meta_sources_clip_chars)
        url = (a.get("url") or "").strip()
        sources_list.append(f"- {title} — {url}")
    sources_text = "\n".join(sources_list)

    # Startbudget enligt config, men vi kommer sänka den om servern klagar
    budget_tokens = max(512, max_ctx - max_out - margin)

    meta_attempts = 8
    last_err: Optional[Exception] = None

    for attempt in range(1, meta_attempts + 1):
        meta_user = _budgeted_meta_user(
            prompts=prompts,
            batch_summaries=batch_summaries,
            sources_text=sources_text,
            budget_tokens=budget_tokens,
            chars_per_token=chars_per_token,
        )

        # checkpoint meta-input (uppdatera varje försök så /resume kan fortsätta här också)
        if cp_enabled and meta_path is not None:
            _atomic_write_json(
                meta_path,
                {
                    "kind": "meta_input",
                    "created_at": int(time.time()),
                    "job_id": job_id,
                    "checkpoint_key": cp_key,
                    "batch_total": len(batches),
                    "article_ids": [a.get("id", "") for a in articles],
                    "meta_system": prompts["meta_system"],
                    "meta_user": meta_user,
                    "meta_budget_tokens": budget_tokens,
                    "batch_article_ids": _batch_article_ids_map(batches),
                    "done_batches": _done_batches_payload(done_map, batches),
                },
            )

        meta_messages = [
            {"role": "system", "content": prompts["meta_system"]},
            {"role": "user", "content": meta_user},
        ]

        try:
            # använd llm.chat direkt här (chat_guarded kan annars kasta PromptTooLongStructural
            # baserat på din estimator som bevisligen inte matchar servern för meta)
            meta = await llm.chat(meta_messages, temperature=0.2)
            break

        except Exception as e:
            last_err = e
            msg = str(e).lower()
            overflow = _extract_overflow_tokens(e)

            if (
                not (
                    ("prompt too long" in msg)
                    or ("max context" in msg)
                    or ("context length" in msg)
                )
                or overflow is None
            ):
                raise

            overflow_i = int(overflow)
            est_prompt = _est_user_tokens(meta_user, chars_per_token)

            # approx: ctx_limit ≈ est_prompt - overflow (enligt serverns error)
            ctx_limit_est = max(2048, est_prompt - overflow_i)

            # sänk budget aggressivt + rejäl buffert (eftersom estimator != server tokenizer)
            new_budget = max(512, ctx_limit_est - 1200)

            logger.warning(
                "Meta too long: server_overflow=%s est_prompt=%s => ctx_limit_est~%s. "
                "Budget %s -> %s (attempt %s/%s)",
                overflow_i,
                est_prompt,
                ctx_limit_est,
                budget_tokens,
                new_budget,
                attempt,
                meta_attempts,
            )

            # om vi inte sjunker, halvera för att undvika loop
            if new_budget >= budget_tokens:
                new_budget = max(512, int(budget_tokens * 0.6))

            budget_tokens = new_budget
    else:
        raise RuntimeError(
            f"Meta misslyckades efter {meta_attempts} försök: {last_err}"
        )

    # checkpoint meta-result
    if cp_enabled and meta_path is not None:
        _atomic_write_json(
            meta_path,
            {
                "kind": "meta_result",
                "created_at": int(time.time()),
                "job_id": job_id,
                "checkpoint_key": cp_key,
                "batch_total": len(batches),
                "article_ids": [a.get("id", "") for a in articles],
                "meta": meta,
                "meta_budget_tokens": budget_tokens,
                "batch_article_ids": _batch_article_ids_map(batches),
                "done_batches": _done_batches_payload(done_map, batches),
            },
        )

    # cleanup checkpoints on success
    if cp_enabled:
        try:
            if cp_path is not None:
                cp_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            if meta_path is not None:
                meta_path.unlink(missing_ok=True)
        except Exception:
            pass

    logger.info("Summary done")
    return meta


async def run_resume_from_checkpoint(
    config: Dict[str, Any],
    store: NewsStore,
    llm: LLMClient,
    job_id: int,
) -> str:
    """
    Resume: läs checkpoint för job_id, ladda article_ids från store, kör summarize_batches_then_meta.
    """
    cp_key = _checkpoint_key(job_id, [])
    cp_path = _checkpoint_path(config, cp_key)
    cp = _load_checkpoint(cp_path)
    if not cp:
        raise RuntimeError(f"Ingen checkpoint hittades för job {job_id} ({cp_path})")

    article_ids = cp.get("article_ids") or []
    if not article_ids:
        raise RuntimeError(f"Checkpoint saknar article_ids för job {job_id}")

    articles = store.get_articles_by_ids(article_ids)
    if not articles:
        raise RuntimeError(
            "Kunde inte ladda artiklar från store för checkpointens article_ids"
        )

    by_id = {str(a.get("id")): a for a in articles if a.get("id")}
    ordered = [by_id[i] for i in article_ids if i in by_id]

    return await summarize_batches_then_meta(
        config, ordered, llm=llm, store=store, job_id=job_id
    )
