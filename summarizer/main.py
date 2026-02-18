# LICENSE HEADER MANAGED BY add-license-header
#
# BSD 3-Clause License
#
# Copyright (c) 2026, Martin Vesterlund
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from llmClient import LLMClient, create_llm_client
from persistence import NewsStore, create_store
from summarizer.helpers import (
    _atomic_write_json,
    _checkpoint_key,
    _checkpoint_path,
    _extract_overflow_tokens,
    _load_checkpoint,
    _meta_ckpt_path,
    interleave_by_source_oldest_first,
    load_prompts,
    setup_logging,
    text_clip,
    set_job
)
from summarizer.ingest import gather_articles_to_store
from summarizer.token_budget import enforce_budget

setup_logging()
logger = logging.getLogger(__name__)

# ----------------------------
# Batching
# ----------------------------
def batch_articles(
    articles: List[dict],
    max_chars_per_batch: int,
    max_articles_per_batch: int,
    article_clip_chars: int = 2500,
) -> List[List[dict]]:
    batches: List[List[dict]] = []
    current: List[dict] = []
    current_chars = 0

    for a in articles:
        per_article_text = text_clip(a.get("text", ""), article_clip_chars)
        estimated = (
            len(per_article_text)
            + len(a.get("title", ""))
            + len(a.get("url", ""))
            + 200
        )

        if current and (
            current_chars + estimated > max_chars_per_batch
            or len(current) >= max_articles_per_batch
        ):
            batches.append(current)
            current = []
            current_chars = 0

        a2 = dict(a)
        a2["text"] = per_article_text
        current.append(a2)
        current_chars += estimated

    if current:
        batches.append(current)

    return batches


# ----------------------------
# Prompt-too-long helpers + stable resume helpers
# ----------------------------
class PromptTooLongStructural(Exception):
    def __init__(self, overflow_tokens: int):
        super().__init__(f"prompt too long (structural), overflow={overflow_tokens}")
        self.overflow_tokens = overflow_tokens


def _choose_trim_action(overflow_tokens: int, structural_threshold: int) -> str:
    if overflow_tokens <= 200:
        return "word_trim"
    if overflow_tokens <= structural_threshold:
        return "drop_one_article"
    return "drop_multiple_articles"


def trim_text_tail_by_words(
    text: str, remove_tokens: int, *, chars_per_token: float
) -> str:
    """
    Tar bort från slutet men alltid på whitespace (ordgräns).
    """
    s = text or ""
    if not s:
        return s

    remove_chars = int(max(1, remove_tokens) * chars_per_token)
    if remove_chars >= len(s):
        return ""

    target = max(0, len(s) - remove_chars)
    cut = max(
        s.rfind(" ", 0, target), s.rfind("\n", 0, target), s.rfind("\t", 0, target)
    )
    if cut <= 0:
        cut = target

    return s[:cut].rstrip() + "\n\n[TRUNCATED FOR CONTEXT WINDOW]\n"


def _trim_last_user_word_boundary(
    messages: List[Dict[str, str]], remove_tokens: int, *, chars_per_token: float
) -> List[Dict[str, str]]:
    out = [dict(m) for m in messages]
    idx = None
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            idx = i
            break
    if idx is None:
        return out
    content = out[idx].get("content") or ""
    out[idx]["content"] = trim_text_tail_by_words(
        content, remove_tokens, chars_per_token=chars_per_token
    )
    return out


def _estimate_article_chars(a: dict) -> int:
    return (
        len(a.get("text", "") or "")
        + len(a.get("title", "") or "")
        + len(a.get("url", "") or "")
        + 200
    )


def _batch_chars(batch: List[dict]) -> int:
    return sum(_estimate_article_chars(x) for x in batch)


def _can_fit_in_batch(
    batch: List[dict], a: dict, *, max_chars_per_batch: int, max_articles_per_batch: int
) -> bool:
    if max_articles_per_batch and len(batch) >= max_articles_per_batch:
        return False
    return (_batch_chars(batch) + _estimate_article_chars(a)) <= max_chars_per_batch


def _move_article_to_tail_batch(
    batches: List[List[dict]],
    a: dict,
    *,
    max_chars_per_batch: int,
    max_articles_per_batch: int,
    avoid_batch: Optional[List[dict]] = None,
) -> None:
    """
    Flytta till sista batch om plats, annars ny batch.
    Viktigt: undvik att lägga tillbaka i samma batch (tail-loop).
    """
    if not batches:
        batches.append([a])
        return

    last = batches[-1]
    if avoid_batch is not None and last is avoid_batch:
        batches.append([a])
        return

    if _can_fit_in_batch(
        last,
        a,
        max_chars_per_batch=max_chars_per_batch,
        max_articles_per_batch=max_articles_per_batch,
    ):
        last.append(a)
    else:
        batches.append([a])


def _batch_article_ids_map(batches_local: List[List[dict]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for i, b in enumerate(batches_local, start=1):
        out[str(i)] = [str(a.get("id", "")) for a in b if a.get("id")]
    return out


def _done_batches_payload(
    done_map_local: Dict[int, str], batches_local: List[List[dict]]
) -> Dict[str, Dict[str, Any]]:
    ids_map = _batch_article_ids_map(batches_local)
    payload: Dict[str, Dict[str, Any]] = {}
    for k, v in sorted(done_map_local.items()):
        sk = str(k)
        payload[sk] = {"article_ids": ids_map.get(sk, []), "summary": v}
    return payload


def _done_map_from_done_batches(cp_done_batches: Dict[str, Any]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    if not isinstance(cp_done_batches, dict):
        return out
    for k, entry in cp_done_batches.items():
        try:
            idx = int(k)
        except Exception:
            continue
        if isinstance(entry, dict) and isinstance(entry.get("summary"), str):
            out[idx] = entry["summary"]
    return out


def _build_batches_from_checkpoint(
    batch_article_ids: Dict[str, Any],
    all_articles: List[dict],
    *,
    clip_chars: int,
) -> List[List[dict]]:
    """
    Återskapa batch-indelning EXAKT från checkpointens batch_article_ids.
    """
    by_id: Dict[str, dict] = {}
    for a in all_articles:
        aid = a.get("id")
        if aid:
            by_id[str(aid)] = a

    def key_int(k: str) -> int:
        try:
            return int(k)
        except Exception:
            return 10**9

    rebuilt: List[List[dict]] = []
    missing: List[str] = []

    for k in sorted(batch_article_ids.keys(), key=key_int):
        ids = batch_article_ids.get(k)
        if not isinstance(ids, list):
            continue

        batch: List[dict] = []
        for aid in ids:
            aid_s = str(aid)
            a = by_id.get(aid_s)
            if not a:
                missing.append(aid_s)
                continue
            a2 = dict(a)
            a2["text"] = text_clip(a2.get("text", ""), clip_chars)
            batch.append(a2)

        if batch:
            rebuilt.append(batch)

    if missing:
        logger.warning(
            "Resume: %d article_ids saknas i store (hoppas över). Ex: %s",
            len(missing),
            ", ".join(missing[:5]),
        )

    if not rebuilt:
        raise RuntimeError(
            "Resume: batch_article_ids fanns men inga batcher kunde återskapas."
        )

    return rebuilt


def _budgeted_meta_user(
    *,
    prompts: Dict[str, str],
    batch_summaries: List[Tuple[int, str]],
    sources_text: str,
    budget_tokens: int,  # <-- NYTT: styr budget direkt
    chars_per_token: float,
) -> str:
    """
    Bygg meta-user inom en *explicit* tokenbudget.
    Skalar ner via clip-levels, käll-clip och decimering.
    """

    def est_tokens(s: str) -> int:
        return max(1, int(len(s) / chars_per_token))

    def render(batch_block: str, src: str) -> str:
        return prompts["meta_user_template"].format(
            batch_summaries=batch_block,
            sources_list=src,
        )

    # Aggressivare stegar än innan (särskilt decimations)
    sources_levels = [len(sources_text), 6000, 3500, 2000, 1200, 700]
    clip_levels = [4200, 3200, 2400, 1800, 1200, 900, 700, 500, 350, 250]
    decimations = [1, 2, 3, 4, 6, 8, 12]  # 1=alla batcher, 2=varannan, ...

    summaries_desc = sorted(batch_summaries, key=lambda x: x[0], reverse=True)

    for src_lim in sources_levels:
        src2 = (
            sources_text
            if len(sources_text) <= src_lim
            else (sources_text[:src_lim].rstrip() + "…")
        )

        for dec in decimations:
            subset = summaries_desc[::dec]

            for clip_n in clip_levels:
                parts: List[str] = []

                for i, s in subset:
                    s2 = (s or "").strip()
                    if len(s2) > clip_n:
                        s2 = s2[:clip_n].rstrip() + "…"

                    candidate_parts = parts + [f"Batch {i}:\n{s2}"]
                    candidate_block = "\n\n====================\n\n".join(
                        candidate_parts
                    )
                    candidate_user = render(candidate_block, src2)

                    if est_tokens(candidate_user) <= budget_tokens:
                        parts = candidate_parts
                    else:
                        if parts:
                            break
                        continue

                if parts:
                    return render("\n\n====================\n\n".join(parts), src2)

    # Sista utväg: utan batch-summaries
    return render(
        "[Inga batch-summaries kunde inkluderas inom context-budget.]",
        (sources_text[:700].rstrip() + "…")
        if len(sources_text) > 700
        else sources_text,
    )


# ----------------------------
# Summarization (LLM + stable checkpoint/resume + budgeted meta)
# ----------------------------
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

    async def chat_guarded(
        messages: List[Dict[str, str]], *, temperature: float = 0.2
    ) -> str:
        """
        - enforce_budget (best effort) + log
        - om overflow <= 200 => trim sista user (word boundary) och retry
        - annars => raise PromptTooLongStructural för batch/meta-logik
        """
        attempt = 1
        current, est, budget = enforce_budget(
            messages,
            max_context_tokens=max_ctx,
            max_output_tokens=max_out,
            safety_margin_tokens=margin,
        )

        while True:
            try:
                return await llm.chat(current, temperature=temperature)
            except Exception as e:
                msg = str(e).lower()
                overflow = _extract_overflow_tokens(e)

                if (
                    "prompt too long" in msg
                    or "max context" in msg
                    or "context length" in msg
                ):
                    if attempt >= max_attempts:
                        raise

                    if overflow is None:
                        # okänt overflow: trim schablon
                        current = _trim_last_user_word_boundary(
                            current, 2048, chars_per_token=chars_per_token
                        )
                        attempt += 1
                        continue

                    overflow_i = int(overflow)
                    action = _choose_trim_action(overflow_i, structural_threshold)

                    if action == "word_trim":
                        remove_tokens = overflow_i + 1024
                        logger.warning(
                            "LLM prompt too long: overflow=%s action=word_trim attempt=%s/%s",
                            overflow_i,
                            attempt,
                            max_attempts,
                        )
                        current = _trim_last_user_word_boundary(
                            current, remove_tokens, chars_per_token=chars_per_token
                        )
                        attempt += 1
                        continue

                    raise PromptTooLongStructural(overflow_i)

                raise

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
                    store
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
                            store
                        )
                except Exception:
                    done_map = {}

    batch_summaries: List[Tuple[int, str]] = [
        (i, done_map[i]) for i in sorted(done_map.keys())
    ]

    def clip_line(s: str, n: int) -> str:
        s = (s or "").strip()
        return s if len(s) <= n else s[:n].rstrip() + "…"

    # --- kör batches (med structural trim + tail-loop-skydd) ---
    idx = 1
    while idx <= len(batches):
        if idx in done_map:
            idx += 1
            continue

        batch = batches[idx - 1]
        set_job(f"Summerar batch {idx}/{len(batches)}...", job_id, store)

        def build_messages_for_batch(
            batch_index: int, batch_items: List[dict]
        ) -> List[Dict[str, str]]:
            parts = []
            for i, a in enumerate(batch_items, start=1):
                parts.append(
                    f"[{i}] {a.get('title', '')}\n"
                    f"Källa: {a.get('source', '')}\n"
                    f"Publicerad: {a.get('published', '')}\n"
                    f"URL: {a.get('url', '')}\n\n"
                    f"{a.get('text', '')}"
                )
            corpus = "\n\n---\n\n".join(parts)
            user_content = prompts["batch_user_template"].format(
                batch_index=batch_index,
                batch_total=len(batches),
                articles_corpus=corpus,
            )
            return [
                {"role": "system", "content": prompts["batch_system"]},
                {"role": "user", "content": user_content},
            ]

        # retry-loop för samma batch vid PromptTooLongStructural
        while True:
            try:
                summary = await chat_guarded(
                    build_messages_for_batch(idx, batch), temperature=0.2
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

    sources_list = []
    for a in articles:
        title = clip_line(a.get("title", ""), meta_sources_clip_chars)
        url = (a.get("url") or "").strip()
        sources_list.append(f"- {title} — {url}")
    sources_text = "\n".join(sources_list)

    # Startbudget enligt config, men vi kommer sänka den om servern klagar
    budget_tokens = max(512, max_ctx - max_out - margin)

    def _est_user_tokens(s: str) -> int:
        return max(1, int(len(s) / chars_per_token))

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
            est_prompt = _est_user_tokens(meta_user)

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


# ----------------------------
# Pipeline (orchestrates only)
# ----------------------------
async def run_pipeline(
    config_path: str = "config.yaml", job_id: Optional[int] = None
) -> Optional[int]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    store = create_store(config.get("store", {}))
    llm = create_llm_client(config)

    if job_id is not None:
        store.update_job(
            job_id,
            status="running",
            started_at=int(time.time()),
            message="Startar ingest...",
        )
        logger.info(f"Startar ingest job {job_id}")

    ins, upd = await gather_articles_to_store(config, store, job_id=job_id)

    if job_id is not None:
        store.update_job(
            job_id,
            message=f"Ingest klart. Inserted={ins}, Updated={upd}. Förbereder summering...",
        )

    to_sum = store.list_unsummarized_articles(limit=200)
    if not to_sum:
        if job_id is not None:
            store.update_job(
                job_id,
                status="done",
                finished_at=int(time.time()),
                message="Klart: inga nya/ändrade artiklar att summera.",
            )
        return None

    summary = await summarize_batches_then_meta(
        config, to_sum, llm=llm, store=store, job_id=job_id
    )

    ids = [a["id"] for a in to_sum]
    summary_id = store.save_summary(summary, ids)
    store.mark_articles_summarized(ids)

    if job_id is not None:
        store.update_job(
            job_id,
            status="done",
            finished_at=int(time.time()),
            message=f"Klart: summerade {len(ids)} artiklar.",
            summary_id=summary_id,
        )

    return summary_id


if __name__ == "__main__":
    asyncio.run(run_pipeline("config.yaml"))
