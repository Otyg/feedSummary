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

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("FeedSummarizer")

_PROMPT_TOO_LONG_RE = re.compile(
    r"exceeded max context length by\s+(\d+)\s+tokens", re.IGNORECASE
)


def _extract_overflow_tokens(err: Exception) -> Optional[int]:
    m = _PROMPT_TOO_LONG_RE.search(str(err))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _last_user_index(msgs: List[Dict[str, str]]) -> Optional[int]:
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            return i
    return None


def _trim_last_user_word_boundary(
    msgs: List[Dict[str, str]],
    remove_tokens: int,
    *,
    chars_per_token: float,
) -> List[Dict[str, str]]:
    out = [dict(m) for m in msgs]
    idx = _last_user_index(out)
    if idx is None:
        return out

    content = out[idx].get("content") or ""
    if not content:
        return out

    remove_chars = int(max(1, remove_tokens) * chars_per_token)
    if remove_chars >= len(content):
        out[idx]["content"] = "[TRUNCATED FOR CONTEXT WINDOW]\n"
        return out

    target = len(content) - remove_chars
    if target < 0:
        target = 0

    cut_space = content.rfind(" ", 0, target)
    cut_nl = content.rfind("\n", 0, target)
    cut_tab = content.rfind("\t", 0, target)
    cut = max(cut_space, cut_nl, cut_tab)
    if cut <= 0:
        cut = target

    out[idx]["content"] = (
        content[:cut].rstrip() + "\n\n[TRUNCATED FOR CONTEXT WINDOW]\n"
    )
    return out


def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


def _estimate_article_chars(a: dict) -> int:
    # _clip_text används om den finns
    t = a.get("_clip_text")
    if not isinstance(t, str):
        t = a.get("text", "") or ""
    return len(t) + len(a.get("title", "")) + len(a.get("url", "")) + 200


def _batch_chars(batch: List[dict]) -> int:
    return sum(_estimate_article_chars(x) for x in batch)


def _can_fit_in_batch(
    batch: List[dict], a: dict, *, max_chars: int, max_n: int
) -> bool:
    if max_n and len(batch) >= max_n:
        return False
    return (_batch_chars(batch) + _estimate_article_chars(a)) <= max_chars


def _move_article_to_tail_batch(
    batches: List[List[dict]], a: dict, *, max_chars: int, max_n: int
) -> None:
    if not batches:
        batches.append([a])
        return
    last = batches[-1]
    if _can_fit_in_batch(last, a, max_chars=max_chars, max_n=max_n):
        last.append(a)
    else:
        batches.append([a])


def _choose_trim_action(overflow_tokens: int, structural_threshold: int) -> str:
    if overflow_tokens <= 200:
        return "word_trim"
    if overflow_tokens <= structural_threshold:
        return "drop_one_article"
    return "drop_multiple_articles"


class PromptTooLongStructural(Exception):
    def __init__(self, overflow_tokens: int):
        super().__init__(f"prompt too long (structural), overflow={overflow_tokens}")
        self.overflow_tokens = overflow_tokens


async def chat_guarded(
    llm: Any,
    config: Dict[str, Any],
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.2,
) -> str:
    """
    Prompt-lab chat: overflow-driven.
    - small overflow: word-trim här
    - medium/large: signalera structural till batch-loopen
    """
    llm_cfg = config.get("llm") or {}
    chars_per_token = float(llm_cfg.get("token_chars_per_token", 2.4))
    max_attempts = int(llm_cfg.get("prompt_too_long_max_attempts", 6))
    structural_threshold = int(
        llm_cfg.get("prompt_too_long_structural_threshold_tokens", 1200)
    )

    attempt = 1
    current = messages

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

                if overflow:
                    action = _choose_trim_action(int(overflow), structural_threshold)
                    if action == "word_trim":
                        remove_tokens = int(overflow) + 512
                        logger.warning(
                            "Prompt-lab prompt too long: overflow=%s action=word_trim attempt=%s/%s remove_tokens~%s",
                            overflow,
                            attempt,
                            max_attempts,
                            remove_tokens,
                        )
                        current = _trim_last_user_word_boundary(
                            current, remove_tokens, chars_per_token=chars_per_token
                        )
                        attempt += 1
                        continue

                    # medium/large: låt batch-loopen flytta artiklar
                    raise PromptTooLongStructural(int(overflow))

                # overflow okänd → word-trim schablon
                logger.warning(
                    "Prompt-lab prompt too long (no overflow parsed): trimming fixed chunk"
                )
                current = _trim_last_user_word_boundary(
                    current, 1024, chars_per_token=chars_per_token
                )
                attempt += 1
                continue

            raise


async def run_promptlab_summarization(
    *,
    config: Dict[str, Any],
    prompts: Dict[str, str],
    store: Any,
    llm: Any,
    job_id: int,
    source_summary_id: int,
    articles: List[dict],
) -> int:
    """
    Kör prompt-lab på befintliga artiklar (ingen ny ingest).
    Lagrar temporärt resultat under job_id via store.put_temp_summary(job_id, payload).
    """

    batching = config.get("batching") or {}
    max_chars = int(batching.get("max_chars_per_batch", 18000))
    max_n = int(batching.get("max_articles_per_batch", 10))
    article_clip_chars = int(batching.get("article_clip_chars", 2500))
    meta_batch_clip_chars = int(batching.get("meta_batch_clip_chars", 2500))
    meta_sources_clip_chars = int(batching.get("meta_sources_clip_chars", 140))

    llm_cfg = config.get("llm") or {}
    chars_per_token = float(llm_cfg.get("token_chars_per_token", 2.4))
    structural_threshold = int(
        llm_cfg.get("prompt_too_long_structural_threshold_tokens", 1200)
    )

    def batch_articles_local(items: List[dict]) -> List[List[dict]]:
        batches: List[List[dict]] = []
        current: List[dict] = []
        current_chars = 0
        for a in items:
            t = _clip(a.get("text", ""), article_clip_chars)
            estimated = len(t) + len(a.get("title", "")) + len(a.get("url", "")) + 200
            if current and (
                current_chars + estimated > max_chars or len(current) >= max_n
            ):
                batches.append(current)
                current = []
                current_chars = 0
            aa = dict(a)
            aa["_clip_text"] = t
            current.append(aa)
            current_chars += estimated
        if current:
            batches.append(current)
        return batches

    def build_messages_for_batch(
        batch_index: int, batch_total: int, batch_items: List[dict]
    ) -> List[Dict[str, str]]:
        parts = []
        for i, a in enumerate(batch_items, start=1):
            parts.append(
                f"[{i}] {a.get('title', '')}\n"
                f"Källa: {a.get('source', '')}\n"
                f"Publicerad: {a.get('published', '')}\n"
                f"URL: {a.get('url', '')}\n\n"
                f"{a.get('_clip_text', '')}"
            )
        corpus = "\n\n---\n\n".join(parts)
        user_content = prompts["batch_user_template"].format(
            batch_index=batch_index,
            batch_total=batch_total,
            articles_corpus=corpus,
        )
        return [
            {"role": "system", "content": prompts["batch_system"]},
            {"role": "user", "content": user_content},
        ]

    batches = batch_articles_local(articles)
    done_map: Dict[int, str] = {}

    idx = 1
    while idx <= len(batches):
        store.update_job(
            job_id, message=f"Prompt-lab: summerar batch {idx}/{len(batches)}..."
        )
        batch = batches[idx - 1]

        while True:
            messages = build_messages_for_batch(idx, len(batches), batch)
            try:
                summary = await chat_guarded(llm, config, messages, temperature=0.2)
                break
            except PromptTooLongStructural as e:
                overflow = int(getattr(e, "overflow_tokens", 0) or 0)
                action = _choose_trim_action(overflow, structural_threshold)

                if len(batch) <= 1:
                    # batch med 1 artikel → klipp text hårdare
                    a0 = batch[0]
                    t = a0.get("_clip_text", "") or ""
                    new_len = max(800, int(len(t) * 0.6))
                    a0["_clip_text"] = _clip(t, new_len)
                    logger.warning(
                        "Prompt-lab batch %s: 1 artikel overflow=%s. Klipper text hårdare och retry.",
                        idx,
                        overflow,
                    )
                    continue

                target_remove_tokens = overflow + 512
                target_remove_chars = int(target_remove_tokens * chars_per_token)

                removed_count = 0
                removed_chars = 0

                if action == "drop_one_article":
                    a = batch.pop()
                    removed_count = 1
                    removed_chars = _estimate_article_chars(a)
                    _move_article_to_tail_batch(
                        batches, a, max_chars=max_chars, max_n=max_n
                    )
                else:
                    while len(batch) > 1 and removed_chars < target_remove_chars:
                        a = batch.pop()
                        removed_count += 1
                        removed_chars += _estimate_article_chars(a)
                        _move_article_to_tail_batch(
                            batches, a, max_chars=max_chars, max_n=max_n
                        )

                logger.warning(
                    "Prompt-lab structural: overflow=%s action=%s removed=%s (chars~%s) from batch=%s. "
                    "Moved to tail. Retrying same batch.",
                    overflow,
                    action,
                    removed_count,
                    removed_chars,
                    idx,
                )
                continue

        done_map[idx] = summary

        # partial temp-save
        store.put_temp_summary(
            job_id,
            {
                "kind": "prompt_lab_partial",
                "job_id": job_id,
                "source_summary_id": source_summary_id,
                "created_at": int(time.time()),
                "batch_done": idx,
                "batch_total": len(batches),
                "batch_summaries": {str(k): v for k, v in sorted(done_map.items())},
                "meta": {"prompts": prompts},
            },
        )

        idx += 1

    store.update_job(job_id, message="Prompt-lab: skapar metasammanfattning...")

    sources_list = [
        f"- {_clip(a.get('title', ''), meta_sources_clip_chars)} — {str(a.get('url', '')).strip()}"
        for a in articles
    ]
    sources_text = "\n".join(sources_list)

    batch_text = "\n\n====================\n\n".join(
        [
            f"Batch {i}:\n{_clip(s, meta_batch_clip_chars)}"
            for i, s in sorted(done_map.items(), key=lambda x: x[0])
        ]
    )

    meta_user = prompts["meta_user_template"].format(
        batch_summaries=batch_text,
        sources_list=sources_text,
    )
    meta_messages = [
        {"role": "system", "content": prompts["meta_system"]},
        {"role": "user", "content": meta_user},
    ]

    # Meta: vid structural → gör word-trim fallback
    try:
        meta = await chat_guarded(llm, config, meta_messages, temperature=0.2)
    except PromptTooLongStructural as e:
        overflow = int(getattr(e, "overflow_tokens", 0) or 0)
        logger.warning(
            "Prompt-lab meta structural overflow=%s. Faller tillbaka till word-trim i meta.",
            overflow,
        )
        trimmed = _trim_last_user_word_boundary(
            meta_messages, overflow + 2048, chars_per_token=chars_per_token
        )
        meta = await llm.chat(trimmed, temperature=0.2)

    store.put_temp_summary(
        job_id,
        {
            "kind": "prompt_lab_result",
            "job_id": job_id,
            "source_summary_id": source_summary_id,
            "created_at": int(time.time()),
            "summary": meta,
            "meta": {"prompts": prompts},
            "batch_summaries": {str(k): v for k, v in sorted(done_map.items())},
        },
    )

    return job_id
