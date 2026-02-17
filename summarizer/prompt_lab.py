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
from typing import Any, Dict, List, Optional, Tuple

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


def _hard_trim_last_user(
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
    remove_chars = int(remove_tokens * chars_per_token)
    if remove_chars <= 0:
        remove_chars = int(512 * chars_per_token)

    if remove_chars >= len(content):
        out[idx]["content"] = "[TRUNCATED FOR CONTEXT WINDOW]\n"
        return out

    out[idx]["content"] = (
        content[: len(content) - remove_chars] + "\n\n[TRUNCATED FOR CONTEXT WINDOW]\n"
    )
    return out


def _estimate_tokens(text: str, *, chars_per_token: float) -> int:
    return max(1, int(len(text) / chars_per_token))


def _messages_to_text(messages: List[Dict[str, str]]) -> str:
    parts = []
    for m in messages:
        parts.append(m.get("role", "user") + ":\n" + (m.get("content") or ""))
    return "\n\n".join(parts)


def _enforce_budget(
    messages: List[Dict[str, str]],
    *,
    max_context_tokens: int,
    max_output_tokens: int,
    safety_margin_tokens: int,
    chars_per_token: float,
) -> Tuple[List[Dict[str, str]], int, int]:
    budget = max_context_tokens - max_output_tokens - safety_margin_tokens
    if budget < 256:
        budget = 256

    est = _estimate_tokens(_messages_to_text(messages), chars_per_token=chars_per_token)
    if est <= budget:
        return messages, est, budget

    out = [dict(m) for m in messages]
    idx = _last_user_index(out)
    if idx is None:
        return messages, est, budget

    target_chars = int(budget * chars_per_token)
    content = out[idx].get("content") or ""
    if len(content) > target_chars:
        out[idx]["content"] = (
            content[:target_chars] + "\n\n[TRUNCATED FOR CONTEXT WINDOW]\n"
        )
    est2 = _estimate_tokens(_messages_to_text(out), chars_per_token=chars_per_token)
    return out, est2, budget


async def chat_guarded(
    llm: Any,
    config: Dict[str, Any],
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.2,
) -> str:
    """Robust chat för prompt-lab: budget + overflow-trim-retry."""
    llm_cfg = config.get("llm") or {}
    max_ctx = int(llm_cfg.get("context_window_tokens", 32768))
    max_out = int(llm_cfg.get("max_output_tokens", 700))
    margin = int(llm_cfg.get("prompt_safety_margin", 1024))
    chars_per_token = float(llm_cfg.get("token_chars_per_token", 2.4))
    max_attempts = int(llm_cfg.get("prompt_too_long_max_attempts", 6))

    msgs2, est, budget = _enforce_budget(
        messages,
        max_context_tokens=max_ctx,
        max_output_tokens=max_out,
        safety_margin_tokens=margin,
        chars_per_token=chars_per_token,
    )
    logger.info(
        f"Prompt-lab LLM budget: est_prompt_tokens={est} budget_tokens={budget}"
    )

    attempt = 1
    current = msgs2
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
                    remove_tokens = overflow + 512
                    logger.warning(
                        "Prompt-lab: prompt too long. overflow=%s tokens. attempt=%s/%s. trimming ~%s tokens...",
                        overflow,
                        attempt,
                        max_attempts,
                        remove_tokens,
                    )
                    current = _hard_trim_last_user(
                        current, remove_tokens, chars_per_token=chars_per_token
                    )
                else:
                    logger.warning(
                        "Prompt-lab: prompt too long (no overflow parsed). attempt=%s/%s. trimming fixed chunk...",
                        attempt,
                        max_attempts,
                    )
                    current = _hard_trim_last_user(
                        current, 1024, chars_per_token=chars_per_token
                    )
                attempt += 1
                continue
            raise


def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


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
    Lagrar temporärt resultat under en tydlig "temp"-nyckel kopplad till job_id.

    Returnerar temp_summary_id (ofta samma som job_id beroende på store-implementation).
    """

    batching = config.get("batching") or {}
    max_chars = int(batching.get("max_chars_per_batch", 18000))
    max_n = int(batching.get("max_articles_per_batch", 10))
    article_clip_chars = int(batching.get("article_clip_chars", 2500))
    meta_batch_clip_chars = int(batching.get("meta_batch_clip_chars", 2500))
    meta_sources_clip_chars = int(batching.get("meta_sources_clip_chars", 140))

    # Använd samma batchning som main, om du har en utility. Annars enkel variant:
    def batch_articles_local(items: List[dict]) -> List[List[dict]]:
        batches: List[List[dict]] = []
        current: List[dict] = []
        current_chars = 0
        for a in items:
            t = _clip(a.get("text", ""), article_clip_chars)
            estimated = len(t) + len(a.get("title", "")) + len(a.get("url", ""))
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

    batches = batch_articles_local(articles)

    done_map: Dict[int, str] = {}

    for idx, batch in enumerate(batches, start=1):
        store.update_job(
            job_id, message=f"Prompt-lab: summerar batch {idx}/{len(batches)}..."
        )

        parts = []
        for i, a in enumerate(batch, start=1):
            parts.append(
                f"[{i}] {a.get('title', '')}\n"
                f"Källa: {a.get('source', '')}\n"
                f"Publicerad: {a.get('published', '')}\n"
                f"URL: {a.get('url', '')}\n\n"
                f"{a.get('_clip_text', '')}"
            )
        corpus = "\n\n---\n\n".join(parts)

        user_content = prompts["batch_user_template"].format(
            batch_index=idx,
            batch_total=len(batches),
            articles_corpus=corpus,
        )
        messages = [
            {"role": "system", "content": prompts["batch_system"]},
            {"role": "user", "content": user_content},
        ]

        summary = await chat_guarded(llm, config, messages, temperature=0.2)
        done_map[idx] = summary

        # checkpoint per batch (i store som temp, så UI kan visa progress om du vill)
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

    meta = await chat_guarded(llm, config, meta_messages, temperature=0.2)

    # slutligt temp-resultat
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
