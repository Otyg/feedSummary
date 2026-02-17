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

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from llmClient import LLMClient
from persistence import NewsStore
from summarizer.helpers import clip_text

logger = logging.getLogger(__name__)


def _clip(s: str, n: int) -> str:
    return clip_text(s=s, n=n)


def build_prompts_from_form(
    form: Dict[str, str], base_config: Dict[str, Any]
) -> Dict[str, str]:
    """
    Tar prompts från webform (om givna) annars från config.yaml.
    """
    defaults = base_config.get("prompts") or {}
    keys = ["batch_system", "batch_user_template", "meta_system", "meta_user_template"]

    out: Dict[str, str] = {}
    for k in keys:
        v = form.get(k)
        if v is None or v.strip() == "":
            v = str(defaults.get(k, ""))
        out[k] = str(v)
    return out


def select_articles_for_promptlab(
    store: NewsStore, summary_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Väljer artiklar från en existerande summary (latest eller specifik).
    Returnerar {summary_id, created_at, article_ids, articles}.
    """
    if summary_id is None:
        s = store.get_latest_summary()
        if not s:
            raise ValueError("Ingen sparad summary finns att använda för prompt-lab.")
    else:
        s = store.get_summary(summary_id)
        if not s:
            raise ValueError("Kunde inte hitta angiven summary_id.")

    article_ids = s.get("article_ids", []) or []
    articles = store.get_articles_by_ids(article_ids)

    return {
        "source_summary_id": s.get("id"),
        "source_created_at": s.get("created_at"),
        "article_ids": article_ids,
        "articles": articles,
    }


def batch_articles(
    articles: List[dict],
    max_chars_per_batch: int,
    max_articles_per_batch: int,
    article_clip_chars: int,
) -> List[List[dict]]:
    batches: List[List[dict]] = []
    current: List[dict] = []
    current_chars = 0

    for a in articles:
        text = _clip(a.get("text", ""), article_clip_chars)
        estimated = len(text) + len(a.get("title", "")) + len(a.get("url", "")) + 200

        if current and (
            current_chars + estimated > max_chars_per_batch
            or len(current) >= max_articles_per_batch
        ):
            batches.append(current)
            current = []
            current_chars = 0

        a2 = dict(a)
        a2["text"] = text
        current.append(a2)
        current_chars += estimated

    if current:
        batches.append(current)

    return batches


async def run_promptlab_summarization(
    *,
    config: Dict[str, Any],
    prompts: Dict[str, str],
    store: NewsStore,
    llm: LLMClient,
    job_id: int,
    source_summary_id: int,
    articles: List[dict],
) -> str:
    """
    Kör batch + meta enligt prompts, men sparar endast som temp_summary kopplat till job_id.
    """
    batching = config.get("batching", {}) or {}
    max_chars = int(batching.get("max_chars_per_batch", 8000))
    max_n = int(batching.get("max_articles_per_batch", 4))
    article_clip_chars = int(batching.get("article_clip_chars", 2000))

    meta_batch_clip_chars = int(batching.get("meta_batch_clip_chars", 2500))
    meta_sources_clip_chars = int(batching.get("meta_sources_clip_chars", 140))

    def set_job(msg: str):
        store.update_job(job_id, message=msg)

    batches = batch_articles(articles, max_chars, max_n, article_clip_chars)

    batch_summaries: List[str] = []
    for idx, batch in enumerate(batches, start=1):
        set_job(f"Prompt-lab: summerar batch {idx}/{len(batches)}...")
        logger.info(f"Prompt-lab: summerar batch {idx}/{len(batches)}...")
        parts = []
        for i, a in enumerate(batch, start=1):
            parts.append(
                f"[{i}] {a.get('title', '')}\nKälla: {a.get('source', '')}\n"
                f"Publicerad: {a.get('published', '')}\nURL: {a.get('url', '')}\n\n{a.get('text', '')}"
            )
        corpus = "\n\n---\n\n".join(parts)

        user_content = prompts["batch_user_template"].format(
            batch_index=idx,
            batch_total=len(batches),
            articles_corpus=corpus,
        )

        summary = await llm.chat(
            [
                {"role": "system", "content": prompts["batch_system"]},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        )
        batch_summaries.append(summary)

    set_job("Prompt-lab: skapar metasammanfattning...")
    logger.info("Prompt-lab: skapar metasammanfattning...")
    # Klipp meta-input för att hålla det snällt för modellen
    clipped_batches = []
    for i, s in enumerate(batch_summaries, start=1):
        clipped_batches.append(f"Batch {i}:\n{_clip(s, meta_batch_clip_chars)}")
    batch_text = "\n\n====================\n\n".join(clipped_batches)

    sources_list = []
    for a in articles:
        title = _clip(a.get("title", ""), meta_sources_clip_chars)
        sources_list.append(f"- {title} — {a.get('url', '').strip()}")
    sources_text = "\n".join(sources_list)

    meta_user = prompts["meta_user_template"].format(
        batch_summaries=batch_text,
        sources_list=sources_text,
    )

    meta = await llm.chat(
        [
            {"role": "system", "content": prompts["meta_system"]},
            {"role": "user", "content": meta_user},
        ],
        temperature=0.2,
    )

    # Spara temp-resultatet i DB (tydligt temporärt)
    store.save_temp_summary(
        job_id=job_id,
        summary_text=meta,
        meta={
            "kind": "prompt_lab",
            "source_summary_id": source_summary_id,
            "created_at": int(time.time()),
            "used_article_count": len(articles),
            "prompts": {
                "batch_system": prompts["batch_system"],
                "batch_user_template": prompts["batch_user_template"],
                "meta_system": prompts["meta_system"],
                "meta_user_template": prompts["meta_user_template"],
            },
        },
    )

    return meta
