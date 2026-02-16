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
import hashlib
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import feedparser
import trafilatura
import yaml
from aiolimiter import AsyncLimiter
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from llmClient import LLMClient, create_llm_client
from persistence import NewsStore, create_store


# ----------------------------
# Hash helpers
# ----------------------------
def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def compute_content_hash(title: str, url: str, text: str) -> str:
    base = f"{(title or '').strip()}|{(url or '').strip()}|{normalize_text(text)}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def stable_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def text_clip(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "…"


class RateLimitError(Exception):
    def __init__(
        self, status: int, retry_after: Optional[float] = None, body: str = ""
    ):
        super().__init__(f"HTTP {status} rate-limited")
        self.status = status
        self.retry_after = retry_after
        self.body = body


# ----------------------------
# Prompt loader (from config)
# ----------------------------
def load_prompts(config: Dict[str, Any]) -> Dict[str, str]:
    defaults = {
        "batch_system": None,
        "batch_user_template": None,
        "meta_system": None,
        "meta_user_template": None,
    }

    p = config.get("prompts", {}) or {}
    out = {k: str(p.get(k, defaults[k])) for k in defaults.keys()}
    if None in out:
        raise
    return out


# ----------------------------
# Fetch/extract
# ----------------------------
async def fetch_rss(
    feed_url: str, session: aiohttp.ClientSession
) -> feedparser.FeedParserDict:
    async with session.get(feed_url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
        resp.raise_for_status()
        content = await resp.read()
    return feedparser.parse(content)


def extract_text_from_html(html: str, url: str) -> str:
    extracted = trafilatura.extract(
        html, url=url, include_comments=False, include_tables=False
    )
    return (extracted or "").strip()


async def fetch_article_html(
    url: str, session: aiohttp.ClientSession, timeout_s: int
) -> str:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
        if resp.status == 429:
            ra = resp.headers.get("Retry-After")
            retry_after = float(ra) if ra and ra.isdigit() else None
            body = await resp.text(errors="ignore")
            raise RateLimitError(429, retry_after=retry_after, body=body[:500])
        resp.raise_for_status()
        return await resp.text(errors="ignore")


@retry(
    wait=wait_exponential_jitter(initial=1, max=30),
    stop=stop_after_attempt(6),
    retry=(
        retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(aiohttp.ClientError)
        | retry_if_exception_type(asyncio.TimeoutError)
    ),
)
async def guarded_fetch_article(
    url: str, session: aiohttp.ClientSession, timeout_s: int
) -> str:
    try:
        return await fetch_article_html(url, session, timeout_s)
    except RateLimitError as e:
        if e.retry_after:
            await asyncio.sleep(min(e.retry_after, 60))
        raise


# ----------------------------
# Ingest (RSS -> article text -> store)
# ----------------------------
async def gather_articles_to_store(
    config: Dict[str, Any],
    store: NewsStore,
    job_id: Optional[int] = None,
) -> Tuple[int, int]:
    def set_job(msg: str):
        if job_id is not None:
            store.update_job(job_id, message=msg)

    feeds = config.get("feeds", [])
    max_items = int(config.get("max_items_per_feed", 8))
    timeout_s = int(config.get("article_timeout_s", 20))

    http_limiter = AsyncLimiter(max_rate=6, time_period=1)
    headers = {"User-Agent": "news-summarizer/2.1 (personal; rate-limited)"}

    inserted = 0
    updated = 0

    async with aiohttp.ClientSession(headers=headers) as session:
        for f in feeds:
            name = f["name"]
            feed_url = f["url"]
            set_job(f"Läser RSS: {name}")

            try:
                feed = await fetch_rss(feed_url, session)
            except Exception as e:
                print(f"[WARN] Kunde inte läsa RSS: {name} ({feed_url}) -> {e}")
                continue

            for entry in feed.entries[:max_items]:
                link = getattr(entry, "link", None)
                if not link:
                    continue

                aid = stable_id(link)
                title = (getattr(entry, "title", "") or "").strip()
                published = (
                    getattr(entry, "published", "")
                    or getattr(entry, "updated", "")
                    or ""
                )

                existing = store.get_article(aid)

                try:
                    async with http_limiter:
                        html = await guarded_fetch_article(link, session, timeout_s)
                    text = extract_text_from_html(html, link)
                    if len(text) < 200:
                        continue

                    chash = compute_content_hash(title, link, text)

                    doc = {
                        "id": aid,
                        "source": name,
                        "title": title,
                        "url": link,
                        "published": published,
                        "fetched_at": int(time.time()),
                        "text": text,
                        "content_hash": chash,
                    }

                    if existing is None:
                        doc["summarized"] = False
                        doc["summarized_at"] = None
                        store.upsert_article(doc)
                        inserted += 1
                    else:
                        if existing.get("content_hash") != chash:
                            doc["summarized"] = False
                            doc["summarized_at"] = None
                            store.upsert_article(doc)
                            updated += 1

                except Exception as e:
                    print(f"[WARN] Artikel misslyckades: {link} -> {e}")

    return inserted, updated


# ----------------------------
# Batching
# ----------------------------
def batch_articles(
    articles: List[dict],
    max_chars_per_batch: int,
    max_articles_per_batch: int,
) -> List[List[dict]]:
    batches: List[List[dict]] = []
    current: List[dict] = []
    current_chars = 0

    for a in articles:
        per_article_text = text_clip(a.get("text", ""), 6000)
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
# Summarization (LLM + prompts from config)
# ----------------------------
async def summarize_batches_then_meta(
    config: Dict[str, Any],
    articles: List[dict],
    llm: LLMClient,
    store: NewsStore,
    job_id: Optional[int] = None,
) -> str:
    def set_job(msg: str):
        if job_id is not None:
            store.update_job(job_id, message=msg)

    prompts = load_prompts(config)

    batching = config.get("batching", {})
    max_chars = int(batching.get("max_chars_per_batch", 18000))
    max_n = int(batching.get("max_articles_per_batch", 10))

    batches = batch_articles(articles, max_chars, max_n)

    batch_summaries: List[Tuple[int, str]] = []

    for idx, batch in enumerate(batches, start=1):
        set_job(f"Summerar batch {idx}/{len(batches)}...")

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

        messages = [
            {"role": "system", "content": prompts["batch_system"]},
            {"role": "user", "content": user_content},
        ]
        summary = await llm.chat(messages, temperature=0.2)
        batch_summaries.append((idx, summary))

    set_job("Skapar metasammanfattning...")

    sources_list = [
        f"- {a.get('title', '').strip()} — {a.get('url', '').strip()}" for a in articles
    ]
    sources_text = "\n".join(sources_list)

    batch_text = "\n\n====================\n\n".join(
        [f"Batch {i}:\n{s}" for i, s in batch_summaries]
    )

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
    return meta


# ----------------------------
# Pipeline (orchestrates only)
# ----------------------------
async def run_pipeline(
    config_path: str = "config.yaml", job_id: Optional[int] = None
) -> Optional[int]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    store = create_store(config.get("store", {}))
    llm = create_llm_client(config.get("llm", {}))

    if job_id is not None:
        store.update_job(
            job_id,
            status="running",
            started_at=int(time.time()),
            message="Startar ingest...",
        )

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
