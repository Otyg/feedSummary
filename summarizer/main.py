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
import datetime
import logging
import re
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
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
from summarizer.helpers import (
    RateLimitError,
    _atomic_write_json,
    _checkpoint_key,
    _checkpoint_path,
    _load_checkpoint,
    _meta_ckpt_path,
    compute_content_hash,
    setup_logging,
    stable_id,
    text_clip,
)
from summarizer.token_budget import enforce_budget

setup_logging()
logger = logging.getLogger(__name__)

_PROMPT_TOO_LONG_RE = re.compile(
    r"exceeded max context length by\s+(\d+)\s+tokens", re.IGNORECASE
)

_DURATION_RE = re.compile(r"^\s*(\d+)\s*([mhdwy])\s*$", re.IGNORECASE)


def _extract_overflow_tokens(err: Exception) -> Optional[int]:
    m = _PROMPT_TOO_LONG_RE.search(str(err))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def parse_lookback_to_seconds(s: str) -> int:
    """
    "90m" -> 5400, "24h" -> 86400, "3d" -> 259200, "2w" -> 1209600
    """
    if not s:
        raise ValueError("lookback är tom")
    m = _DURATION_RE.match(s)
    if not m:
        raise ValueError(
            f"Ogiltigt lookback-format: {s!r} (förväntar t.ex. 90m, 24h, 3d, 2w)"
        )
    n = int(m.group(1))
    unit = m.group(2).lower()
    HOURS = 60 * 60
    DAYS = 24 * HOURS
    WEEKS = 7 * DAYS
    MONTHS = 4 * WEEKS
    YEARS = 12 * MONTHS
    if unit == "h":
        return n * HOURS
    if unit == "d":
        return n * DAYS
    if unit == "w":
        return n * WEEKS
    if unit == "m":
        return n * MONTHS
    if unit == "y":
        return n * YEARS
    raise ValueError(f"Okänd enhet: {unit}")


def entry_published_ts(entry: feedparser.FeedParserDict) -> Optional[int]:
    """
    Försök få ett unix-timestamp för entry.
    Prioriterar feedparser's *_parsed (struct_time) men kan även parse:a text.
    """
    for attr in ("published_parsed", "updated_parsed"):
        st = getattr(entry, attr, None)
        if st:
            try:
                return int(time.mktime(st))
            except Exception:
                pass

    for attr in ("published", "updated"):
        s = getattr(entry, attr, None)
        if s:
            try:
                dt = parsedate_to_datetime(s)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.timezone.utc)
                return int(dt.timestamp())
            except Exception:
                pass
    return None


def load_prompts(config: Dict[str, Any]) -> Dict[str, str]:
    defaults = {
        "batch_system": "",
        "batch_user_template": "",
        "meta_system": "",
        "meta_user_template": "",
    }
    p = config.get("prompts", {}) or {}
    out = {k: str(p.get(k, defaults[k])) for k in defaults.keys()}
    return out


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


async def gather_articles_to_store(
    config: Dict[str, Any],
    store: NewsStore,
    job_id: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Ingest (RSS -> article text -> store)
    - config.ingest.lookback: "24h", "3d", "2w", "90m"
    - max_items_per_feed används som safety cap
    - om lookback saknas: gammalt beteende (entries[:max_items])
    """

    def set_job(msg: str):
        if job_id is not None:
            store.update_job(job_id, message=msg)

    feeds = config.get("feeds", [])
    ingest_cfg = config.get("ingest") or {}
    lookback = ingest_cfg.get("lookback")  # t.ex. "24h", "3d"
    max_items = int(
        ingest_cfg.get("max_items_per_feed", config.get("max_items_per_feed", 8))
    )
    timeout_s = int(config.get("article_timeout_s", 20))

    http_limiter = AsyncLimiter(max_rate=6, time_period=1)
    headers = {"User-Agent": "news-summarizer/2.2 (personal; rate-limited)"}

    inserted = 0
    updated = 0

    async with aiohttp.ClientSession(headers=headers) as session:
        for f in feeds:
            name = f["name"]
            feed_url = f["url"]
            set_job(f"Läser RSS: {name}")

            try:
                logger.info(f"Hämtar RSS: {name} ({feed_url})")
                feed = await fetch_rss(feed_url, session)
            except Exception as e:
                logger.warning(f"Kunde inte läsa RSS: {name} ({feed_url}) -> {e}")
                continue

            entries = list(feed.entries or [])

            cutoff_ts: Optional[int] = None
            if lookback:
                now = int(time.time())
                cutoff_ts = now - parse_lookback_to_seconds(str(lookback))
                filtered: List[feedparser.FeedParserDict] = []
                for entry in entries:
                    ts = entry_published_ts(entry)
                    if ts is None:
                        continue
                    if ts >= cutoff_ts:
                        filtered.append(entry)
                filtered.sort(key=lambda e: entry_published_ts(e) or 0, reverse=True)
                if max_items and len(filtered) > max_items:
                    filtered = filtered[:max_items]
                logger.info(
                    "RSS %s: %d entries (filtered=%d, lookback=%s, cap=%s)",
                    name,
                    len(entries),
                    len(filtered),
                    lookback,
                    max_items,
                )
                entries_to_process = filtered
            else:
                entries_to_process = entries[:max_items]
                logger.info(
                    "RSS %s: %d entries (cap=%s, lookback=none)",
                    name,
                    len(entries),
                    max_items,
                )

            for entry in entries_to_process:
                link = getattr(entry, "link", None)
                if not link:
                    continue

                aid = stable_id(link)
                title = (getattr(entry, "title", "") or "").strip()
                published = (
                    getattr(entry, "published", "")
                    or getattr(entry, "updated", "")
                    or ""
                ).strip()

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
                        logger.info(f"Inserted {title} från {name} som {aid}")
                    else:
                        if existing.get("content_hash") != chash:
                            doc["summarized"] = False
                            doc["summarized_at"] = None
                            store.upsert_article(doc)
                            updated += 1
                            logger.info(f"Uppdaterade {title} från {name} som {aid}")

                except Exception as e:
                    logger.warning(f"Artikel misslyckades: {link} -> {e}")

    return inserted, updated


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


def _estimate_article_chars(a: dict) -> int:
    return (
        len(a.get("text", "")) + len(a.get("title", "")) + len(a.get("url", "")) + 200
    )


def _batch_chars(batch: List[dict]) -> int:
    return sum(_estimate_article_chars(x) for x in batch)


def _can_fit_in_batch(
    batch: List[dict],
    a: dict,
    *,
    max_chars_per_batch: int,
    max_articles_per_batch: int,
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
) -> None:
    if not batches:
        batches.append([a])
        return
    last = batches[-1]
    if _can_fit_in_batch(
        last,
        a,
        max_chars_per_batch=max_chars_per_batch,
        max_articles_per_batch=max_articles_per_batch,
    ):
        last.append(a)
    else:
        batches.append([a])


def _choose_trim_action(overflow_tokens: int, structural_threshold: int) -> str:
    """
    Väljer strategi baserat på overflow.
    - small overflow: word_trim
    - medium: drop_one_article
    - large: drop_multiple_articles
    """
    if overflow_tokens <= 200:
        return "word_trim"
    if overflow_tokens <= structural_threshold:
        return "drop_one_article"
    return "drop_multiple_articles"


def _trim_last_user_word_boundary(
    messages: List[Dict[str, str]],
    remove_tokens: int,
    *,
    chars_per_token: float,
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


async def summarize_batches_then_meta(
    config: Dict[str, Any],
    articles: List[dict],
    llm: LLMClient,
    store: NewsStore,
    job_id: Optional[int] = None,
) -> str:
    """
    Summerar artiklar i batcher + metasammanfattning.

    Patchar:
    - checkpointing (batch + meta)
    - token budget (best effort)
    - prompt-too-long: välj strategi baserat på overflow
        * liten overflow -> word-boundary trim
        * större overflow -> flytta ut hela artiklar från batchens slut till sista batchen / ny batch
    """

    def set_job(msg: str):
        if job_id is not None:
            store.update_job(job_id, message=msg)

    prompts = load_prompts(config)

    batching = config.get("batching", {}) or {}
    max_chars = int(batching.get("max_chars_per_batch", 18000))
    max_n = int(batching.get("max_articles_per_batch", 10))

    article_clip_chars = int(batching.get("article_clip_chars", 6000))
    meta_batch_clip_chars = int(batching.get("meta_batch_clip_chars", 3500))
    meta_sources_clip_chars = int(batching.get("meta_sources_clip_chars", 140))

    def clip_text(s: str, n: int) -> str:
        s = (s or "").strip()
        return s if len(s) <= n else s[:n] + "…"

    def clip_line(s: str, n: int) -> str:
        s = (s or "").strip()
        return s if len(s) <= n else s[:n] + "…"

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
        msgs2, est, budget = enforce_budget(
            messages,
            max_context_tokens=max_ctx,
            max_output_tokens=max_out,
            safety_margin_tokens=margin,
        )
        logger.info(f"LLM budget: est_prompt_tokens={est} budget_tokens={budget}")

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
                        action = _choose_trim_action(
                            int(overflow), structural_threshold
                        )

                        if action == "word_trim":
                            remove_tokens = int(overflow) + 512
                            logger.warning(
                                "LLM prompt too long: overflow=%s. action=word_trim attempt=%s/%s remove_tokens~%s",
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

                        # För medium/large vill vi göra strukturell justering i batch-loop (inte här),
                        # så vi signalerar med en special-exception.
                        raise PromptTooLongStructural(int(overflow))

                    # overflow okänd → word_trim schablon
                    logger.warning(
                        "LLM prompt too long (no overflow parsed): action=word_trim_fixed attempt=%s/%s",
                        attempt,
                        max_attempts,
                    )
                    current = _trim_last_user_word_boundary(
                        current, 1024, chars_per_token=chars_per_token
                    )
                    attempt += 1
                    continue

                raise

    class PromptTooLongStructural(Exception):
        def __init__(self, overflow_tokens: int):
            super().__init__(
                f"prompt too long (structural), overflow={overflow_tokens}"
            )
            self.overflow_tokens = overflow_tokens

    # --- checkpoint setup ---
    cp_cfg = config.get("checkpointing") or {}
    cp_enabled = bool(cp_cfg.get("enabled", True))
    cp_key = _checkpoint_key(job_id, articles)
    cp_path: Optional[Path] = _checkpoint_path(config, cp_key) if cp_enabled else None
    meta_path: Optional[Path] = _meta_ckpt_path(config, cp_key) if cp_enabled else None

    batches = batch_articles(
        articles, max_chars, max_n, article_clip_chars=article_clip_chars
    )

    # Resume meta if done
    if cp_enabled and meta_path is not None:
        meta_cp = _load_checkpoint(meta_path)
        if meta_cp and meta_cp.get("kind") == "meta_result":
            cached = (meta_cp.get("meta") or "").strip()
            if cached:
                set_job("Återupptar: meta redan klar (från checkpoint).")
                return cached

    # Resume batch summaries
    done_map: Dict[int, str] = {}
    if cp_enabled and cp_path is not None:
        cp = _load_checkpoint(cp_path)
        if cp and cp.get("kind") == "batch_summaries":
            # OBS: batch_total måste matcha för att vi säkert ska kunna mappa index
            if cp.get("batch_total") == len(batches):
                done = cp.get("done") or {}
                try:
                    for k, v in done.items():
                        done_map[int(k)] = str(v)
                    if done_map:
                        set_job(
                            f"Återupptar från checkpoint: {len(done_map)}/{len(batches)} batcher klara."
                        )
                except Exception:
                    done_map = {}

    batch_summaries: List[Tuple[int, str]] = [
        (i, done_map[i]) for i in sorted(done_map.keys())
    ]

    # ---- RUN batches with structural trimming ----
    idx = 1
    while idx <= len(batches):
        if idx in done_map:
            idx += 1
            continue

        batch = batches[idx - 1]
        set_job(f"Summerar batch {idx}/{len(batches)}...")

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
                    f"{clip_text(a.get('text', ''), article_clip_chars)}"
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

        attempts_here = 0
        while True:
            attempts_here += 1
            messages = build_messages_for_batch(idx, batch)
            try:
                summary = await chat_guarded(messages, temperature=0.2)
                break
            except PromptTooLongStructural as e:
                overflow = int(getattr(e, "overflow_tokens", 0) or 0)
                action = _choose_trim_action(overflow, structural_threshold)

                if len(batch) <= 1:
                    # Batch med 1 artikel: kan inte flytta artikel → klipp artikeln hårdare och retry
                    logger.warning(
                        "Batch %s har 1 artikel men overflow=%s. Klipper article text hårdare och retry.",
                        idx,
                        overflow,
                    )
                    a0 = batch[0]
                    a0["text"] = clip_text(
                        a0.get("text", ""), max(800, int(article_clip_chars * 0.6))
                    )
                    continue

                # Hur mycket vi behöver minska? overflow + buffert.
                target_remove_tokens = overflow + 512
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
                    )
                else:
                    # drop_multiple_articles tills vi tror vi sparat nog
                    while len(batch) > 1 and removed_chars < target_remove_chars:
                        a = batch.pop()
                        removed_count += 1
                        removed_chars += _estimate_article_chars(a)
                        _move_article_to_tail_batch(
                            batches,
                            a,
                            max_chars_per_batch=max_chars,
                            max_articles_per_batch=max_n,
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

                # retry samma idx med reducerad batch
                continue

        done_map[idx] = summary
        batch_summaries.append((idx, summary))

        if cp_enabled and cp_path is not None:
            payload = {
                "kind": "batch_summaries",
                "created_at": int(time.time()),
                "job_id": job_id,
                "checkpoint_key": cp_key,
                "batch_total": len(batches),
                "done": {str(k): v for k, v in sorted(done_map.items())},
                "article_ids": [a.get("id", "") for a in articles],
            }
            _atomic_write_json(cp_path, payload)

        idx += 1

    # ---- META ----
    set_job("Skapar metasammanfattning...")

    sources_list = [
        f"- {clip_line(a.get('title', ''), meta_sources_clip_chars)} — {a.get('url', '').strip()}"
        for a in articles
    ]
    sources_text = "\n".join(sources_list)

    batch_text = "\n\n====================\n\n".join(
        [
            f"Batch {i}:\n{clip_text(s, meta_batch_clip_chars)}"
            for i, s in sorted(batch_summaries, key=lambda x: x[0])
        ]
    )

    meta_user = prompts["meta_user_template"].format(
        batch_summaries=batch_text, sources_list=sources_text
    )

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
            },
        )

    meta_messages = [
        {"role": "system", "content": prompts["meta_system"]},
        {"role": "user", "content": meta_user},
    ]

    # Meta: vid prompt-too-long kör vi word-trim i chat_guarded (structural triggas men batch flytt gäller ej meta),
    # så vi fångar PromptTooLongStructural och gör word-trim retry lokalt.
    try:
        meta = await chat_guarded(meta_messages, temperature=0.2)
    except PromptTooLongStructural as e:
        overflow = int(getattr(e, "overflow_tokens", 0) or 0)
        logger.warning(
            "Meta prompt too long structural overflow=%s. Faller tillbaka till word-trim i meta.",
            overflow,
        )
        trimmed = _trim_last_user_word_boundary(
            meta_messages, overflow + 2048, chars_per_token=chars_per_token
        )
        meta = await llm.chat(trimmed, temperature=0.2)

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
            },
        )

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
