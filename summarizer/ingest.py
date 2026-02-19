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
# ----------------------------
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
import feedparser
import trafilatura
from aiolimiter import AsyncLimiter
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from persistence import NewsStore
from summarizer.helpers import (
    RateLimitError,
    compute_content_hash,
    entry_published_ts,
    load_feeds_into_config,
    parse_lookback_to_seconds,
    set_job,
    stable_id,
)

logger = logging.getLogger(__name__)


def _norm_cat(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _entry_categories(entry: feedparser.FeedParserDict) -> Set[str]:
    cats: Set[str] = set()

    try:
        c = getattr(entry, "category", None)
        if isinstance(c, str) and c.strip():
            cats.add(_norm_cat(c))
    except Exception:
        pass

    try:
        cs = getattr(entry, "categories", None)
        if isinstance(cs, list):
            for x in cs:
                if isinstance(x, str) and x.strip():
                    cats.add(_norm_cat(x))
                elif isinstance(x, dict):
                    term = x.get("term") or x.get("label") or x.get("name")
                    if isinstance(term, str) and term.strip():
                        cats.add(_norm_cat(term))
    except Exception:
        pass

    try:
        tags = getattr(entry, "tags", None)
        if isinstance(tags, list):
            for t in tags:
                if isinstance(t, dict):
                    term = t.get("term") or t.get("label") or t.get("name")
                    if isinstance(term, str) and term.strip():
                        cats.add(_norm_cat(term))
    except Exception:
        pass

    return cats


def _passes_category_filter(
    entry: feedparser.FeedParserDict, feed_cfg: Dict[str, Any]
) -> bool:
    inc = feed_cfg.get("category_include") or feed_cfg.get("categories_include")
    exc = feed_cfg.get("category_exclude") or feed_cfg.get("categories_exclude")

    inc_set: Set[str] = set()
    exc_set: Set[str] = set()

    if isinstance(inc, list):
        inc_set = {_norm_cat(str(x)) for x in inc if str(x).strip()}
    elif isinstance(inc, str) and inc.strip():
        inc_set = {_norm_cat(inc)}

    if isinstance(exc, list):
        exc_set = {_norm_cat(str(x)) for x in exc if str(x).strip()}
    elif isinstance(exc, str) and exc.strip():
        exc_set = {_norm_cat(exc)}

    if not inc_set and not exc_set:
        return True

    cats = _entry_categories(entry)

    if exc_set and (cats & exc_set):
        return False

    if inc_set and not (cats & inc_set):
        return False

    return True


async def fetch_rss(
    feed_url: str, session: aiohttp.ClientSession
) -> feedparser.FeedParserDict:
    async with session.get(feed_url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
        resp.raise_for_status()
        content = await resp.read()
        logger.info(f"{feed_url} hämtad")
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

    - config.ingest.lookback: "24h", "3d", "2w", "90m" osv.
    - max_items_per_feed används fortfarande som safety cap.
    - Per-feed filter:
        category_include / category_exclude (matchar entry.tags[].term m.fl.)

    ✅ Artikel-store ska bara hålla artiklar:
      - vi sätter inte summarized/summarized_at här längre
    """
    config = load_feeds_into_config(config, base_config_path="config.yaml")
    feeds = config.get("feeds", [])
    ingest_cfg = config.get("ingest") or {}
    lookback = ingest_cfg.get("lookback")
    max_items = int(
        ingest_cfg.get("max_items_per_feed", config.get("max_items_per_feed", 8))
    )

    timeout_s = int(config.get("article_timeout_s", 20))
    http_limiter = AsyncLimiter(max_rate=6, time_period=1)
    headers = {"User-Agent": "news-summarizer/2.1 (personal; rate-limited)"}

    inserted = 0
    updated = 0
    connector = aiohttp.TCPConnector(limit=50, ttl_dns_cache=300)
    async with aiohttp.ClientSession(
        headers=headers,
        connector=connector,
        max_line_size=16384,
        max_field_size=32768,
    ) as session:
        for f in feeds:
            name = f["name"]
            feed_url = f["url"]
            set_job(f"Läser RSS: {name}", job_id, store)

            try:
                logger.info(f"Hämtar RSS: {name}")
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

            # category filter per feed
            before_cat = len(entries_to_process)
            if isinstance(f, dict) and (
                f.get("category_include")
                or f.get("categories_include")
                or f.get("category_exclude")
                or f.get("categories_exclude")
            ):
                tmp: List[feedparser.FeedParserDict] = []
                for e in entries_to_process:
                    if _passes_category_filter(e, f):
                        tmp.append(e)
                entries_to_process = tmp

                logger.info(
                    "RSS %s: category filter applied (kept=%d/%d include=%s exclude=%s)",
                    name,
                    len(entries_to_process),
                    before_cat,
                    f.get("category_include") or f.get("categories_include"),
                    f.get("category_exclude") or f.get("categories_exclude"),
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
                )

                existing = store.get_article(aid)

                try:
                    async with http_limiter:
                        html = await guarded_fetch_article(link, session, timeout_s)
                    text = extract_text_from_html(html, link)
                    if len(text) < 200:
                        continue

                    chash = compute_content_hash(title, link, text)
                    ts = entry_published_ts(entry)
                    doc = {
                        "id": aid,
                        "source": name,
                        "title": title,
                        "url": link,
                        "published": published,
                        "published_ts": ts or 0,
                        "fetched_at": int(time.time()),
                        "text": text,
                        "content_hash": chash,
                    }

                    if existing is None:
                        store.upsert_article(doc)
                        inserted += 1
                    else:
                        if existing.get("content_hash") != chash:
                            store.upsert_article(doc)
                            updated += 1

                except Exception as e:
                    logger.warning(f"Artikel misslyckades: {link} -> {e}")
    logger.info(f"Articles: {str(inserted)} inserted, {str(updated)} updated")
    return inserted, updated
