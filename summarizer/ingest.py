# ----------------------------
# Fetch/extract
# ----------------------------
import time
from aiolimiter import AsyncLimiter
from typing import Any, Dict, List, Optional, Tuple
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
    parse_lookback_to_seconds,
    set_job,
    stable_id,
)
import logging
import trafilatura
import asyncio
import aiohttp
import feedparser

logger = logging.getLogger(__name__)


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
    - Om lookback saknas: bakåtkompatibelt (entries[:max_items]).
    """

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
