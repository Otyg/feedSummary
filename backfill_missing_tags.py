#!/usr/bin/env python3
"""Kör om taggningen för artiklar som helt saknar taggar och lägg till CVE-taggar.

Denna modul tillhandahåller två huvudfunktioner:
1. backfill_missing_tags: Använder LLM för att tagga artiklar som helt saknar taggar
2. backfill_missing_cve_tags: Extraherar CVE-identifikatorer från artikeltext/titel 
   och lägger till CVE-taggar på artiklar som saknar dem

CVE-taggar läggs till enligt samma logik som används i summary_tags.py:
- CVE-identifikatorer extraheras med hjälp av regex-mönstret från tagging_rules
- CVE-taggar skapas automatiskt om de inte redan finns
- Taggarna läggs till i kategorin VULNERABILITY
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from feedsummary_core.llm_client import create_llm_client
from feedsummary_core.persistence import NewsStore, create_store
from feedsummary_core.summarizer.tagging_integration import tag_articles_safe
from feedsummary_core.tagging_rules import extract_cve_ids, is_cve_tag

log = logging.getLogger("missing_tags_backfill")
DEFAULT_BATCH_SIZE = 20


def _resolve_config_path(raw_path: Optional[str]) -> Path:
    candidate = raw_path or os.environ.get("FEEDSUMMARY_CONFIG") or "config.yaml"
    return Path(os.path.expandvars(os.path.expanduser(candidate))).resolve()


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Konfigurationsfilen saknas: {path}")

    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Konfigurationen måste vara ett YAML-objekt: {path}")

    store_config = config.get("store")
    if isinstance(store_config, dict) and store_config.get("path"):
        resolved_store = dict(store_config)
        store_path = Path(
            os.path.expandvars(os.path.expanduser(str(store_config["path"])))
        )
        if not store_path.is_absolute():
            store_path = path.parent / store_path
        resolved_store["path"] = str(store_path.resolve())
        config["store"] = resolved_store

    return config


def _find_untagged_article_ids(store: NewsStore, limit: Optional[int]) -> List[str]:
    """Return ids of articles that have zero tags, oldest first.

    Loading the tagged ids in bulk is important for remote stores such as
    CouchDB.  Calling ``get_article_tags`` once per article otherwise causes two
    HTTP queries per article (one for relations and one for tag documents).
    """
    iter_with_tags = getattr(store, "iter_articles_with_tags", None)
    tagged_article_ids: Optional[set[str]] = None
    if callable(iter_with_tags):
        tagged_article_ids = set()
        for item in iter_with_tags():
            article = item.get("article") if isinstance(item, dict) else None
            tags = item.get("tags") if isinstance(item, dict) else None
            if not isinstance(article, dict) or not tags:
                continue
            article_id = str(article.get("id") or article.get("_id") or "").strip()
            if article_id:
                tagged_article_ids.add(article_id)

    untagged: List[str] = []
    for article in store.iter_articles():
        article_id = str(article.get("id") or article.get("_id") or "").strip()
        if not article_id:
            continue
        has_tags = (
            article_id in tagged_article_ids
            if tagged_article_ids is not None
            else bool(store.get_article_tags(article_id))
        )
        if not has_tags:
            untagged.append(article_id)
            if limit is not None and len(untagged) >= limit:
                break
    return untagged


def _chunked(items: Sequence[str], size: int) -> List[List[str]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def _find_articles_without_cve_tags(store: NewsStore, limit: Optional[int]) -> List[Dict[str, Any]]:
    """Find articles with CVE identifiers in content that lack CVE tags.
    
    Returns a list of dicts with 'article', 'cve_ids' keys.
    """
    iter_with_tags = getattr(store, "iter_articles_with_tags", None)
    tagged_article_ids: Optional[set[str]] = None
    
    if callable(iter_with_tags):
        tagged_article_ids = set()
        for item in iter_with_tags():
            article = item.get("article") if isinstance(item, dict) else None
            tags = item.get("tags") if isinstance(item, dict) else None
            if not isinstance(article, dict) or not tags:
                continue
            article_id = str(article.get("id") or article.get("_id") or "").strip()
            if article_id and any(
                isinstance(tag, dict) and is_cve_tag(tag.get("name"))
                for tag in tags
            ):
                tagged_article_ids.add(article_id)

    candidates: List[Dict[str, Any]] = []
    for article in store.iter_articles():
        article_id = str(article.get("id") or article.get("_id") or "").strip()
        if not article_id:
            continue
        
        has_cve_tags = (
            article_id in tagged_article_ids
            if tagged_article_ids is not None
            else any(
                isinstance(tag, dict) and is_cve_tag(tag.get("name"))
                for tag in (store.get_article_tags(article_id) or [])
            )
        )
        
        if has_cve_tags:
            continue
        
        title = str(article.get("title") or "").strip()
        content = str(article.get("content") or article.get("text") or "").strip()
        combined = f"{title} {content}"
        
        cve_ids = extract_cve_ids(combined)
        if cve_ids:
            candidates.append({
                "article": article,
                "cve_ids": cve_ids,
            })
            if limit is not None and len(candidates) >= limit:
                break
    
    return candidates


async def backfill_missing_cve_tags(
    store: NewsStore,
    config: Dict[str, Any],
    *,
    apply: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Add CVE tags to articles that mention CVEs but lack CVE tags.
    
    Returns counters for reporting.
    """
    candidates = _find_articles_without_cve_tags(store, limit)
    result: Dict[str, Any] = {
        "apply": apply,
        "articles_with_cves": len(candidates),
        "cve_tags_added": 0,
        "already_had_cve_tags": 0,
        "failed": 0,
        "errors": [],
    }
    
    if not apply or not candidates:
        return result
    
    for candidate in candidates:
        article = candidate.get("article")
        if not isinstance(article, dict):
            continue
        
        article_id = str(article.get("id") or article.get("_id") or "").strip()
        if not article_id:
            continue
        
        cve_ids = candidate.get("cve_ids") or []
        
        for cve_id in cve_ids:
            try:
                cve_tag = store.get_tag_by_name(cve_id)
                cve_tag_id = int(cve_tag["id"]) if cve_tag and cve_tag.get("id") else None
                
                if cve_tag_id is None:
                    cve_tag_id = store.add_tag(cve_id, category="VULNERABILITY")
                    if not cve_tag_id:
                        result["failed"] += 1
                        result["errors"].append({
                            "article_id": article_id,
                            "cve_id": cve_id,
                            "error": "Kunde inte skapa CVE-taggen"
                        })
                        continue
                    cve_tag_id = int(cve_tag_id)
                
                if store.add_tag_to_article(article_id, cve_tag_id):
                    result["cve_tags_added"] += 1
                    log.info(
                        "Lade till CVE-tagg %s på artikel %s (%s)",
                        cve_id,
                        article.get("title") or "utan titel",
                        article_id,
                    )
                else:
                    result["failed"] += 1
                    result["errors"].append({
                        "article_id": article_id,
                        "cve_id": cve_id,
                        "error": "Associationen kunde inte skapas"
                    })
            except Exception as error:
                result["failed"] += 1
                result["errors"].append({
                    "article_id": article_id,
                    "cve_id": cve_id,
                    "error": str(error)
                })
    
    return result



async def backfill_missing_tags(
    store: NewsStore,
    config: Dict[str, Any],
    *,
    apply: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Retag articles without any tags and return counters for reporting."""
    untagged_ids = _find_untagged_article_ids(store, limit)
    result: Dict[str, Any] = {
        "apply": apply,
        "untagged_articles": len(untagged_ids),
        "tagged": 0,
    }
    if not apply or not untagged_ids:
        return result

    llm_client = create_llm_client(config)
    tagged = 0
    for batch in _chunked(untagged_ids, batch_size):
        tagged += await tag_articles_safe(
            store=store,
            llm_client=llm_client,
            article_ids=batch,
            config=config,
        )
        log.info("Taggade %d/%d artiklar hittills", tagged, len(untagged_ids))

    result["tagged"] = tagged
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Kör om taggningen på artiklar som saknar taggar och/eller lägg till CVE-taggar. "
            "Kör utan --apply för förhandsgranskning."
        )
    )
    parser.add_argument("--config", help="Sökväg till config.yaml (standard: config.yaml)")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Utför taggningen; utan flaggan görs endast en förhandsgranskning",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Antal artiklar per taggnings-batch (standard: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max antal otaggade artiklar att behandla (standard: alla)",
    )
    parser.add_argument(
        "--cve-only",
        action="store_true",
        help="Lägg endast till CVE-taggar, hoppa över LLM-taggning",
    )
    parser.add_argument(
        "--skip-cve",
        action="store_true",
        help="Hoppa över CVE-taggning",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    config_path = _resolve_config_path(args.config)
    config = _load_config(config_path)
    store = create_store(config.get("store", {}))
    try:
        results: Dict[str, Dict[str, Any]] = {}
        
        if not args.cve_only:
            result = await backfill_missing_tags(
                store,
                config,
                apply=args.apply,
                batch_size=args.batch_size,
                limit=args.limit,
            )
            results["regular"] = result
            mode = "ÄNDRING" if args.apply else "FÖRHANDSGRANSKNING"
            print(
                f"{mode} (standard taggning): {result['untagged_articles']} artiklar saknade taggar, "
                f"{result['tagged']} taggades."
            )
            if not args.apply and result["untagged_articles"]:
                print("Kör med --apply för att faktiskt taggera artiklarna.")
        
        if not args.skip_cve:
            cve_result = await backfill_missing_cve_tags(
                store,
                config,
                apply=args.apply,
                limit=args.limit,
            )
            results["cve"] = cve_result
            mode = "ÄNDRING" if args.apply else "FÖRHANDSGRANSKNING"
            print(
                f"{mode} (CVE-taggar): {cve_result['articles_with_cves']} artiklar med CVE-mentions, "
                f"{cve_result['cve_tags_added']} CVE-taggar tillagda, "
                f"{cve_result['failed']} fel."
            )
            if not args.apply and cve_result["articles_with_cves"]:
                print("Kör med --apply för att faktiskt lägga till CVE-taggarna.")
            for error in cve_result.get("errors", []):
                log.error(
                    "%s / %s: %s",
                    error.get("article_id"),
                    error.get("cve_id"),
                    error.get("error")
                )
        
        return 0
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        return asyncio.run(_run(args))
    except Exception as error:
        log.error("Backfill misslyckades: %s", error, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
