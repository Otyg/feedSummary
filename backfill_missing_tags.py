#!/usr/bin/env python3
"""Kör om taggningen för artiklar som helt saknar taggar."""

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
    """Return ids of articles that have zero tags, oldest first."""
    untagged: List[str] = []
    for article in store.iter_articles():
        article_id = str(article.get("id") or article.get("_id") or "").strip()
        if not article_id:
            continue
        if not store.get_article_tags(article_id):
            untagged.append(article_id)
            if limit is not None and len(untagged) >= limit:
                break
    return untagged


def _chunked(items: Sequence[str], size: int) -> List[List[str]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


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
            "Kör om taggningen på artiklar som saknar taggar. "
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
    return parser


async def _run(args: argparse.Namespace) -> int:
    config_path = _resolve_config_path(args.config)
    config = _load_config(config_path)
    store = create_store(config.get("store", {}))

    result = await backfill_missing_tags(
        store,
        config,
        apply=args.apply,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    mode = "ÄNDRING" if args.apply else "FÖRHANDSGRANSKNING"
    print(
        f"{mode}: {result['untagged_articles']} artiklar saknade taggar, "
        f"{result['tagged']} taggades."
    )
    if not args.apply and result["untagged_articles"]:
        print("Kör med --apply för att faktiskt taggera artiklarna.")
    return 0


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
