#!/usr/bin/env python3
"""Backfill the separate similarity and tagging vectors for stored articles."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from feedsummary_core.llm_client import (
    create_llm_client,
    get_client_embedding_dimensions,
    get_client_embedding_model,
    has_local_embedding_provider,
)
from feedsummary_core.persistence import create_store
from feedsummary_core.summarizer.batching import (
    SIMILARITY_EMBEDDING_INSTRUCTION,
    SIMILARITY_EMBEDDING_PURPOSE,
    TAGGING_EMBEDDING_INSTRUCTION,
    TAGGING_EMBEDDING_PURPOSE,
    article_embedding_text,
    cached_embedding,
    ensure_article_embedding,
)

from run_long_term_clustering import _profile_config, _resolve_sources, _timestamp


def _config_path(raw: str | None) -> Path:
    value = raw or os.environ.get("FEEDSUMMARY_CONFIG") or "config.yaml"
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Konfigurationen måste vara ett YAML-objekt: {path}")
    return data


def _store_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    raw = config.get("store")
    if not isinstance(raw, dict):
        raise TypeError("Konfigurationen saknar 'store'")
    result = dict(raw)
    if result.get("path"):
        path = Path(os.path.expandvars(os.path.expanduser(str(result["path"]))))
        if not path.is_absolute():
            path = config_path.parent / path
        result["path"] = str(path.resolve())
    return result


def _article_selected(
    article: dict[str, Any],
    *,
    sources: set[str] | None,
    since_ts: int | None,
    until_ts: int | None,
) -> bool:
    if sources is not None and str(article.get("source") or "") not in sources:
        return False
    article_ts = int(article.get("published_ts") or article.get("fetched_at") or 0)
    if since_ts is not None and article_ts < since_ts:
        return False
    return until_ts is None or article_ts <= until_ts


async def _run(
    config_path: Path,
    limit: int | None,
    *,
    purpose: str = "both",
    profile_id: str | None = None,
    since_ts: int | None = None,
    until_ts: int | None = None,
    dry_run: bool = False,
) -> tuple[int, int]:
    config = _load_config(config_path)
    if not has_local_embedding_provider(config):
        raise ValueError("Konfigurationen saknar en ollama_local-provider")

    sources: set[str] | None = None
    if profile_id:
        _, profile = _profile_config(config, profile_id)
        sources = set(_resolve_sources(config, config_path, profile))
    store_config = _store_config(config, config_path)
    if dry_run:
        store_config["initialize_schema"] = False
    store = create_store(store_config)
    client = create_llm_client(config)
    completed = 0
    failed = 0
    try:
        iterator = getattr(store, "iter_articles", None)
        if not callable(iterator):
            raise TypeError("Lagringsbackend saknar iter_articles()")
        batching = config.get("batching", {}) or {}
        tagging_ml = ((config.get("tagging", {}) or {}).get("ml", {}) or {})
        text_chars = int(
            tagging_ml.get(
                "embedding_text_chars", batching.get("embedding_text_chars", 2000)
            )
        )
        model = get_client_embedding_model(client)
        dimensions = get_client_embedding_dimensions(client)
        similarity_instruction = str(
            batching.get("embedding_instruction", SIMILARITY_EMBEDDING_INSTRUCTION)
        ).strip()
        tagging_instruction = str(
            tagging_ml.get("embedding_instruction", TAGGING_EMBEDDING_INSTRUCTION)
        ).strip()

        selected = 0
        for article in iterator():
            if not _article_selected(
                article,
                sources=sources,
                since_ts=since_ts,
                until_ts=until_ts,
            ):
                continue
            selected += 1
            if limit is not None and selected > limit:
                break
            requested = []
            if purpose in {"similarity", "both"}:
                requested.append(
                    (SIMILARITY_EMBEDDING_PURPOSE, similarity_instruction)
                )
            if purpose in {"tagging", "both"}:
                requested.append((TAGGING_EMBEDDING_PURPOSE, tagging_instruction))
            if dry_run:
                text = article_embedding_text(article, text_chars)
                results = [
                    cached_embedding(
                        article,
                        text,
                        model,
                        purpose=item_purpose,
                        instruction=instruction,
                        dimensions=dimensions,
                    )
                    for item_purpose, instruction in requested
                ]
            else:
                results = await asyncio.gather(
                    *(
                        ensure_article_embedding(
                            article,
                            client.embed,
                            store=store,
                            embedding_model=model,
                            embedding_text_chars=text_chars,
                            purpose=item_purpose,
                            instruction=instruction,
                            dimensions=dimensions,
                        )
                        for item_purpose, instruction in requested
                    )
                )
            if all(result is not None for result in results):
                completed += 1
            else:
                failed += 1
            if selected % 100 == 0:
                print(f"Bearbetade {selected} valda artiklar …", file=sys.stderr)
        return completed, failed
    finally:
        close_client = getattr(client, "aclose", None)
        if callable(close_client):
            await close_client()
        close_store = getattr(store, "close", None)
        if callable(close_store):
            close_store()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Standard: FEEDSUMMARY_CONFIG eller config.yaml")
    parser.add_argument("--limit", type=int, help="Bearbeta högst detta antal artiklar")
    parser.add_argument(
        "--purpose",
        choices=("similarity", "tagging", "both"),
        default="both",
        help="Vilken embeddingtyp som ska fyllas i (standard: both)",
    )
    parser.add_argument("--profile", help="Begränsa till en long_term_analysis-profil")
    parser.add_argument("--since", help="Starttid som Unix-tid eller ISO-8601")
    parser.add_argument("--until", help="Sluttid som Unix-tid eller ISO-8601")
    parser.add_argument("--dry-run", action="store_true", help="Räkna utan LLM-anrop eller skrivningar")
    args = parser.parse_args()
    try:
        completed, failed = asyncio.run(
            _run(
                _config_path(args.config),
                max(1, args.limit) if args.limit else None,
                purpose=args.purpose,
                profile_id=args.profile,
                since_ts=_timestamp(args.since),
                until_ts=_timestamp(args.until),
                dry_run=args.dry_run,
            )
        )
        label = "har redan valda vektorer" if args.dry_run else "har valda vektorer"
        print(f"Klart: {completed} artiklar {label}; {failed} saknar eller misslyckades.")
        return 0 if failed == 0 else 1
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"Backfill misslyckades: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
