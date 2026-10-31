#!/usr/bin/env python3
"""Replace tags for every stored article using the current tagging configuration."""

from __future__ import annotations

import argparse
import asyncio
import copy
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml

from feedsummary_core.llm_client import create_llm_client
from feedsummary_core.persistence import create_store
from feedsummary_core.summarizer.tagging_integration import tag_articles


def _config_path(raw: Optional[str]) -> Path:
    value = raw or os.environ.get("FEEDSUMMARY_CONFIG") or "config.yaml"
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Konfigurationen måste vara ett YAML-objekt: {path}")
    return data


def _runtime_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Return a copy suited for deterministic, batched full retagging."""
    result = copy.deepcopy(config)
    tagging = result.setdefault("tagging", {})
    ml = tagging.setdefault("ml", {})
    # Use the same frozen classifier throughout the run. It is retrained once
    # after all tag associations have been replaced.
    ml["auto_retrain"] = False
    model_path = ml.get("model_path")
    if model_path:
        path = Path(os.path.expandvars(os.path.expanduser(str(model_path))))
        if not path.is_absolute():
            ml["model_path"] = str((config_path.parent / path).resolve())
    # Similarity consistency only adds tags after selection and is not part of
    # replacing each article's own tag set.
    consistency = tagging.setdefault("similarity_consistency", {})
    consistency["enabled"] = False
    return result


async def _run(
    config_path: Path,
    *,
    batch_size: int,
    offset: int,
    limit: Optional[int],
    max_tags: int,
) -> tuple[int, int]:
    config = _runtime_config(_load_config(config_path), config_path)
    store = create_store(config.get("store", {}))
    client = create_llm_client(config)
    processed = 0
    with_tags = 0
    try:
        iterator = getattr(store, "iter_articles", None)
        if not callable(iterator):
            raise TypeError("Lagringsbackend saknar iter_articles()")
        all_article_ids = [
            str(article.get("id"))
            for article in iterator(limit=limit)
            if article.get("id")
        ]
        article_ids = all_article_ids[offset:]
        total = len(article_ids)
        print(
            f"Startar omtaggning av {total} artiklar från offset {offset} "
            f"({len(all_article_ids)} valda totalt).",
            flush=True,
        )

        for start in range(0, total, batch_size):
            batch = article_ids[start : start + batch_size]
            results = await tag_articles(
                store=store,
                llm_client=client,
                article_ids=batch,
                config=config,
                max_tags_per_article=max_tags,
                skip_if_already_tagged=False,
                enable_embedding_matching=True,
            )
            processed += len(batch)
            with_tags += sum(bool(results.get(article_id)) for article_id in batch)
            print(
                f"Bearbetade {processed}/{total} artiklar i denna körning "
                f"(absolut {offset + processed}/{len(all_article_ids)}); "
                f"{with_tags} fick minst en tagg.",
                flush=True,
            )
        return processed, with_tags
    finally:
        close_client = getattr(client, "aclose", None)
        if callable(close_client):
            await close_client()
        close_store = getattr(store, "close", None)
        if callable(close_store):
            close_store()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Standard: FEEDSUMMARY_CONFIG eller config.yaml")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Hoppa över detta antal artiklar i äldst-först-ordningen",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tags", type=int, default=20)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        processed, with_tags = asyncio.run(
            _run(
                _config_path(args.config),
                batch_size=max(1, args.batch_size),
                offset=max(0, args.offset),
                limit=max(1, args.limit) if args.limit else None,
                max_tags=max(1, args.max_tags),
            )
        )
        print(
            f"Omtaggning klar: {processed} artiklar bearbetades; "
            f"{with_tags} fick minst en tagg."
        )
        return 0
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(
            f"Omtaggningen misslyckades ({type(error).__name__}): {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
