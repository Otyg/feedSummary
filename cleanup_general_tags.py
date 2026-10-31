#!/usr/bin/env python3
"""Remove GENERAL tags that are attached to at most one article."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from feedsummary_core.persistence import NewsStore, create_store


MAX_USES = 1
QUERY_CHUNK_SIZE = 400


def _resolve_config_path(raw_path: Optional[str]) -> Path:
    candidate = raw_path or os.environ.get("FEEDSUMMARY_CONFIG") or "config.yaml"
    return Path(os.path.expandvars(os.path.expanduser(candidate))).resolve()


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Konfigurationsfilen saknas: {path}")

    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Konfigurationen måste vara ett YAML-objekt: {path}")

    # Store paths are normally resolved from the process working directory.
    # Resolve them from the config file instead so the script works anywhere.
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


def _general_tags(store: NewsStore) -> List[Dict[str, Any]]:
    tags = []
    for tag in store.get_all_tags() or []:
        category = str(tag.get("category") or "GENERAL").strip()
        if category.casefold() == "general":
            tags.append(tag)
    return tags


def _articles_for_tag_names(
    store: NewsStore,
    tag_names: Sequence[str],
) -> List[Dict[str, Any]]:
    articles_by_id: Dict[str, Dict[str, Any]] = {}
    names = [str(name).strip() for name in tag_names if str(name).strip()]

    for offset in range(0, len(names), QUERY_CHUNK_SIZE):
        chunk = names[offset : offset + QUERY_CHUNK_SIZE]
        for article in store.get_articles_by_tags(chunk, match_mode="any") or []:
            article_id = str(article.get("id") or "").strip()
            if not article_id:
                raise ValueError("En taggad artikel saknar id; rensningen avbryts")
            articles_by_id[article_id] = article

    return list(articles_by_id.values())


def find_cleanup_candidates(store: NewsStore) -> List[Dict[str, Any]]:
    """Return GENERAL tags used by zero or one distinct article."""
    general_tags = _general_tags(store)
    if not general_tags:
        return []

    tag_by_id = {int(tag["id"]): tag for tag in general_tags}
    usage_by_id = {tag_id: 0 for tag_id in tag_by_id}
    tag_names = [str(tag.get("name") or "").strip() for tag in general_tags]

    for article in _articles_for_tag_names(store, tag_names):
        article_id = str(article["id"])
        # Count a tag at most once per article even if a malformed store happens
        # to contain duplicate relation rows.
        article_tag_ids = {
            int(tag["id"])
            for tag in store.get_article_tags(article_id) or []
            if tag.get("id") is not None
        }
        for tag_id in article_tag_ids & tag_by_id.keys():
            usage_by_id[tag_id] += 1

    candidates = [
        {
            "tag_id": tag_id,
            "tag": str(tag_by_id[tag_id].get("name") or "").strip(),
            "category": str(tag_by_id[tag_id].get("category") or "GENERAL"),
            "uses": uses,
        }
        for tag_id, uses in usage_by_id.items()
        if uses <= MAX_USES
    ]
    candidates.sort(key=lambda item: (item["uses"], item["tag"].casefold()))
    return candidates


def _current_use_count(store: NewsStore, tag_name: str) -> int:
    articles = _articles_for_tag_names(store, [tag_name])
    return len(articles)


def _current_tag_by_id(store: NewsStore, tag_id: int) -> Optional[Dict[str, Any]]:
    return next(
        (
            tag
            for tag in store.get_all_tags() or []
            if int(tag.get("id") or 0) == tag_id
        ),
        None,
    )


def delete_candidates(
    store: NewsStore,
    candidates: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Recheck and delete candidates, returning one result per tag."""
    results: List[Dict[str, Any]] = []

    for candidate in candidates:
        result = dict(candidate)
        result.update({"deleted": False, "error": None})
        try:
            tag_id = int(candidate["tag_id"])
            current_tag = _current_tag_by_id(store, tag_id)
            if current_tag is None:
                result["error"] = "Taggen finns inte längre"
                results.append(result)
                continue

            current_name = str(current_tag.get("name") or "").strip()
            current_category = str(current_tag.get("category") or "GENERAL").strip()
            if current_name != str(candidate["tag"]):
                result["error"] = "Taggens namn ändrades under körningen"
                results.append(result)
                continue
            if current_category.casefold() != "general":
                result["error"] = "Taggen tillhör inte längre kategorin GENERAL"
                results.append(result)
                continue

            current_uses = _current_use_count(store, current_name)
            result["uses"] = current_uses
            if current_uses > MAX_USES:
                result["error"] = "Taggen används nu av fler än en artikel"
            elif store.delete_tag(tag_id):
                result["deleted"] = True
            else:
                result["error"] = "Taggen kunde inte tas bort"
        except Exception as exc:
            result["error"] = str(exc)
        results.append(result)

    return results


def _print_candidates(candidates: Sequence[Dict[str, Any]]) -> None:
    if not candidates:
        print("Inga GENERAL-taggar används en gång eller mindre.")
        return

    print(f"Hittade {len(candidates)} GENERAL-taggar med högst {MAX_USES} användning:")
    for candidate in candidates:
        print(
            f"- {candidate['tag']} "
            f"(id={candidate['tag_id']}, användningar={candidate['uses']})"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hitta GENERAL-taggar som används av högst en artikel. "
            "Standardläget är en torrkörning."
        )
    )
    parser.add_argument(
        "--config",
        help="Sökväg till config.yaml (standard: FEEDSUMMARY_CONFIG eller ./config.yaml)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Ta faktiskt bort taggarna och deras artikelkopplingar",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        config_path = _resolve_config_path(args.config)
        config = _load_config(config_path)
        store_config = config.get("store", {})
        if not isinstance(store_config, dict):
            raise ValueError("Konfigurationens 'store' måste vara ett YAML-objekt")
        store = create_store(store_config)

        candidates = find_cleanup_candidates(store)
        _print_candidates(candidates)

        if not args.apply:
            if candidates:
                print("Torrkörning: inget togs bort. Kör igen med --apply för att rensa.")
            return 0

        results = delete_candidates(store, candidates)
        deleted = [result for result in results if result["deleted"]]
        failed = [result for result in results if result["error"]]
        print(f"Borttagna taggar: {len(deleted)}.")
        for result in failed:
            print(f"Kunde inte ta bort {result['tag']}: {result['error']}", file=sys.stderr)
        return 2 if failed else 0
    except KeyboardInterrupt:
        print("Rensningen avbröts.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Rensningen misslyckades: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
