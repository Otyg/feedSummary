#!/usr/bin/env python3
"""Preview or delete tags whose article usage is below a configured minimum."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from feedsummary_core.persistence import NewsStore, create_store


QUERY_CHUNK_SIZE = 400
PROTECTED_TAG_PREFIXES = ("cve-", "cwe-")


def _resolve_config_path(raw_path: Optional[str]) -> Path:
    candidate = raw_path or os.environ.get("FEEDSUMMARY_CONFIG") or "config.yaml"
    return Path(os.path.expandvars(os.path.expanduser(candidate))).resolve()


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Konfigurationsfilen saknas: {path}")

    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Konfigurationen måste vara ett YAML-objekt: {path}")

    # Resolve file-based stores relative to the selected configuration file so
    # the command behaves consistently regardless of the working directory.
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


def _validate_min_uses(min_uses: int) -> int:
    if min_uses < 0:
        raise ValueError("--min-uses måste vara 0 eller större")
    return min_uses


def _is_protected_identifier_tag(tag_name: Any) -> bool:
    """Return whether a tag is a CVE or CWE identifier that must be preserved."""
    normalized = str(tag_name or "").strip().casefold()
    return normalized.startswith(PROTECTED_TAG_PREFIXES)


def _tags_in_category(
    store: NewsStore,
    category: str,
) -> List[Dict[str, Any]]:
    wanted = str(category).strip()
    if not wanted:
        raise ValueError("--category får inte vara tom")
    return [
        tag
        for tag in store.get_all_tags() or []
        if str(tag.get("category") or "GENERAL").strip().casefold()
        == wanted.casefold()
        and not _is_protected_identifier_tag(tag.get("name"))
    ]


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


def find_prune_candidates(
    store: NewsStore,
    *,
    category: str,
    min_uses: int,
) -> List[Dict[str, Any]]:
    """Return tags in ``category`` used by fewer than ``min_uses`` articles."""
    min_uses = _validate_min_uses(min_uses)
    category_tags = _tags_in_category(store, category)
    if not category_tags or min_uses == 0:
        return []

    tag_by_id = {int(tag["id"]): tag for tag in category_tags}
    usage_by_id = {tag_id: 0 for tag_id in tag_by_id}
    tag_names = [str(tag.get("name") or "").strip() for tag in category_tags]

    for article in _articles_for_tag_names(store, tag_names):
        article_id = str(article["id"])
        # Count every tag at most once per article, even if malformed relation
        # data happens to contain duplicates.
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
            "category": str(tag_by_id[tag_id].get("category") or "GENERAL").strip(),
            "uses": uses,
        }
        for tag_id, uses in usage_by_id.items()
        if uses < min_uses
    ]
    candidates.sort(key=lambda item: (item["uses"], item["tag"].casefold()))
    return candidates


def _current_tag_by_id(
    store: NewsStore,
    tag_id: int,
) -> Optional[Dict[str, Any]]:
    return next(
        (
            tag
            for tag in store.get_all_tags() or []
            if int(tag.get("id") or 0) == tag_id
        ),
        None,
    )


def _current_use_count(store: NewsStore, tag_name: str) -> int:
    return len(_articles_for_tag_names(store, [tag_name]))


def delete_candidates(
    store: NewsStore,
    candidates: Sequence[Dict[str, Any]],
    *,
    category: str,
    min_uses: int,
) -> List[Dict[str, Any]]:
    """Recheck and delete candidates, returning one result per tag."""
    min_uses = _validate_min_uses(min_uses)
    expected_category = str(category).strip()
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
            current_category = str(
                current_tag.get("category") or "GENERAL"
            ).strip()
            if current_name != str(candidate["tag"]):
                result["error"] = "Taggens namn ändrades under körningen"
                results.append(result)
                continue
            if current_category.casefold() != expected_category.casefold():
                result["error"] = (
                    f"Taggen tillhör inte längre kategorin {expected_category}"
                )
                results.append(result)
                continue

            current_uses = _current_use_count(store, current_name)
            result["uses"] = current_uses
            if current_uses >= min_uses:
                result["error"] = (
                    f"Taggen används nu {current_uses} gånger "
                    f"(gränsvärde {min_uses})"
                )
            elif store.delete_tag(tag_id):
                result["deleted"] = True
            else:
                result["error"] = "Taggen kunde inte tas bort"
        except Exception as exc:
            result["error"] = str(exc)
        results.append(result)

    return results


def _print_candidates(
    candidates: Sequence[Dict[str, Any]],
    *,
    category: str,
    min_uses: int,
) -> None:
    if not candidates:
        print(
            f"Inga taggar i kategorin {category} används färre än "
            f"{min_uses} gånger."
        )
        return

    print(
        f"Hittade {len(candidates)} taggar i kategorin {category} som används "
        f"färre än {min_uses} gånger:"
    )
    for candidate in candidates:
        print(
            f"- {candidate['tag']} "
            f"(id={candidate['tag_id']}, användningar={candidate['uses']})"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hitta taggar i en kategori som används av färre än ett angivet "
            "antal artiklar. CVE- och CWE-taggar ignoreras alltid. "
            "Standardläget är en torrkörning."
        )
    )
    parser.add_argument(
        "--category",
        required=True,
        help="Taggkategori att rensa, exempelvis GENERAL eller PRODUCT",
    )
    parser.add_argument(
        "--min-uses",
        required=True,
        type=int,
        help=(
            "Minsta antal artikelanvändningar som krävs för att behålla taggen; "
            "taggar under gränsen raderas"
        ),
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
        min_uses = _validate_min_uses(args.min_uses)
        category = str(args.category).strip()
        if not category:
            raise ValueError("--category får inte vara tom")

        config_path = _resolve_config_path(args.config)
        config = _load_config(config_path)
        store_config = config.get("store", {})
        if not isinstance(store_config, dict):
            raise ValueError("Konfigurationens 'store' måste vara ett YAML-objekt")
        store = create_store(store_config)

        candidates = find_prune_candidates(
            store,
            category=category,
            min_uses=min_uses,
        )
        _print_candidates(candidates, category=category, min_uses=min_uses)

        if not args.apply:
            if candidates:
                print(
                    "Torrkörning: inget togs bort. Kör igen med --apply för att rensa."
                )
            return 0

        results = delete_candidates(
            store,
            candidates,
            category=category,
            min_uses=min_uses,
        )
        deleted = [result for result in results if result["deleted"]]
        failed = [result for result in results if result["error"]]
        print(f"Borttagna taggar: {len(deleted)}.")
        for result in failed:
            print(
                f"Kunde inte ta bort {result['tag']}: {result['error']}",
                file=sys.stderr,
            )
        return 2 if failed else 0
    except KeyboardInterrupt:
        print("Rensningen avbröts.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Rensningen misslyckades: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
