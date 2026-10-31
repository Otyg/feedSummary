#!/usr/bin/env python3
"""Interactively turn selected tag synonyms into child tags."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import yaml

from feedsummary_core.llm_client import create_llm_client
from feedsummary_core.persistence import NewsStore, create_store
from feedsummary_core.summarizer.tagging import TagManager


log = logging.getLogger("tag_synonym_audit")
DEFAULT_CATEGORY = "LOCATION"
AUDIT_VERSION = 1


class _CategoryOnlyStore:
    """Expose one tag category while delegating every other store operation."""

    def __init__(self, store: NewsStore, category: str):
        self._store = store
        self._category = category

    def get_all_tags(self) -> List[Dict[str, Any]]:
        return [
            tag
            for tag in self._store.get_all_tags()
            if _category(tag).casefold() == self._category.casefold()
        ]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)


def _category(tag: Dict[str, Any]) -> str:
    return str(tag.get("category") or "GENERAL").strip()


def resolve_category_name(store: NewsStore, requested: str) -> str:
    """Resolve a category case-insensitively against definitions and stored tags."""
    name = str(requested or "").strip()
    if not name:
        raise ValueError("Kategorin får inte vara tom")

    available: Dict[str, str] = {}
    category_reader = getattr(store, "get_all_categories", None)
    if callable(category_reader):
        for category in category_reader() or []:
            if not isinstance(category, dict):
                continue
            candidate = str(category.get("name") or "").strip()
            if candidate:
                available[candidate.casefold()] = candidate
    for tag in store.get_all_tags() or []:
        if not isinstance(tag, dict):
            continue
        candidate = str(tag.get("category") or "GENERAL").strip()
        if candidate:
            available[candidate.casefold()] = candidate

    resolved = available.get(name.casefold())
    if not resolved:
        choices = ", ".join(sorted(available.values(), key=str.casefold))
        suffix = f" Tillgängliga kategorier: {choices}." if choices else ""
        raise ValueError(f"Kategorin {name!r} finns inte.{suffix}")
    return resolved


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


def _clean_synonyms(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    synonyms: List[str] = []
    seen: set[str] = set()
    for raw_synonym in value:
        synonym = str(raw_synonym or "").strip()
        normalized = synonym.casefold()
        if synonym and normalized not in seen:
            seen.add(normalized)
            synonyms.append(synonym)
    return synonyms


def find_tags_with_synonyms(
    store: NewsStore,
    category: str,
) -> List[Dict[str, Any]]:
    """Return tags in one category having at least one non-empty unique synonym."""
    normalized_category = str(category or "").strip().casefold()
    found: List[Dict[str, Any]] = []
    for raw_tag in store.get_all_tags() or []:
        if (
            not isinstance(raw_tag, dict)
            or _category(raw_tag).casefold() != normalized_category
        ):
            continue
        synonyms = _clean_synonyms(raw_tag.get("synonyms"))
        if synonyms:
            found.append({**raw_tag, "synonyms": synonyms})
    found.sort(key=lambda tag: str(tag.get("name") or "").casefold())
    return found


def find_location_tags_with_synonyms(store: NewsStore) -> List[Dict[str, Any]]:
    """Backward-compatible LOCATION-specific tag lookup."""
    return find_tags_with_synonyms(store, DEFAULT_CATEGORY)


def _tag_by_id(store: NewsStore, tag_id: int) -> Optional[Dict[str, Any]]:
    return next(
        (
            tag
            for tag in store.get_all_tags() or []
            if int(tag.get("id") or 0) == int(tag_id)
        ),
        None,
    )


def _tag_by_exact_name(store: NewsStore, name: str) -> Optional[Dict[str, Any]]:
    normalized = str(name or "").strip().casefold()
    return next(
        (
            tag
            for tag in store.get_all_tags() or []
            if str(tag.get("name") or "").strip().casefold() == normalized
        ),
        None,
    )


def _remove_synonym(store: NewsStore, parent: Dict[str, Any], synonym: str) -> None:
    normalized = synonym.casefold()
    remaining = [
        item
        for item in _clean_synonyms(parent.get("synonyms"))
        if item.casefold() != normalized
    ]
    updated = store.update_tag(int(parent["id"]), synonyms=remaining)
    if not isinstance(updated, dict):
        raise RuntimeError(f"Kunde inte uppdatera synonymer för {parent['name']!r}")


def create_child_from_synonym(
    store: NewsStore,
    parent: Dict[str, Any],
    synonym: str,
) -> tuple[Dict[str, Any], bool]:
    """Create/reuse a same-category tag, attach it as a child, and remove the synonym."""
    parent_id = int(parent["id"])
    parent_category = _category(parent)
    existing = _tag_by_exact_name(store, synonym)
    created = False
    child = existing

    if child is not None and int(child.get("id") or 0) == parent_id:
        raise ValueError("Synonymen är samma namn som föräldrataggen")
    if child is not None and _category(child) != parent_category:
        raise ValueError(
            f"Taggen {child.get('name')!r} finns redan i kategorin {_category(child)}"
        )

    if child is None:
        child = store.create_tag(
            synonym,
            category=parent_category,
            description=f"Tidigare synonym till {parent.get('name') or ''}.",
            synonyms=[],
        )
        if not isinstance(child, dict):
            child = _tag_by_exact_name(store, synonym)
        if not isinstance(child, dict):
            raise RuntimeError(
                f"Kunde inte skapa taggen {synonym!r} i kategorin {parent_category}"
            )
        created = True

    child_id = int(child["id"])
    relation_reader = getattr(store, "get_tag_relations", None)
    previous_parent_ids = []
    if callable(relation_reader):
        relations = relation_reader(child_id) or {}
        previous_parent_ids = [
            int(tag["id"])
            for tag in relations.get("parents", [])
            if isinstance(tag, dict) and tag.get("id")
        ]
    parent_ids = sorted(set(previous_parent_ids) | {parent_id})

    relation_written = False
    try:
        store.set_tag_relations(child_id, parent_ids=parent_ids)
        relation_written = True
        fresh_parent = _tag_by_id(store, parent_id)
        if not isinstance(fresh_parent, dict):
            raise RuntimeError(f"Föräldrataggen med ID {parent_id} saknas")
        _remove_synonym(store, fresh_parent, synonym)
    except Exception:
        if relation_written:
            try:
                store.set_tag_relations(child_id, parent_ids=previous_parent_ids)
            except Exception:
                log.exception("Kunde inte återställa relationerna för tagg %s", child_id)
        if created:
            try:
                store.delete_tag(child_id)
            except Exception:
                log.exception("Kunde inte återställa den skapade taggen %s", child_id)
        raise

    refreshed = _tag_by_id(store, child_id)
    return (refreshed or child), created


async def suggest_category_tags(
    store: NewsStore,
    llm_client: Any,
    article: Dict[str, Any],
    current_tags: List[Dict[str, Any]],
    *,
    category: str,
    max_tags: int,
) -> List[Dict[str, Any]]:
    """Run the existing suggestion flow against one tag category only."""
    category_store = _CategoryOnlyStore(store, category)
    tagger = TagManager(category_store)
    suggestions = await tagger.reclassify_article_with_existing_tags(
        llm_client=llm_client,
        article=article,
        current_article_tags=current_tags,
        max_tags=max_tags,
    )
    allowed_ids = {
        int(tag["id"])
        for tag in category_store.get_all_tags()
        if tag.get("id") is not None
    }
    return [
        tag
        for tag in suggestions
        if isinstance(tag, dict)
        and int(tag.get("id") or 0) in allowed_ids
        and _category(tag).casefold() == category.casefold()
    ]


async def suggest_location_tags(
    store: NewsStore,
    llm_client: Any,
    article: Dict[str, Any],
    current_tags: List[Dict[str, Any]],
    *,
    max_tags: int,
) -> List[Dict[str, Any]]:
    """Backward-compatible LOCATION-specific suggestion flow."""
    return await suggest_category_tags(
        store,
        llm_client,
        article,
        current_tags,
        category=DEFAULT_CATEGORY,
        max_tags=max_tags,
    )


async def retag_parent_articles(
    store: NewsStore,
    llm_client: Any,
    parent: Dict[str, Any],
    child: Dict[str, Any],
    *,
    category: Optional[str] = None,
    max_tags: int = 10,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Replace the parent on articles for which the new child is suggested."""
    get_articles = getattr(store, "get_articles_by_tags", None)
    if not callable(get_articles):
        raise RuntimeError("Store saknar get_articles_by_tags()")

    articles = list(get_articles([str(parent["name"])], match_mode="any") or [])
    articles.sort(key=lambda article: str(article.get("id") or ""))
    if limit is not None:
        articles = articles[:limit]

    parent_id = int(parent["id"])
    child_id = int(child["id"])
    selected_category = str(category or _category(parent)).strip()
    results: List[Dict[str, Any]] = []
    replaced = 0
    errors = 0

    for index, article in enumerate(articles, start=1):
        article_id = str(article.get("id") or "").strip()
        title = str(article.get("title") or "").strip()
        result = {
            "article_id": article_id,
            "title": title,
            "suggested_tags": [],
            "child_suggested": False,
            "replaced": False,
            "error": None,
        }
        print(f"  Artikel {index}/{len(articles)}: {title or article_id}")
        try:
            current_tags = list(store.get_article_tags(article_id) or [])
            if not any(int(tag.get("id") or 0) == parent_id for tag in current_tags):
                result["error"] = "Föräldrataggen saknas redan på artikeln"
                errors += 1
                results.append(result)
                continue

            # Keep an already assigned child available for relevance assessment.
            suggestion_current_tags = [
                tag for tag in current_tags if int(tag.get("id") or 0) != child_id
            ]
            suggestions = await suggest_category_tags(
                store,
                llm_client,
                article,
                suggestion_current_tags,
                category=selected_category,
                max_tags=max_tags,
            )
            result["suggested_tags"] = [
                str(tag.get("name") or "") for tag in suggestions
            ]
            result["child_suggested"] = any(
                int(tag.get("id") or 0) == child_id for tag in suggestions
            )
            if not result["child_suggested"]:
                results.append(result)
                continue

            already_has_child = any(
                int(tag.get("id") or 0) == child_id for tag in current_tags
            )
            child_added = already_has_child or store.add_tag_to_article(article_id, child_id)
            if not child_added:
                child_added = any(
                    int(tag.get("id") or 0) == child_id
                    for tag in store.get_article_tags(article_id) or []
                )
            if not child_added:
                raise RuntimeError("Kunde inte lägga till barntaggen")
            if not store.remove_article_tag(article_id, parent_id):
                if not already_has_child:
                    store.remove_article_tag(article_id, child_id)
                raise RuntimeError("Kunde inte ta bort föräldrataggen")

            result["replaced"] = True
            replaced += 1
        except Exception as exc:
            result["error"] = str(exc)
            errors += 1
            log.exception("Omtaggning misslyckades för artikel %s", article_id)
        results.append(result)

    return {
        "articles_found": len(articles),
        "articles_replaced": replaced,
        "errors": errors,
        "articles": results,
    }


def ask_action(
    parent: Dict[str, Any],
    synonym: str,
    *,
    input_fn: Callable[[str], str] = input,
) -> str:
    """Ask whether one synonym should remain, become a child, be skipped, or quit."""
    prompt = (
        f"\n{parent.get('name')} → {synonym!r}\n"
        "[b]ehåll synonym, skapa [n]y barntagg, [s]kippa tills nästa körning, "
        "[q] avbryt: "
    )
    while True:
        answer = input_fn(prompt).strip().casefold()
        if answer in {"", "b", "behåll", "behall", "keep"}:
            return "keep"
        if answer in {"n", "ny", "child", "barn"}:
            return "child"
        if answer in {"s", "skip", "skippa", "hoppa"}:
            return "skip"
        if answer in {"q", "quit", "avbryt"}:
            return "quit"
        print("Ogiltigt val. Ange b, n, s eller q.")


async def audit_tag_synonyms(
    store: NewsStore,
    llm_client: Any,
    *,
    category: str = DEFAULT_CATEGORY,
    max_tags: int = 10,
    article_limit: Optional[int] = None,
    dry_run: bool = False,
    input_fn: Callable[[str], str] = input,
) -> Dict[str, Any]:
    selected_category = resolve_category_name(store, category)
    targets = find_tags_with_synonyms(store, selected_category)
    decisions: List[Dict[str, Any]] = []
    totals = {
        "tags_with_synonyms": len(targets),
        "synonyms_found": sum(len(tag["synonyms"]) for tag in targets),
        "kept": 0,
        "skipped": 0,
        "child_tags_created": 0,
        "existing_tags_reused": 0,
        "articles_checked": 0,
        "articles_replaced": 0,
        "errors": 0,
    }

    synonym_label = "synonym" if totals["synonyms_found"] == 1 else "synonymer"
    tag_label = "tagg" if len(targets) == 1 else "taggar"
    print(
        f"Hittade {totals['synonyms_found']} {synonym_label} på "
        f"{len(targets)} {tag_label} i kategorin {selected_category}."
    )
    if dry_run:
        print("Dry-run: inga taggar, relationer eller artikelkopplingar ändras.")

    aborted = False
    for parent_snapshot in targets:
        for synonym in parent_snapshot["synonyms"]:
            action = ask_action(parent_snapshot, synonym, input_fn=input_fn)
            decision: Dict[str, Any] = {
                "parent_id": int(parent_snapshot["id"]),
                "parent": str(parent_snapshot.get("name") or ""),
                "synonym": synonym,
                "decision": action,
                "status": action,
                "child": None,
                "retagging": None,
                "error": None,
            }
            if action == "quit":
                decisions.append(decision)
                aborted = True
                break
            if action == "keep":
                totals["kept"] += 1
                decisions.append(decision)
                continue
            if action == "skip":
                totals["skipped"] += 1
                decisions.append(decision)
                continue
            if dry_run:
                decision["status"] = "would_create_child"
                decisions.append(decision)
                continue

            try:
                fresh_parent = _tag_by_id(store, int(parent_snapshot["id"]))
                if not isinstance(fresh_parent, dict):
                    raise RuntimeError("Föräldrataggen saknas")
                child, created = create_child_from_synonym(
                    store, fresh_parent, synonym
                )
                decision["child"] = {
                    "id": int(child["id"]),
                    "name": str(child.get("name") or ""),
                    "created": created,
                }
                if created:
                    totals["child_tags_created"] += 1
                else:
                    totals["existing_tags_reused"] += 1

                print(
                    f"  {child.get('name')} är nu barn till {fresh_parent.get('name')}. "
                    "Granskar artiklar med föräldrataggen …"
                )
                retagging = await retag_parent_articles(
                    store,
                    llm_client,
                    fresh_parent,
                    child,
                    category=selected_category,
                    max_tags=max_tags,
                    limit=article_limit,
                )
                decision["retagging"] = retagging
                decision["status"] = "child_created" if created else "child_reused"
                totals["articles_checked"] += retagging["articles_found"]
                totals["articles_replaced"] += retagging["articles_replaced"]
                totals["errors"] += retagging["errors"]
            except Exception as exc:
                decision["status"] = "error"
                decision["error"] = str(exc)
                totals["errors"] += 1
                log.exception(
                    "Kunde inte konvertera synonymen %r för %r",
                    synonym,
                    parent_snapshot.get("name"),
                )
            decisions.append(decision)
        if aborted:
            break

    return {
        "audit_version": AUDIT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "category": selected_category,
        "dry_run": dry_run,
        "aborted": aborted,
        "decisions": decisions,
        "totals": totals,
    }


async def audit_location_synonyms(
    store: NewsStore,
    llm_client: Any,
    *,
    max_tags: int = 10,
    article_limit: Optional[int] = None,
    dry_run: bool = False,
    input_fn: Callable[[str], str] = input,
) -> Dict[str, Any]:
    """Backward-compatible LOCATION-specific audit entry point."""
    return await audit_tag_synonyms(
        store,
        llm_client,
        category=DEFAULT_CATEGORY,
        max_tags=max_tags,
        article_limit=article_limit,
        dry_run=dry_run,
        input_fn=input_fn,
    )


async def _close_client(llm_client: Any) -> None:
    close = getattr(llm_client, "aclose", None)
    if callable(close):
        await close()


def _write_report(report: Dict[str, Any], raw_path: Optional[str]) -> Optional[Path]:
    if not raw_path:
        return None
    path = Path(raw_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Rapport sparad: {path}")
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Granska synonymer i en taggkategori interaktivt och konvertera valda "
            "synonymer till barntaggar."
        )
    )
    parser.add_argument(
        "--config",
        help="Sökväg till config.yaml (standard: FEEDSUMMARY_CONFIG eller ./config.yaml)",
    )
    parser.add_argument("--output", help="Spara en JSON-rapport till denna sökväg")
    parser.add_argument(
        "--category",
        default=DEFAULT_CATEGORY,
        help=f"Kategori att granska, skiftlägesokänsligt (standard: {DEFAULT_CATEGORY})",
    )
    parser.add_argument(
        "--max-suggestions",
        type=int,
        default=10,
        help="Max antal förslag från den valda kategorin per artikel (standard: 10)",
    )
    parser.add_argument(
        "--article-limit",
        type=int,
        help="Granska högst detta antal artiklar per skapad barntagg",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Samla interaktiva beslut utan att ändra databasen eller anropa LLM",
    )
    parser.add_argument("--verbose", action="store_true", help="Visa debugloggning")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.max_suggestions < 1:
        parser.error("--max-suggestions måste vara minst 1")
    if args.article_limit is not None and args.article_limit < 1:
        parser.error("--article-limit måste vara minst 1")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    try:
        config_path = _resolve_config_path(args.config)
        config = _load_config(config_path)
        store = create_store(config.get("store", {}))
        selected_category = resolve_category_name(store, args.category)
        llm_client = None if args.dry_run else create_llm_client(config)

        async def run() -> Dict[str, Any]:
            try:
                return await audit_tag_synonyms(
                    store,
                    llm_client,
                    category=selected_category,
                    max_tags=args.max_suggestions,
                    article_limit=args.article_limit,
                    dry_run=args.dry_run,
                )
            finally:
                if llm_client is not None:
                    await _close_client(llm_client)

        report = asyncio.run(run())
        _write_report(report, args.output)
        totals = report["totals"]
        print(
            "Klart: "
            f"{totals['kept']} synonymer behölls, "
            f"{totals['child_tags_created']} barntaggar skapades, "
            f"{totals['existing_tags_reused']} befintliga taggar återanvändes, "
            f"{totals['articles_checked']} artiklar granskades och "
            f"{totals['articles_replaced']} föräldrataggar ersattes."
        )
        if not args.output:
            print("Ange --output SÖKVÄG för att spara en fullständig JSON-rapport.")
        if report["aborted"]:
            return 130
        return 2 if totals["errors"] else 0
    except KeyboardInterrupt:
        log.warning("Avbruten av användaren")
        return 130
    except Exception as exc:
        log.error("Synonymrevisionen misslyckades: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
