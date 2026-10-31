#!/usr/bin/env python3
"""Bulk-import tags and aliases from a JSON file."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from feedsummary_core.persistence import NewsStore, create_store


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


def _normalized(value: Any) -> str:
    return str(value or "").strip().casefold()


def _clean_aliases(values: Sequence[str], *, tag_name: str) -> List[str]:
    aliases: List[str] = []
    seen = {_normalized(tag_name)}
    for value in values:
        alias = value.strip()
        normalized = _normalized(alias)
        if normalized and normalized not in seen:
            aliases.append(alias)
            seen.add(normalized)
    return aliases


def _merge_entries_with_shared_identities(
    entries: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge entries connected by a shared alias or an alias matching a name."""
    parents = list(range(len(entries)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        # Keep the first entry in the file as the component's main object.
        first_root, second_root = sorted((left_root, right_root))
        parents[second_root] = first_root

    identity_owner: Dict[str, int] = {}
    for index, entry in enumerate(entries):
        identities = [entry["name"], *entry.get("aliases", [])]
        for identity in identities:
            normalized = _normalized(identity)
            previous_owner = identity_owner.get(normalized)
            if previous_owner is None:
                identity_owner[normalized] = index
            else:
                union(index, previous_owner)

    components: Dict[int, List[int]] = {}
    for index in range(len(entries)):
        components.setdefault(find(index), []).append(index)

    merged_entries: List[Dict[str, Any]] = []
    for indexes in sorted(components.values(), key=lambda values: values[0]):
        main = entries[indexes[0]]
        categories = {
            _normalized(entries[index]["category"])
            for index in indexes
        }
        if len(categories) > 1:
            names = ", ".join(repr(entries[index]["name"]) for index in indexes)
            raise ValueError(
                f"Kan inte slå ihop {names} eftersom objekten har olika kategorier"
            )

        alias_candidates = list(main.get("aliases", []))
        for index in indexes[1:]:
            merged = entries[index]
            alias_candidates.append(merged["name"])
            alias_candidates.extend(merged.get("aliases", []))

        merged_entries.append(
            {
                "name": main["name"],
                "category": main["category"],
                "aliases": _clean_aliases(
                    alias_candidates,
                    tag_name=main["name"],
                ),
            }
        )

    return merged_entries


def load_tag_import(path: Path) -> List[Dict[str, Any]]:
    """Load and validate one tag object or a list of tag objects."""
    if not path.is_file():
        raise FileNotFoundError(f"Importfilen saknas: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Ogiltig JSON i {path} på rad {error.lineno}, kolumn {error.colno}"
        ) from error

    if isinstance(payload, dict):
        raw_entries = [payload]
    elif isinstance(payload, list):
        raw_entries = payload
    else:
        raise ValueError("Importfilen måste innehålla ett objekt eller en lista av objekt")

    entries: List[Dict[str, Any]] = []
    imported_names = set()
    for index, item in enumerate(raw_entries):
        prefix = f"post {index + 1}"
        if not isinstance(item, dict):
            raise ValueError(f"{prefix} måste vara ett JSON-objekt")

        name = item.get("name")
        category = item.get("category")
        aliases = item.get("aliases", [])
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{prefix}.name måste vara en icke-tom sträng")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"{prefix}.category måste vara en icke-tom sträng")
        if not isinstance(aliases, list) or any(
            not isinstance(alias, str) or not alias.strip() for alias in aliases
        ):
            raise ValueError(
                f"{prefix}.aliases måste vara en lista med icke-tomma strängar"
            )

        clean_name = name.strip()
        normalized_name = _normalized(clean_name)
        if normalized_name in imported_names:
            raise ValueError(f"Taggen {clean_name!r} förekommer flera gånger i filen")
        imported_names.add(normalized_name)
        entries.append(
            {
                "name": clean_name,
                "category": category.strip(),
                "aliases": _clean_aliases(aliases, tag_name=clean_name),
            }
        )

    return _merge_entries_with_shared_identities(entries)


def _tags_by_name(store: NewsStore) -> Dict[str, Dict[str, Any]]:
    return {
        _normalized(tag.get("name")): tag
        for tag in (store.get_all_tags() or [])
        if isinstance(tag, dict) and _normalized(tag.get("name"))
    }


def _merge_aliases(tag: Dict[str, Any], aliases: Sequence[str]) -> tuple[List[str], List[str]]:
    merged: List[str] = []
    seen = {_normalized(tag.get("name"))}
    for value in tag.get("synonyms", []) or []:
        alias = str(value).strip()
        normalized = _normalized(alias)
        if normalized and normalized not in seen:
            merged.append(alias)
            seen.add(normalized)

    added: List[str] = []
    for value in aliases:
        alias = value.strip()
        normalized = _normalized(alias)
        if normalized and normalized not in seen:
            merged.append(alias)
            added.append(alias)
            seen.add(normalized)
    return merged, added


def _migration_candidates(
    tags_by_name: Dict[str, Dict[str, Any]],
    main_tag_id: int,
    added_aliases: Sequence[str],
) -> List[int]:
    return sorted(
        {
            int(match["id"])
            for alias in added_aliases
            if (match := tags_by_name.get(_normalized(alias))) is not None
            and match.get("id") is not None
            and int(match["id"]) != main_tag_id
        }
    )


def import_tags(
    store: NewsStore,
    entries: Sequence[Dict[str, Any]],
    *,
    apply: bool = False,
) -> Dict[str, Any]:
    """Create/update tags and migrate stand-alone tags matching new aliases."""
    result: Dict[str, Any] = {
        "apply": apply,
        "created": 0,
        "updated": 0,
        "unchanged": 0,
        "articles_migrated": 0,
        "synonym_tags_deleted": 0,
        "entries": [],
    }

    for entry in entries:
        name = str(entry["name"]).strip()
        category = str(entry["category"]).strip()
        aliases = _clean_aliases(entry.get("aliases", []), tag_name=name)
        tags_by_name = _tags_by_name(store)
        tag = tags_by_name.get(_normalized(name))

        if tag is None:
            entry_result = {
                "name": name,
                "action": "created",
                "aliases_added": aliases,
                "migration_candidates": [
                    tags_by_name[_normalized(alias)]["name"]
                    for alias in aliases
                    if _normalized(alias) in tags_by_name
                ],
            }
            result["created"] += 1
            if apply:
                try:
                    tag = store.create_tag(name, category, "", aliases)
                except TypeError as error:
                    raise RuntimeError(
                        "Den valda databas-backenden saknar stöd för synonymer"
                    ) from error
                if tag is None:
                    # Another writer may have created it after our lookup.
                    tag = _tags_by_name(store).get(_normalized(name))
                    if tag is None:
                        raise RuntimeError(f"Kunde inte skapa taggen {name!r}")
                    merged, added = _merge_aliases(tag, aliases)
                    if added:
                        tag = store.update_tag(int(tag["id"]), synonyms=merged)
                        if tag is None:
                            raise RuntimeError(f"Kunde inte uppdatera taggen {name!r}")
                added_aliases = aliases
            else:
                added_aliases = aliases
                tag = {"id": -1, "name": name}
        else:
            merged_aliases, added_aliases = _merge_aliases(tag, aliases)
            if added_aliases:
                entry_result = {
                    "name": name,
                    "action": "updated",
                    "aliases_added": added_aliases,
                    "migration_candidates": [
                        tags_by_name[_normalized(alias)]["name"]
                        for alias in added_aliases
                        if _normalized(alias) in tags_by_name
                        and int(tags_by_name[_normalized(alias)]["id"]) != int(tag["id"])
                    ],
                }
                result["updated"] += 1
                if apply:
                    updated = store.update_tag(int(tag["id"]), synonyms=merged_aliases)
                    if updated is None:
                        raise RuntimeError(f"Kunde inte uppdatera taggen {name!r}")
                    tag = updated
            else:
                entry_result = {
                    "name": name,
                    "action": "unchanged",
                    "aliases_added": [],
                    "migration_candidates": [],
                }
                result["unchanged"] += 1

        main_tag_id = int(tag["id"])
        synonym_tag_ids = _migration_candidates(
            tags_by_name, main_tag_id, added_aliases
        )
        if apply and synonym_tag_ids:
            migrate = getattr(store, "migrate_synonym_to_main_tag", None)
            if not callable(migrate):
                raise RuntimeError(
                    "Den valda databas-backenden saknar stöd för synonymmigrering"
                )
            migrated, deleted = migrate(main_tag_id, synonym_tag_ids)
            result["articles_migrated"] += int(migrated)
            result["synonym_tags_deleted"] += int(deleted)

        result["entries"].append(entry_result)

    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bulk-importera taggar och alias från JSON. Utan --apply görs endast "
            "en förhandsgranskning."
        )
    )
    parser.add_argument("json_file", type=Path, help="JSON-fil med en tagg eller en lista")
    parser.add_argument(
        "--config",
        help="Sökväg till config.yaml (standard: FEEDSUMMARY_CONFIG eller ./config.yaml)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Skriv ändringarna till databasen",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    store: Optional[NewsStore] = None
    try:
        entries = load_tag_import(args.json_file.resolve())
        config = _load_config(_resolve_config_path(args.config))
        store_config = config.get("store", {})
        if not isinstance(store_config, dict):
            raise ValueError("Konfigurationens 'store' måste vara ett YAML-objekt")
        store = create_store(store_config)
        result = import_tags(store, entries, apply=args.apply)

        mode = "IMPORT" if args.apply else "FÖRHANDSGRANSKNING"
        print(
            f"{mode}: {result['created']} nya, {result['updated']} uppdaterade, "
            f"{result['unchanged']} oförändrade taggar."
        )
        for item in result["entries"]:
            aliases = ", ".join(item["aliases_added"]) or "inga nya alias"
            print(f"- {item['name']}: {item['action']} ({aliases})")
            if item["migration_candidates"]:
                print(
                    "  Fristående synonymtaggar: "
                    + ", ".join(item["migration_candidates"])
                )
        if args.apply:
            print(
                f"Migrerade artikelkopplingar: {result['articles_migrated']}; "
                f"borttagna synonymtaggar: {result['synonym_tags_deleted']}."
            )
        else:
            print("Inget ändrades. Kör igen med --apply för att importera.")
        return 0
    except KeyboardInterrupt:
        print("Importen avbröts.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Importen misslyckades: {error}", file=sys.stderr)
        return 2
    finally:
        if store is not None:
            close = getattr(store, "close", None)
            if callable(close):
                close()


if __name__ == "__main__":
    raise SystemExit(main())
