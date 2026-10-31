#!/usr/bin/env python3
"""Find related existing tags in a category, then review their relationships."""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from feedsummary_core.persistence import NewsStore, create_store

from audit_location_synonyms import (
    _load_config,
    _resolve_config_path,
    resolve_category_name,
)


log = logging.getLogger("tag_synonym_candidates")


def _key(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def _identities(tag: dict[str, Any]) -> set[str]:
    return {
        key
        for value in [tag.get("name"), *(tag.get("synonyms") or [])]
        if (key := _key(value))
    }


def _pair_match(
    left: dict[str, Any],
    right: dict[str, Any],
    rare_words: set[str],
    anchor_names: set[frozenset[str]],
) -> Optional[tuple[str, int]]:
    """Return a direct, sufficiently specific match and its review priority."""
    left_name, right_name = _key(left.get("name")), _key(right.get("name"))
    if str(left.get("name") or "").strip().casefold() == str(right.get("name") or "").strip().casefold():
        return "samma namn", 100
    if any(len(identity.replace(" ", "")) >= 3 for identity in _identities(left) & _identities(right)):
        return "samma namn eller befintligt alias", 95
    if (len(left_name.replace(" ", "")) >= 5
            and left_name.replace(" ", "") == right_name.replace(" ", "")):
        return "samma namn utan skiljetecken", 90
    left_order, right_order = left_name.split(), right_name.split()
    left_words, right_words = set(left_order), set(right_order)
    shared = left_words & right_words
    if not shared:
        return None
    informative = {word for word in shared if len(word) >= 5}
    if not informative:
        return None
    # A single word in a long product name is usually a vendor or platform,
    # not evidence that the products themselves are interchangeable or related.
    if len(shared) / len(left_words) < 0.5 or len(shared) / len(right_words) < 0.5:
        return None
    if left_words <= right_words or right_words <= left_words:
        if len(informative) >= 2 or informative & rare_words:
            return "namn ingår i annat namn", 70 + len(informative)
    # A shared vendor or product-type word in the same position (Microsoft
    # Azure/Edge, Huly/Logix Platform) is weak evidence. An actual base tag
    # such as NetScaler or Windows Server can anchor related variants.
    shifted = any(left_order.index(word) != right_order.index(word) for word in informative)
    anchored = any(
        frozenset(words) in anchor_names
        and width / len(left_words) >= 0.5
        and width / len(right_words) >= 0.5
        for width in (1, 2)
        for words in combinations(sorted(shared), width)
    )
    if not shifted and not anchored:
        return None
    if len(informative) >= 2:
        return "flera gemensamma namnord", 60 + len(informative)
    if informative & rare_words:
        return "sällsynt gemensamt namnord", 50
    return None


def find_candidate_groups(store: NewsStore, category: str) -> list[dict[str, Any]]:
    """Scan the category and form groups whose members all match directly."""
    tags = sorted(
        (
            tag for tag in store.get_all_tags() or []
             if isinstance(tag, dict)
             and str(tag.get("category") or "GENERAL").casefold() == category.casefold()
             and tag.get("id") is not None and _key(tag.get("name"))),
        key=lambda tag: (_key(tag["name"]), int(tag["id"])),
    )
    word_counts = Counter(word for tag in tags for word in set(_key(tag["name"]).split()))
    rare_words = {
        word for word, count in word_counts.items()
        if len(word) >= 8 and count <= max(12, len(tags) // 100)
    }
    anchor_names = {
        frozenset(_key(tag["name"]).split())
        for tag in tags
        if 1 <= len(set(_key(tag["name"]).split())) <= 2
    }

    # Only compare pairs sharing an identity, a useful name word or a compact
    # spelling. This keeps a category-wide scan practical for large tag sets.
    indexed: dict[str, list[int]] = defaultdict(list)
    for index, tag in enumerate(tags):
        name_key = _key(tag["name"])
        for identity in _identities(tag):
            indexed[f"identity:{identity}"].append(index)
        for word in set(name_key.split()):
            if len(word) >= 5:
                indexed[f"word:{word}"].append(index)
        if len(name_key.replace(" ", "")) >= 5:
            indexed[f"compact:{name_key.replace(' ', '')}"].append(index)
    candidate_pairs: set[tuple[int, int]] = set()
    for indexes in indexed.values():
        for position, left in enumerate(indexes):
            for right in indexes[position + 1:]:
                candidate_pairs.add((min(left, right), max(left, right)))

    matches: dict[tuple[int, int], tuple[str, int]] = {}
    for left, right in sorted(candidate_pairs):
        match = _pair_match(tags[left], tags[right], rare_words, anchor_names)
        if match:
            matches[(left, right)] = match

    # Connected components allow unrelated endpoints to be joined through a
    # chain of generic words. Merge only when every cross-pair matches.
    components: dict[int, set[int]] = {index: {index} for index in range(len(tags))}
    component_of = list(range(len(tags)))
    for (left, right), (_, score) in sorted(
        matches.items(), key=lambda item: (-item[1][1], item[0])
    ):
        left_component, right_component = component_of[left], component_of[right]
        if left_component == right_component:
            continue
        if not all(
            (min(a, b), max(a, b)) in matches
            for a in components[left_component]
            for b in components[right_component]
        ):
            continue
        for index in components[right_component]:
            component_of[index] = left_component
        components[left_component].update(components.pop(right_component))

    groups = []
    for member_indexes in sorted(components.values(), key=min):
        if len(member_indexes) < 2:
            continue
        indexes = sorted(member_indexes)
        groups.append({
            "tags": [{"id": int(tags[index]["id"]), "name": str(tags[index]["name"]),
                      "synonyms": list(tags[index].get("synonyms") or [])} for index in indexes],
            "links": [
                {"left_id": int(tags[left]["id"]), "right_id": int(tags[right]["id"]),
                 "reason": matches[(left, right)][0]}
                for position, left in enumerate(indexes)
                for right in indexes[position + 1:]
            ],
        })
    return groups


def _fresh_tag(store: NewsStore, tag_id: int, category: str) -> dict[str, Any]:
    tag = next((item for item in store.get_all_tags() or [] if int(item.get("id") or 0) == tag_id), None)
    if not tag or str(tag.get("category") or "GENERAL").casefold() != category.casefold():
        raise ValueError(f"Tagg {tag_id} saknas i kategorin {category}")
    return tag


def _relation_ids(store: NewsStore, tag_id: int, side: str) -> list[int]:
    relations = store.get_tag_relations(tag_id) or {}
    return [int(tag["id"]) for tag in relations.get(side, []) if tag.get("id") is not None]


def apply_choice(store: NewsStore, category: str, main_id: int, other_id: int, action: str) -> dict[str, Any]:
    """Apply a reviewed synonym or child choice to existing same-category tags."""
    if main_id == other_id or action not in {"synonym", "child"}:
        raise ValueError("Ogiltigt taggval")
    main = _fresh_tag(store, main_id, category)
    other = _fresh_tag(store, other_id, category)
    if action == "child":
        parents = _relation_ids(store, other_id, "parents")
        child_names = _identities(other)
        previous_aliases = list(main.get("synonyms") or [])
        retained_aliases = [alias for alias in previous_aliases if _key(alias) not in child_names]
        store.set_tag_relations(other_id, parent_ids=sorted(set(parents) | {main_id}))
        if retained_aliases != previous_aliases:
            try:
                updated = store.update_tag(main_id, synonyms=retained_aliases)
                if not isinstance(updated, dict):
                    raise RuntimeError("Kunde inte ta bort barntaggens alias från huvudtaggen")
            except Exception:
                try:
                    store.set_tag_relations(other_id, parent_ids=parents)
                except Exception:
                    log.exception("Kunde inte återställa barntaggens relationer")
                raise
        return {"status": "child", "articles_migrated": 0}

    if _relation_ids(store, other_id, "parents") or _relation_ids(store, other_id, "children"):
        raise ValueError(f"{other['name']} har relationer; välj barntagg eller granska relationerna först")
    migrate = getattr(store, "migrate_synonym_to_main_tag", None)
    if not callable(migrate):
        raise RuntimeError("Databasen saknar stöd för synonymmigrering")
    aliases = list(main.get("synonyms") or [])
    seen = {_key(main["name"]), *(_key(alias) for alias in aliases)}
    for alias in [other["name"], *(other.get("synonyms") or [])]:
        key = _key(alias)
        if key and key not in seen:
            aliases.append(str(alias).strip())
            seen.add(key)
    try:
        updated = store.update_tag(main_id, synonyms=aliases)
    except TypeError as exc:
        raise RuntimeError("Databasen saknar stöd för sparade synonymer") from exc
    if not isinstance(updated, dict):
        raise RuntimeError("Kunde inte uppdatera huvudtaggens synonymer")
    try:
        migrated, deleted = migrate(main_id, [other_id])
        if int(deleted) != 1:
            raise RuntimeError("Synonymtaggen kunde inte tas bort efter migreringen")
    except Exception:
        try:
            store.update_tag(main_id, synonyms=list(main.get("synonyms") or []))
        except Exception:
            log.exception("Kunde inte återställa huvudtaggens synonymer")
        raise
    return {"status": "synonym", "articles_migrated": int(migrated)}


def _ask_main(group: dict[str, Any], input_fn: Callable[[str], str]) -> Optional[int | str]:
    tags = group["tags"]
    while True:
        answer = input_fn("Välj huvudtaggens nummer, [s] hoppa över gruppen, [q] avbryt: ").strip().casefold()
        if answer in {"s", ""}:
            return None
        if answer == "q":
            return "quit"
        if answer.isdecimal() and 1 <= int(answer) <= len(tags):
            return int(tags[int(answer) - 1]["id"])
        print("Ogiltigt val.")


def _ask_relation(name: str, input_fn: Callable[[str], str]) -> str:
    while True:
        answer = input_fn(f"{name}: [y] synonym, [b] barntagg, [s] hoppa över, [q] avbryt: ").strip().casefold()
        if answer in {"y", "synonym"}:
            return "synonym"
        if answer in {"b", "barn", "child"}:
            return "child"
        if answer in {"s", ""}:
            return "skip"
        if answer == "q":
            return "quit"
        print("Ogiltigt val.")


def audit_candidates(
    store: NewsStore,
    *,
    category: str,
    dry_run: bool = False,
    input_fn: Callable[[str], str] = input,
) -> dict[str, Any]:
    selected_category = resolve_category_name(store, category)
    groups = find_candidate_groups(store, selected_category)
    print(f"Hittade {len(groups)} kandidatgrupp{'er' if len(groups) != 1 else ''} i {selected_category}:")
    for number, group in enumerate(groups, 1):
        print(f"  {number}. " + ", ".join(tag["name"] for tag in group["tags"]))
    if dry_run:
        print("Förhandsgranskning: inga databasändringar görs.")
    elif groups:
        print("Synonymval flyttar artikelkopplingar till huvudtaggen och tar bort den andra taggen.")

    decisions: list[dict[str, Any]] = []
    aborted = False
    for number, group in enumerate(groups, 1):
        print(f"\nGrupp {number}/{len(groups)}:")
        for index, tag in enumerate(group["tags"], 1):
            aliases = ", ".join(str(alias) for alias in tag["synonyms"])
            suffix = f"; befintliga alias: {aliases}" if aliases else ""
            print(f"  {index}. {tag['name']} (ID {tag['id']}){suffix}")
        main_id = _ask_main(group, input_fn)
        if main_id == "quit":
            aborted = True
            break
        if main_id is None:
            continue
        main_name = next(tag["name"] for tag in group["tags"] if tag["id"] == main_id)
        selections = []
        for tag in group["tags"]:
            if tag["id"] == main_id:
                continue
            action = _ask_relation(tag["name"], input_fn)
            if action == "quit":
                aborted = True
                break
            selections.append((tag, action))
        if aborted:
            break
        for tag, action in selections:
            decision = {"main_id": main_id, "main": main_name,
                        "tag_id": tag["id"], "tag": tag["name"],
                        "choice": action, "status": action, "error": None}
            if action in {"synonym", "child"}:
                if dry_run:
                    decision["status"] = f"would_{action}"
                else:
                    try:
                        decision.update(apply_choice(store, selected_category, main_id, tag["id"], action))
                    except Exception as exc:
                        decision["status"] = "error"
                        decision["error"] = str(exc)
                        log.exception("Kunde inte behandla taggen %s", tag["name"])
            decisions.append(decision)
    return {
        "audit_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "category": selected_category,
        "dry_run": dry_run,
        "aborted": aborted,
        "groups": groups,
        "decisions": decisions,
        "totals": {"groups_found": len(groups),
                   "synonyms": sum(item["status"] == "synonym" for item in decisions),
                   "children": sum(item["status"] == "child" for item in decisions),
                   "errors": sum(item["status"] == "error" for item in decisions),
                   "articles_migrated": sum(item.get("articles_migrated", 0) for item in decisions)},
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Hitta troliga taggsynonymer i en kategori och välj huvudtagg samt relation.")
    parser.add_argument("--config", help="Sökväg till config.yaml")
    parser.add_argument("--category", required=True, help="Taggkategori att granska")
    parser.add_argument("--output", help="Spara JSON-rapport")
    parser.add_argument("--dry-run", action="store_true", help="Förhandsgranska val utan databasändringar")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = _load_config(_resolve_config_path(args.config))
        store = create_store(config.get("store", {}))
        report = audit_candidates(store, category=args.category, dry_run=args.dry_run)
        if args.output:
            path = Path(args.output).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Rapport sparad: {path}")
        print("Klart: " + ", ".join(f"{key}={value}" for key, value in report["totals"].items()))
        return 130 if report["aborted"] else (2 if report["totals"]["errors"] else 0)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        log.error("Granskningen misslyckades: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
