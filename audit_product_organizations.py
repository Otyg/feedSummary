#!/usr/bin/env python3
"""Pair PRODUCT tags with organizations and audit their article assignments."""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from feedsummary_core.persistence import NewsStore, create_store

from audit_location_synonyms import _load_config, _resolve_config_path


log = logging.getLogger("product_organization_audit")
PRODUCT = "PRODUCT"
ORGANIZATION = "ORGANIZATION"


def _tokens(value: Any) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return tuple(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def _aliases(tag: dict[str, Any]) -> list[str]:
    values = tag.get("synonyms") or []
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _category_tags(tags: Sequence[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    return sorted(
        (tag for tag in tags if str(tag.get("category") or "GENERAL").casefold() == category.casefold()
         and tag.get("id") is not None and _tokens(tag.get("name"))),
        key=lambda tag: (str(tag["name"]).casefold(), int(tag["id"])),
    )


def discover_pairs(tags: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Match an organization's name or alias at the start of a product name."""
    products = _category_tags(tags, PRODUCT)
    organizations = _category_tags(tags, ORGANIZATION)
    if not products or not organizations:
        raise ValueError("Både PRODUCT och ORGANIZATION måste innehålla taggar")

    identities: list[tuple[dict[str, Any], str, str, tuple[str, ...]]] = []
    for organization in organizations:
        for kind, value in [("name", str(organization["name"])),
                            *(("alias", alias) for alias in _aliases(organization))]:
            words = _tokens(value)
            # A one- or two-character alias is too easy to match accidentally.
            if words and (kind == "name" or len("".join(words)) >= 3):
                identities.append((organization, kind, value, words))

    pairs: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    for product in products:
        product_names = [("name", str(product["name"])),
                         *(("alias", alias) for alias in _aliases(product))]
        matches: list[tuple[tuple[int, int, int, int], dict[str, Any], str, str]] = []
        for product_kind, product_name in product_names:
            product_words = _tokens(product_name)
            if not product_words:
                continue
            for organization, organization_kind, organization_name, organization_words in identities:
                if product_words[:len(organization_words)] != organization_words:
                    continue
                score = (
                    int(product_kind == "name"),
                    len(organization_words),
                    int(organization_kind == "name"),
                    len("".join(organization_words)),
                )
                matches.append((score, organization, product_name, organization_name))

        if not matches:
            unmatched.append({"product_id": int(product["id"]), "product": str(product["name"])})
            continue
        best_score = max(match[0] for match in matches)
        best = [match for match in matches if match[0] == best_score]
        organization_ids = {int(match[1]["id"]) for match in best}
        if len(organization_ids) != 1:
            ambiguous.append({
                "product_id": int(product["id"]),
                "product": str(product["name"]),
                "organization_candidates": sorted(
                    {match[1]["name"] for match in best}, key=str.casefold
                ),
            })
            continue
        _, organization, product_name, organization_name = best[0]
        pairs.append({
            "product_id": int(product["id"]),
            "product": str(product["name"]),
            "organization_id": int(organization["id"]),
            "organization": str(organization["name"]),
            "matched_product_name": product_name,
            "matched_organization_name": organization_name,
        })
    return {"pairs": pairs, "ambiguous": ambiguous, "unmatched": unmatched,
            "product_tags": len(products), "organization_tags": len(organizations)}


def load_reviewed_pairs(path: Path, tags: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Load selected pairs from a previous report and validate current tag IDs."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("pairs"), list):
        raise ValueError("Parfilen måste vara en JSON-rapport med listan 'pairs'")
    by_id = {int(tag["id"]): tag for tag in tags if isinstance(tag, dict) and tag.get("id") is not None}
    result: list[dict[str, Any]] = []
    seen_products: set[int] = set()
    for index, item in enumerate(payload["pairs"], 1):
        if not isinstance(item, dict):
            raise ValueError(f"Par {index} måste vara ett JSON-objekt")
        try:
            product_id, organization_id = int(item["product_id"]), int(item["organization_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Par {index} saknar giltiga tagg-ID") from exc
        product, organization = by_id.get(product_id), by_id.get(organization_id)
        if (not product or not organization
                or str(product.get("category") or "").casefold() != PRODUCT.casefold()
                or str(organization.get("category") or "").casefold() != ORGANIZATION.casefold()
                or str(item.get("product") or "").casefold() != str(product.get("name") or "").casefold()
                or str(item.get("organization") or "").casefold() != str(organization.get("name") or "").casefold()):
            raise ValueError(f"Par {index} stämmer inte längre med taggarna i databasen")
        if product_id in seen_products:
            raise ValueError(f"Produkten {product['name']} finns i flera par")
        seen_products.add(product_id)
        result.append({
            "product_id": product_id, "product": str(product["name"]),
            "organization_id": organization_id,
            "organization": str(organization["name"]),
            "matched_product_name": item.get("matched_product_name"),
            "matched_organization_name": item.get("matched_organization_name"),
        })
    return result


def audit_articles(store: NewsStore, pairs: Sequence[dict[str, Any]], *, apply: bool = False) -> dict[str, Any]:
    """Check each article once; add each missing organization once if requested."""
    articles: dict[str, dict[str, Any]] = {}
    for offset in range(0, len(pairs), 200):
        names = [pair["product"] for pair in pairs[offset:offset + 200]]
        for article in store.get_articles_by_tags(names, match_mode="any") or []:
            article_id = str(article.get("id") or "").strip()
            if article_id:
                articles[article_id] = {"id": article_id, "title": str(article.get("title") or "")}

    by_product_id = {int(pair["product_id"]): pair for pair in pairs}
    missing: list[dict[str, Any]] = []
    present = 0
    errors = 0
    added = 0
    checked = 0
    for article_id, article in sorted(articles.items()):
        article_tags = store.get_article_tags(article_id) or []
        tag_ids = {int(tag["id"]) for tag in article_tags if tag.get("id") is not None}
        required: dict[int, list[dict[str, Any]]] = {}
        for product_id in sorted(tag_ids & by_product_id.keys()):
            pair = by_product_id[product_id]
            required.setdefault(int(pair["organization_id"]), []).append(pair)
        if not required:
            continue
        checked += 1
        for organization_id, trigger_pairs in sorted(required.items()):
            if organization_id in tag_ids:
                present += 1
                continue
            item = {
                "article_id": article_id,
                "title": article["title"],
                "organization_id": organization_id,
                "organization": trigger_pairs[0]["organization"],
                "products": [pair["product"] for pair in trigger_pairs],
                "status": "missing",
                "error": None,
            }
            if apply:
                try:
                    success = store.add_tag_to_article(article_id, organization_id)
                    if not success:
                        success = any(
                            int(tag.get("id") or 0) == organization_id
                            for tag in store.get_article_tags(article_id) or []
                        )
                    if not success:
                        raise RuntimeError("Kunde inte lägga till organisationstaggen")
                    item["status"] = "added"
                    tag_ids.add(organization_id)
                    added += 1
                except Exception as exc:
                    item["status"] = "error"
                    item["error"] = str(exc)
                    errors += 1
                    log.exception("Kunde inte uppdatera artikel %s", article_id)
            missing.append(item)
    return {
        "articles_checked": checked,
        "articles_with_missing": len({item["article_id"] for item in missing}),
        "organization_tags_present": present,
        "organization_tags_missing": len(missing),
        "organization_tags_added": added,
        "errors": errors,
        "missing": missing,
    }


def run_audit(store: NewsStore, *, apply: bool = False,
              pairs_file: Optional[Path] = None) -> dict[str, Any]:
    tags = [tag for tag in store.get_all_tags() or [] if isinstance(tag, dict)]
    discovery = discover_pairs(tags)
    pairs = load_reviewed_pairs(pairs_file, tags) if pairs_file else discovery["pairs"]
    article_audit = audit_articles(store, pairs, apply=apply)
    return {
        "audit_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "apply": apply,
        "pair_source": str(pairs_file) if pairs_file else "discovered",
        "pairs": pairs,
        "ambiguous": discovery["ambiguous"],
        "unmatched": discovery["unmatched"],
        "totals": {
            "product_tags": discovery["product_tags"],
            "organization_tags": discovery["organization_tags"],
            "pairs": len(pairs),
            "ambiguous_products": len(discovery["ambiguous"]),
            "unmatched_products": len(discovery["unmatched"]),
            **{key: value for key, value in article_audit.items() if key != "missing"},
        },
        "missing": article_audit["missing"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Para PRODUCT med ORGANIZATION och kontrollera artiklarnas organisationstaggar."
    )
    parser.add_argument("--config", help="Sökväg till config.yaml")
    parser.add_argument("--output", help="Spara par och artikelgranskning som JSON")
    parser.add_argument("--pairs-file", type=Path,
                        help="Använd en granskad JSON-rapport som parlista")
    parser.add_argument("--apply", action="store_true",
                        help="Lägg till saknade organisationstaggar på artiklarna")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = _load_config(_resolve_config_path(args.config))
        store = create_store(config.get("store", {}))
        report = run_audit(store, apply=args.apply, pairs_file=args.pairs_file)
        for pair in report["pairs"]:
            print(f"{pair['product']} → {pair['organization']}")
        totals = report["totals"]
        print(
            f"{totals['pairs']} par; {totals['ambiguous_products']} tvetydiga och "
            f"{totals['unmatched_products']} produkter utan automatiskt par; "
            f"{totals['articles_checked']} artiklar kontrollerade; "
            f"{totals['organization_tags_missing']} organisationstaggar saknades; "
            f"{totals['organization_tags_added']} tillagda; {totals['errors']} fel."
        )
        for item in report["missing"][:20]:
            print(f"  {item['article_id']}: {', '.join(item['products'])} → "
                  f"{item['organization']} ({item['status']})")
        if len(report["missing"]) > 20:
            print(f"  ... ytterligare {len(report['missing']) - 20} poster finns i rapporten")
        if args.output:
            path = Path(args.output).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Rapport sparad: {path}")
        else:
            print("Ange --output SÖKVÄG för att spara alla par och artikelresultat.")
        return 2 if totals["errors"] else 0
    except Exception as exc:
        log.error("Granskningen misslyckades: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
