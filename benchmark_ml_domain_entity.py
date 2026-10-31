#!/usr/bin/env python3
"""Read-only audit of production ML suggestions against stored DOMAIN_ENTITY tags."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from feedsummary_core.llm_client import get_local_embedding_model
from feedsummary_core.persistence import NewsStore, create_store
from feedsummary_core.tagging_ml.embedding_sgd import (
    EmbeddingClassifierSettings,
    EmbeddingClassifierTagger,
)

log = logging.getLogger("ml_domain_entity_benchmark")
CATEGORY = "DOMAIN_ENTITY"
REPORT_SCHEMA_VERSION = 1


def _resolve_config_path(raw_path: str | None) -> Path:
    candidate = raw_path or os.environ.get("FEEDSUMMARY_CONFIG") or "config.yaml"
    return Path(os.path.expandvars(os.path.expanduser(candidate))).resolve()


def _resolve_relative_path(raw_path: str, config_path: Path) -> str:
    path = Path(os.path.expandvars(os.path.expanduser(raw_path)))
    if not path.is_absolute():
        path = config_path.parent / path
    return str(path.resolve())


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Konfigurationsfilen saknas: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Konfigurationen måste vara ett YAML-objekt: {path}")
    return config


def _normalized(name: Any) -> str:
    return str(name or "").strip().casefold()


def _metric_block(true_positive: int, false_positive: int, false_negative: int) -> dict[str, Any]:
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _iter_articles(store: NewsStore, limit: int | None) -> Iterable[dict[str, Any]]:
    iterator = getattr(store, "iter_articles", None)
    if callable(iterator):
        yield from iterator(limit=limit)
        return
    maximum = int(limit) if limit is not None else 2_147_483_647
    yield from store.list_articles(limit=maximum)


def _actual_domain_tags(store: NewsStore, article_id: str) -> list[dict[str, Any]]:
    tags = [
        dict(tag)
        for tag in store.get_article_tags(article_id)
        if isinstance(tag, dict)
        and str(tag.get("category") or "GENERAL").strip().upper() == CATEGORY
        and str(tag.get("name") or "").strip()
    ]
    tags.sort(key=lambda tag: _normalized(tag.get("name")))
    return tags


def _article_identity(article: dict[str, Any]) -> dict[str, Any]:
    return {
        "article_id": str(article.get("id") or ""),
        "title": str(article.get("title") or "").strip(),
        "url": str(article.get("url") or "").strip(),
        "source": str(article.get("source") or "").strip(),
        "published_ts": article.get("published_ts"),
    }


def audit_predictions(
    store: NewsStore,
    tagger: EmbeddingClassifierTagger,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Compare model suggestions with stored DOMAIN_ENTITY tags without writing data."""
    if not tagger.ready:
        raise ValueError("Modellartefakten måste vara laddad före granskningen")
    if CATEGORY not in tagger.settings.categories:
        raise ValueError(f"Den laddade modellen omfattar inte kategorin {CATEGORY}")

    vocabulary = {_normalized(name) for name in tagger.label_names if _normalized(name)}
    skipped_reasons: Counter[str] = Counter()
    actual_frequency: Counter[str] = Counter()
    suggested_frequency: Counter[str] = Counter()
    correct_frequency: Counter[str] = Counter()
    addition_frequency: Counter[str] = Counter()
    missed_frequency: Counter[str] = Counter()
    article_rows: list[dict[str, Any]] = []

    totals: Counter[str] = Counter()
    eligible_tp = eligible_fp = eligible_fn = 0
    all_tp = all_fp = all_fn = 0

    for article in _iter_articles(store, limit):
        article_id = str(article.get("id") or "").strip()
        if not article_id:
            totals["articles_without_id"] += 1
            continue
        totals["articles"] += 1

        actual_tags = _actual_domain_tags(store, article_id)
        actual_by_key = {
            _normalized(tag.get("name")): str(tag.get("name") or "").strip()
            for tag in actual_tags
        }
        actual_keys = set(actual_by_key)
        identity = _article_identity(article)
        incompatibility = tagger.embedding_incompatibility_reason(article)
        if incompatibility is not None:
            totals["skipped_articles"] += 1
            skipped_reasons[incompatibility] += 1
            article_rows.append(
                {
                    **identity,
                    "status": "not_evaluated",
                    "skip_reason": incompatibility,
                    "actual_tags": sorted(actual_by_key.values(), key=str.casefold),
                    "suggested_tags": [],
                    "matching_tags": [],
                    "suggested_additions": [],
                    "actual_not_suggested": [],
                    "actual_outside_model_vocabulary": sorted(
                        (
                            actual_by_key[key]
                            for key in actual_keys - vocabulary
                        ),
                        key=str.casefold,
                    ),
                    "review_candidate": False,
                }
            )
            continue

        totals["evaluated_articles"] += 1
        for key in actual_keys:
            actual_frequency[key] += 1
        scores = tagger.score_names(article)
        predicted = tagger.predict_tags(article, store, scores=scores)
        predicted = [
            tag
            for tag in predicted
            if str(tag.get("category") or "").strip().upper() == CATEGORY
            and str(tag.get("name") or "").strip()
        ]
        suggested_by_key = {
            _normalized(tag.get("name")): {
                "name": str(tag.get("name") or "").strip(),
                "probability": float(tag.get("ml_probability") or 0.0),
            }
            for tag in predicted
        }
        suggested_keys = set(suggested_by_key)
        for key in suggested_keys:
            suggested_frequency[key] += 1

        matching = actual_keys & suggested_keys
        additions = suggested_keys - actual_keys
        missing_all = actual_keys - suggested_keys
        missing_eligible = (actual_keys & vocabulary) - suggested_keys
        outside_vocabulary = actual_keys - vocabulary

        for key in matching:
            correct_frequency[key] += 1
        for key in additions:
            addition_frequency[key] += 1
        for key in missing_all:
            missed_frequency[key] += 1

        tp = len(matching)
        fp = len(additions)
        eligible_tp += tp
        eligible_fp += fp
        eligible_fn += len(missing_eligible)
        all_tp += tp
        all_fp += fp
        all_fn += len(missing_all)

        totals["actual_assignments"] += len(actual_keys)
        totals["suggested_assignments"] += len(suggested_keys)
        totals["matching_assignments"] += tp
        totals["suggested_additions"] += fp
        totals["actual_not_suggested"] += len(missing_all)
        totals["actual_outside_model_vocabulary"] += len(outside_vocabulary)

        if not actual_keys and suggested_keys:
            status = "untagged_with_suggestions"
            totals["untagged_articles_with_suggestions"] += 1
        elif additions and missing_all:
            status = "review_additions_and_missing"
        elif additions:
            status = "review_suggested_additions"
        elif missing_all:
            status = "model_missed_actual_tags"
        else:
            status = "exact_match"
            totals["exact_match_articles"] += 1

        if additions or missing_all:
            totals["articles_with_differences"] += 1
        if additions:
            totals["articles_with_suggested_additions"] += 1
        if missing_all:
            totals["articles_with_missing_predictions"] += 1

        article_rows.append(
            {
                **identity,
                "status": status,
                "skip_reason": None,
                "actual_tags": sorted(actual_by_key.values(), key=str.casefold),
                "suggested_tags": sorted(
                    suggested_by_key.values(),
                    key=lambda item: (-item["probability"], item["name"].casefold()),
                ),
                "matching_tags": sorted(
                    (actual_by_key[key] for key in matching), key=str.casefold
                ),
                "suggested_additions": sorted(
                    (suggested_by_key[key] for key in additions),
                    key=lambda item: (-item["probability"], item["name"].casefold()),
                ),
                "actual_not_suggested": sorted(
                    (actual_by_key[key] for key in missing_all), key=str.casefold
                ),
                "actual_outside_model_vocabulary": sorted(
                    (actual_by_key[key] for key in outside_vocabulary), key=str.casefold
                ),
                "review_candidate": bool(additions or missing_all),
            }
        )

    tag_names = sorted(
        set(actual_frequency) | set(suggested_frequency) | vocabulary,
        key=str.casefold,
    )
    tag_comparison = [
        {
            "tag": name,
            "in_model_vocabulary": name in vocabulary,
            "actual_articles": actual_frequency[name],
            "suggested_articles": suggested_frequency[name],
            "matching_articles": correct_frequency[name],
            "suggested_additions": addition_frequency[name],
            "actual_not_suggested": missed_frequency[name],
        }
        for name in tag_names
    ]

    evaluated = totals["evaluated_articles"]
    summary_counts = {
        key: totals[key]
        for key in (
            "articles",
            "articles_without_id",
            "evaluated_articles",
            "skipped_articles",
            "exact_match_articles",
            "articles_with_differences",
            "articles_with_suggested_additions",
            "articles_with_missing_predictions",
            "untagged_articles_with_suggestions",
            "actual_assignments",
            "suggested_assignments",
            "matching_assignments",
            "suggested_additions",
            "actual_not_suggested",
            "actual_outside_model_vocabulary",
        )
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "category": CATEGORY,
            "database_scope": "all_articles",
            "evaluation_type": "in_sample_full_database_audit",
            "limit": limit,
            "read_only": True,
        },
        "model": tagger.model_metadata,
        "summary": {
            **summary_counts,
            "embedding_coverage": evaluated / totals["articles"] if totals["articles"] else 0.0,
            "exact_match_rate": (
                totals["exact_match_articles"] / evaluated if evaluated else 0.0
            ),
            "skipped_by_reason": dict(sorted(skipped_reasons.items())),
            "eligible_vocabulary_metrics": _metric_block(
                eligible_tp, eligible_fp, eligible_fn
            ),
            "all_actual_tags_metrics": _metric_block(all_tp, all_fp, all_fn),
        },
        "tag_comparison": tag_comparison,
        "articles": article_rows,
        "warnings": [
            (
                "Detta är en in-sample-audit mot samma databas som modellen tränats "
                "från och är därför inte ett opartiskt mått på generalisering."
            ),
            (
                "Föreslagna tillägg är granskningskandidater, inte automatiska "
                "korrigeringar. Faktiska taggar som modellen missar bör inte tas bort "
                "utan separat innehållsgranskning."
            ),
            (
                "Artiklar utan kompatibel embedding ingår i täckningen men inte i "
                "precision/recall/F1."
            ),
        ],
    }


def _markdown_escape(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ").strip()


def _tag_list(values: Sequence[Any]) -> str:
    rendered = []
    for value in values:
        if isinstance(value, dict):
            rendered.append(f"{value.get('name', '')} ({float(value.get('probability', 0)):.3f})")
        else:
            rendered.append(str(value))
    return _markdown_escape(", ".join(rendered) or "—")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    eligible = summary["eligible_vocabulary_metrics"]
    all_actual = summary["all_actual_tags_metrics"]
    model = report["model"]
    lines = [
        "# ML-audit: DOMAIN_ENTITY",
        "",
        f"Skapad: `{report['created_at']}`",
        "",
        "## Sammanfattning",
        "",
        f"- Artiklar i databasen: **{summary.get('articles', 0)}**",
        (
            f"- Utvärderade med kompatibel embedding: "
            f"**{summary.get('evaluated_articles', 0)}** "
            f"({summary.get('embedding_coverage', 0):.1%})"
        ),
        (
            f"- Artiklar med exakt match: **{summary.get('exact_match_articles', 0)}** "
            f"({summary.get('exact_match_rate', 0):.1%} av utvärderade)"
        ),
        f"- Artiklar med föreslagna tillägg: **{summary.get('articles_with_suggested_additions', 0)}**",
        f"- Helt otaggade artiklar med förslag: **{summary.get('untagged_articles_with_suggestions', 0)}**",
        f"- Artiklar där modellen missar faktiska taggar: **{summary.get('articles_with_missing_predictions', 0)}**",
        f"- Ej utvärderade artiklar: **{summary.get('skipped_articles', 0)}**",
        "",
        "## Modell",
        "",
        f"- Classifier: `{model.get('classifier')}`",
        f"- Kategorier: `{', '.join(model.get('categories') or [])}`",
        f"- Tröskel: `{float(model.get('threshold') or 0):.6f}`",
        f"- Träningsartiklar: `{model.get('training_articles')}`",
        f"- Modellens taggar: `{model.get('label_count')}`",
        f"- Tränad: `{model.get('trained_at')}`",
        "",
        "## Mått",
        "",
        "| Omfång | Precision | Recall | F1 | TP | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Endast taggar i modellens vokabulär | {eligible['precision']:.3f} | "
            f"{eligible['recall']:.3f} | {eligible['f1']:.3f} | "
            f"{eligible['true_positive']} | {eligible['false_positive']} | "
            f"{eligible['false_negative']} |"
        ),
        (
            f"| Alla faktiska DOMAIN_ENTITY-taggar | {all_actual['precision']:.3f} | "
            f"{all_actual['recall']:.3f} | {all_actual['f1']:.3f} | "
            f"{all_actual['true_positive']} | {all_actual['false_positive']} | "
            f"{all_actual['false_negative']} |"
        ),
        "",
        (
            "Precision anger hur ofta ett modellförslag redan finns bland de faktiska "
            "taggarna. Ett modellförslag som saknas i databasen kan vara antingen ett "
            "bra nytt tillägg eller ett falskt positivt och måste granskas manuellt."
        ),
        "",
        "## Föreslagna tillägg att granska",
        "",
        "| Artikel | Faktiska taggar | Föreslagna tillägg |",
        "| --- | --- | --- |",
    ]

    additions = [row for row in report["articles"] if row["suggested_additions"]]
    additions.sort(
        key=lambda row: (
            -max(item["probability"] for item in row["suggested_additions"]),
            row["article_id"],
        )
    )
    if additions:
        for row in additions:
            article = _markdown_escape(row["title"] or row["article_id"])
            lines.append(
                f"| {article} | {_tag_list(row['actual_tags'])} | "
                f"{_tag_list(row['suggested_additions'])} |"
            )
    else:
        lines.append("| — | — | Inga |")

    lines.extend(
        [
            "",
            "## Jämförelse per tagg",
            "",
            "| Tagg | I modellen | Faktisk | Föreslagen | Match | Möjliga tillägg | Missad faktisk |",
            "| --- | :---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["tag_comparison"]:
        lines.append(
            f"| {_markdown_escape(row['tag'])} | {'ja' if row['in_model_vocabulary'] else 'nej'} | "
            f"{row['actual_articles']} | {row['suggested_articles']} | "
            f"{row['matching_articles']} | {row['suggested_additions']} | "
            f"{row['actual_not_suggested']} |"
        )

    lines.extend(
        [
            "",
            "## Alla artiklar",
            "",
            "| Status | Artikel | Faktiska taggar | Modellförslag | Saknas i databasen | Faktisk men ej föreslagen |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report["articles"]:
        article = _markdown_escape(row["title"] or row["article_id"])
        status = row["status"]
        if row.get("skip_reason"):
            status = f"{status}: {row['skip_reason']}"
        lines.append(
            f"| {_markdown_escape(status)} | {article} | {_tag_list(row['actual_tags'])} | "
            f"{_tag_list(row['suggested_tags'])} | {_tag_list(row['suggested_additions'])} | "
            f"{_tag_list(row['actual_not_suggested'])} |"
        )

    lines.extend(["", "## Begränsningar", ""])
    lines.extend(f"- {warning}" for warning in report["warnings"])
    lines.append("")
    return "\n".join(lines)


def write_reports(report: dict[str, Any], output_dir: str) -> tuple[Path, Path]:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "report.json"
    markdown_path = directory / "report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Jämför en tränad produktionsmodells DOMAIN_ENTITY-förslag med "
            "faktiska taggar på alla artiklar. Databasen ändras aldrig."
        )
    )
    parser.add_argument(
        "--config",
        help="Sökväg till config.yaml (standard: FEEDSUMMARY_CONFIG eller ./config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results/ml-domain-entity-audit",
        help="Katalog för report.json och report.md",
    )
    parser.add_argument("--limit", type=int, help="Utvärdera högst detta antal artiklar")
    parser.add_argument("--verbose", action="store_true", help="Visa debugloggning")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit måste vara minst 1")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    store = None
    raw_uri = ""
    try:
        config_path = _resolve_config_path(args.config)
        config = _load_config(config_path)
        settings = EmbeddingClassifierSettings.from_config(
            config,
            embedding_model=get_local_embedding_model(config),
        )
        if CATEGORY not in settings.categories:
            raise ValueError(f"tagging.ml.categories måste innehålla {CATEGORY}")
        settings = replace(
            settings,
            model_path=_resolve_relative_path(settings.model_path, config_path),
        )
        tagger = EmbeddingClassifierTagger(settings)
        if not tagger.load():
            raise FileNotFoundError(
                "Ingen kompatibel modellartefakt kunde laddas. Kör "
                "retrain_tagging_ml.py först."
            )

        store_config = config.get("store")
        if not isinstance(store_config, dict):
            raise TypeError("Konfigurationen saknar 'store'")
        read_only_store_config = dict(store_config)
        read_only_store_config["initialize_schema"] = False
        raw_uri = str(read_only_store_config.get("uri") or "")
        store = create_store(read_only_store_config)

        report = audit_predictions(store, tagger, limit=args.limit)
        json_path, markdown_path = write_reports(report, args.output_dir)
        summary = report["summary"]
        print(f"JSON-rapport: {json_path}")
        print(f"Markdownrapport: {markdown_path}")
        print(
            "Klart: "
            f"{summary.get('articles', 0)} artiklar, "
            f"{summary.get('evaluated_articles', 0)} utvärderade, "
            f"{summary.get('articles_with_suggested_additions', 0)} med möjliga "
            "taggtillägg."
        )
        return 0
    except KeyboardInterrupt:
        log.warning("Avbruten av användaren")
        return 130
    except Exception as error:  # noqa: BLE001 - CLI boundary converts failures to exit codes
        message = str(error)
        if raw_uri:
            message = message.replace(raw_uri, "<mongodb-uri>")
        log.error("ML-granskningen misslyckades (%s): %s", type(error).__name__, message)
        return 2
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    sys.exit(main())
