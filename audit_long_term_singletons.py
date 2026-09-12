#!/usr/bin/env python3
"""Audit singleton event clusters without changing long-term state."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from feedsummary_core.long_term import ClusterStatus
from feedsummary_core.persistence import create_store

from calibrate_long_term_clustering import (
    calculate_similarity_report,
    collect_compatible_articles,
    select_review_pairs,
    write_review_template,
)
from run_long_term_clustering import (
    _config_path,
    _load_config,
    _profile_config,
    _resolved_store_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--pairs-per-band", type=int, default=15)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--review-template", type=Path)
    return parser


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _threshold_counts(scores: list[float], thresholds: Sequence[float]) -> dict[str, int]:
    return {
        str(threshold): sum(score >= threshold for score in scores)
        for threshold in thresholds
    }


def execute_audit(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _config_path(args.config)
    config = _load_config(config_path)
    long_term, _ = _profile_config(config, args.profile)
    clustering = long_term.get("clustering") or {}
    reports = long_term.get("reports") or {}
    threshold = float(clustering.get("similarity_threshold", 0.80))
    ambiguity_margin = float(clustering.get("ambiguity_margin", 0.03))
    candidate_window_days = int(clustering.get("candidate_window_days", 60))
    max_clusters = int(reports.get("max_clusters", 10000))
    store_config = _resolved_store_config(config, config_path)
    store_config["initialize_schema"] = False
    store = create_store(store_config)
    try:
        clusters = store.list_threat_clusters(
            args.profile,
            statuses=[status.value for status in ClusterStatus],
            limit=max_clusters,
        )
        size_distribution = Counter(int(row.get("member_count") or 0) for row in clusters)
        singleton_clusters = [
            row for row in clusters if int(row.get("member_count") or 0) == 1
        ]
        memberships: list[dict[str, Any]] = []
        for cluster in singleton_clusters:
            rows = store.list_cluster_memberships(str(cluster["id"]))
            if len(rows) == 1:
                memberships.append(rows[0])

        article_ids = [str(row.get("article_id") or "") for row in memberships]
        documents = store.get_articles_by_ids([value for value in article_ids if value])
        articles, selection = collect_compatible_articles(documents)
        article_by_id = {article.article_id: article for article in articles}
        compatible_ids = set(article_by_id)
        compatible_memberships = [
            row for row in memberships if str(row.get("article_id") or "") in compatible_ids
        ]

        assignment_reasons = Counter(
            str(row.get("assignment_reason") or "unknown")
            for row in compatible_memberships
        )
        rejected_scores = [
            float(row["similarity"])
            for row in compatible_memberships
            if row.get("assignment_reason") == "below_similarity_threshold"
            and row.get("similarity") is not None
        ]
        indicator_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for membership in compatible_memberships:
            for indicator in membership.get("strong_indicators") or []:
                indicator_members[str(indicator)].append(membership)
        repeated_indicators = []
        for indicator, rows in indicator_members.items():
            cluster_ids = {str(row.get("cluster_id") or "") for row in rows}
            if len(cluster_ids) < 2:
                continue
            examples = []
            for row in rows[:5]:
                evidence = row.get("evidence") or {}
                examples.append(
                    {
                        "cluster_id": str(row.get("cluster_id") or ""),
                        "title": str(evidence.get("title") or ""),
                        "source": str(evidence.get("source") or ""),
                    }
                )
            repeated_indicators.append(
                {
                    "indicator": indicator,
                    "cluster_count": len(cluster_ids),
                    "examples": examples,
                }
            )
        repeated_indicators.sort(
            key=lambda row: (
                not str(row["indicator"]).startswith("cve:"),
                -int(row["cluster_count"]),
                str(row["indicator"]),
            )
        )

        review_rows = select_review_pairs(
            articles,
            threshold=threshold,
            ambiguity_margin=ambiguity_margin,
            pairs_per_band=args.pairs_per_band,
            max_time_distance_days=candidate_window_days,
        )
        if args.review_template:
            write_review_template(args.review_template.resolve(), review_rows)

        total_articles = sum(size * count for size, count in size_distribution.items())
        total_clusters = sum(size_distribution.values())
        thresholds = (0.75, 0.78, 0.80, 0.81, threshold)
        result = {
            "schema_version": 1,
            "read_only": True,
            "profile_id": args.profile,
            "configuration": {
                "similarity_threshold": threshold,
                "ambiguity_margin": ambiguity_margin,
                "candidate_window_days": candidate_window_days,
            },
            "cluster_sizes": {
                "articles": total_articles,
                "clusters": total_clusters,
                "mean": total_articles / total_clusters if total_clusters else 0.0,
                "distribution": {
                    str(size): count for size, count in sorted(size_distribution.items())
                },
            },
            "singletons": {
                "count": len(singleton_clusters),
                "share_of_clusters": (
                    len(singleton_clusters) / total_clusters if total_clusters else 0.0
                ),
                "compatible_articles": len(articles),
                "assignment_reasons": dict(sorted(assignment_reasons.items())),
                "rejected_similarity_at_or_above": _threshold_counts(
                    rejected_scores, thresholds
                ),
                "repeated_strong_indicator_count": len(repeated_indicators),
                "repeated_strong_indicators": repeated_indicators[:50],
            },
            "singleton_similarity": calculate_similarity_report(
                articles,
                thresholds=thresholds,
            ),
            "review": {
                "pair_count": len(review_rows),
                "pairs_per_band": args.pairs_per_band,
                "candidate_window_days": candidate_window_days,
                "template": str(args.review_template.resolve())
                if args.review_template
                else None,
            },
            "selection": selection,
        }
        if args.report:
            _write_new_json(args.report.resolve(), result)
        return result
    finally:
        store.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = execute_audit(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"Singelton-audit misslyckades: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
