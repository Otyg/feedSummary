#!/usr/bin/env python3
"""Read-only calibration report for long-term event-clustering thresholds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from feedsummary_core.persistence import create_store

REPORT_SCHEMA_VERSION = 1
DEFAULT_THRESHOLDS = (0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.90)


@dataclass(frozen=True)
class EmbeddedArticle:
    article_id: str
    title: str
    source: str
    published_ts: int
    vector: tuple[float, ...]
    signature: tuple[str, int, str]


def _config_path(raw: str | None) -> Path:
    value = raw or os.environ.get("FEEDSUMMARY_CONFIG") or "config.yaml"
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Konfigurationen måste vara ett YAML-objekt: {path}")
    return config


def _store_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    raw = config.get("store")
    if not isinstance(raw, dict):
        raise TypeError("Konfigurationen saknar 'store'")
    result = dict(raw)
    result["initialize_schema"] = False
    if result.get("path"):
        path = Path(os.path.expandvars(os.path.expanduser(str(result["path"]))))
        if not path.is_absolute():
            path = config_path.parent / path
        result["path"] = str(path.resolve())
    return result


def _article_ts(article: dict[str, Any]) -> int:
    for field in ("published_ts", "fetched_at"):
        try:
            value = int(article.get(field) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0


def collect_compatible_articles(
    articles: Iterable[dict[str, Any]],
    *,
    requested_model: str = "",
) -> tuple[list[EmbeddedArticle], dict[str, Any]]:
    """Select the most common compatible similarity-embedding signature."""

    candidates: list[EmbeddedArticle] = []
    skipped: Counter[str] = Counter()
    signature_counts: Counter[tuple[str, int, str]] = Counter()
    requested_model = requested_model.strip()
    total = 0

    for article in articles:
        total += 1
        article_id = str(article.get("id") or "").strip()
        raw_vector = article.get("similarity_embedding_vector")
        model = str(article.get("similarity_embedding_model") or "").strip()
        instruction = str(article.get("similarity_embedding_instruction") or "").strip()
        if not article_id:
            skipped["missing_article_id"] += 1
            continue
        if requested_model and model != requested_model:
            skipped["different_model"] += 1
            continue
        if not isinstance(raw_vector, list) or not raw_vector:
            skipped["missing_embedding"] += 1
            continue
        try:
            vector = tuple(float(value) for value in raw_vector)
        except (TypeError, ValueError):
            skipped["non_numeric_embedding"] += 1
            continue
        if not all(math.isfinite(value) for value in vector):
            skipped["non_finite_embedding"] += 1
            continue
        if math.sqrt(sum(value * value for value in vector)) == 0.0:
            skipped["zero_norm_embedding"] += 1
            continue
        if not model:
            skipped["missing_model"] += 1
            continue
        signature = (model, len(vector), instruction)
        signature_counts[signature] += 1
        candidates.append(
            EmbeddedArticle(
                article_id=article_id,
                title=str(article.get("title") or "").strip(),
                source=str(article.get("source") or "").strip(),
                published_ts=_article_ts(article),
                vector=vector,
                signature=signature,
            )
        )

    if not signature_counts:
        raise ValueError("Inga användbara similarity-embeddings hittades")
    selected_signature = min(
        signature_counts,
        key=lambda signature: (-signature_counts[signature], signature),
    )
    selected = [item for item in candidates if item.signature == selected_signature]
    skipped["incompatible_signature"] += len(candidates) - len(selected)
    model, dimensions, instruction = selected_signature
    return selected, {
        "articles_seen": total,
        "articles_selected": len(selected),
        "coverage": len(selected) / total if total else 0.0,
        "embedding_signature": {
            "model": model,
            "dimensions": dimensions,
            "instruction": instruction,
        },
        "available_signatures": [
            {
                "model": signature[0],
                "dimensions": signature[1],
                "instruction": signature[2],
                "articles": signature_counts[signature],
            }
            for signature in sorted(
                signature_counts,
                key=lambda signature: (-signature_counts[signature], signature),
            )
        ],
        "skipped_by_reason": dict(sorted(skipped.items())),
    }


def calculate_similarity_report(
    articles: list[EmbeddedArticle],
    *,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    labels: dict[str, str] | None = None,
    block_size: int = 256,
) -> dict[str, Any]:
    """Calculate pair and nearest-neighbour statistics without retaining an N×N matrix."""

    if len(articles) < 2:
        raise ValueError("Minst två kompatibla artiklar krävs")
    thresholds = tuple(sorted({float(value) for value in thresholds}))
    if not thresholds or any(value < -1.0 or value > 1.0 for value in thresholds):
        raise ValueError("Tröskelvärden måste ligga mellan -1 och 1")

    matrix = np.asarray([item.vector for item in articles], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, np.finfo(np.float32).eps, None)
    nearest = np.full(len(articles), -1.0, dtype=np.float32)
    pair_counts = {threshold: 0 for threshold in thresholds}
    label_counts = {
        threshold: Counter({"true_positive": 0, "false_positive": 0, "false_negative": 0})
        for threshold in thresholds
    }
    labelled_ids = set(labels or {})

    for start in range(0, len(articles), max(1, block_size)):
        end = min(len(articles), start + max(1, block_size))
        similarities = matrix[start:end] @ matrix.T
        row_indices = np.arange(start, end)
        similarities[np.arange(end - start), row_indices] = -1.0
        nearest[start:end] = similarities.max(axis=1)

        for local_row, left_index in enumerate(row_indices):
            if left_index + 1 >= len(articles):
                continue
            scores = similarities[local_row, left_index + 1 :]
            for threshold in thresholds:
                pair_counts[threshold] += int(np.count_nonzero(scores >= threshold))
            if not labels or articles[left_index].article_id not in labelled_ids:
                continue
            left_label = labels[articles[left_index].article_id]
            for offset, score in enumerate(scores, start=left_index + 1):
                right_id = articles[offset].article_id
                if right_id not in labelled_ids:
                    continue
                actual_same = left_label == labels[right_id]
                for threshold in thresholds:
                    predicted_same = float(score) >= threshold
                    if predicted_same and actual_same:
                        label_counts[threshold]["true_positive"] += 1
                    elif predicted_same:
                        label_counts[threshold]["false_positive"] += 1
                    elif actual_same:
                        label_counts[threshold]["false_negative"] += 1

    quantiles = {
        str(quantile): float(np.quantile(nearest, quantile))
        for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    }
    threshold_rows = []
    for threshold in thresholds:
        row: dict[str, Any] = {
            "threshold": threshold,
            "pairs_at_or_above": pair_counts[threshold],
            "articles_with_neighbour_at_or_above": int(np.count_nonzero(nearest >= threshold)),
        }
        counts = label_counts[threshold]
        if labels:
            tp = counts["true_positive"]
            fp = counts["false_positive"]
            fn = counts["false_negative"]
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            row["labelled_pair_metrics"] = {
                **dict(counts),
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0,
            }
        threshold_rows.append(row)
    return {
        "nearest_neighbour_cosine_quantiles": quantiles,
        "thresholds": threshold_rows,
        "labelled_articles": sum(
            article.article_id in labelled_ids for article in articles
        ),
    }


def _load_labels(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"article_id", "event_id"}.issubset(reader.fieldnames):
            raise ValueError("Etikettfilen måste ha kolumnerna article_id,event_id")
        labels = {
            str(row["article_id"]).strip(): str(row["event_id"]).strip()
            for row in reader
            if str(row.get("article_id") or "").strip()
            and str(row.get("event_id") or "").strip()
        }
    if not labels:
        raise ValueError("Etikettfilen innehåller inga giltiga rader")
    return labels


def _parse_thresholds(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("trösklar måste vara kommaseparerade tal") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Standard: FEEDSUMMARY_CONFIG eller config.yaml")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--thresholds", type=_parse_thresholds, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--labels", type=Path, help="CSV med article_id,event_id")
    args = parser.parse_args()

    store = None
    try:
        config_path = _config_path(args.config)
        config = _load_config(config_path)
        store = create_store(_store_config(config, config_path))
        iterator = getattr(store, "iter_articles", None)
        raw_articles = (
            iterator(limit=max(2, args.limit))
            if callable(iterator)
            else store.list_articles(limit=max(2, args.limit))
        )
        articles, selection = collect_compatible_articles(
            raw_articles,
            requested_model=args.embedding_model,
        )
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "read_only": True,
            "selection": selection,
            "similarity": calculate_similarity_report(
                articles,
                thresholds=args.thresholds,
                labels=_load_labels(args.labels),
            ),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(json.dumps({"error": str(error)}, ensure_ascii=False))
        return 2
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    raise SystemExit(main())
