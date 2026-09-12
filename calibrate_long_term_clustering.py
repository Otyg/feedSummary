#!/usr/bin/env python3
"""Read-only calibration report for long-term event-clustering thresholds."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from feedsummary_core.long_term.clustering import (
    AssignmentAction,
    ClusteringSettings,
    add_cluster_member,
    assign_article,
    create_cluster,
)
from feedsummary_core.long_term.models import ClusterStatus, EmbeddingSignature
from feedsummary_core.persistence import create_store

REPORT_SCHEMA_VERSION = 1
PORTABLE_DATASET_SCHEMA_VERSION = 1
DEFAULT_THRESHOLDS = (0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.90)
_CVE_PATTERN = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.IGNORECASE)
_PORTABLE_BASE_TS = 1_700_000_000


@dataclass(frozen=True)
class EmbeddedArticle:
    article_id: str
    title: str
    source: str
    published_ts: int
    vector: tuple[float, ...]
    signature: tuple[str, int, str]
    strong_indicators: tuple[str, ...] = ()


@dataclass(frozen=True)
class LabelledPair:
    left_article_id: str
    right_article_id: str
    judgement: str
    notes: str = ""


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


def resolve_profile_sources(
    config_path: Path, profile_id: str
) -> tuple[set[str], dict[str, Any]]:
    """Resolve a long-term profile's feed categories to source names."""

    config = _load_config(config_path)
    long_term = config.get("long_term_analysis")
    profiles = long_term.get("profiles") if isinstance(long_term, dict) else None
    if not isinstance(profiles, dict) or profile_id not in profiles:
        raise ValueError(f"Okänd långtidsprofil: {profile_id}")
    profile = profiles[profile_id]
    if not isinstance(profile, dict):
        raise TypeError(f"Långtidsprofilen måste vara ett YAML-objekt: {profile_id}")
    sources = {
        str(source).strip()
        for source in profile.get("sources") or []
        if str(source).strip()
    }
    categories = {
        str(category).strip()
        for category in profile.get("categories") or []
        if str(category).strip()
    }
    if categories:
        feeds_config = config.get("feeds") or {}
        raw_path = feeds_config.get("path") if isinstance(feeds_config, dict) else None
        if not raw_path:
            raise ValueError("Profilen har categories men feeds.path saknas")
        feeds_path = Path(os.path.expandvars(os.path.expanduser(str(raw_path))))
        if not feeds_path.is_absolute():
            feeds_path = config_path.parent / feeds_path
        feeds = yaml.safe_load(feeds_path.read_text(encoding="utf-8")) or []
        if not isinstance(feeds, list):
            raise TypeError(f"Feedkonfigurationen måste vara en YAML-lista: {feeds_path}")
        for feed in feeds:
            if not isinstance(feed, dict):
                continue
            topics = {str(topic).strip() for topic in feed.get("topics") or []}
            if categories.intersection(topics):
                source = str(feed.get("name") or "").strip()
                if source:
                    sources.add(source)
    include_all = bool(profile.get("include_all_sources", False))
    if not sources and not include_all:
        raise ValueError(
            "Profilen löser inte till några källor; ange sources/categories eller "
            "include_all_sources"
        )
    effective_sources = set() if include_all else sources
    return effective_sources, {
        "profile_id": profile_id,
        "profile_config": str(config_path),
        "include_all_sources": include_all,
        "categories": sorted(categories),
        "sources": sorted(sources),
    }


def _article_ts(article: dict[str, Any]) -> int:
    for field in ("published_ts", "fetched_at"):
        try:
            value = int(article.get(field) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0


def _article_strong_indicators(article: dict[str, Any]) -> tuple[str, ...]:
    text = "\n".join(
        str(article.get(field) or "")
        for field in ("title", "text", "content", "summary")
    )
    return tuple(
        sorted({f"cve:{match.casefold()}" for match in _CVE_PATTERN.findall(text)})
    )


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
                strong_indicators=_article_strong_indicators(article),
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
    pair_labels: tuple[LabelledPair, ...] | None = None,
    block_size: int = 256,
) -> dict[str, Any]:
    """Calculate pair and nearest-neighbour statistics without retaining an N×N matrix."""

    if len(articles) < 2:
        raise ValueError("Minst två kompatibla artiklar krävs")
    thresholds = tuple(sorted({float(value) for value in thresholds}))
    if not thresholds or any(value < -1.0 or value > 1.0 for value in thresholds):
        raise ValueError("Tröskelvärden måste ligga mellan -1 och 1")
    if labels and pair_labels:
        raise ValueError("Använd antingen klusteretiketter eller paretiketter, inte båda")
    for pair in pair_labels or ():
        if pair.judgement not in {"same", "different", "ambiguous"}:
            raise ValueError(f"Ogiltig paretikett: {pair.judgement}")
        if pair.left_article_id == pair.right_article_id:
            raise ValueError("Ett etiketterat par måste innehålla två artiklar")

    matrix = np.asarray([item.vector for item in articles], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, np.finfo(np.float32).eps, None)
    nearest = np.full(len(articles), -1.0, dtype=np.float32)
    pair_counts = {threshold: 0 for threshold in thresholds}
    label_counts = {
        threshold: Counter(
            {
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 0,
                "true_negative": 0,
                "ambiguous": 0,
                "ambiguous_predicted_same": 0,
            }
        )
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
                    else:
                        label_counts[threshold]["true_negative"] += 1

    if pair_labels:
        article_indices = {
            article.article_id: index for index, article in enumerate(articles)
        }
        missing_ids = sorted(
            {
                article_id
                for pair in pair_labels
                for article_id in (pair.left_article_id, pair.right_article_id)
                if article_id not in article_indices
            }
        )
        if missing_ids:
            preview = ", ".join(missing_ids[:3])
            raise ValueError(
                f"{len(missing_ids)} etiketterade artikel-ID:n saknas i urvalet: {preview}"
            )
        for pair in pair_labels:
            left_index = article_indices[pair.left_article_id]
            right_index = article_indices[pair.right_article_id]
            score = float(matrix[left_index] @ matrix[right_index])
            for threshold in thresholds:
                predicted_same = score >= threshold
                counts = label_counts[threshold]
                if pair.judgement == "ambiguous":
                    counts["ambiguous"] += 1
                    counts["ambiguous_predicted_same"] += int(predicted_same)
                elif pair.judgement == "same" and predicted_same:
                    counts["true_positive"] += 1
                elif pair.judgement == "same":
                    counts["false_negative"] += 1
                elif predicted_same:
                    counts["false_positive"] += 1
                else:
                    counts["true_negative"] += 1

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
        if labels or pair_labels:
            tp = counts["true_positive"]
            fp = counts["false_positive"]
            fn = counts["false_negative"]
            tn = counts["true_negative"]
            evaluated = tp + fp + fn + tn
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            row["labelled_pair_metrics"] = {
                **dict(counts),
                "evaluated_pairs": evaluated,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0,
                "accuracy": (tp + tn) / evaluated if evaluated else 0.0,
            }
        threshold_rows.append(row)
    return {
        "nearest_neighbour_cosine_quantiles": quantiles,
        "thresholds": threshold_rows,
        "labelled_articles": sum(
            article.article_id in labelled_ids for article in articles
        )
        if labels
        else len(
            {
                article_id
                for pair in pair_labels or ()
                for article_id in (pair.left_article_id, pair.right_article_id)
            }
        ),
        "labelled_pairs": len(pair_labels or ()),
    }


def select_review_pairs(
    articles: list[EmbeddedArticle],
    *,
    threshold: float = 0.80,
    ambiguity_margin: float = 0.03,
    pairs_per_band: int = 10,
    max_time_distance_days: int | None = None,
) -> list[dict[str, Any]]:
    """Select deterministic high, boundary and near-negative pairs for review."""

    if len(articles) < 2:
        raise ValueError("Minst två kompatibla artiklar krävs")
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("Tröskelvärdet måste ligga mellan -1 och 1")
    if not 0.0 <= ambiguity_margin <= 2.0:
        raise ValueError("Tvetydighetsmarginalen måste ligga mellan 0 och 2")
    if pairs_per_band < 1:
        raise ValueError("Antal granskningspar per band måste vara positivt")
    if max_time_distance_days is not None and max_time_distance_days < 1:
        raise ValueError("Tidsfönstret måste vara positivt")
    max_time_distance_seconds = (
        max_time_distance_days * 86400
        if max_time_distance_days is not None
        else None
    )

    matrix = np.asarray([item.vector for item in articles], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, np.finfo(np.float32).eps, None)
    bands: dict[str, list[tuple[float, int, int]]] = {
        "high_similarity": [],
        "threshold_boundary": [],
        "near_negative": [],
    }

    for left_index in range(len(articles) - 1):
        scores = matrix[left_index + 1 :] @ matrix[left_index]
        for offset, raw_score in enumerate(scores, start=left_index + 1):
            if (
                max_time_distance_seconds is not None
                and abs(
                    articles[left_index].published_ts
                    - articles[offset].published_ts
                )
                > max_time_distance_seconds
            ):
                continue
            score = float(raw_score)
            if score >= threshold + ambiguity_margin:
                band = "high_similarity"
            elif abs(score - threshold) <= ambiguity_margin:
                band = "threshold_boundary"
            elif threshold - 0.10 <= score < threshold - ambiguity_margin:
                band = "near_negative"
            else:
                continue
            bands[band].append((score, left_index, offset))
            if len(bands[band]) > pairs_per_band * 4:
                bands[band] = _rank_review_band(
                    band, bands[band], articles, threshold, pairs_per_band
                )

    result = []
    for band in ("high_similarity", "threshold_boundary", "near_negative"):
        for score, left_index, right_index in _rank_review_band(
            band, bands[band], articles, threshold, pairs_per_band
        ):
            left = articles[left_index]
            right = articles[right_index]
            result.append(
                {
                    "left_article_id": left.article_id,
                    "right_article_id": right.article_id,
                    "judgement": "",
                    "similarity": round(score, 6),
                    "review_band": band,
                    "left_title": left.title,
                    "right_title": right.title,
                    "left_source": left.source,
                    "right_source": right.source,
                    "notes": "",
                }
            )
    return result


def _rank_review_band(
    band: str,
    rows: list[tuple[float, int, int]],
    articles: list[EmbeddedArticle],
    threshold: float,
    limit: int,
) -> list[tuple[float, int, int]]:
    def source_rank(row: tuple[float, int, int]) -> bool:
        return articles[row[1]].source == articles[row[2]].source

    if band == "high_similarity":
        key = lambda row: (source_rank(row), -row[0], row[1], row[2])
    elif band == "threshold_boundary":
        key = lambda row: (
            source_rank(row),
            abs(row[0] - threshold),
            row[1],
            row[2],
        )
    else:
        key = lambda row: (source_rank(row), -row[0], row[1], row[2])
    return sorted(rows, key=key)[:limit]


def write_review_template(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a new review CSV without overwriting an existing analyst file."""

    fieldnames = [
        "left_article_id",
        "right_article_id",
        "judgement",
        "similarity",
        "review_band",
        "left_title",
        "right_title",
        "left_source",
        "right_source",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_pair_review_packet(
    articles: list[EmbeddedArticle],
    pair_labels: tuple[LabelledPair, ...],
    article_documents: dict[str, dict[str, Any]],
    *,
    text_chars: int = 0,
) -> dict[str, Any]:
    """Build a blind full-text packet without golden labels or raw article IDs."""

    if text_chars < 0:
        raise ValueError("Granskningstextens teckengräns får inte vara negativ")
    article_ids = sorted(
        {
            article_id
            for pair in pair_labels
            for article_id in (pair.left_article_id, pair.right_article_id)
        }
    )
    articles_by_id = {article.article_id: article for article in articles}
    missing_ids = set(article_ids).difference(articles_by_id)
    if missing_ids:
        raise ValueError("Granskningspaketet saknar en eller flera artiklar")
    aliases = {
        article_id: f"article-{index:03d}"
        for index, article_id in enumerate(article_ids, start=1)
    }

    normalized_vectors = {}
    for article_id in article_ids:
        vector = np.asarray(articles_by_id[article_id].vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm <= np.finfo(np.float32).eps:
            raise ValueError("Granskningspaketet innehåller en nollvektor")
        normalized_vectors[article_id] = vector / norm

    portable_articles = []
    for article_id in article_ids:
        article = articles_by_id[article_id]
        raw = article_documents.get(article_id) or {}
        body = str(
            raw.get("text")
            or raw.get("content")
            or raw.get("summary")
            or ""
        )
        if text_chars:
            body = body[:text_chars]
        portable_articles.append(
            {
                "article_id": aliases[article_id],
                "title": article.title,
                "source": article.source,
                "published_ts": article.published_ts,
                "url": str(raw.get("url") or ""),
                "text": body,
            }
        )

    portable_pairs = []
    for index, pair in enumerate(pair_labels, start=1):
        similarity = float(
            normalized_vectors[pair.left_article_id]
            @ normalized_vectors[pair.right_article_id]
        )
        portable_pairs.append(
            {
                "pair_id": f"pair-{index:03d}",
                "left_article_id": aliases[pair.left_article_id],
                "right_article_id": aliases[pair.right_article_id],
                "similarity": round(similarity, 6),
                "reviewer_judgement": "",
                "reviewer_notes": "",
            }
        )
    return {
        "schema_version": 1,
        "blind_review": True,
        "instructions": {
            "allowed_judgements": ["same", "different", "ambiguous"],
            "criterion": "Bedöm om artiklarna beskriver samma verkliga händelse.",
            "independence": "Granska utan tillgång till golden-etikettfilen.",
        },
        "article_count": len(portable_articles),
        "pair_count": len(portable_pairs),
        "text_chars": text_chars,
        "articles": portable_articles,
        "pairs": portable_pairs,
    }


def write_pair_review_packet(path: Path, packet: dict[str, Any]) -> None:
    """Write a blind review packet without overwriting an analyst's work."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(packet, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def select_competing_background_articles(
    articles: list[EmbeddedArticle],
    pair_labels: tuple[LabelledPair, ...],
    *,
    candidate_window_days: int = 60,
    neighbors_per_article: int = 3,
    min_similarity: float = 0.75,
) -> list[EmbeddedArticle]:
    """Select nearest earlier non-golden articles that can compete for assignment."""

    if candidate_window_days < 1:
        raise ValueError("Bakgrundens kandidatfönster måste vara positivt")
    if neighbors_per_article < 0:
        raise ValueError("Antal bakgrundsgrannar får inte vara negativt")
    if not -1.0 <= min_similarity <= 1.0:
        raise ValueError("Bakgrundens minsta likhet måste ligga mellan -1 och 1")
    if not neighbors_per_article:
        return []
    golden_ids = {
        article_id
        for pair in pair_labels
        for article_id in (pair.left_article_id, pair.right_article_id)
    }
    articles_by_id = {article.article_id: article for article in articles}
    missing_ids = golden_ids.difference(articles_by_id)
    if missing_ids:
        raise ValueError("Bakgrundsurvalet saknar etiketterade artiklar")
    signatures = {article.signature for article in articles}
    if len(signatures) != 1:
        raise ValueError("Bakgrundsurvalet kräver en gemensam embedding-signatur")

    matrix = np.asarray([article.vector for article in articles], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, np.finfo(np.float32).eps, None)
    index_by_id = {
        article.article_id: index for index, article in enumerate(articles)
    }
    window_seconds = candidate_window_days * 86400
    selected_ids = set()
    for golden_id in sorted(golden_ids):
        target_index = index_by_id[golden_id]
        target = articles[target_index]
        target_order = (target.published_ts, target.article_id)
        earliest_ts = target.published_ts - window_seconds
        scores = matrix @ matrix[target_index]
        candidates = [
            (float(scores[index]), article.article_id)
            for index, article in enumerate(articles)
            if article.article_id not in golden_ids
            and earliest_ts <= article.published_ts
            and (article.published_ts, article.article_id) < target_order
            and float(scores[index]) >= min_similarity
        ]
        candidates.sort(key=lambda row: (-row[0], row[1]))
        selected_ids.update(
            article_id for _, article_id in candidates[:neighbors_per_article]
        )
    return sorted(
        (articles_by_id[article_id] for article_id in selected_ids),
        key=lambda article: (article.published_ts, article.article_id),
    )


def build_portable_dataset(
    articles: list[EmbeddedArticle],
    pair_labels: tuple[LabelledPair, ...],
    *,
    profile_id: str,
    background_articles: tuple[EmbeddedArticle, ...] = (),
    background_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Anonymize labelled articles into a self-contained float32 fixture."""

    if not pair_labels:
        raise ValueError("Paretiketter krävs för en portabel fixture")
    articles_by_id = {article.article_id: article for article in articles}
    selected_ids = sorted(
        {
            article_id
            for pair in pair_labels
            for article_id in (pair.left_article_id, pair.right_article_id)
        }
    )
    missing_ids = [article_id for article_id in selected_ids if article_id not in articles_by_id]
    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} etiketterade artikel-ID:n saknas i exportunderlaget"
        )
    selected = [articles_by_id[article_id] for article_id in selected_ids]
    duplicate_background_ids = set(selected_ids).intersection(
        article.article_id for article in background_articles
    )
    if duplicate_background_ids:
        raise ValueError("Bakgrundsartiklar får inte överlappa golden-artiklar")
    export_articles = [*selected, *background_articles]
    signatures = {article.signature for article in export_articles}
    if len(signatures) != 1:
        raise ValueError("Alla exporterade artiklar måste ha samma embedding-signatur")
    signature = next(iter(signatures))
    if signature[1] < 1:
        raise ValueError("Embedding-dimensionen måste vara positiv")

    article_aliases = {
        article_id: f"article-{index:03d}"
        for index, article_id in enumerate(selected_ids, start=1)
    }
    background_aliases = {
        article.article_id: f"background-{index:03d}"
        for index, article in enumerate(
            sorted(background_articles, key=lambda item: item.article_id), start=1
        )
    }
    all_aliases = {**article_aliases, **background_aliases}
    source_aliases = {
        source: f"source-{index:02d}"
        for index, source in enumerate(
            sorted({article.source for article in export_articles}), start=1
        )
    }
    indicator_aliases = {
        indicator: f"cve:fixture-{index:03d}"
        for index, indicator in enumerate(
            sorted(
                {
                    indicator
                    for article in export_articles
                    for indicator in article.strong_indicators
                    if indicator.startswith("cve:")
                }
            ),
            start=1,
        )
    }
    first_ts = min(article.published_ts for article in export_articles)

    def portable_article(article: EmbeddedArticle) -> dict[str, Any]:
        vector = np.asarray(article.vector, dtype="<f4")
        if vector.size != signature[1] or not np.all(np.isfinite(vector)):
            raise ValueError(f"Ogiltig exportvektor för artikel {article.article_id}")
        return {
            "article_id": all_aliases[article.article_id],
            "published_ts": _PORTABLE_BASE_TS + article.published_ts - first_ts,
            "source_id": source_aliases[article.source],
            "strong_indicators": [
                indicator_aliases[indicator]
                for indicator in article.strong_indicators
                if indicator in indicator_aliases
            ],
            "vector_f32_base64": base64.b64encode(vector.tobytes()).decode("ascii"),
        }

    portable_articles = [portable_article(article) for article in selected]
    portable_background = [
        portable_article(article) for article in background_articles
    ]
    portable_pairs = [
        {
            "left_article_id": article_aliases[pair.left_article_id],
            "right_article_id": article_aliases[pair.right_article_id],
            "judgement": pair.judgement,
        }
        for pair in pair_labels
    ]
    online_scenarios = _build_portable_online_scenarios(signature)
    return {
        "schema_version": PORTABLE_DATASET_SCHEMA_VERSION,
        "dataset_kind": "stratified_pair_review",
        "review_status": "draft",
        "profile_id": str(profile_id or "anonymous-profile"),
        "embedding_signature": {
            "model": signature[0],
            "dimensions": signature[1],
            "instruction": signature[2],
        },
        "vector_encoding": "little-endian-float32-base64",
        "article_count": len(portable_articles),
        "pair_count": len(portable_pairs),
        "source_count": len(source_aliases),
        "background_article_count": len(portable_background),
        "background_selection": background_selection or {},
        "online_scenario_count": len(online_scenarios),
        "articles": portable_articles,
        "background_articles": portable_background,
        "pairs": portable_pairs,
        "online_scenarios": online_scenarios,
    }


def _encoded_portable_vector(
    values: tuple[float, ...], dimensions: int
) -> str:
    vector = np.zeros(dimensions, dtype="<f4")
    vector[: len(values)] = values
    return base64.b64encode(vector.tobytes()).decode("ascii")


def _build_portable_online_scenarios(
    signature: tuple[str, int, str],
) -> list[dict[str, Any]]:
    if signature[1] < 2:
        raise ValueError("Online-scenarierna kräver minst två embedding-dimensioner")
    cosine_30 = math.sqrt(3) / 2

    def article(
        article_id: str,
        published_ts: int,
        vector: tuple[float, float],
        indicators: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        return {
            "article_id": article_id,
            "published_ts": published_ts,
            "source_id": f"source-{article_id}",
            "strong_indicators": list(indicators),
            "vector_f32_base64": _encoded_portable_vector(vector, signature[1]),
        }

    return [
        {
            "scenario_id": "equal-candidates-needs-review",
            "probe_article_id": "scenario-ambiguity-probe",
            "expected_action": "needs_review",
            "thresholds": [0.80, 0.82],
            "articles": [
                article(
                    "scenario-ambiguity-left",
                    _PORTABLE_BASE_TS,
                    (cosine_30, -0.5),
                ),
                article(
                    "scenario-ambiguity-right",
                    _PORTABLE_BASE_TS + 1,
                    (cosine_30, 0.5),
                ),
                article(
                    "scenario-ambiguity-probe",
                    _PORTABLE_BASE_TS + 2,
                    (1.0, 0.0),
                ),
            ],
        },
        {
            "scenario_id": "conflicting-cve-removes-ambiguity",
            "probe_article_id": "scenario-cve-probe",
            "expected_action": "match",
            "expected_cluster_seed_id": "scenario-cve-left",
            "thresholds": [0.80, 0.82],
            "articles": [
                article(
                    "scenario-cve-left",
                    _PORTABLE_BASE_TS,
                    (cosine_30, -0.5),
                    ("cve:fixture-scenario-a",),
                ),
                article(
                    "scenario-cve-right",
                    _PORTABLE_BASE_TS + 1,
                    (cosine_30, 0.5),
                    ("cve:fixture-scenario-b",),
                ),
                article(
                    "scenario-cve-probe",
                    _PORTABLE_BASE_TS + 2,
                    (1.0, 0.0),
                    ("cve:fixture-scenario-a",),
                ),
            ],
        },
    ]


def write_portable_dataset(path: Path, dataset: dict[str, Any]) -> None:
    """Write a portable fixture without overwriting an existing review artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dataset, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def load_portable_dataset(
    path: Path,
) -> tuple[list[EmbeddedArticle], tuple[LabelledPair, ...], dict[str, Any]]:
    """Load and validate an anonymized calibration fixture."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Den portabla fixturen måste vara ett JSON-objekt")
    if payload.get("schema_version") != PORTABLE_DATASET_SCHEMA_VERSION:
        raise ValueError("Okänd schemaversion för portabel fixture")
    if payload.get("vector_encoding") != "little-endian-float32-base64":
        raise ValueError("Okänd vektorkodning för portabel fixture")
    raw_signature = payload.get("embedding_signature")
    if not isinstance(raw_signature, dict):
        raise TypeError("Den portabla fixturen saknar embedding-signatur")
    signature = (
        str(raw_signature.get("model") or ""),
        int(raw_signature.get("dimensions") or 0),
        str(raw_signature.get("instruction") or ""),
    )
    if not signature[0] or signature[1] < 1:
        raise ValueError("Ogiltig embedding-signatur i portabel fixture")

    articles = []
    seen_ids = set()
    for raw_article in payload.get("articles") or []:
        article = _decode_portable_article(raw_article, signature)
        article_id = article.article_id
        if not article_id or article_id in seen_ids:
            raise ValueError("Tomt eller duplicerat artikel-ID i portabel fixture")
        seen_ids.add(article_id)
        articles.append(article)
    background_articles = []
    for raw_article in payload.get("background_articles") or []:
        article = _decode_portable_article(raw_article, signature)
        if not article.article_id or article.article_id in seen_ids:
            raise ValueError(
                "Tomt eller överlappande bakgrunds-ID i portabel fixture"
            )
        seen_ids.add(article.article_id)
        background_articles.append(article)
    pairs = tuple(
        LabelledPair(
            left_article_id=str(raw_pair.get("left_article_id") or ""),
            right_article_id=str(raw_pair.get("right_article_id") or ""),
            judgement=str(raw_pair.get("judgement") or ""),
        )
        for raw_pair in payload.get("pairs") or []
        if isinstance(raw_pair, dict)
    )
    if len(articles) != int(payload.get("article_count") or -1):
        raise ValueError("Artikelantalet stämmer inte i portabel fixture")
    if len(pairs) != int(payload.get("pair_count") or -1):
        raise ValueError("Antalet par stämmer inte i portabel fixture")
    if len(background_articles) != int(
        payload.get("background_article_count") or 0
    ):
        raise ValueError("Antalet bakgrundsartiklar stämmer inte i portabel fixture")
    online_scenarios = []
    for raw_scenario in payload.get("online_scenarios") or []:
        if not isinstance(raw_scenario, dict):
            raise TypeError("Ogiltigt online-scenario i portabel fixture")
        scenario_articles = tuple(
            _decode_portable_article(raw_article, signature)
            for raw_article in raw_scenario.get("articles") or []
        )
        scenario = {
            "scenario_id": str(raw_scenario.get("scenario_id") or "").strip(),
            "probe_article_id": str(
                raw_scenario.get("probe_article_id") or ""
            ).strip(),
            "expected_action": str(raw_scenario.get("expected_action") or "").strip(),
            "expected_cluster_seed_id": str(
                raw_scenario.get("expected_cluster_seed_id") or ""
            ).strip(),
            "thresholds": tuple(float(value) for value in raw_scenario.get("thresholds") or []),
            "articles": scenario_articles,
        }
        if not scenario["scenario_id"] or len(scenario_articles) < 3:
            raise ValueError("Online-scenario saknar identitet eller tre artiklar")
        if scenario["probe_article_id"] not in {
            article.article_id for article in scenario_articles
        }:
            raise ValueError("Online-scenariots probe saknas bland artiklarna")
        if scenario["expected_action"] not in {
            action.value for action in AssignmentAction
        }:
            raise ValueError("Online-scenariot har okänd förväntad action")
        online_scenarios.append(scenario)
    if len(online_scenarios) != int(payload.get("online_scenario_count") or -1):
        raise ValueError("Antalet online-scenarier stämmer inte i portabel fixture")
    calculate_similarity_report(articles, thresholds=(0.8,), pair_labels=pairs)
    metadata = {
        key: payload.get(key)
        for key in (
            "schema_version",
            "dataset_kind",
            "review_status",
            "profile_id",
            "article_count",
            "pair_count",
            "source_count",
            "background_article_count",
            "background_selection",
            "online_scenario_count",
            "embedding_signature",
            "vector_encoding",
        )
    }
    metadata["background_articles"] = tuple(background_articles)
    metadata["online_scenarios"] = tuple(online_scenarios)
    return articles, pairs, metadata


def _decode_portable_article(
    raw_article: Any,
    signature: tuple[str, int, str],
) -> EmbeddedArticle:
    if not isinstance(raw_article, dict):
        raise TypeError("Ogiltig artikel i portabel fixture")
    article_id = str(raw_article.get("article_id") or "").strip()
    try:
        vector_bytes = base64.b64decode(
            str(raw_article.get("vector_f32_base64") or ""), validate=True
        )
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Ogiltig base64-vektor för {article_id}") from exc
    vector_array = np.frombuffer(vector_bytes, dtype="<f4")
    if vector_array.size != signature[1] or not np.all(np.isfinite(vector_array)):
        raise ValueError(f"Ogiltig vektordimension för {article_id}")
    published_ts = int(raw_article.get("published_ts") or 0)
    if published_ts < 1:
        raise ValueError(f"Ogiltig tidsstämpel för {article_id}")
    return EmbeddedArticle(
        article_id=article_id,
        title=article_id,
        source=str(raw_article.get("source_id") or "anonymous-source"),
        published_ts=published_ts,
        vector=tuple(float(value) for value in vector_array),
        signature=signature,
        strong_indicators=tuple(
            sorted(
                {
                    str(value).strip()
                    for value in raw_article.get("strong_indicators") or []
                    if str(value).strip()
                }
            )
        ),
    )


def _run_online_assignment_sequence(
    articles: list[EmbeddedArticle],
    *,
    threshold: float,
    ambiguity_margin: float,
    candidate_window_days: int,
    profile_id: str,
) -> tuple[list[Any], dict[str, tuple[str, AssignmentAction, Any]], Counter[str]]:
    signatures = {article.signature for article in articles}
    if len(signatures) != 1:
        raise ValueError("Onlineutvärderingen kräver en gemensam embedding-signatur")
    signature = EmbeddingSignature(*next(iter(signatures)))
    clusters = []
    assignments: dict[str, tuple[str, AssignmentAction, Any]] = {}
    actions: Counter[str] = Counter()
    settings = ClusteringSettings(
        similarity_threshold=threshold,
        ambiguity_margin=ambiguity_margin,
        candidate_window_days=candidate_window_days,
    )
    for article in sorted(
        articles, key=lambda item: (item.published_ts, item.article_id)
    ):
        decision = assign_article(
            profile_id=profile_id,
            article_ts=article.published_ts,
            embedding=article.vector,
            signature=signature,
            candidates=clusters,
            strong_indicators=article.strong_indicators,
            settings=settings,
        )
        if decision.action is AssignmentAction.MATCH:
            cluster_index = next(
                index
                for index, cluster in enumerate(clusters)
                if cluster.id == decision.cluster_id
            )
            cluster = add_cluster_member(
                clusters[cluster_index],
                article_ts=article.published_ts,
                embedding=article.vector,
                strong_indicators=article.strong_indicators,
            )
            clusters[cluster_index] = cluster
        else:
            cluster = create_cluster(
                profile_id=profile_id,
                article_id=article.article_id,
                article_ts=article.published_ts,
                embedding=article.vector,
                signature=signature,
                strong_indicators=article.strong_indicators,
            )
            if decision.action is AssignmentAction.NEEDS_REVIEW:
                cluster = replace(cluster, status=ClusterStatus.NEEDS_REVIEW)
            clusters.append(cluster)
        assignments[article.article_id] = (cluster.id, decision.action, decision)
        actions[decision.action.value] += 1
    return clusters, assignments, actions


def evaluate_online_clustering(
    articles: list[EmbeddedArticle],
    pair_labels: tuple[LabelledPair, ...],
    *,
    thresholds: tuple[float, ...] = (0.80, 0.82),
    ambiguity_margin: float = 0.03,
    candidate_window_days: int = 60,
    background_articles: tuple[EmbeddedArticle, ...] = (),
) -> dict[str, Any]:
    """Evaluate the real online assignment functions over labelled articles."""

    labelled_ids = {
        article_id
        for pair in pair_labels
        for article_id in (pair.left_article_id, pair.right_article_id)
    }
    selected = sorted(
        (article for article in articles if article.article_id in labelled_ids),
        key=lambda article: (article.published_ts, article.article_id),
    )
    if {article.article_id for article in selected} != labelled_ids:
        raise ValueError("Onlineutvärderingen saknar en eller flera etiketterade artiklar")
    if labelled_ids.intersection(article.article_id for article in background_articles):
        raise ValueError("Bakgrundsartiklar får inte överlappa etiketterade artiklar")
    processing_articles = [*selected, *background_articles]
    rows = []

    for threshold in sorted({float(value) for value in thresholds}):
        clusters, assignments, actions = _run_online_assignment_sequence(
            processing_articles,
            threshold=threshold,
            ambiguity_margin=ambiguity_margin,
            candidate_window_days=candidate_window_days,
            profile_id="golden-fixture",
        )

        counts: Counter[str] = Counter(
            {
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 0,
                "true_negative": 0,
                "ambiguous": 0,
                "ambiguous_same_cluster": 0,
                "ambiguous_needs_review": 0,
            }
        )
        error_cases = []
        for pair in pair_labels:
            left_cluster, left_action, _ = assignments[pair.left_article_id]
            right_cluster, right_action, _ = assignments[pair.right_article_id]
            predicted_same = left_cluster == right_cluster
            pair_needs_review = AssignmentAction.NEEDS_REVIEW in {
                left_action,
                right_action,
            }
            if pair.judgement == "ambiguous":
                counts["ambiguous"] += 1
                counts["ambiguous_same_cluster"] += int(predicted_same)
                counts["ambiguous_needs_review"] += int(pair_needs_review)
            elif pair.judgement == "same" and predicted_same:
                counts["true_positive"] += 1
            elif pair.judgement == "same":
                counts["false_negative"] += 1
                error_cases.append(
                    {
                        "left_article_id": pair.left_article_id,
                        "right_article_id": pair.right_article_id,
                        "error": "false_negative",
                        "left_action": left_action.value,
                        "right_action": right_action.value,
                    }
                )
            elif predicted_same:
                counts["false_positive"] += 1
                error_cases.append(
                    {
                        "left_article_id": pair.left_article_id,
                        "right_article_id": pair.right_article_id,
                        "error": "false_positive",
                        "left_action": left_action.value,
                        "right_action": right_action.value,
                    }
                )
            else:
                counts["true_negative"] += 1
        tp = counts["true_positive"]
        fp = counts["false_positive"]
        fn = counts["false_negative"]
        tn = counts["true_negative"]
        evaluated = tp + fp + fn + tn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append(
            {
                "threshold": threshold,
                "cluster_count": len(clusters),
                "needs_review_cluster_count": sum(
                    cluster.status is ClusterStatus.NEEDS_REVIEW for cluster in clusters
                ),
                "assignment_actions": dict(sorted(actions.items())),
                "error_cases": error_cases,
                "labelled_pair_metrics": {
                    **dict(counts),
                    "evaluated_pairs": evaluated,
                    "precision": precision,
                    "recall": recall,
                    "f1": 2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0,
                    "accuracy": (tp + tn) / evaluated if evaluated else 0.0,
                },
            }
        )
    return {
        "article_count": len(selected),
        "background_article_count": len(background_articles),
        "processed_article_count": len(processing_articles),
        "pair_count": len(pair_labels),
        "ambiguity_margin": ambiguity_margin,
        "candidate_window_days": candidate_window_days,
        "thresholds": rows,
    }


def evaluate_online_scenarios(
    scenarios: tuple[dict[str, Any], ...],
    *,
    ambiguity_margin: float = 0.03,
    candidate_window_days: int = 60,
) -> dict[str, Any]:
    """Run isolated triplets and verify expected ambiguity/CVE decisions."""

    results = []
    for scenario in scenarios:
        for threshold in scenario["thresholds"]:
            clusters, assignments, _ = _run_online_assignment_sequence(
                list(scenario["articles"]),
                threshold=threshold,
                ambiguity_margin=ambiguity_margin,
                candidate_window_days=candidate_window_days,
                profile_id=f"golden-scenario-{scenario['scenario_id']}",
            )
            probe_cluster_id, probe_action, decision = assignments[
                scenario["probe_article_id"]
            ]
            expected_action = scenario["expected_action"]
            action_matches = probe_action.value == expected_action
            expected_seed = scenario["expected_cluster_seed_id"]
            seed_matches = True
            if expected_seed:
                seed_cluster_id, _, _ = assignments[expected_seed]
                seed_matches = probe_cluster_id == seed_cluster_id
            results.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "threshold": threshold,
                    "expected_action": expected_action,
                    "actual_action": probe_action.value,
                    "best_similarity": decision.similarity,
                    "second_similarity": decision.second_similarity,
                    "cluster_count": len(clusters),
                    "passed": action_matches and seed_matches,
                }
            )
    return {
        "scenario_count": len(scenarios),
        "checks": len(results),
        "passed": all(result["passed"] for result in results),
        "results": results,
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


def load_pair_labels(path: Path | None) -> tuple[LabelledPair, ...] | None:
    if path is None:
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"left_article_id", "right_article_id", "judgement"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Paretikettfilen måste ha kolumnerna "
                "left_article_id,right_article_id,judgement"
            )
        pairs: dict[tuple[str, str], LabelledPair] = {}
        for line_number, row in enumerate(reader, start=2):
            left = str(row.get("left_article_id") or "").strip()
            right = str(row.get("right_article_id") or "").strip()
            judgement = str(row.get("judgement") or "").strip().casefold()
            if not left and not right and not judgement:
                continue
            if not left or not right or left == right:
                raise ValueError(f"Ogiltigt artikelpar på rad {line_number}")
            if judgement not in {"same", "different", "ambiguous"}:
                raise ValueError(
                    f"Ogiltig bedömning på rad {line_number}: {judgement or '<tom>'}"
                )
            pair_key = tuple(sorted((left, right)))
            pair = LabelledPair(
                left_article_id=pair_key[0],
                right_article_id=pair_key[1],
                judgement=judgement,
                notes=str(row.get("notes") or "").strip(),
            )
            if pair_key in pairs and pairs[pair_key].judgement != judgement:
                raise ValueError(f"Motstridiga bedömningar för paret på rad {line_number}")
            pairs[pair_key] = pair
    if not pairs:
        raise ValueError("Paretikettfilen innehåller inga giltiga rader")
    return tuple(pairs[key] for key in sorted(pairs))


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
    parser.add_argument("--portable-dataset", type=Path)
    parser.add_argument("--profile")
    parser.add_argument(
        "--profile-config",
        type=Path,
        help="YAML med long_term_analysis-profil; standard är --config",
    )
    parser.add_argument("--thresholds", type=_parse_thresholds, default=DEFAULT_THRESHOLDS)
    labels = parser.add_mutually_exclusive_group()
    labels.add_argument("--labels", type=Path, help="CSV med article_id,event_id")
    labels.add_argument(
        "--pair-labels",
        type=Path,
        help="CSV med left_article_id,right_article_id,judgement",
    )
    parser.add_argument("--write-review-template", type=Path)
    parser.add_argument(
        "--write-pair-review-packet",
        type=Path,
        help="Skriv blindat JSON-paket för fulltextgranskning av --pair-labels",
    )
    parser.add_argument(
        "--pair-review-text-chars",
        type=int,
        default=0,
        help="Max tecken per artikel i granskningspaketet; 0 betyder full text",
    )
    parser.add_argument("--review-pairs-per-band", type=int, default=10)
    parser.add_argument("--review-threshold", type=float, default=0.80)
    parser.add_argument("--ambiguity-margin", type=float, default=0.03)
    parser.add_argument("--candidate-window-days", type=int, default=60)
    parser.add_argument("--background-neighbors-per-article", type=int, default=0)
    parser.add_argument("--background-min-similarity", type=float, default=0.75)
    parser.add_argument("--evaluate-online", action="store_true")
    parser.add_argument("--export-portable-dataset", type=Path)
    args = parser.parse_args()

    store = None
    try:
        if args.portable_dataset:
            if args.labels or args.pair_labels:
                raise ValueError(
                    "En portabel fixture innehåller redan paretiketter"
                )
            if (
                args.write_review_template
                or args.write_pair_review_packet
                or args.export_portable_dataset
            ):
                raise ValueError(
                    "Granskningsunderlag och export kräver databasunderlag"
                )
            articles, pair_labels, portable_metadata = load_portable_dataset(
                args.portable_dataset
            )
            online_scenarios = portable_metadata["online_scenarios"]
            background_articles = portable_metadata["background_articles"]
            background_selection = portable_metadata["background_selection"]
            signature = portable_metadata["embedding_signature"]
            selection = {
                "articles_seen": len(articles),
                "articles_selected": len(articles),
                "coverage": 1.0,
                "embedding_signature": signature,
                "available_signatures": [{**signature, "articles": len(articles)}],
                "skipped_by_reason": {},
            }
            profile_metadata = {
                "profile_id": portable_metadata["profile_id"],
                "portable_dataset": str(args.portable_dataset.resolve()),
                "review_status": portable_metadata["review_status"],
            }
        else:
            config_path = _config_path(args.config)
            config = _load_config(config_path)
            store = create_store(_store_config(config, config_path))
            profile_metadata = None
            profile_sources: set[str] = set()
            if args.profile:
                profile_path = (args.profile_config or config_path).resolve()
                profile_sources, profile_metadata = resolve_profile_sources(
                    profile_path, args.profile
                )
            iterator = getattr(store, "iter_articles", None)
            raw_articles: Iterable[dict[str, Any]] = (
                iterator(limit=None)
                if callable(iterator)
                else store.list_articles(limit=max(2, args.limit))
            )
            if profile_sources:
                raw_articles = (
                    article
                    for article in raw_articles
                    if str(article.get("source") or "").strip() in profile_sources
                )
            raw_articles = islice(raw_articles, max(2, args.limit))
            article_documents: dict[str, dict[str, Any]] = {}
            if args.write_pair_review_packet:
                raw_articles = list(raw_articles)
                article_documents = {
                    str(article.get("id") or ""): article
                    for article in raw_articles
                    if str(article.get("id") or "")
                }
            articles, selection = collect_compatible_articles(
                raw_articles,
                requested_model=args.embedding_model,
            )
            pair_labels = load_pair_labels(args.pair_labels)
            online_scenarios = ()
            background_articles = ()
            background_selection = {}
            if pair_labels and args.background_neighbors_per_article:
                background_articles = tuple(
                    select_competing_background_articles(
                        articles,
                        pair_labels,
                        candidate_window_days=args.candidate_window_days,
                        neighbors_per_article=args.background_neighbors_per_article,
                        min_similarity=args.background_min_similarity,
                    )
                )
                background_selection = {
                    "candidate_window_days": args.candidate_window_days,
                    "neighbors_per_article": args.background_neighbors_per_article,
                    "min_similarity": args.background_min_similarity,
                }
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "read_only": True,
            "selection": selection,
            "similarity": calculate_similarity_report(
                articles,
                thresholds=args.thresholds,
                labels=_load_labels(args.labels),
                pair_labels=pair_labels,
            ),
        }
        if profile_metadata:
            report["profile"] = profile_metadata
        if args.evaluate_online:
            if not pair_labels:
                raise ValueError("Onlineutvärdering kräver paretiketter")
            report["online_clustering"] = evaluate_online_clustering(
                articles,
                pair_labels,
                thresholds=args.thresholds,
                ambiguity_margin=args.ambiguity_margin,
                candidate_window_days=args.candidate_window_days,
                background_articles=background_articles,
            )
            if online_scenarios:
                report["online_ambiguity_scenarios"] = evaluate_online_scenarios(
                    online_scenarios,
                    ambiguity_margin=args.ambiguity_margin,
                    candidate_window_days=args.candidate_window_days,
                )
        if args.write_review_template:
            review_rows = select_review_pairs(
                articles,
                threshold=args.review_threshold,
                ambiguity_margin=args.ambiguity_margin,
                pairs_per_band=args.review_pairs_per_band,
            )
            write_review_template(args.write_review_template, review_rows)
            report["review_template"] = {
                "path": str(args.write_review_template),
                "pairs": len(review_rows),
                "threshold": args.review_threshold,
                "ambiguity_margin": args.ambiguity_margin,
            }
        if args.write_pair_review_packet:
            if not pair_labels:
                raise ValueError("Blindat granskningspaket kräver --pair-labels")
            packet = build_pair_review_packet(
                articles,
                pair_labels,
                article_documents,
                text_chars=args.pair_review_text_chars,
            )
            write_pair_review_packet(args.write_pair_review_packet, packet)
            report["pair_review_packet"] = {
                "path": str(args.write_pair_review_packet),
                "article_count": packet["article_count"],
                "pair_count": packet["pair_count"],
                "text_chars": packet["text_chars"],
                "blind_review": True,
            }
        if args.export_portable_dataset:
            if not pair_labels:
                raise ValueError("Portabel export kräver --pair-labels")
            dataset = build_portable_dataset(
                articles,
                pair_labels,
                profile_id=args.profile or "anonymous-profile",
                background_articles=background_articles,
                background_selection=background_selection,
            )
            write_portable_dataset(args.export_portable_dataset, dataset)
            report["portable_export"] = {
                "path": str(args.export_portable_dataset),
                "article_count": dataset["article_count"],
                "pair_count": dataset["pair_count"],
                "source_count": dataset["source_count"],
                "background_article_count": dataset["background_article_count"],
                "online_scenario_count": dataset["online_scenario_count"],
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
