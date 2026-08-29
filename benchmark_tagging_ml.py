#!/usr/bin/env python3
"""Benchmark lightweight multilabel classifiers on article tags in MongoDB.

This command is deliberately independent from the production tagging pipeline.
It reads existing articles and tag assignments, compares established
scikit-learn algorithms on identical chronological splits, and writes a report
that can be used to choose an implementation later.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import resource
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from feedsummary_core.persistence import MongoDBStore, create_store


log = logging.getLogger("tagging_ml_benchmark")
DEFAULT_CATEGORIES: Tuple[str, ...] = (
    "DOMAIN_ENTITY",
    "LOCATION",
    "DATATYPE",
    "SECTOR",
    "THREAT",
)
DEFAULT_REPRESENTATIONS: Tuple[str, ...] = (
    "tfidf",
    "hashing",
    "embedding",
    "hybrid",
)


@dataclass(frozen=True)
class ArticleExample:
    article_id: str
    text: str
    timestamp: int
    labels: Tuple[str, ...]
    embedding: Optional[Tuple[float, ...]] = None
    embedding_model: str = ""


@dataclass(frozen=True)
class Candidate:
    name: str
    factory: Callable[[int, int], Any]
    requires_nonnegative: bool = False


def _resolve_config_path(raw_path: Optional[str]) -> Path:
    candidate = raw_path or os.environ.get("FEEDSUMMARY_CONFIG") or "config.yaml"
    return Path(os.path.expandvars(os.path.expanduser(candidate))).resolve()


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Konfigurationsfilen saknas: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Konfigurationen måste vara ett YAML-objekt: {path}")
    return config


def _article_text(article: Dict[str, Any], max_chars: int) -> str:
    title = str(article.get("title") or "").strip()
    source = str(article.get("source") or "").strip()
    body = str(
        article.get("text")
        or article.get("content")
        or article.get("summary")
        or article.get("description")
        or ""
    ).strip()
    # Repeating the short title gives it useful weight without language-specific logic.
    text = "\n".join(part for part in (title, title, source, body) if part)
    return text[:max_chars] if max_chars > 0 else text


def _safe_timestamp(article: Dict[str, Any]) -> int:
    for key in ("published_ts", "_mongo_sort_ts", "fetched_at", "created_at"):
        value = article.get(key)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def load_mongodb_examples(
    store: MongoDBStore,
    categories: Sequence[str],
    *,
    max_chars: int,
    max_articles: Optional[int] = None,
) -> Tuple[List[ArticleExample], Dict[str, Any]]:
    """Load articles and target-category labels using MongoDB collection queries."""
    category_set = {str(category).strip().upper() for category in categories}
    tags = list(
        store.db.tags.find(
            {"category": {"$in": sorted(category_set)}},
            {"_id": 1, "name": 1, "category": 1},
        )
    )
    tag_names = {
        int(tag["_id"]): str(tag.get("name") or "").strip()
        for tag in tags
        if str(tag.get("name") or "").strip()
    }
    if not tag_names:
        raise ValueError(
            "MongoDB innehåller inga taggar i kategorierna " + ", ".join(categories)
        )

    labels_by_article: Dict[str, set[str]] = {}
    for relation in store.db.article_tags.find(
        {"tag_id": {"$in": list(tag_names)}}, {"article_id": 1, "tag_id": 1}
    ):
        name = tag_names.get(int(relation.get("tag_id") or 0))
        article_id = str(relation.get("article_id") or "").strip()
        if name and article_id:
            labels_by_article.setdefault(article_id, set()).add(name)

    cursor = store.db.articles.find(
        {},
        {
            "_id": 1,
            "id": 1,
            "title": 1,
            "source": 1,
            "text": 1,
            "content": 1,
            "summary": 1,
            "description": 1,
            "published_ts": 1,
            "_mongo_sort_ts": 1,
            "fetched_at": 1,
            "created_at": 1,
            "embedding_vector": 1,
            "embedding_model": 1,
        },
    ).sort([("_mongo_sort_ts", 1), ("_id", 1)])
    if max_articles is not None:
        cursor = cursor.limit(max(0, int(max_articles)))

    examples: List[ArticleExample] = []
    skipped_empty = 0
    for article in cursor:
        article_id = str(article.get("id") or article.get("_id") or "").strip()
        text = _article_text(article, max_chars)
        if not article_id or not text:
            skipped_empty += 1
            continue
        raw_embedding = article.get("embedding_vector")
        embedding: Optional[Tuple[float, ...]] = None
        if (
            isinstance(raw_embedding, list)
            and raw_embedding
            and all(isinstance(value, (int, float)) for value in raw_embedding)
        ):
            normalized_embedding = tuple(float(value) for value in raw_embedding)
            if all(value == value and abs(value) != float("inf") for value in normalized_embedding):
                embedding = normalized_embedding

        examples.append(
            ArticleExample(
                article_id=article_id,
                text=text,
                timestamp=_safe_timestamp(article),
                labels=tuple(sorted(labels_by_article.get(article_id, set()))),
                embedding=embedding,
                embedding_model=str(article.get("embedding_model") or ""),
            )
        )

    examples.sort(key=lambda item: (item.timestamp, item.article_id))
    metadata = {
        "articles": len(examples),
        "articles_with_target_labels": sum(bool(item.labels) for item in examples),
        "target_tags": len(tag_names),
        "target_relations": sum(len(item.labels) for item in examples),
        "skipped_empty_articles": skipped_empty,
        "categories": [
            category
            for category in (str(item).strip().upper() for item in categories)
            if category
        ],
        "articles_with_embeddings": sum(item.embedding is not None for item in examples),
    }
    return examples, metadata


def _embedding_subset(
    examples: Sequence[ArticleExample], requested_model: Optional[str] = None
) -> Tuple[List[ArticleExample], Dict[str, Any]]:
    """Select one compatible embedding model and dimension, preserving article order."""
    signatures = Counter(
        (item.embedding_model, len(item.embedding))
        for item in examples
        if item.embedding is not None
        and (not requested_model or item.embedding_model == requested_model)
    )
    if not signatures:
        model_note = f" för modellen {requested_model}" if requested_model else ""
        raise ValueError(f"Inga användbara artikel-embeddings hittades{model_note}")

    selected_model, selected_dimension = min(
        signatures,
        key=lambda signature: (-signatures[signature], signature[0], signature[1]),
    )
    selected = [
        item
        for item in examples
        if item.embedding is not None
        and item.embedding_model == selected_model
        and len(item.embedding) == selected_dimension
    ]
    return selected, {
        "embedding_model": selected_model,
        "embedding_dimension": selected_dimension,
        "embedding_articles": len(selected),
        "embedding_coverage": len(selected) / max(1, len(examples)),
        "excluded_without_compatible_embedding": len(examples) - len(selected),
        "available_embedding_signatures": [
            {"model": model, "dimension": dimension, "articles": count}
            for (model, dimension), count in sorted(
                signatures.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
            )
        ],
    }


def _select_common_embedding_examples(
    examples: Sequence[ArticleExample],
    dataset_metadata: Dict[str, Any],
    requested_model: Optional[str] = None,
) -> Tuple[List[ArticleExample], Dict[str, Any], Dict[str, Any]]:
    """Create the common embedding-covered corpus used by every representation."""
    selected, embedding_metadata = _embedding_subset(examples, requested_model)
    selected_metadata = {
        **dataset_metadata,
        "source_articles": len(examples),
        "articles": len(selected),
        "articles_with_target_labels": sum(bool(item.labels) for item in selected),
        "target_relations": sum(len(item.labels) for item in selected),
        "articles_with_embeddings": len(selected),
    }
    return selected, selected_metadata, embedding_metadata


def chronological_split(
    examples: Sequence[ArticleExample], validation_fraction: float, test_fraction: float
) -> Tuple[List[ArticleExample], List[ArticleExample], List[ArticleExample]]:
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("Validerings- och testandel måste ligga mellan 0 och 1")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validerings- och testandel måste tillsammans vara mindre än 1")
    if len(examples) < 10:
        raise ValueError("Minst 10 artiklar krävs för ett meningsfullt benchmark")

    train_end = max(1, int(len(examples) * (1 - validation_fraction - test_fraction)))
    validation_end = max(train_end + 1, int(len(examples) * (1 - test_fraction)))
    validation_end = min(validation_end, len(examples) - 1)
    return (
        list(examples[:train_end]),
        list(examples[train_end:validation_end]),
        list(examples[validation_end:]),
    )


def eligible_labels(
    train: Sequence[ArticleExample],
    validation: Sequence[ArticleExample],
    test: Sequence[ArticleExample],
    min_support: int,
) -> Tuple[List[str], Dict[str, str]]:
    partitions = (train, validation, test)
    counts: List[Dict[str, int]] = []
    for partition in partitions:
        current: Dict[str, int] = {}
        for example in partition:
            for label in example.labels:
                current[label] = current.get(label, 0) + 1
        counts.append(current)

    all_labels = sorted(set().union(*(set(count) for count in counts)))
    included: List[str] = []
    excluded: Dict[str, str] = {}
    for label in all_labels:
        total = sum(count.get(label, 0) for count in counts)
        if total < min_support:
            excluded[label] = f"support {total} < {min_support}"
        elif counts[0].get(label, 0) == 0:
            excluded[label] = "saknas i träningsdelen"
        elif counts[0].get(label, 0) == len(train):
            excluded[label] = "saknar negativa träningsexempel"
        else:
            included.append(label)
    return included, excluded


def _candidate_factories() -> Dict[str, Candidate]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.naive_bayes import ComplementNB, MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC

    return {
        "logistic_regression": Candidate(
            "logistic_regression",
            lambda seed, jobs: LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=seed
            ),
        ),
        "sgd": Candidate(
            "sgd",
            lambda seed, jobs: SGDClassifier(
                loss="log_loss", class_weight="balanced", max_iter=2000,
                tol=1e-4, random_state=seed
            ),
        ),
        "multinomial_nb": Candidate(
            "multinomial_nb", lambda seed, jobs: MultinomialNB(alpha=0.5),
            requires_nonnegative=True,
        ),
        "complement_nb": Candidate(
            "complement_nb", lambda seed, jobs: ComplementNB(alpha=0.5),
            requires_nonnegative=True,
        ),
        "knn": Candidate(
            "knn",
            lambda seed, jobs: KNeighborsClassifier(
                n_neighbors=5, weights="distance", metric="cosine", algorithm="brute",
                n_jobs=jobs,
            ),
        ),
        "random_forest": Candidate(
            "random_forest",
            lambda seed, jobs: RandomForestClassifier(
                n_estimators=200, class_weight="balanced_subsample", random_state=seed,
                n_jobs=jobs,
            ),
        ),
        "linear_svm": Candidate(
            "linear_svm",
            lambda seed, jobs: LinearSVC(class_weight="balanced", random_state=seed),
        ),
    }


def _vectorizer(name: str, max_features: int) -> Any:
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
    from sklearn.pipeline import FeatureUnion

    if name == "tfidf":
        return FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=(1, 2), min_df=2, max_features=max_features,
                        sublinear_tf=True, strip_accents="unicode",
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                        max_features=max_features, sublinear_tf=True,
                    ),
                ),
            ]
        )
    if name == "hashing":
        feature_count = max(1024, max_features)
        return FeatureUnion(
            [
                (
                    "word",
                    HashingVectorizer(
                        ngram_range=(1, 2), n_features=feature_count,
                        alternate_sign=False, strip_accents="unicode",
                    ),
                ),
                (
                    "char",
                    HashingVectorizer(
                        analyzer="char_wb", ngram_range=(3, 5),
                        n_features=feature_count, alternate_sign=False,
                    ),
                ),
            ]
        )
    raise ValueError(f"Okänd textrepresentation: {name}")


def _embedding_array(examples: Sequence[ArticleExample]) -> Any:
    import numpy as np

    if any(item.embedding is None for item in examples):
        raise ValueError("Embeddingrepresentationen fick en artikel utan embedding")
    return np.asarray([item.embedding for item in examples], dtype=float)


def _fit_features(
    representation: str,
    candidate: Candidate,
    train: Sequence[ArticleExample],
    *,
    max_features: int,
    embedding_weight: float,
) -> Tuple[Any, Dict[str, Any]]:
    from scipy import sparse
    from sklearn.preprocessing import MinMaxScaler, Normalizer

    if representation in ("tfidf", "hashing"):
        text_transformer = _vectorizer(representation, max_features)
        x_train = text_transformer.fit_transform([item.text for item in train])
        return x_train, {"text_transformer": text_transformer}

    raw_embedding = _embedding_array(train)
    embedding_transformer = (
        MinMaxScaler(clip=True)
        if candidate.requires_nonnegative
        else Normalizer(norm="l2")
    )
    embedding_train = embedding_transformer.fit_transform(raw_embedding) * embedding_weight
    artifact: Dict[str, Any] = {"embedding_transformer": embedding_transformer}
    if representation == "embedding":
        return embedding_train, artifact
    if representation != "hybrid":
        raise ValueError(f"Okänd feature-representation: {representation}")

    text_transformer = _vectorizer("tfidf", max_features)
    text_train = text_transformer.fit_transform([item.text for item in train])
    artifact["text_transformer"] = text_transformer
    return sparse.hstack(
        (text_train, sparse.csr_matrix(embedding_train)), format="csr"
    ), artifact


def _transform_features(
    representation: str,
    feature_artifact: Dict[str, Any],
    examples: Sequence[ArticleExample],
    *,
    embedding_weight: float,
) -> Any:
    from scipy import sparse

    if representation in ("tfidf", "hashing"):
        return feature_artifact["text_transformer"].transform(
            [item.text for item in examples]
        )

    embedding_matrix = feature_artifact["embedding_transformer"].transform(
        _embedding_array(examples)
    ) * embedding_weight
    if representation == "embedding":
        return embedding_matrix
    if representation != "hybrid":
        raise ValueError(f"Okänd feature-representation: {representation}")
    text_matrix = feature_artifact["text_transformer"].transform(
        [item.text for item in examples]
    )
    return sparse.hstack(
        (text_matrix, sparse.csr_matrix(embedding_matrix)), format="csr"
    )


def _scores(model: Any, matrix: Any) -> Any:
    import numpy as np

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(matrix)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(matrix)
    else:
        scores = model.predict(matrix)
    scores = np.asarray(scores)
    return scores.reshape((-1, 1)) if scores.ndim == 1 else scores


def choose_threshold(scores: Any, truth: Any, min_precision: float) -> Dict[str, Any]:
    """Choose one candidate-local threshold using validation data only."""
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    flat_truth = np.asarray(truth).ravel()
    flat_scores = np.asarray(scores).ravel()
    precision, recall, thresholds = precision_recall_curve(flat_truth, flat_scores)
    choices = [
        (float(recall[index]), float(precision[index]), float(threshold))
        for index, threshold in enumerate(thresholds)
        if precision[index] >= min_precision
    ]
    if choices:
        selected = max(choices, key=lambda row: (row[0], row[1], row[2]))
        return {
            "value": selected[2], "precision": selected[1], "recall": selected[0],
            "qualified": True,
        }

    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    index = int(np.argmax(f1)) if len(f1) else 0
    fallback = float(thresholds[index]) if len(thresholds) else 0.0
    return {
        "value": fallback,
        "precision": float(precision[index]),
        "recall": float(recall[index]),
        "qualified": False,
    }


def _metrics(truth: Any, scores: Any, threshold: float) -> Dict[str, float]:
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support

    truth = np.asarray(truth)
    predicted = np.asarray(scores) >= threshold
    result: Dict[str, float] = {}
    for average in ("micro", "macro"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, predicted, average=average, zero_division=0
        )
        result[f"{average}_precision"] = float(precision)
        result[f"{average}_recall"] = float(recall)
        result[f"{average}_f1"] = float(f1)

    correct_articles = np.logical_and(truth.astype(bool), predicted).any(axis=1)
    relevant_articles = truth.astype(bool).any(axis=1)
    result["article_hit_rate"] = float(
        correct_articles[relevant_articles].mean() if relevant_articles.any() else 0.0
    )

    k = min(5, scores.shape[1])
    if k:
        top_indices = np.argpartition(scores, -k, axis=1)[:, -k:]
        hits = sum(
            int(truth[row_index, label_index])
            for row_index, row in enumerate(top_indices)
            for label_index in row
        )
        result["precision_at_5"] = float(hits / (len(truth) * k))
    else:
        result["precision_at_5"] = 0.0
    return result


def _artifact_size_mb(model: Any) -> float:
    import joblib

    with tempfile.TemporaryDirectory(prefix="tagging-benchmark-") as temp_dir:
        path = Path(temp_dir) / "model.joblib"
        joblib.dump(model, path, compress=3)
        return path.stat().st_size / (1024 * 1024)


def benchmark_candidate(
    candidate: Candidate,
    representation: str,
    train: Sequence[ArticleExample],
    validation: Sequence[ArticleExample],
    test: Sequence[ArticleExample],
    labels: Sequence[str],
    *,
    max_features: int,
    min_precision: float,
    seed: int,
    jobs: int,
    embedding_weight: float,
) -> Dict[str, Any]:
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer

    label_set = set(labels)
    binarizer = MultiLabelBinarizer(classes=list(labels))
    y_train = binarizer.fit_transform([label_set.intersection(item.labels) for item in train])
    y_validation = binarizer.transform(
        [label_set.intersection(item.labels) for item in validation]
    )
    y_test = binarizer.transform([label_set.intersection(item.labels) for item in test])

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    train_started = time.perf_counter()
    x_train, feature_artifact = _fit_features(
        representation,
        candidate,
        train,
        max_features=max_features,
        embedding_weight=embedding_weight,
    )
    model = OneVsRestClassifier(candidate.factory(seed, jobs), n_jobs=jobs)
    model.fit(x_train, y_train)
    train_seconds = time.perf_counter() - train_started
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    x_validation = _transform_features(
        representation,
        feature_artifact,
        validation,
        embedding_weight=embedding_weight,
    )
    validation_scores = _scores(model, x_validation)
    threshold = choose_threshold(validation_scores, y_validation, min_precision)

    inference_started = time.perf_counter()
    x_test = _transform_features(
        representation,
        feature_artifact,
        test,
        embedding_weight=embedding_weight,
    )
    test_scores = _scores(model, x_test)
    inference_seconds = time.perf_counter() - inference_started
    metrics = _metrics(y_test, test_scores, float(threshold["value"]))

    return {
        "algorithm": candidate.name,
        "representation": representation,
        "vectorizer": representation,
        "status": "ok",
        "threshold": threshold,
        "metrics": metrics,
        "performance": {
            "train_seconds": train_seconds,
            "test_inference_seconds": inference_seconds,
            "inference_ms_per_article": 1000 * inference_seconds / max(1, len(test)),
            "peak_rss_delta_mb": max(0, rss_after - rss_before) / 1024,
            "artifact_size_mb": _artifact_size_mb(
                {
                    "features": feature_artifact,
                    "model": model,
                    "labels": list(labels),
                    "representation": representation,
                }
            ),
            "feature_count": int(x_train.shape[1]),
        },
    }


def select_recommendation(
    results: Sequence[Dict[str, Any]],
    *,
    min_precision: float = 0.90,
    max_train_seconds: float = 10.0,
) -> Optional[Dict[str, Any]]:
    successful = [result for result in results if result.get("status") == "ok"]
    if not successful:
        return None

    qualified = [
        result
        for result in successful
        if result["threshold"]["qualified"]
        and result["metrics"]["micro_precision"] >= min_precision
        and result["performance"]["train_seconds"] <= max_train_seconds
    ]
    pool = qualified or successful
    winner = max(
        pool,
        key=lambda result: (
            result["metrics"]["micro_recall"],
            result["metrics"]["article_hit_rate"],
            -result["performance"]["train_seconds"],
            -result["performance"]["artifact_size_mb"],
        ),
    )
    return {
        "algorithm": winner["algorithm"],
        "representation": winner.get("representation", winner.get("vectorizer")),
        "vectorizer": winner.get("representation", winner.get("vectorizer")),
        "meets_quality_gate": winner in qualified,
        "note": (
            "Rekommendationen klarar kvalitets- och tidsgränserna."
            if winner in qualified
            else (
                "Ingen kandidat klarade alla gränser; rekommendationen är endast "
                "bäst av de jämförda."
            )
        ),
    }


def _combination_comparison(
    benchmarks: Sequence[Dict[str, Any]],
    base_category: str,
    *,
    min_precision: float,
    max_train_seconds: float,
) -> Dict[str, Any]:
    """Rank the base category and its pairwise category combinations by micro-F1."""
    entries: List[Dict[str, Any]] = []
    seen_category_sets: set[Tuple[str, ...]] = set()
    for scope in benchmarks:
        categories = tuple(scope.get("categories") or ())
        if (
            scope.get("status") != "ok"
            or base_category not in categories
            or len(categories) > 2
            or categories in seen_category_sets
        ):
            continue
        seen_category_sets.add(categories)

        successful = [
            result
            for suite in scope.get("representations", ())
            if suite.get("status") == "ok"
            for result in suite.get("results", ())
            if result.get("status") == "ok"
        ]
        if not successful:
            continue
        winner = max(
            successful,
            key=lambda result: (
                result["metrics"]["micro_f1"],
                result["metrics"]["micro_recall"],
                result["metrics"]["micro_precision"],
                -result["performance"]["train_seconds"],
            ),
        )
        meets_quality_gate = bool(
            winner["threshold"]["qualified"]
            and winner["metrics"]["micro_precision"] >= min_precision
            and winner["performance"]["train_seconds"] <= max_train_seconds
        )
        entries.append(
            {
                "name": scope["name"],
                "categories": list(categories),
                "added_category": next(
                    (category for category in categories if category != base_category),
                    None,
                ),
                "algorithm": winner["algorithm"],
                "representation": winner.get(
                    "representation", winner.get("vectorizer")
                ),
                "meets_quality_gate": meets_quality_gate,
                "metrics": dict(winner["metrics"]),
                "performance": dict(winner["performance"]),
            }
        )

    entries.sort(
        key=lambda entry: (
            -entry["metrics"]["micro_f1"],
            -entry["metrics"]["micro_recall"],
            -entry["metrics"]["micro_precision"],
        )
    )
    return {
        "base_category": base_category,
        "ranking_metric": "micro_f1",
        "entries": entries,
    }


def _markdown_report(report: Dict[str, Any]) -> str:
    lines = [
        "# Benchmark: lättvikts-ML för artikeltaggar",
        "",
        f"Skapad: {report['created_at']}",
    ]
    comparison = report.get("combination_comparison") or {}
    comparison_entries = comparison.get("entries") or []
    if comparison_entries:
        lines.extend(
            [
                "",
                f"## Kombinationer med {comparison['base_category']}",
                "",
                "Bästa algoritm och representation per kategoriomfång, rankad efter "
                "micro-F1.",
                "",
                "| Kategorier | Algoritm | Representation | Precision | Recall | F1 | Kvalitetsgrind |",
                "|---|---|---|---:|---:|---:|:---:|",
            ]
        )
        for entry in comparison_entries:
            metrics = entry["metrics"]
            lines.append(
                "| {categories} | {algorithm} | {representation} | {precision:.3f} | "
                "{recall:.3f} | {f1:.3f} | {quality} |".format(
                    categories=" + ".join(entry["categories"]),
                    algorithm=entry["algorithm"],
                    representation=entry["representation"],
                    precision=metrics["micro_precision"],
                    recall=metrics["micro_recall"],
                    f1=metrics["micro_f1"],
                    quality="ja" if entry["meets_quality_gate"] else "nej",
                )
            )
    for scope in report["benchmarks"]:
        category_label = ", ".join(scope["categories"])
        lines.extend(["", f"## {scope['name']}: {category_label}", ""])
        if scope.get("status") != "ok":
            lines.append(f"Körningen hoppades över: {scope.get('error', 'okänt fel')}")
            continue
        for suite in scope["representations"]:
            lines.extend(["", f"### {suite['representation']}", ""])
            if suite.get("status") != "ok":
                lines.append(f"Körningen hoppades över: {suite.get('error', 'okänt fel')}")
                continue
            dataset = suite["dataset"]
            lines.append(
                f"Artiklar: {dataset['articles']} "
                f"(träning {dataset['train_articles']}, validering "
                f"{dataset['validation_articles']}, test {dataset['test_articles']}); "
                f"utvärderade taggar: {len(dataset['evaluated_labels'])}."
            )
            if suite.get("embedding"):
                embedding = suite["embedding"]
                lines.append(
                    f"Embedding: `{embedding['embedding_model']}` / "
                    f"{embedding['embedding_dimension']} dimensioner; "
                    f"täckning {embedding['embedding_coverage']:.1%}."
                )
            lines.extend(
                [
                    "",
                    "| Algoritm | Precision | Recall | F1 | P@5 | Träning (s) | ms/artikel | MB |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for result in suite["results"]:
                if result.get("status") != "ok":
                    lines.append(f"| {result.get('algorithm')} | FEL | | | | | | |")
                    continue
                metrics = result["metrics"]
                performance = result["performance"]
                lines.append(
                    "| {algorithm} | {precision:.3f} | {recall:.3f} | {f1:.3f} | "
                    "{p5:.3f} | {train:.3f} | {inference:.3f} | {size:.2f} |".format(
                        algorithm=result["algorithm"],
                        precision=metrics["micro_precision"],
                        recall=metrics["micro_recall"],
                        f1=metrics["micro_f1"],
                        p5=metrics["precision_at_5"],
                        train=performance["train_seconds"],
                        inference=performance["inference_ms_per_article"],
                        size=performance["artifact_size_mb"],
                    )
                )
            recommendation = suite.get("recommendation")
            lines.extend(["", "Rekommendation: "])
            if recommendation:
                lines[-1] += (
                    f"**{recommendation['algorithm']}** — {recommendation['note']}"
                )
            else:
                lines[-1] += "Ingen kandidat kunde utvärderas."
    lines.extend(
        [
            "",
            "Rekommendationen är beslutsunderlag och aktiverar ingen modell i produktionsflödet.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_names(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _format_default_names(names: Iterable[str] | str) -> str:
    """Format a CLI default without treating one string as an iterable of chars."""
    if isinstance(names, str):
        return names
    return ",".join(names)


def _merge_categories(
    defaults: Sequence[str], manually_requested: Sequence[str]
) -> List[str]:
    """Combine defaults and CLI additions case-insensitively, preserving order."""
    merged: List[str] = []
    seen: set[str] = set()
    for raw_category in (*defaults, *manually_requested):
        category = str(raw_category).strip().upper()
        if category and category not in seen:
            seen.add(category)
            merged.append(category)
    return merged


def _category_scopes(
    categories: Sequence[str], base_category: str = "DOMAIN_ENTITY"
) -> List[Tuple[str, List[str]]]:
    scopes = [("Alla kategorier", list(categories))]
    scopes.extend((f"Kategori {category}", [category]) for category in categories)
    if base_category in categories:
        for category in categories:
            pair = [base_category, category]
            if category != base_category and pair != list(categories):
                scopes.append((f"Kombination {base_category} + {category}", pair))
    return scopes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmarka scikit-learn-algoritmer mot befintliga MongoDB-taggar."
    )
    parser.add_argument("--config", default=None, help="Sökväg till config.yaml")
    parser.add_argument("--output-dir", default="benchmark_results/tagging_ml")
    parser.add_argument(
        "--categories",
        default="",
        help=(
            "Kommaseparerade kategorier som läggs till efter DEFAULT_CATEGORIES; "
            "standardkategorierna körs alltid."
        ),
    )
    parser.add_argument(
        "--combination-base-category",
        default="DOMAIN_ENTITY",
        help=(
            "Kategori som jämförs ensam och parvis med övriga kategorier "
            "(standard: DOMAIN_ENTITY)."
        ),
    )
    parser.add_argument("--algorithms", default=_format_default_names(_candidate_factories()))
    parser.add_argument(
        "--representations",
        "--vectorizers",
        dest="representations",
        default=_format_default_names(DEFAULT_REPRESENTATIONS),
        help="tfidf, hashing, embedding och/eller hybrid",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Kräv en viss sparad embeddingmodell; annars väljs den vanligaste.",
    )
    parser.add_argument(
        "--embedding-weight",
        type=float,
        default=1.0,
        help="Vikt för embeddingdelen i hybridrepresentationen.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--min-label-support", type=int, default=3)
    parser.add_argument("--min-precision", type=float, default=0.90)
    parser.add_argument("--max-train-seconds", type=float, default=10.0)
    parser.add_argument("--max-features", type=int, default=32768)
    parser.add_argument("--max-article-chars", type=int, default=10000)
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _run_representation(
    *,
    representation: str,
    examples: Sequence[ArticleExample],
    base_metadata: Dict[str, Any],
    candidates: Dict[str, Candidate],
    algorithm_names: Sequence[str],
    embedding_metadata: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    selected_examples = list(examples)

    train, validation, test = chronological_split(
        selected_examples, args.validation_fraction, args.test_fraction
    )
    labels, excluded_labels = eligible_labels(
        train, validation, test, args.min_label_support
    )
    if not labels:
        raise ValueError("Inga taggar har tillräckligt stöd i träningsdatan")

    results: List[Dict[str, Any]] = []
    for algorithm_name in algorithm_names:
        log.info("Kör %s med %s", algorithm_name, representation)
        try:
            result = benchmark_candidate(
                candidates[algorithm_name],
                representation,
                train,
                validation,
                test,
                labels,
                max_features=args.max_features,
                min_precision=args.min_precision,
                seed=args.seed,
                jobs=args.jobs,
                embedding_weight=args.embedding_weight,
            )
        except Exception as exc:
            log.exception("Benchmark misslyckades för %s/%s", algorithm_name, representation)
            result = {
                "algorithm": algorithm_name,
                "representation": representation,
                "vectorizer": representation,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)

    suite = {
        "representation": representation,
        "status": "ok",
        "dataset": {
            **base_metadata,
            "articles": len(selected_examples),
            "train_articles": len(train),
            "validation_articles": len(validation),
            "test_articles": len(test),
            "evaluated_labels": labels,
            "excluded_labels": excluded_labels,
        },
        "results": results,
        "recommendation": select_recommendation(
            results,
            min_precision=args.min_precision,
            max_train_seconds=args.max_train_seconds,
        ),
    }
    suite["embedding"] = embedding_metadata
    return suite


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config_path = _resolve_config_path(args.config)
    config = _load_config(config_path)
    store_config = config.get("store") or {}
    if str(store_config.get("provider") or "").lower() not in ("mongo", "mongodb"):
        raise ValueError("Benchmarken kräver att store.provider är mongodb")

    store = create_store(store_config)
    if not isinstance(store, MongoDBStore):
        raise TypeError("Den konfigurerade storen är inte MongoDBStore")

    candidates = _candidate_factories()
    algorithm_names = _parse_names(args.algorithms)
    unknown_algorithms = sorted(set(algorithm_names) - set(candidates))
    if unknown_algorithms:
        raise ValueError("Okända algoritmer: " + ", ".join(unknown_algorithms))
    representations = _parse_names(args.representations)
    unknown_representations = sorted(
        set(representations) - set(DEFAULT_REPRESENTATIONS)
    )
    if unknown_representations:
        raise ValueError(
            "Okända feature-representationer: " + ", ".join(unknown_representations)
        )
    if args.embedding_weight <= 0:
        raise ValueError("--embedding-weight måste vara större än noll")

    manual_categories = _parse_names(args.categories)
    categories = _merge_categories(DEFAULT_CATEGORIES, manual_categories)
    combination_base_category = str(args.combination_base_category).strip().upper()
    if not combination_base_category:
        raise ValueError("--combination-base-category får inte vara tom")
    if combination_base_category not in categories:
        raise ValueError(
            "--combination-base-category måste finnas bland benchmarkkategorierna: "
            + ", ".join(categories)
        )
    benchmarks: List[Dict[str, Any]] = []
    any_success = False
    for scope_name, scope_categories in _category_scopes(
        categories, combination_base_category
    ):
        log.info("Benchmarkomfång: %s", ", ".join(scope_categories))
        try:
            examples, dataset_metadata = load_mongodb_examples(
                store,
                scope_categories,
                max_chars=args.max_article_chars,
                max_articles=args.max_articles,
            )
            selected_examples, selected_metadata, embedding_metadata = (
                _select_common_embedding_examples(
                    examples, dataset_metadata, args.embedding_model
                )
            )
        except Exception as exc:
            log.exception("Kunde inte läsa benchmarkomfånget %s", scope_name)
            benchmarks.append(
                {
                    "name": scope_name,
                    "categories": scope_categories,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "representations": [],
                }
            )
            continue

        suites: List[Dict[str, Any]] = []
        for representation in representations:
            try:
                suite = _run_representation(
                    representation=representation,
                    examples=selected_examples,
                    base_metadata=selected_metadata,
                    candidates=candidates,
                    algorithm_names=algorithm_names,
                    embedding_metadata=embedding_metadata,
                    args=args,
                )
                any_success = any(
                    result.get("status") == "ok" for result in suite["results"]
                ) or any_success
            except Exception as exc:
                log.exception(
                    "Kunde inte köra %s för %s", representation, scope_name
                )
                suite = {
                    "representation": representation,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "results": [],
                    "recommendation": None,
                }
            suites.append(suite)
        benchmarks.append(
            {
                "name": scope_name,
                "categories": scope_categories,
                "status": "ok",
                "representations": suites,
            }
        )

    import numpy
    import sklearn

    report = {
        "schema_version": 3,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
            "numpy": numpy.__version__,
        },
        "config": {
            "default_categories": list(DEFAULT_CATEGORIES),
            "manually_added_categories": manual_categories,
            "categories": categories,
            "combination_base_category": combination_base_category,
            "algorithms": algorithm_names,
            "representations": representations,
            "embedding_model": args.embedding_model,
            "embedding_weight": args.embedding_weight,
            "validation_fraction": args.validation_fraction,
            "test_fraction": args.test_fraction,
            "min_label_support": args.min_label_support,
            "min_precision": args.min_precision,
            "max_train_seconds": args.max_train_seconds,
            "max_features": args.max_features,
            "max_article_chars": args.max_article_chars,
            "seed": args.seed,
        },
        "benchmarks": benchmarks,
    }
    report["combination_comparison"] = _combination_comparison(
        benchmarks,
        combination_base_category,
        min_precision=args.min_precision,
        max_train_seconds=args.max_train_seconds,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark.json"
    markdown_path = output_dir / "benchmark.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    log.info("Skrev %s och %s", json_path, markdown_path)
    return 0 if any_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
