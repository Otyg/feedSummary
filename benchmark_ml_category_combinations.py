#!/usr/bin/env python3
"""Train and compare the three requested embedding tag category combinations.

The benchmark uses one shared, chronological and content-hash-grouped split for
all combinations.  Thresholds are selected on validation data, metrics are
reported on held-out test data, and a final production-compatible artifact is
then fitted on the complete embedding-covered corpus for each combination.
MongoDB is only read; the active application configuration is not changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from feedsummary_core.persistence import MongoDBStore, create_store
from feedsummary_core.tagging_ml.benchmark import (
    _aggregate_metrics,
    _coverage_adjusted_f1,
    _per_label_metrics,
    _threshold_predictions,
    _tune_threshold,
)
from feedsummary_core.tagging_ml.embedding_sgd import EmbeddingClassifierSettings
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, Normalizer

log = logging.getLogger("ml_category_combination_benchmark")
ARTIFACT_VERSION = 2
CATEGORY_COMBINATIONS: tuple[tuple[str, ...], ...] = (
    ("DOMAIN_ENTITY", "DATATYPE"),
    ("DOMAIN_ENTITY", "SECTOR"),
    ("DOMAIN_ENTITY", "DATATYPE", "SECTOR"),
)


@dataclass(frozen=True)
class Example:
    article_id: str
    timestamp: int
    group_key: str
    embedding: tuple[float, ...]
    embedding_model: str
    source_hash: str
    labels_by_category: dict[str, tuple[str, ...]]

    def labels_for(self, categories: Sequence[str]) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    label
                    for category in categories
                    for label in self.labels_by_category.get(category, ())
                }
            )
        )


def _config_path(raw_path: str | None) -> Path:
    candidate = raw_path or os.environ.get("FEEDSUMMARY_CONFIG") or "config.yaml"
    return Path(os.path.expandvars(os.path.expanduser(candidate))).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise TypeError("Configuration root must be a mapping")
    return config


def _timestamp(article: dict[str, Any]) -> int:
    for key in ("published_ts", "_mongo_sort_ts", "fetched_at", "created_at"):
        try:
            value = int(article.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0


def _load_examples(
    store: MongoDBStore,
    categories: Sequence[str],
    requested_embedding_model: str,
) -> tuple[list[Example], dict[str, Any]]:
    staged: list[Example] = []
    signatures: Counter[tuple[str, int]] = Counter()
    exported_rows = 0
    invalid_embeddings = 0

    for row in store.iter_articles_with_tags(categories=list(categories)):
        exported_rows += 1
        article = row.get("article") if isinstance(row, dict) else None
        if not isinstance(article, dict):
            continue
        vector = article.get("tagging_embedding_vector")
        model = str(article.get("tagging_embedding_model") or "").strip()
        if (
            not isinstance(vector, list)
            or not vector
            or not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in vector
            )
        ):
            invalid_embeddings += 1
            continue
        if requested_embedding_model and model != requested_embedding_model:
            continue

        article_id = str(article.get("id") or article.get("_id") or "").strip()
        if not article_id:
            continue
        labels_by_category: dict[str, set[str]] = defaultdict(set)
        for tag in row.get("tags") or ():
            if not isinstance(tag, dict):
                continue
            category = str(tag.get("category") or "").strip().upper()
            name = str(tag.get("name") or "").strip().casefold()
            if category in categories and name:
                labels_by_category[category].add(name)

        signature = (model, len(vector))
        signatures[signature] += 1
        content_hash = str(article.get("content_hash") or "").strip()
        staged.append(
            Example(
                article_id=article_id,
                timestamp=_timestamp(article),
                group_key=f"hash:{content_hash}"
                if content_hash
                else f"article:{article_id}",
                embedding=tuple(float(value) for value in vector),
                embedding_model=model,
                source_hash=str(
                    article.get("tagging_embedding_source_hash") or content_hash
                ),
                labels_by_category={
                    category: tuple(sorted(names))
                    for category, names in labels_by_category.items()
                },
            )
        )

    if not signatures:
        raise ValueError("No compatible persisted article embeddings were found")
    selected_signature = min(
        signatures,
        key=lambda signature: (-signatures[signature], signature[0], signature[1]),
    )
    examples = [
        item
        for item in staged
        if (item.embedding_model, len(item.embedding)) == selected_signature
    ]
    examples.sort(key=lambda item: (item.timestamp, item.article_id))
    if len(examples) < 10:
        raise ValueError("At least 10 compatible examples are required")
    return examples, {
        "exported_tagged_articles": exported_rows,
        "compatible_articles": len(examples),
        "invalid_or_missing_embeddings": invalid_embeddings,
        "embedding_model": selected_signature[0],
        "embedding_dimension": selected_signature[1],
        "available_embedding_signatures": [
            {"model": model, "dimension": dimension, "articles": count}
            for (model, dimension), count in sorted(
                signatures.items(), key=lambda item: (-item[1], item[0])
            )
        ],
    }


def _grouped_chronological_split(
    examples: Sequence[Example],
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[example.group_key].append(index)
    ordered = sorted(
        groups.items(),
        key=lambda item: (min(examples[index].timestamp for index in item[1]), item[0]),
    )
    train_target = max(1, int(len(examples) * 0.60))
    validation_target = max(1, int(len(examples) * 0.20))
    result = {"train": [], "validation": [], "test": []}
    for _, indices in ordered:
        if len(result["train"]) < train_target:
            destination = "train"
        elif len(result["validation"]) < validation_target:
            destination = "validation"
        else:
            destination = "test"
        result[destination].extend(indices)
    if any(not result[name] for name in result):
        raise ValueError("The grouped corpus cannot produce three non-empty partitions")
    return result


def _label_vocabulary(
    examples: Sequence[Example],
    categories: Sequence[str],
    train_indices: Sequence[int],
    min_support: int,
) -> tuple[list[str], dict[str, Any]]:
    support: Counter[str] = Counter()
    category_by_label: dict[str, str] = {}
    for example in examples:
        for category in categories:
            for label in example.labels_by_category.get(category, ()):
                support[label] += 1
                category_by_label[label] = category
    train_support: Counter[str] = Counter(
        label
        for index in train_indices
        for label in examples[index].labels_for(categories)
    )
    eligible = sorted(
        label
        for label, count in support.items()
        if count >= min_support and 0 < train_support[label] < len(train_indices)
    )
    excluded = {
        label: (
            f"support {count} < {min_support}"
            if count < min_support
            else "missing positive or negative training examples"
        )
        for label, count in sorted(support.items())
        if label not in eligible
    }
    if not eligible:
        raise ValueError("No labels have enough support in the training corpus")
    return eligible, {
        "observed_labels": len(support),
        "eligible_labels": len(eligible),
        "observed_assignments": sum(support.values()),
        "eligible_assignments": sum(support[label] for label in eligible),
        "assignment_coverage": (
            sum(support[label] for label in eligible) / sum(support.values())
            if support
            else 0.0
        ),
        "label_support": dict(sorted(support.items())),
        "excluded_labels": excluded,
        "category_by_label": category_by_label,
    }


def _estimator(settings: EmbeddingClassifierSettings) -> Any:
    if settings.algorithm == "logistic_regression":
        return LogisticRegression(
            C=settings.regularization_c,
            class_weight="balanced",
            max_iter=settings.max_iter,
            tol=settings.tolerance,
            random_state=settings.random_state,
        )
    return SGDClassifier(
        loss="log_loss",
        alpha=settings.alpha,
        class_weight="balanced",
        max_iter=settings.max_iter,
        tol=settings.tolerance,
        random_state=settings.random_state,
    )


def _fit(
    examples: Sequence[Example],
    categories: Sequence[str],
    labels: Sequence[str],
    settings: EmbeddingClassifierSettings,
) -> tuple[Any, Normalizer, str, np.ndarray]:
    features = np.asarray([item.embedding for item in examples], dtype=float)
    transformer = Normalizer(norm="l2")
    transformed = transformer.fit_transform(features)
    label_set = set(labels)
    binarizer = MultiLabelBinarizer(classes=list(labels))
    targets = binarizer.fit_transform(
        [label_set.intersection(item.labels_for(categories)) for item in examples]
    )
    estimator = _estimator(settings)
    if targets.shape[1] == 1:
        classifier = estimator.fit(transformed, targets[:, 0])
        kind = "binary"
    else:
        classifier = OneVsRestClassifier(estimator, n_jobs=1).fit(transformed, targets)
        kind = "multilabel"
    return classifier, transformer, kind, targets


def _probabilities(classifier: Any, kind: str, features: np.ndarray) -> np.ndarray:
    raw = np.asarray(classifier.predict_proba(features), dtype=float)
    if kind == "binary":
        return raw[:, 1].reshape((-1, 1))
    return raw


def _subset(values: Sequence[Example], indices: Sequence[int]) -> list[Example]:
    return [values[index] for index in indices]


def _category_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    labels: Sequence[str],
    category_by_label: dict[str, str],
    categories: Sequence[str],
    ml: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for category in categories:
        indices = [
            index
            for index, label in enumerate(labels)
            if category_by_label[label] == category
        ]
        if indices:
            result[category] = _aggregate_metrics(
                truth[:, indices], predicted[:, indices], ml
            )
    return result


def _fingerprint(
    examples: Sequence[Example],
    categories: Sequence[str],
    settings: EmbeddingClassifierSettings,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"categories": categories, "settings": asdict(settings)},
            default=str,
            sort_keys=True,
        ).encode()
    )
    for example in examples:
        digest.update(example.article_id.encode())
        digest.update(example.source_hash.encode())
        digest.update(np.asarray(example.embedding, dtype=np.float64).tobytes())
        digest.update("\0".join(example.labels_for(categories)).encode())
    return digest.hexdigest()


def _train_combination(
    examples: Sequence[Example],
    split: dict[str, list[int]],
    categories: tuple[str, ...],
    base_settings: EmbeddingClassifierSettings,
    min_support: int,
    output_dir: Path,
    ml: dict[str, Any],
) -> dict[str, Any]:
    labels, vocabulary = _label_vocabulary(
        examples, categories, split["train"], min_support
    )
    settings = replace(
        base_settings,
        enabled=True,
        categories=categories,
        min_label_support=min_support,
        model_path=str(output_dir / "model.joblib"),
    )
    train_examples = _subset(examples, split["train"])
    classifier, transformer, kind, train_targets = _fit(
        train_examples, categories, labels, settings
    )
    validation_examples = _subset(examples, split["validation"])
    validation_features = transformer.transform(
        np.asarray([item.embedding for item in validation_examples], dtype=float)
    )
    binarizer = MultiLabelBinarizer(classes=labels)
    label_set = set(labels)
    validation_targets = binarizer.fit_transform(
        [
            label_set.intersection(item.labels_for(categories))
            for item in validation_examples
        ]
    )
    validation_scores = _probabilities(classifier, kind, validation_features)
    threshold, _, validation_metrics = _tune_threshold(
        validation_scores,
        validation_targets,
        settings.max_tags_per_article,
        "predict_proba",
        ml,
    )
    observed_validation_assignments = sum(
        len(example.labels_for(categories)) for example in validation_examples
    )
    eligible_validation_assignments = int(validation_targets.sum())
    validation_assignment_coverage = (
        eligible_validation_assignments / observed_validation_assignments
        if observed_validation_assignments
        else 0.0
    )

    test_examples = _subset(examples, split["test"])
    test_features = transformer.transform(
        np.asarray([item.embedding for item in test_examples], dtype=float)
    )
    test_targets = binarizer.transform(
        [label_set.intersection(item.labels_for(categories)) for item in test_examples]
    )
    test_scores = _probabilities(classifier, kind, test_features)
    test_predictions = _threshold_predictions(
        test_scores, threshold, settings.max_tags_per_article, ml
    )
    test_metrics = _aggregate_metrics(test_targets, test_predictions, ml)
    category_metrics = _category_metrics(
        test_targets,
        test_predictions,
        labels,
        vocabulary["category_by_label"],
        categories,
        ml,
    )
    observed_test_assignments = sum(
        len(example.labels_for(categories)) for example in test_examples
    )
    eligible_test_assignments = int(test_targets.sum())
    test_assignment_coverage = (
        eligible_test_assignments / observed_test_assignments
        if observed_test_assignments
        else 0.0
    )

    final_classifier, final_transformer, final_kind, _ = _fit(
        examples, categories, labels, settings
    )
    fingerprint = _fingerprint(examples, categories, settings)
    trained_at = datetime.now(timezone.utc).isoformat()
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "algorithm": settings.algorithm,
        "representation": "embedding",
        "classifier": final_classifier,
        "classifier_kind": final_kind,
        "feature_transformer": final_transformer,
        "classes": list(labels),
        "categories": list(categories),
        "embedding_model": examples[0].embedding_model,
        "embedding_dimension": len(examples[0].embedding),
        "threshold": float(threshold),
        "max_tags_per_article": settings.max_tags_per_article,
        "corpus_fingerprint": fingerprint,
        "training_articles": len(examples),
        "trained_at": trained_at,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    model_path = output_dir / "model.joblib"
    joblib.dump(artifact, model_path)
    snippet = {
        "tagging": {
            "ml": {
                "enabled": True,
                "classifier": settings.algorithm,
                "representation": "embedding",
                "categories": list(categories),
                "model_path": str(model_path),
                "min_label_support": min_support,
                "max_tags_per_article": settings.max_tags_per_article,
                "threshold": float(threshold),
            }
        }
    }
    (output_dir / "config-snippet.yaml").write_text(
        yaml.safe_dump(snippet, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return {
        "categories": list(categories),
        "status": "ok",
        "classifier": settings.algorithm,
        "labels": labels,
        "vocabulary": vocabulary,
        "validation": validation_metrics,
        "validation_assignment_coverage": validation_assignment_coverage,
        "coverage_adjusted_validation_micro_f1": _coverage_adjusted_f1(
            validation_metrics, validation_assignment_coverage
        ),
        "test": test_metrics,
        "test_by_category": category_metrics,
        "test_assignment_coverage": test_assignment_coverage,
        "coverage_adjusted_test_micro_f1": _coverage_adjusted_f1(
            test_metrics, test_assignment_coverage
        ),
        "test_per_label": _per_label_metrics(
            test_targets, test_predictions, labels, ml
        ),
        "threshold": float(threshold),
        "training": {
            "train_target_shape": list(train_targets.shape),
            "final_training_articles": len(examples),
            "trained_at": trained_at,
            "corpus_fingerprint": fingerprint,
        },
        "artifact": {
            "model": str(model_path),
            "config_snippet": str(output_dir / "config-snippet.yaml"),
            "size_bytes": model_path.stat().st_size,
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Benchmark: DOMAIN_ENTITY med datadomän och sektor",
        "",
        f"Skapad: `{report['generated_at']}`",
        "",
        "Datadomän motsvarar den lagrade kategorinyckeln `DATATYPE`.",
        "Alla modeller använder samma kronologiska, innehållshash-grupperade delning.",
        "Tröskeln valdes på valideringsdelen; nedanstående mått kommer från testdelen.",
        "",
        "| Rang | Kategorier | Etiketter | Val justerad F1 | Testtäckning | Testprecision | Testrecall | Test micro-F1 | Test justerad F1 | DOMAIN_ENTITY F1 | Tröskel |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, item in enumerate(report["ranking"], start=1):
        core = item["test_by_category"].get("DOMAIN_ENTITY", {})
        lines.append(
            f"| {rank} | {' + '.join(item['categories'])} | "
            f"{item['vocabulary']['eligible_labels']}/{item['vocabulary']['observed_labels']} | "
            f"{item['coverage_adjusted_validation_micro_f1']:.3f} | "
            f"{item['test_assignment_coverage']:.1%} | "
            f"{item['test']['micro_precision']:.3f} | "
            f"{item['test']['micro_recall']:.3f} | "
            f"{item['test']['micro_f1']:.3f} | "
            f"{item['coverage_adjusted_test_micro_f1']:.3f} | "
            f"{core.get('micro_f1', 0.0):.3f} | {item['threshold']:.6f} |"
        )
    lines.extend(
        [
            "",
            (
                "Rangordningen använder täckningsjusterad micro-F1 på valideringsdelen. "
                "Testmåtten är orörda slutmått och har inte använts för modellvalet."
            ),
            "",
            "## Dataset",
            "",
            f"- Kompatibla artiklar: **{report['dataset']['compatible_articles']}**",
            f"- Träning: **{report['split']['train']}**",
            f"- Validering: **{report['split']['validation']}**",
            f"- Test: **{report['split']['test']}**",
            (
                f"- Embedding: `{report['dataset']['embedding_model']}` / "
                f"{report['dataset']['embedding_dimension']} dimensioner"
            ),
            "",
            (
                "De tre fulltränade modellartefakterna ligger i respektive underkatalog. "
                "Ingen av dem har aktiverats i produktionskonfigurationen."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run(
    config_path: Path, output_parent: Path, min_support: int | None = None
) -> dict[str, Any]:
    config = _load_config(config_path)
    store_config = config.get("store")
    if not isinstance(store_config, dict):
        raise TypeError("Configuration must contain a store mapping")
    provider = str(
        store_config.get("provider") or store_config.get("type") or ""
    ).lower()
    if provider not in {"mongo", "mongodb"}:
        raise ValueError("This benchmark requires MongoDB")
    tagging = config.get("tagging") if isinstance(config.get("tagging"), dict) else {}
    ml_config = tagging.get("ml") if isinstance(tagging.get("ml"), dict) else {}
    base_settings = EmbeddingClassifierSettings.from_config(config)
    if base_settings.algorithm not in {"logistic_regression", "sgd"}:
        raise ValueError("tagging.ml.classifier must be logistic_regression or sgd")
    selected_min_support = int(
        min_support
        if min_support is not None
        else ml_config.get("min_label_support", 3)
    )
    if selected_min_support < 2:
        raise ValueError("min_label_support must be at least 2")

    read_only_config = dict(store_config)
    read_only_config["initialize_schema"] = False
    store = create_store(read_only_config)
    if not isinstance(store, MongoDBStore):
        raise TypeError("Configured store is not MongoDB")
    try:
        all_categories = tuple(
            dict.fromkeys(
                category for scope in CATEGORY_COMBINATIONS for category in scope
            )
        )
        examples, dataset = _load_examples(
            store, all_categories, base_settings.embedding_model
        )
        split = _grouped_chronological_split(examples)
        generated_at = datetime.now(timezone.utc).isoformat()
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = output_parent.expanduser().resolve() / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        ml = {
            "np": np,
            "accuracy_score": accuracy_score,
            "hamming_loss": hamming_loss,
        }
        results = []
        for categories in CATEGORY_COMBINATIONS:
            log.info("Training and evaluating %s", " + ".join(categories))
            slug = "__".join(category.lower() for category in categories)
            results.append(
                _train_combination(
                    examples,
                    split,
                    categories,
                    base_settings,
                    selected_min_support,
                    run_dir / slug,
                    ml,
                )
            )
        ranking = sorted(
            results,
            key=lambda item: (
                item["coverage_adjusted_validation_micro_f1"],
                item["validation"]["micro_f1"],
                item["validation"]["micro_recall"],
                -len(item["categories"]),
            ),
            reverse=True,
        )
        report = {
            "generated_at": generated_at,
            "category_aliases": {"DATADOMAIN": "DATATYPE", "Datadomän": "DATATYPE"},
            "classifier": base_settings.algorithm,
            "min_label_support": selected_min_support,
            "dataset": dataset,
            "split": {name: len(indices) for name, indices in split.items()},
            "results": results,
            "ranking": ranking,
            "winner": ranking[0]["categories"],
            "selection_rule": "coverage-adjusted chronological validation micro-F1",
            "production_activated": False,
            "output_directory": str(run_dir),
        }
        (run_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (run_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
        return report
    finally:
        store.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="FeedSummary config.yaml")
    parser.add_argument(
        "--output-dir",
        default="benchmark_results/ml-category-combinations",
        help="Parent directory for timestamped benchmark runs",
    )
    parser.add_argument("--min-label-support", type=int)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    raw_uri = ""
    try:
        config_path = _config_path(args.config)
        config = _load_config(config_path)
        store = config.get("store") if isinstance(config.get("store"), dict) else {}
        raw_uri = str(store.get("uri") or "")
        report = run(
            config_path,
            Path(args.output_dir),
            min_support=args.min_label_support,
        )
        print(f"Winner: {' + '.join(report['winner'])}")
        print(f"Output: {report['output_directory']}")
        return 0
    except KeyboardInterrupt:
        log.warning("Interrupted")
        return 130
    except Exception as error:  # noqa: BLE001
        message = (
            str(error).replace(raw_uri, "<mongodb-uri>") if raw_uri else str(error)
        )
        log.error("Benchmark failed (%s): %s", type(error).__name__, message)
        return 2


if __name__ == "__main__":
    sys.exit(main())
