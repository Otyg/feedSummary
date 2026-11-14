#!/usr/bin/env python3
"""Compare representative FeedSummary operations in CouchDB and MongoDB.

The read/filter tests use documents that exist in both stores.  The write test
uses temporary article ids and removes them from both databases in ``finally``.
No production document is modified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from feedsummary_core.persistence import CouchDBStore, MongoDBStore


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _measure(
    operation: Callable[[], Any], *, warmups: int, iterations: int, units: int = 1
) -> Dict[str, Any]:
    for _ in range(warmups):
        operation()
    samples = []
    result_size: Optional[int] = None
    for _ in range(iterations):
        started = time.perf_counter_ns()
        result = operation()
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
        if result_size is None and hasattr(result, "__len__"):
            result_size = len(result)
    median = statistics.median(samples)
    return {
        "iterations": iterations,
        "median_ms": round(median, 3),
        "p95_ms": round(_percentile(samples, 0.95), 3),
        "min_ms": round(min(samples), 3),
        "max_ms": round(max(samples), 3),
        "median_ms_per_item": round(median / max(1, units), 4),
        "items_per_iteration": units,
        "result_size": result_size,
        "samples_ms": [round(value, 3) for value in samples],
    }


def _article_id(article: Dict[str, Any]) -> str:
    return str(article.get("id") or article.get("_id") or "").strip()


def _timestamp(article: Dict[str, Any]) -> int:
    for field in ("published_ts", "fetched_at", "_mongo_sort_ts"):
        try:
            value = int(article.get(field) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return 0


def _common_sample(
    couch: CouchDBStore, mongo: MongoDBStore, sample_size: int
) -> Tuple[List[Dict[str, Any]], List[str]]:
    candidates = list(couch.iter_articles(limit=max(sample_size * 3, sample_size)))
    ids = [_article_id(article) for article in candidates if _article_id(article)]
    mongo_ids = {
        _article_id(article) for article in mongo.get_articles_by_ids(ids) if _article_id(article)
    }
    articles = [article for article in candidates if _article_id(article) in mongo_ids][:sample_size]
    if not articles:
        raise RuntimeError("Databaserna saknar gemensamma artiklar att mäta")
    return articles, [_article_id(article) for article in articles]


def _filter_arguments(articles: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    timestamps = [value for value in (_timestamp(article) for article in articles) if value > 0]
    sources = sorted(
        {str(article.get("source") or "").strip() for article in articles}
        - {""}
    )
    return {
        "sources": sources[: min(5, len(sources))],
        "since_ts": min(timestamps) if timestamps else 0,
        "until_ts": max(timestamps) if timestamps else None,
        "limit": max(1, len(articles)),
    }


def _common_tag_names(
    couch: CouchDBStore,
    mongo: MongoDBStore,
    article_ids: Sequence[str],
    limit: int = 3,
) -> List[str]:
    names: List[str] = []
    for article_id in article_ids:
        for tag in couch.get_article_tags(article_id):
            name = str(tag.get("name") or "").strip().lower()
            if name and name not in names and mongo.get_tag_by_name(name) is not None:
                names.append(name)
                if len(names) >= limit:
                    return names
    return names


def _temporary_articles(prefix: str, count: int, iterations: int) -> List[List[Dict[str, Any]]]:
    now = int(time.time())
    return [
        [
            {
                "id": f"{prefix}{iteration}_{index}",
                "url": f"https://benchmark.invalid/{prefix}{iteration}_{index}",
                "title": f"FeedSummary database benchmark {iteration}/{index}",
                "text": "Temporary benchmark article. This document is safe to delete.",
                "source": "__feedsummary_database_benchmark__",
                "published_ts": now + iteration * count + index,
                "fetched_at": now + iteration * count + index,
                "summarized": False,
            }
            for index in range(count)
        ]
        for iteration in range(iterations)
    ]


def _benchmark_store(
    store: Any,
    *,
    sample_ids: List[str],
    filter_args: Dict[str, Any],
    tag_names: List[str],
    write_batches: List[List[Dict[str, Any]]],
    warmups: int,
    iterations: int,
) -> Dict[str, Any]:
    first_id = sample_ids[0]
    operations: Dict[str, Dict[str, Any]] = {
        "read_one_article": _measure(
            lambda: store.get_article(first_id), warmups=warmups, iterations=iterations
        ),
        "read_articles_by_ids": _measure(
            lambda: store.get_articles_by_ids(sample_ids),
            warmups=warmups,
            iterations=iterations,
            units=len(sample_ids),
        ),
        "filter_articles_by_source_and_time": _measure(
            lambda: store.list_articles_by_filter(**filter_args),
            warmups=warmups,
            iterations=iterations,
        ),
        "list_unsummarized_articles": _measure(
            lambda: store.list_unsummarized_articles(limit=len(sample_ids)),
            warmups=warmups,
            iterations=iterations,
        ),
    }
    if tag_names:
        operations["filter_articles_by_tag"] = _measure(
            lambda: store.get_articles_by_tags(tag_names, match_mode="any"),
            warmups=warmups,
            iterations=iterations,
        )

    write_iteration = iter(write_batches)

    def write_batch() -> None:
        for document in next(write_iteration):
            store.upsert_article(document)

    # No warmup: a warmup would turn the measured inserts into updates.
    operations["write_articles_upsert"] = _measure(
        write_batch,
        warmups=0,
        iterations=iterations,
        units=len(write_batches[0]),
    )
    return operations


def _delete_temporary(store: Any, document_ids: List[str]) -> int:
    # Query exact ids and source so cleanup cannot remove a production article.
    result = store.db.articles.delete_many(
        {
            "source": "__feedsummary_database_benchmark__",
            "_id": {"$in": document_ids},
        }
    )
    return int(result.deleted_count)


def _comparison(results: Dict[str, Any]) -> Dict[str, Any]:
    comparison = {}
    couch_ops = results["couchdb"]
    mongo_ops = results["mongodb"]
    for name in sorted(couch_ops.keys() & mongo_ops.keys()):
        couch_ms = float(couch_ops[name]["median_ms"])
        mongo_ms = float(mongo_ops[name]["median_ms"])
        comparison[name] = {
            "faster": "mongodb" if mongo_ms < couch_ms else "couchdb",
            "couchdb_over_mongodb_ratio": (
                round(couch_ms / mongo_ms, 2) if mongo_ms > 0 else None
            ),
        }
    return comparison


def benchmark(
    *,
    couch_url: str,
    couch_database: str,
    mongo_uri: str,
    mongo_database: str,
    couch_username: Optional[str] = None,
    couch_password: Optional[str] = None,
    sample_size: int = 100,
    write_count: int = 20,
    warmups: int = 1,
    iterations: int = 5,
) -> Dict[str, Any]:
    if min(sample_size, write_count, iterations) <= 0 or warmups < 0:
        raise ValueError("sample-size, write-count och iterations måste vara positiva")
    couch = CouchDBStore(
        url=couch_url,
        database=couch_database,
        username=couch_username,
        password=couch_password,
        initialize_schema=False,
    )
    mongo = MongoDBStore(
        uri=mongo_uri, database=mongo_database, initialize_schema=False
    )
    # CouchDB reserves document ids beginning with an underscore.
    prefix = f"fs-benchmark-{uuid.uuid4().hex}-"
    write_batches = _temporary_articles(prefix, write_count, iterations)
    temporary_ids = [document["id"] for batch in write_batches for document in batch]
    try:
        couch_status, _ = couch.client.request("GET", "/")
        if couch_status != 200:
            raise RuntimeError(f"CouchDB svarade med HTTP {couch_status}")
        mongo.client.admin.command("ping")
        sample_articles, sample_ids = _common_sample(couch, mongo, sample_size)
        filter_args = _filter_arguments(sample_articles)
        tag_names = _common_tag_names(couch, mongo, sample_ids)
        results = {
            "couchdb": _benchmark_store(
                couch,
                sample_ids=sample_ids,
                filter_args=filter_args,
                tag_names=tag_names,
                write_batches=write_batches,
                warmups=warmups,
                iterations=iterations,
            ),
            "mongodb": _benchmark_store(
                mongo,
                sample_ids=sample_ids,
                filter_args=filter_args,
                tag_names=tag_names,
                write_batches=write_batches,
                warmups=warmups,
                iterations=iterations,
            ),
        }
        return {
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "sample_size": len(sample_ids),
                "write_count": write_count,
                "warmups": warmups,
                "iterations": iterations,
                "couch_database": couch_database,
                "mongo_database": mongo_database,
            },
            "workload": {"filter": filter_args, "tag_names": tag_names},
            "results": results,
            "comparison": _comparison(results),
        }
    finally:
        try:
            _delete_temporary(couch, temporary_ids)
        finally:
            try:
                _delete_temporary(mongo, temporary_ids)
            finally:
                couch.close()
                mongo.close()


def _print_table(report: Dict[str, Any]) -> None:
    print(f"{'Operation':42} {'Couch median':>13} {'Mongo median':>13} {'C/M':>7}")
    print("-" * 80)
    couch = report["results"]["couchdb"]
    mongo = report["results"]["mongodb"]
    for name in couch:
        if name not in mongo:
            continue
        ratio = report["comparison"][name]["couchdb_over_mongodb_ratio"]
        print(
            f"{name:42} {couch[name]['median_ms']:10.3f} ms "
            f"{mongo[name]['median_ms']:10.3f} ms {str(ratio):>7}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jämför CouchDB och MongoDB för FeedSummary.")
    parser.add_argument("--couch-url", default=os.environ.get("FEEDSUMMARY_COUCHDB_URL", "http://localhost:5984"))
    parser.add_argument("--couch-database", default=os.environ.get("FEEDSUMMARY_COUCHDB_DATABASE", "feedsummary"))
    parser.add_argument("--couch-username", default=os.environ.get("FEEDSUMMARY_COUCHDB_USER"))
    parser.add_argument("--couch-password", default=os.environ.get("FEEDSUMMARY_COUCHDB_PASSWORD"))
    parser.add_argument("--mongo-uri", default=os.environ.get("FEEDSUMMARY_MONGODB_URI", "mongodb://localhost:27017"))
    parser.add_argument("--mongo-database", default=os.environ.get("FEEDSUMMARY_MONGODB_DATABASE", "feedsummary"))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--write-count", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Spara full rapport som JSON")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = benchmark(
            couch_url=args.couch_url,
            couch_database=args.couch_database,
            couch_username=args.couch_username,
            couch_password=args.couch_password,
            mongo_uri=args.mongo_uri,
            mongo_database=args.mongo_database,
            sample_size=args.sample_size,
            write_count=args.write_count,
            warmups=args.warmups,
            iterations=args.iterations,
        )
    except Exception as exc:
        print(f"Benchmark misslyckades: {exc}", file=sys.stderr)
        return 1
    _print_table(report)
    if args.output:
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nFull rapport: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
