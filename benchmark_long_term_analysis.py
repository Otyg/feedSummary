#!/usr/bin/env python3
"""Run the deterministic 90/180-day long-term analysis capacity benchmark."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from feedsummary_core.long_term import (
    PerformanceBenchmarkSettings,
    run_long_term_performance_benchmark,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", nargs="+", type=int, default=[90, 180])
    parser.add_argument("--events-per-day", type=int, default=4)
    parser.add_argument("--embedding-dimensions", type=int, default=64)
    parser.add_argument("--candidate-probes", type=int, default=100)
    parser.add_argument("--max-total-ms", type=float, default=15_000)
    parser.add_argument("--max-peak-memory-mb", type=float, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    results = [
        run_long_term_performance_benchmark(
            PerformanceBenchmarkSettings(
                days=days,
                events_per_day=args.events_per_day,
                embedding_dimensions=args.embedding_dimensions,
                candidate_probes=args.candidate_probes,
            )
        )
        for days in args.days
    ]
    violations = [
        {
            "days": result["dataset"]["days"],
            "total_ms": result["timings_ms"]["total"],
            "peak_memory_mb": result["peak_memory_mb"],
        }
        for result in results
        if result["timings_ms"]["total"] > args.max_total_ms
        or result["peak_memory_mb"] > args.max_peak_memory_mb
    ]
    print(
        json.dumps(
            {
                "schema_version": 1,
                "limits": {
                    "max_total_ms": args.max_total_ms,
                    "max_peak_memory_mb": args.max_peak_memory_mb,
                },
                "passed": not violations,
                "violations": violations,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not violations else 2


if __name__ == "__main__":
    raise SystemExit(main())
