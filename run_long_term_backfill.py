#!/usr/bin/env python3
"""Create a complete, resumable long-term baseline before publishing a report."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from collections.abc import Sequence

from run_long_term_clustering import execute_long_term_clustering
from run_long_term_map import execute_long_term_map
from run_long_term_report import execute_long_term_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--article-limit", type=int, default=5000)
    parser.add_argument("--map-limit", type=int, default=10000)
    parser.add_argument("--max-map-passes", type=int, default=4)
    parser.add_argument("--all-sources", action="store_true")
    parser.add_argument("--skip-clustering", action="store_true")
    return parser


def _progress(stage: str, **fields: object) -> None:
    print(json.dumps({"stage": stage, **fields}, ensure_ascii=False), flush=True)


async def execute_backfill(args: argparse.Namespace) -> dict[str, object]:
    if args.article_limit < 1 or args.map_limit < 1 or args.max_map_passes < 1:
        raise ValueError("limits and max-map-passes must be positive")

    if not args.skip_clustering:
        frozen_until = int(time.time())
        while True:
            clustering = execute_long_term_clustering(
                argparse.Namespace(
                    config=args.config,
                    profile=args.profile,
                    limit=args.article_limit,
                    since=None,
                    until=str(frozen_until),
                    all_sources=args.all_sources,
                    dry_run=False,
                )
            )
            _progress(
                "clustering",
                fetched=clustering["fetched"],
                assigned=clustering["assigned"],
                existing=clustering["existing"],
                quarantined=clustering["quarantined"],
                blocked_reason=clustering["blocked_reason"],
            )
            if clustering["blocked_reason"]:
                raise RuntimeError(str(clustering["blocked_reason"]))
            if int(clustering["fetched"]) < args.article_limit:
                break

    total_updates = 0
    total_failures = 0
    for pass_number in range(1, args.max_map_passes + 1):
        def map_progress(
            event: dict[str, object], *, _pass_number: int = pass_number
        ) -> None:
            _progress("map_cluster", pass_number=_pass_number, **event)

        mapped = await execute_long_term_map(
            argparse.Namespace(
                config=args.config,
                profile=args.profile,
                cluster=None,
                limit=args.map_limit,
                force=True,
                continue_on_error=True,
                dry_run=False,
            ),
            progress_callback=map_progress,
        )
        updates = list(mapped.get("updates") or [])
        failures = list(mapped.get("failures") or [])
        total_updates += len(updates)
        total_failures += len(failures)
        failure_types = Counter(
            str(row.get("error_type") or "unknown") for row in failures
        )
        failed_cluster_ids = [row.get("cluster_id") for row in failures]
        _progress(
            "map",
            pass_number=pass_number,
            updates=len(updates),
            failures=len(failures),
            failure_type_counts=dict(sorted(failure_types.items())),
            failed_cluster_ids=failed_cluster_ids[:20],
            omitted_failed_cluster_ids=max(0, len(failed_cluster_ids) - 20),
        )

        remaining = await execute_long_term_map(
            argparse.Namespace(
                config=args.config,
                profile=args.profile,
                cluster=None,
                limit=1,
                force=True,
                continue_on_error=False,
                dry_run=True,
            )
        )
        if not remaining.get("eligible_clusters"):
            break
    else:
        raise RuntimeError(
            f"Map-backfill är inte komplett efter {args.max_map_passes} pass"
        )

    def report_progress(event: dict[str, object]) -> None:
        _progress("reduce", **event)

    report = await execute_long_term_report(
        argparse.Namespace(
            config=args.config,
            profile=args.profile,
            at=None,
            dry_run=False,
        ),
        progress_callback=report_progress,
    )
    result = {
        "profile_id": args.profile,
        "map_updates": total_updates,
        "map_failures_observed": total_failures,
        "report_id": report.get("report_id"),
    }
    _progress("done", **result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        asyncio.run(execute_backfill(args))
        return 0
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"Långtidsbackfill misslyckades: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
