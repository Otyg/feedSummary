#!/usr/bin/env python3
"""Build read-only deterministic rolling metrics for a threat-landscape profile."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from typing import Any

from feedsummary_core.long_term import (
    LandscapeMetricSettings,
    build_landscape_metrics,
)
from feedsummary_core.persistence import create_store

from run_long_term_clustering import (
    _config_path,
    _load_config,
    _profile_config,
    _resolved_store_config,
    _timestamp,
)


def _metric_settings(long_term: dict[str, Any]) -> LandscapeMetricSettings:
    reports = long_term.get("reports") or {}
    if not isinstance(reports, dict):
        raise TypeError("long_term_analysis.reports måste vara ett YAML-objekt")
    return LandscapeMetricSettings(
        windows_days=tuple(int(value) for value in reports.get("windows_days", (7, 30, 90))),
        bucket_days=int(reports.get("bucket_days", 7)),
        min_independent_events_for_trend=int(
            reports.get("min_independent_events_for_trend", 3)
        ),
        source_concentration_threshold=float(
            reports.get("source_concentration_threshold", 0.75)
        ),
        source_mix_change_threshold=float(
            reports.get("source_mix_change_threshold", 0.35)
        ),
        max_clusters=int(reports.get("max_clusters", 10000)),
        max_memberships_per_cluster=int(
            reports.get("max_memberships_per_cluster", 10000)
        ),
        min_snapshot_member_count=int(
            reports.get("min_snapshot_member_count", 1)
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML-konfiguration (standard: config.yaml)")
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--at",
        help="Fryst periodslut som Unix-tid eller ISO-8601 (standard: nu)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    store = None
    try:
        args = _parser().parse_args(argv)
        config_path = _config_path(args.config)
        config = _load_config(config_path)
        long_term, _ = _profile_config(config, args.profile)
        store_config = _resolved_store_config(config, config_path)
        store_config["initialize_schema"] = False
        store = create_store(store_config)
        raw_period_end = _timestamp(args.at)
        metrics = build_landscape_metrics(
            store,
            profile_id=args.profile,
            period_end_ts=(
                raw_period_end if raw_period_end is not None else int(time.time())
            ),
            settings=_metric_settings(long_term),
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"Långtidsmått misslyckades: {error}", file=sys.stderr)
        return 2
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    raise SystemExit(main())
