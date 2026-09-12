#!/usr/bin/env python3
"""Preview or generate one validated long-term threat-landscape report."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from feedsummary_core.llm_client import create_llm_client
from feedsummary_core.long_term import (
    LongTermLeaseHeartbeat,
    ReduceSettings,
    build_landscape_metrics,
    run_landscape_reduce,
    select_report_snapshots,
)
from feedsummary_core.persistence import create_store

from run_long_term_clustering import (
    _config_path,
    _load_config,
    _profile_config,
    _resolved_store_config,
    _timestamp,
)
from run_long_term_metrics import _metric_settings


def _prompt_package(config_path: Path, name: str) -> dict[str, Any]:
    path = config_path.parent / f"config/prompts/long_term/{name}.yaml"
    package = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(package, dict):
        raise TypeError(f"Reduce-prompten måste vara ett YAML-objekt: {path}")
    return package


def _reduce_llm_config(config: dict[str, Any], provider_name: str) -> dict[str, Any]:
    if provider_name == "configured_chain":
        return config
    if provider_name not in {"cloud", "local"}:
        raise ValueError(
            "long_term_analysis.llm.reduce_provider måste vara cloud, local "
            "eller configured_chain"
        )
    configured = config.get("llm")
    entries = configured if isinstance(configured, list) else [configured]
    provider = "ollama_cloud" if provider_name == "cloud" else "ollama_local"
    selected = [
        dict(item)
        for item in entries
        if isinstance(item, dict)
        and str(item.get("provider") or "").strip().lower() == provider
    ]
    fallback = config.get("llm_fallback")
    if (
        isinstance(fallback, dict)
        and str(fallback.get("provider") or "").strip().lower() == provider
    ):
        selected.append(dict(fallback))
    if not selected:
        raise ValueError(
            f"Reduce-provider {provider_name} kräver en {provider}-konfiguration"
        )
    result = dict(config)
    result["llm"] = selected[0] if len(selected) == 1 else selected
    result.pop("llm_fallback", None)
    return result


def _provider_entries(llm_config: dict[str, Any]) -> list[dict[str, Any]]:
    configured = llm_config.get("llm")
    entries = configured if isinstance(configured, list) else [configured]
    values = [dict(item) for item in entries if isinstance(item, dict)]
    fallback = llm_config.get("llm_fallback")
    if isinstance(fallback, dict):
        values.append(dict(fallback))
    if not values:
        raise ValueError("Reduce kräver minst en LLM-konfiguration")
    return values


def _reduce_settings(
    long_term: dict[str, Any], llm_config: dict[str, Any]
) -> ReduceSettings:
    reports = long_term.get("reports") or {}
    llm = long_term.get("llm") or {}
    if not isinstance(reports, dict) or not isinstance(llm, dict):
        raise TypeError("long_term_analysis.reports och llm måste vara YAML-objekt")
    providers = _provider_entries(llm_config)
    return ReduceSettings(
        max_context_tokens=min(
            int(provider.get("context_window_tokens", 8192)) for provider in providers
        ),
        max_output_tokens=int(llm.get("reduce_max_output_tokens", 2500)),
        safety_margin_tokens=max(
            int(provider.get("prompt_safety_margin", 512)) for provider in providers
        ),
        max_snapshots_per_segment=int(reports.get("max_snapshots_per_segment", 12)),
        max_snapshot_records=int(reports.get("max_snapshot_records", 10000)),
        format_repair_attempts=1,
        generation_attempts=int(reports.get("generation_attempts", 3)),
    )


def _model_label(llm_config: dict[str, Any]) -> str:
    def provider_label(item: dict[str, Any]) -> str:
        provider = str(item.get("provider") or "unknown")
        model = str(item.get("model") or "unknown")
        return f"{provider}:{model}"

    return " -> ".join(
        provider_label(item)
        for item in _provider_entries(llm_config)
    )


def _long_term_mode(long_term: dict[str, Any]) -> str:
    mode = str(long_term.get("mode") or "publish").strip().lower()
    if mode not in {"shadow", "publish"}:
        raise ValueError("long_term_analysis.mode måste vara shadow eller publish")
    return mode


def _report_cluster_ids(metrics: dict[str, Any]) -> tuple[set[str], set[str]]:
    event_ids = {
        str(cluster_id)
        for window in metrics.get("windows") or []
        for cluster_id in window.get("event_cluster_ids") or []
    }
    snapshot_eligible_ids = {
        str(cluster_id)
        for window in metrics.get("windows") or []
        for cluster_id in (
            window.get("snapshot_eligible_cluster_ids")
            if "snapshot_eligible_cluster_ids" in window
            else window.get("event_cluster_ids")
        )
        or []
    }
    return event_ids, snapshot_eligible_ids


async def execute_long_term_report(
    args: argparse.Namespace,
    *,
    config_path: Path | None = None,
    config: dict[str, Any] | None = None,
    store: Any = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute one report publication for CLI or worker callers."""

    resolved_config_path = config_path or _config_path(args.config)
    resolved_config = config or _load_config(resolved_config_path)
    long_term, profile = _profile_config(resolved_config, args.profile)
    if not args.dry_run and not long_term.get("enabled", False):
        raise ValueError(
            "long_term_analysis.enabled är false; använd --dry-run eller aktivera funktionen"
        )
    mode = _long_term_mode(long_term)
    raw_period_end = _timestamp(args.at)
    period_end_ts = raw_period_end if raw_period_end is not None else int(time.time())
    provider_name = str((long_term.get("llm") or {}).get("reduce_provider") or "configured_chain")
    llm_config = _reduce_llm_config(resolved_config, provider_name)
    reduce_settings = _reduce_settings(long_term, llm_config)
    owns_store = store is None
    active_store = store
    if active_store is None:
        store_config = _resolved_store_config(
            resolved_config, resolved_config_path
        )
        if args.dry_run:
            store_config["initialize_schema"] = False
        active_store = create_store(store_config)
    llm_client = None
    owner_id = str(
        getattr(args, "owner_id", "")
        or f"{socket.gethostname()}:{os.getpid()}:report"
    )
    lease_claimed = False
    heartbeat = None
    try:
        if not args.dry_run:
            lease_seconds = int(long_term.get("report_lease_seconds", 3600))
            lease_claimed = active_store.claim_long_term_lease(
                args.profile,
                owner_id,
                now_ts=int(time.time()),
                lease_seconds=lease_seconds,
            )
            if not lease_claimed:
                raise RuntimeError(f"profilens lease används redan: {args.profile}")
            heartbeat = LongTermLeaseHeartbeat(
                active_store,
                profile_id=args.profile,
                owner_id=owner_id,
                lease_seconds=lease_seconds,
            )
            await heartbeat.start()
        metrics = build_landscape_metrics(
            active_store,
            profile_id=args.profile,
            period_end_ts=period_end_ts,
            settings=_metric_settings(long_term),
        )
        snapshots = select_report_snapshots(
            active_store,
            profile_id=args.profile,
            metrics=metrics,
            settings=reduce_settings,
        )
        if args.dry_run:
            event_ids, snapshot_eligible_ids = _report_cluster_ids(metrics)
            snapshot_cluster_ids = {
                str(snapshot.get("cluster_id") or "") for snapshot in snapshots
            }
            return {
                "profile_id": args.profile,
                "dry_run": True,
                "period_end_ts": period_end_ts,
                "event_cluster_count": len(event_ids),
                "snapshot_eligible_cluster_count": len(snapshot_eligible_ids),
                "atomic_observation_cluster_count": len(
                    event_ids - snapshot_eligible_ids
                ),
                "snapshot_ids": [str(snapshot["id"]) for snapshot in snapshots],
                "missing_snapshot_cluster_ids": sorted(
                    snapshot_eligible_ids - snapshot_cluster_ids
                ),
                "quality_warning_codes": metrics["quality"]["warning_codes"],
            }
        llm_client = create_llm_client(llm_config)
        reports = long_term.get("reports") or {}
        result = await run_landscape_reduce(
            active_store,
            llm_client,
            profile_id=args.profile,
            profile_context={"id": args.profile, **profile},
            metrics=metrics,
            final_prompt_package=_prompt_package(
                resolved_config_path, "landscape_report_v1"
            ),
            segment_prompt_package=_prompt_package(
                resolved_config_path, "landscape_segment_v1"
            ),
            model=_model_label(llm_config),
            now_ts=int(time.time()),
            forecast_horizon_days=frozenset(
                int(value) for value in reports.get("forecast_horizon_days", (30, 90))
            ),
            settings=reduce_settings,
            mirror_to_summary_docs=bool(
                mode == "publish" and reports.get("mirror_to_summary_docs", True)
            ),
            lease_guard=heartbeat,
            progress_callback=progress_callback,
        )
        return {
            "profile_id": args.profile,
            "dry_run": False,
            "mode": mode,
            **asdict(result),
        }
    finally:
        if heartbeat is not None:
            await heartbeat.stop()
        if lease_claimed:
            active_store.release_long_term_lease(args.profile, owner_id)
        close_llm = getattr(llm_client, "aclose", None)
        if callable(close_llm):
            await close_llm()
        close_store = getattr(active_store, "close", None)
        if owns_store and callable(close_store):
            close_store()


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    def progress(event: dict[str, Any]) -> None:
        print(json.dumps({"stage": "reduce", **event}, ensure_ascii=False), flush=True)

    return await execute_long_term_report(args, progress_callback=progress)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML-konfiguration (standard: config.yaml)")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--at", help="Fryst periodslut som Unix-tid eller ISO-8601")
    parser.add_argument("--dry-run", action="store_true", help="Inspektera utan LLM eller skrivning")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        print(
            json.dumps(
                asyncio.run(_run(_parser().parse_args(argv))),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"Långtidsrapport misslyckades: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
