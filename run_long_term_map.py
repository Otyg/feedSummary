#!/usr/bin/env python3
"""Preview or execute evidence-bound Map updates for threat clusters."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from feedsummary_core.llm_client import create_llm_client
from feedsummary_core.long_term import (
    ClusterStatus,
    LongTermLeaseHeartbeat,
    MapSettings,
    ThreatCluster,
    cluster_needs_map_update,
    update_cluster_map_snapshot,
)
from feedsummary_core.persistence import create_store

from run_long_term_clustering import (
    _config_path,
    _load_config,
    _profile_config,
    _resolved_store_config,
)


def _prompt_package(config_path: Path) -> dict[str, Any]:
    path = config_path.parent / "config/prompts/long_term/cluster_update_v1.yaml"
    package = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(package, dict):
        raise TypeError(f"Map-prompten måste vara ett YAML-objekt: {path}")
    return package


def _map_llm_config(config: dict[str, Any], map_provider: str) -> dict[str, Any]:
    if map_provider == "configured_chain":
        return config
    if map_provider != "local":
        raise ValueError("long_term_analysis.llm.map_provider måste vara local eller configured_chain")
    configured = config.get("llm")
    entries = configured if isinstance(configured, list) else [configured]
    local = next(
        (
            item
            for item in entries
            if isinstance(item, dict)
            and str(item.get("provider") or "").strip().lower() == "ollama_local"
        ),
        None,
    )
    if local is None:
        raise ValueError("Map-provider local kräver en ollama_local-konfiguration")
    result = dict(config)
    result["llm"] = dict(local)
    result.pop("llm_fallback", None)
    return result


def _map_settings(long_term: dict[str, Any], llm_config: dict[str, Any]) -> MapSettings:
    updates = long_term.get("cluster_updates") or {}
    llm = long_term.get("llm") or {}
    configured = llm_config.get("llm")
    provider = configured[0] if isinstance(configured, list) else configured
    provider = provider if isinstance(provider, dict) else {}
    return MapSettings(
        min_pending_articles=int(updates.get("min_pending_articles", 3)),
        max_update_delay_hours=int(updates.get("max_update_delay_hours", 24)),
        max_articles_per_call=int(updates.get("max_articles_per_call", 12)),
        article_clip_chars=int(updates.get("article_clip_chars", 2500)),
        max_context_tokens=int(provider.get("context_window_tokens", 8192)),
        max_output_tokens=int(llm.get("map_max_output_tokens", 1200)),
        safety_margin_tokens=int(provider.get("prompt_safety_margin", 512)),
        format_repair_attempts=1,
    )


async def execute_long_term_map(
    args: argparse.Namespace,
    *,
    config_path: Path | None = None,
    config: dict[str, Any] | None = None,
    store: Any = None,
) -> dict[str, Any]:
    """Execute bounded Map updates for CLI or worker callers."""

    resolved_config_path = config_path or _config_path(args.config)
    resolved_config = config or _load_config(resolved_config_path)
    long_term, _ = _profile_config(resolved_config, args.profile)
    if not args.dry_run and not long_term.get("enabled", False):
        raise ValueError(
            "long_term_analysis.enabled är false; använd --dry-run eller aktivera funktionen"
        )
    provider_name = str((long_term.get("llm") or {}).get("map_provider") or "local")
    llm_config = _map_llm_config(resolved_config, provider_name)
    settings = _map_settings(long_term, llm_config)
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
        or f"{socket.gethostname()}:{os.getpid()}:map"
    )
    lease_claimed = False
    heartbeat = None
    try:
        now_ts = int(time.time())
        if args.cluster:
            document = active_store.get_threat_cluster(args.cluster)
            documents = [document] if document else []
        else:
            documents = active_store.list_threat_clusters(
                args.profile,
                statuses=[ClusterStatus.ACTIVE.value, ClusterStatus.DORMANT.value],
                limit=max(1, args.limit),
            )
        clusters = [ThreatCluster.from_document(doc) for doc in documents if doc]
        selected = [
            cluster
            for cluster in clusters
            if args.force
            or cluster_needs_map_update(cluster, now_ts=now_ts, settings=settings)
        ][: max(1, args.limit)]
        if args.dry_run:
            return {
                "profile_id": args.profile,
                "dry_run": True,
                "eligible_clusters": [
                    {
                        "cluster_id": cluster.id,
                        "membership_revision": cluster.membership_revision,
                        "summarized_revision": cluster.summarized_revision,
                        "pending_articles": (
                            cluster.membership_revision - cluster.summarized_revision
                        ),
                    }
                    for cluster in selected
                ],
            }
        if not selected:
            return {"profile_id": args.profile, "dry_run": False, "updates": []}
        lease_seconds = int(long_term.get("map_lease_seconds", 1800))
        lease_claimed = active_store.claim_long_term_lease(
            args.profile,
            owner_id,
            now_ts=now_ts,
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
        llm_client = create_llm_client(llm_config)
        package = _prompt_package(resolved_config_path)
        results = []
        for cluster in selected:
            result = await update_cluster_map_snapshot(
                active_store,
                llm_client,
                cluster_id=cluster.id,
                prompt_package=package,
                now_ts=int(time.time()),
                settings=settings,
                force=args.force,
                lease_guard=heartbeat,
            )
            results.append(asdict(result))
        return {"profile_id": args.profile, "dry_run": False, "updates": results}
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
    return await execute_long_term_map(args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML-konfiguration (standard: config.yaml)")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--cluster", help="Bearbeta endast detta kluster-ID")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--force", action="store_true", help="Ignorera update-tröskeln")
    parser.add_argument("--dry-run", action="store_true", help="Lista endast valbara kluster")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = asyncio.run(_run(_parser().parse_args(argv)))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"Map-uppdatering misslyckades: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
