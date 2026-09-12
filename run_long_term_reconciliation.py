#!/usr/bin/env python3
"""Propose reconciliation of existing long-term clusters without writing state."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from feedsummary_core.long_term import (
    ClusterStatus,
    ReconciliationSettings,
    build_cluster_merge_operation,
    propose_cluster_reconciliation,
    validate_cluster_merge_operation,
)
from feedsummary_core.persistence import create_store

from run_long_term_clustering import (
    _config_path,
    _load_config,
    _profile_config,
    _resolved_store_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--auto-threshold", type=float)
    parser.add_argument("--review-threshold", type=float)
    parser.add_argument("--window-days", type=int)
    parser.add_argument("--max-review-pairs", type=int)
    parser.add_argument("--max-clusters", type=int)
    parser.add_argument("--output", type=Path, help="Skriv fullständiga förslag som JSON")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--prepare-group",
        metavar="PRIMARY_CLUSTER_ID",
        help="Exportera en exakt operation för en föreslagen auto-grupp",
    )
    mode.add_argument(
        "--apply-operation",
        type=Path,
        help="Applicera en tidigare exporterad och granskad operation",
    )
    parser.add_argument(
        "--operation-output",
        type=Path,
        help="Ny JSON-fil för --prepare-group; befintlig fil skrivs inte över",
    )
    parser.add_argument(
        "--confirm-apply",
        help="Måste exakt motsvara operationens ID vid --apply-operation",
    )
    return parser


def _settings(long_term: dict[str, Any], args: argparse.Namespace) -> ReconciliationSettings:
    raw = long_term.get("reconciliation") or {}
    if not isinstance(raw, dict):
        raise TypeError("long_term_analysis.reconciliation måste vara ett YAML-objekt")
    clustering = long_term.get("clustering") or {}
    return ReconciliationSettings(
        auto_merge_similarity=float(
            args.auto_threshold
            if args.auto_threshold is not None
            else raw.get("auto_merge_similarity", 0.88)
        ),
        review_similarity=float(
            args.review_threshold
            if args.review_threshold is not None
            else raw.get("review_similarity", clustering.get("similarity_threshold", 0.82))
        ),
        candidate_window_days=int(
            args.window_days
            if args.window_days is not None
            else raw.get("candidate_window_days", clustering.get("candidate_window_days", 60))
        ),
        max_review_pairs=int(
            args.max_review_pairs
            if args.max_review_pairs is not None
            else raw.get("max_review_pairs", 500)
        ),
    )


def execute_reconciliation(
    args: argparse.Namespace,
    *,
    config_path: Path | None = None,
    config: dict[str, Any] | None = None,
    store: Any = None,
) -> dict[str, Any]:
    """Load current state and return proposals; intentionally has no apply mode."""

    resolved_path = config_path or _config_path(args.config)
    resolved_config = config or _load_config(resolved_path)
    long_term, _ = _profile_config(resolved_config, args.profile)
    reports = long_term.get("reports") or {}
    max_clusters = int(
        args.max_clusters
        if args.max_clusters is not None
        else reports.get("max_clusters", 10000)
    )
    owns_store = store is None
    active_store = store
    try:
        if active_store is None:
            store_config = _resolved_store_config(resolved_config, resolved_path)
            store_config["initialize_schema"] = False
            active_store = create_store(store_config)
        clusters = active_store.list_threat_clusters(
            args.profile,
            statuses=[status.value for status in ClusterStatus],
            limit=max_clusters,
        )
        memberships = {
            str(cluster["id"]): active_store.list_cluster_memberships(str(cluster["id"]))
            for cluster in clusters
        }
        result = propose_cluster_reconciliation(
            clusters,
            memberships,
            settings=_settings(long_term, args),
        )
        return {
            "mode": "proposal_only",
            "profile_id": args.profile,
            **result.to_document(),
        }
    finally:
        close = getattr(active_store, "close", None)
        if owns_store and callable(close):
            close()


def _summary(document: dict[str, Any]) -> dict[str, Any]:
    groups = document.get("groups") or []
    return {
        "mode": document["mode"],
        "profile_id": document["profile_id"],
        "cluster_count": document["cluster_count"],
        "compared_pair_count": document["compared_pair_count"],
        "auto_merge_edge_count": len(document.get("auto_merge_edges") or []),
        "review_edge_count": len(document.get("review_edges") or []),
        "blocked_strict_cve_pair_count": document["blocked_strict_cve_pair_count"],
        "merge_group_count": len(groups),
        "clusters_in_merge_groups": sum(len(group["cluster_ids"]) for group in groups),
        "largest_merge_group": max((len(group["cluster_ids"]) for group in groups), default=0),
    }


def _stable_operation_id(
    profile_id: str,
    primary_cluster_id: str,
    cluster_documents: Sequence[dict[str, Any]],
) -> str:
    identity = {
        "profile_id": profile_id,
        "primary_cluster_id": primary_cluster_id,
        "sources": sorted(
            (
                str(row["id"]),
                int(row["membership_revision"]),
            )
            for row in cluster_documents
        ),
        "policy": "cluster-reconciliation-v1",
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"threat_reconciliation_{digest[:32]}"


def prepare_reconciliation_operation(
    args: argparse.Namespace,
    *,
    reconciled_at: int | None = None,
    config_path: Path | None = None,
    config: dict[str, Any] | None = None,
    store: Any = None,
) -> dict[str, Any]:
    """Freeze one current auto-group into a reviewable, optimistic operation."""

    primary_id = str(args.prepare_group or "").strip()
    if not primary_id:
        raise ValueError("--prepare-group kräver ett primärt kluster-ID")
    resolved_path = config_path or _config_path(args.config)
    resolved_config = config or _load_config(resolved_path)
    owns_store = store is None
    active_store = store
    try:
        if active_store is None:
            store_config = _resolved_store_config(resolved_config, resolved_path)
            store_config["initialize_schema"] = False
            active_store = create_store(store_config)
        proposals = execute_reconciliation(
            args,
            config_path=resolved_path,
            config=resolved_config,
            store=active_store,
        )
        group = next(
            (
                row
                for row in proposals.get("groups") or []
                if str(row.get("primary_cluster_id") or "") == primary_id
            ),
            None,
        )
        if group is None:
            raise ValueError("det primära klustret finns inte i en aktuell auto-grupp")
        cluster_ids = [str(value) for value in group["cluster_ids"]]
        clusters = [active_store.get_threat_cluster(value) for value in cluster_ids]
        if any(row is None for row in clusters):
            raise ValueError("ett kluster försvann medan operationen förbereddes")
        cluster_documents = [dict(row) for row in clusters if row is not None]
        memberships = {
            cluster_id: active_store.list_cluster_memberships(cluster_id)
            for cluster_id in cluster_ids
        }
        operation_id = _stable_operation_id(
            args.profile, primary_id, cluster_documents
        )
        return build_cluster_merge_operation(
            cluster_documents,
            memberships,
            primary_cluster_id=primary_id,
            reconciliation_id=operation_id,
            reconciled_at=int(reconciled_at or time.time()),
        )
    finally:
        close = getattr(active_store, "close", None)
        if owns_store and callable(close):
            close()


def apply_prepared_reconciliation(
    args: argparse.Namespace,
    *,
    operation: dict[str, Any] | None = None,
    config_path: Path | None = None,
    config: dict[str, Any] | None = None,
    store: Any = None,
) -> dict[str, Any]:
    """Apply an explicitly confirmed operation with backend revision checks."""

    resolved_path = config_path or _config_path(args.config)
    resolved_config = config or _load_config(resolved_path)
    long_term, _ = _profile_config(resolved_config, args.profile)
    if not long_term.get("enabled", False):
        raise ValueError("long_term_analysis.enabled måste vara true för apply")
    if operation is None:
        if args.apply_operation is None:
            raise ValueError("--apply-operation saknas")
        operation = json.loads(
            args.apply_operation.expanduser().resolve().read_text(encoding="utf-8")
        )
    document = validate_cluster_merge_operation(operation)
    if document["profile_id"] != args.profile:
        raise ValueError("operationens profil matchar inte --profile")
    if str(args.confirm_apply or "") != document["id"]:
        raise ValueError("--confirm-apply måste exakt motsvara operationens ID")
    owns_store = store is None
    active_store = store
    try:
        if active_store is None:
            active_store = create_store(
                _resolved_store_config(resolved_config, resolved_path)
            )
        if not active_store.apply_cluster_reconciliation(document):
            raise ValueError("operationen är stale eller kolliderar med aktuell revision")
        stored = active_store.get_cluster_reconciliation(document["id"])
        return {
            "mode": "applied",
            "profile_id": args.profile,
            "reconciliation_id": document["id"],
            "primary_cluster_id": document["primary_cluster_id"],
            "source_cluster_ids": document["source_cluster_ids"],
            "membership_count": int((stored or {}).get("membership_count") or 0),
            "snapshot_policy": document["snapshot_policy"],
        }
    finally:
        close = getattr(active_store, "close", None)
        if owns_store and callable(close):
            close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.apply_operation:
            document = apply_prepared_reconciliation(args)
            print(json.dumps(document, ensure_ascii=False, sort_keys=True))
        elif args.prepare_group:
            if args.operation_output is None:
                raise ValueError("--prepare-group kräver --operation-output")
            document = prepare_reconciliation_operation(args)
            output = args.operation_output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            print(
                json.dumps(
                    {
                        "mode": "operation_prepared",
                        "reconciliation_id": document["id"],
                        "primary_cluster_id": document["primary_cluster_id"],
                        "cluster_count": len(document["source_cluster_ids"]),
                        "operation_output": str(output),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        else:
            document = execute_reconciliation(args)
            if not args.output:
                print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))
                return 0
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({**_summary(document), "output": str(output)}, ensure_ascii=False))
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"Fel: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
