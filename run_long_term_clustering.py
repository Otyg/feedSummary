#!/usr/bin/env python3
"""Run or preview one resumable long-term threat-clustering batch."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from feedsummary_core.long_term import (
    ClusteringSettings,
    IncrementalSettings,
    run_incremental_clustering,
)
from feedsummary_core.persistence import create_store


def _config_path(raw: str | None) -> Path:
    value = raw or os.environ.get("FEEDSUMMARY_CONFIG") or "config.yaml"
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_config(path: Path) -> dict[str, Any]:
    config = _load_yaml(path) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Konfigurationen måste vara ett YAML-objekt: {path}")
    return config


def _resolved_store_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    raw = config.get("store")
    if not isinstance(raw, dict):
        raise TypeError("Konfigurationen saknar ett giltigt 'store'-objekt")
    result = dict(raw)
    if result.get("path"):
        path = Path(os.path.expandvars(os.path.expanduser(str(result["path"]))))
        if not path.is_absolute():
            path = config_path.parent / path
        result["path"] = str(path.resolve())
    return result


def _profile_config(config: dict[str, Any], profile_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    long_term = config.get("long_term_analysis")
    if not isinstance(long_term, dict):
        raise TypeError("Konfigurationen saknar ett giltigt 'long_term_analysis'-objekt")
    profiles = long_term.get("profiles")
    if not isinstance(profiles, dict):
        raise TypeError("long_term_analysis.profiles måste vara ett YAML-objekt")
    if profile_id not in profiles:
        raise ValueError(f"Okänd långtidsprofil: {profile_id}")
    raw_profile = profiles[profile_id]
    if not isinstance(raw_profile, dict):
        raise TypeError(f"Långtidsprofilen måste vara ett YAML-objekt: {profile_id}")
    profile = dict(raw_profile)
    if not profile.get("enabled", False):
        raise ValueError(f"Långtidsprofilen är avstängd: {profile_id}")
    return long_term, profile


def _resolve_sources(
    config: dict[str, Any], config_path: Path, profile: dict[str, Any]
) -> list[str]:
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
        if isinstance(feeds_config, list):
            # feeds already resolved into a list (e.g. by the worker's load_config)
            feeds = feeds_config
        else:
            raw_path = feeds_config.get("path") if isinstance(feeds_config, dict) else None
            if not raw_path:
                raise ValueError("Profilen har categories men feeds.path saknas")
            feeds_path = Path(os.path.expandvars(os.path.expanduser(str(raw_path))))
            if not feeds_path.is_absolute():
                feeds_path = config_path.parent / feeds_path
            feeds = _load_yaml(feeds_path) or []
        if not isinstance(feeds, list):
            raise TypeError("Feedkonfigurationen måste vara en YAML-lista")
        for feed in feeds:
            if not isinstance(feed, dict):
                continue
            topics = {str(topic).strip() for topic in feed.get("topics") or []}
            if categories.intersection(topics):
                name = str(feed.get("name") or "").strip()
                if name:
                    sources.add(name)
    if not sources and not profile.get("include_all_sources", False):
        raise ValueError(
            "Profilen löser inte till några källor; ange sources/categories eller "
            "include_all_sources"
        )
    return sorted(sources)


def _timestamp(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp())


def _settings(long_term: dict[str, Any], limit: int | None) -> IncrementalSettings:
    clustering = long_term.get("clustering") or {}
    if not isinstance(clustering, dict):
        raise TypeError("long_term_analysis.clustering måste vara ett YAML-objekt")
    return IncrementalSettings(
        batch_size=int(limit or long_term.get("batch_size") or 200),
        lease_seconds=int(long_term.get("lease_seconds") or 300),
        quarantine_retry_limit=int(long_term.get("quarantine_retry_limit", 50)),
        dormant_after_days=int(clustering.get("dormant_after_days", 30)),
        close_after_days=int(clustering.get("close_after_days", 180)),
        clustering=ClusteringSettings(
            similarity_threshold=float(clustering.get("similarity_threshold", 0.80)),
            ambiguity_margin=float(clustering.get("ambiguity_margin", 0.03)),
            candidate_window_days=int(clustering.get("candidate_window_days", 60)),
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML-konfiguration (standard: config.yaml)")
    parser.add_argument("--profile", required=True, help="Profil under long_term_analysis.profiles")
    parser.add_argument("--limit", type=int, help="Maximalt antal artiklar i batchen")
    parser.add_argument("--since", help="Starttid som Unix-tid eller ISO-8601")
    parser.add_argument("--until", help="Sluttid som Unix-tid eller ISO-8601")
    parser.add_argument(
        "--all-sources",
        action="store_true",
        help="Ignorera profilens källfilter för denna körning",
    )
    parser.add_argument("--dry-run", action="store_true", help="Beräkna utan skrivningar")
    return parser


def execute_long_term_clustering(
    args: argparse.Namespace,
    *,
    config_path: Path | None = None,
    config: dict[str, Any] | None = None,
    store: Any = None,
) -> dict[str, Any]:
    """Execute one clustering batch for CLI or worker callers."""

    resolved_config_path = config_path or _config_path(args.config)
    resolved_config = config or _load_config(resolved_config_path)
    owns_store = store is None
    active_store = store
    try:
        long_term, profile = _profile_config(resolved_config, args.profile)
        if not args.dry_run and not long_term.get("enabled", False):
            raise ValueError(
                "long_term_analysis.enabled är false; använd --dry-run eller aktivera funktionen"
            )
        sources = (
            None
            if getattr(args, "all_sources", False)
            else _resolve_sources(resolved_config, resolved_config_path, profile)
        )
        if active_store is None:
            store_config = _resolved_store_config(
                resolved_config, resolved_config_path
            )
            if args.dry_run:
                store_config["initialize_schema"] = False
            active_store = create_store(store_config)
        owner_id = str(
            getattr(args, "owner_id", "")
            or f"{socket.gethostname()}:{os.getpid()}"
        )
        result = run_incremental_clustering(
            active_store,
            profile_id=args.profile,
            owner_id=owner_id,
            sources=sources or None,
            required_tags=[str(tag) for tag in profile.get("required_tags") or []],
            required_tags_match=str(profile.get("required_tags_match") or "all"),
            since_fetched_at=_timestamp(args.since),
            until_fetched_at=_timestamp(args.until),
            settings=_settings(long_term, args.limit),
            dry_run=args.dry_run,
        )
        return asdict(result)
    finally:
        close = getattr(active_store, "close", None)
        if owns_store and callable(close):
            close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = execute_long_term_clustering(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("blocked_reason") is None else 3
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"Långtidsklustring misslyckades: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
