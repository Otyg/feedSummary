"""Archive old long-term clusters and rebuild tagged clusters in new profiles."""

from __future__ import annotations

import argparse
import copy
import json
import os
import socket
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Self

import yaml
from run_long_term_clustering import (
    _config_path,
    _load_config,
    _profile_config,
    _resolve_sources,
    _resolved_store_config,
    _settings,
)

from feedsummary_core.long_term import run_incremental_clustering
from feedsummary_core.persistence import create_store


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Källkonfiguration (standard: config.yaml)")
    parser.add_argument(
        "--profile",
        action="append",
        dest="profiles",
        help="Profil att bygga om; upprepa för flera. Standard: alla aktiva basprofiler.",
    )
    parser.add_argument("--target-suffix", default="_tagged_v2")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--until", type=int, help="Fryst övre gräns för fetched_at")
    parser.add_argument(
        "--output-config",
        help="Ny konfigurationsfil med nya profiler; krävs första gången med --apply",
    )
    parser.add_argument("--all-sources", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Stäng gamla och skapa nya kluster")
    return parser


def _selected_profiles(config: dict[str, Any], requested: Sequence[str] | None) -> list[str]:
    long_term = config.get("long_term_analysis") or {}
    profiles = long_term.get("profiles") or {}
    if not isinstance(profiles, dict):
        raise TypeError("long_term_analysis.profiles måste vara ett YAML-objekt")
    selected = list(
        dict.fromkeys(
            requested
            or [
                name
                for name, profile in profiles.items()
                if isinstance(profile, dict)
                and profile.get("enabled", False)
                and not profile.get("rebuild_source_profile_id")
            ]
        )
    )
    if not selected:
        raise ValueError("Inga aktiva basprofiler finns att bygga om")
    for profile_id in selected:
        _, profile = _profile_config(config, profile_id)
        if profile.get("rebuild_source_profile_id"):
            raise ValueError(f"En återbyggnadsprofil kan inte vara källa: {profile_id}")
    return selected


def _prepared_config(
    config: dict[str, Any], selected: Sequence[str], suffix: str, until_ts: int
) -> tuple[dict[str, Any], dict[str, str]]:
    if not suffix or not suffix.strip() or until_ts < 1:
        raise ValueError("target-suffix och until måste vara giltiga")
    prepared = copy.deepcopy(config)
    long_term = prepared["long_term_analysis"]
    profiles = long_term["profiles"]
    targets = {source: f"{source}{suffix}" for source in selected}
    if len(set(targets.values())) != len(targets):
        raise ValueError("Målprofilerna måste vara unika")
    for source, target in targets.items():
        if source == target:
            raise ValueError("Käll- och målprofil får inte vara samma")
        existing = profiles.get(target)
        if existing is not None:
            if (
                not isinstance(existing, dict)
                or existing.get("rebuild_source_profile_id") != source
            ):
                raise ValueError(f"Målprofilen finns redan utan rätt återbyggnadsmarkör: {target}")
            if int(existing.get("rebuild_until_fetched_at") or 0) != until_ts:
                raise ValueError(f"Målprofilen har en annan fryst sluttid: {target}")
        else:
            target_profile = copy.deepcopy(profiles[source])
            target_profile.update(
                enabled=True,
                clustering_enabled=True,
                rebuild_source_profile_id=source,
                rebuild_until_fetched_at=until_ts,
            )
            profiles[target] = target_profile
        profiles[source]["clustering_enabled"] = False
    categories = _settings(long_term, 1).tag_scope_categories
    if not categories:
        raise ValueError("Taggkategorier måste vara aktiverade för återbyggnaden")
    clustering = long_term.get("clustering") or {}
    clustering["tag_scope_categories"] = list(categories)
    long_term["clustering"] = clustering
    return prepared, targets


def _all_clusters(
    store: Any, profile_id: str, page_size: int, *, include_vectors: bool = False
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = store.list_threat_clusters(
            profile_id, limit=page_size, offset=offset, include_vectors=include_vectors
        )
        rows.extend(page)
        if len(page) < page_size:
            return rows
        offset += len(page)


def _count_articles_for_rebuild(
    store: Any,
    *,
    sources: list[str] | None,
    until_fetched_at: int,
    page_size: int,
) -> int:
    count = 0
    after_fetched_at = 0
    after_article_id = ""
    while True:
        page = store.list_articles_for_long_term(
            after_fetched_at=after_fetched_at,
            after_article_id=after_article_id,
            until_fetched_at=until_fetched_at,
            sources=sources,
            limit=page_size,
        ) or []
        count += len(page)
        if len(page) < page_size:
            return count
        last = page[-1]
        after_fetched_at = int(last.get("fetched_at") or 0)
        after_article_id = str(last.get("id") or "")


def _write_config(path: Path, config: dict[str, Any], source_path: Path) -> None:
    if path.resolve() == source_path.resolve():
        raise ValueError("output-config får inte skriva över källkonfigurationen")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(content)


def _replace_config(path: Path, config: dict[str, Any]) -> None:
    """Replace the active config atomically after the rebuild plan is valid."""

    content = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


class _SourceLeaseHeartbeat:
    """Keep the archived profile locked while synchronous clustering batches run."""

    def __init__(self, store: Any, profile_id: str, owner_id: str, lease_seconds: int):
        self.store = store
        self.profile_id = profile_id
        self.owner_id = owner_id
        self.lease_seconds = lease_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._failure: RuntimeError | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def _renew(self) -> None:
        with self._lock:
            if self._failure is not None:
                return
            try:
                renewed = self.store.renew_long_term_lease(
                    self.profile_id,
                    self.owner_id,
                    now_ts=int(time.time()),
                    lease_seconds=self.lease_seconds,
                )
            except Exception as error:  # noqa: BLE001 - surface background failures
                self._failure = RuntimeError(
                    f"Låsförnyelse misslyckades för {self.profile_id}: {error}"
                )
                return
            if not renewed:
                self._failure = RuntimeError(f"Låset förlorades för {self.profile_id}")

    def _run(self) -> None:
        while not self._stop.wait(max(0.1, self.lease_seconds / 3)):
            self._renew()
            if self._failure is not None:
                return

    def ensure_owned(self) -> None:
        self._renew()
        if self._failure is not None:
            raise self._failure


def execute_rebuild(
    args: argparse.Namespace,
    *,
    config_path: Path | None = None,
    config: dict[str, Any] | None = None,
    store: Any = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Preview or run a resumable rebuild without changing historical memberships."""

    if args.batch_size < 1 or args.page_size < 1:
        raise ValueError("batch-size och page-size måste vara positiva")
    resolved_path = config_path or _config_path(args.config)
    source_config = config if config is not None else _load_config(resolved_path)
    selected = _selected_profiles(source_config, args.profiles)
    existing_profiles = source_config["long_term_analysis"]["profiles"]
    requested_until = int(args.until if args.until is not None else time.time())
    if args.until is None:
        stored_until = {
            int(existing_profiles[f"{profile_id}{args.target_suffix}"]["rebuild_until_fetched_at"])
            for profile_id in selected
            if f"{profile_id}{args.target_suffix}" in existing_profiles
        }
        if len(stored_until) > 1:
            raise ValueError("Målprofilerna har olika frysta sluttider; ange --until")
        if stored_until:
            requested_until = stored_until.pop()
    prepared, targets = _prepared_config(
        source_config, selected, args.target_suffix, requested_until
    )
    if args.apply and not source_config["long_term_analysis"].get("enabled", False):
        raise ValueError("long_term_analysis.enabled måste vara true för --apply")
    needs_output = any(target not in existing_profiles for target in targets.values())
    if args.output_config and Path(args.output_config).resolve() == resolved_path.resolve():
        raise ValueError("output-config får inte vara samma fil som --config")
    if (
        args.output_config
        and Path(args.output_config).resolve().parent != resolved_path.resolve().parent
    ):
        raise ValueError("output-config måste ligga bredvid --config så relativa sökvägar behålls")

    owns_store = store is None
    active_store = store
    try:
        if active_store is None:
            store_config = _resolved_store_config(source_config, resolved_path)
            if not args.apply:
                store_config["initialize_schema"] = False
            active_store = create_store(store_config)
        plans = []
        for source in selected:
            target = targets[source]
            sources = (
                None
                if args.all_sources
                else _resolve_sources(source_config, resolved_path, existing_profiles[source])
            )
            old_clusters = _all_clusters(active_store, source, args.page_size)
            target_clusters = _all_clusters(active_store, target, args.page_size)
            target_cursor = active_store.get_long_term_cursor(target)
            if needs_output and (target_clusters or target_cursor.get("cursor_fetched_at")):
                raise ValueError(
                    f"Målprofilen {target} har redan data; återuppta med den genererade konfigurationen"
                )
            plans.append(
                {
                    "source_profile": source,
                    "target_profile": target,
                    "source_count": len(old_clusters),
                    "source_statuses": dict(
                        Counter(str(row.get("status") or "") for row in old_clusters)
                    ),
                    "target_count_before": len(target_clusters),
                    "target_cursor_before": [
                        int(target_cursor.get("cursor_fetched_at") or 0),
                        str(target_cursor.get("cursor_article_id") or ""),
                    ],
                    "articles_available": _count_articles_for_rebuild(
                        active_store,
                        sources=sources,
                        until_fetched_at=requested_until,
                        page_size=args.page_size,
                    ),
                    "sources": sources,
                }
            )
        result: dict[str, Any] = {
            "dry_run": not args.apply,
            "until_fetched_at": requested_until,
            "output_config": str(Path(args.output_config).resolve())
            if args.output_config
            else str(resolved_path.resolve()) if args.apply else None,
            "profiles": plans,
        }
        if not args.apply:
            return result
        settings = _settings(prepared["long_term_analysis"], args.batch_size)
        owner_id = f"rebuild:{socket.gethostname()}:{os.getpid()}"
        config_written = False
        for plan in plans:
            source = plan["source_profile"]
            target = plan["target_profile"]
            now_ts = int(time.time())
            if not active_store.claim_long_term_lease(
                source, owner_id, now_ts=now_ts, lease_seconds=settings.lease_seconds
            ):
                state = active_store.get_long_term_cursor(source)
                held_until = int(state.get("lease_until") or 0)
                held_until_utc = (
                    datetime.fromtimestamp(held_until, timezone.utc).isoformat()
                    if held_until
                    else "okänt"
                )
                raise RuntimeError(
                    f"Källprofilen används av ett annat långtidsjobb: {source}; "
                    f"ägare={state.get('lease_owner') or 'okänd'}, "
                    f"låset gäller till {held_until_utc}. "
                    "Vänta tills jobbet är klart och kör sedan om med den "
                    "genererade konfigurationsfilen om den redan finns."
                )
            try:
                if not args.output_config and not config_written:
                    _replace_config(resolved_path, prepared)
                    config_written = True
                if args.output_config and not config_written:
                    _write_config(Path(args.output_config), prepared, resolved_path)
                    config_written = True
                with _SourceLeaseHeartbeat(
                    active_store, source, owner_id, settings.lease_seconds
                ) as heartbeat:
                    closed = 0
                    for row in _all_clusters(
                        active_store, source, args.page_size, include_vectors=True
                    ):
                        heartbeat.ensure_owned()
                        if row.get("status") == "closed":
                            continue
                        updated = {
                            **row,
                            "status": "closed",
                            "rebuild_target_profile_id": target,
                            "rebuild_closed_at": now_ts,
                            "updated_at": now_ts,
                        }
                        saved = active_store.save_threat_cluster(
                            updated,
                            expected_membership_revision=int(row.get("membership_revision") or 0),
                        )
                        if not saved:
                            raise RuntimeError(f"Klustret ändrades samtidigt: {row['id']}")
                        closed += 1
                    totals: Counter[str] = Counter()
                    batches = 0
                    while True:
                        heartbeat.ensure_owned()
                        batch = run_incremental_clustering(
                            active_store,
                            profile_id=target,
                            owner_id=owner_id,
                            sources=plan["sources"] or None,
                            required_tags=[
                                str(tag)
                                for tag in existing_profiles[source].get("required_tags") or []
                            ],
                            required_tags_match=str(
                                existing_profiles[source].get("required_tags_match") or "all"
                            ),
                            until_fetched_at=requested_until,
                            settings=settings,
                        )
                        batches += 1
                        totals.update(
                            {
                                field: getattr(batch, field)
                                for field in (
                                    "assigned",
                                    "created",
                                    "matched",
                                    "existing",
                                    "filtered",
                                    "quarantined",
                                )
                            }
                        )
                        if batch.blocked_reason:
                            raise RuntimeError(
                                f"Återbyggnaden blockerades vid {batch.blocked_article_id}: {batch.blocked_reason}"
                            )
                        remaining = active_store.list_articles_for_long_term(
                            after_fetched_at=batch.cursor_fetched_at,
                            after_article_id=batch.cursor_article_id,
                            until_fetched_at=requested_until,
                            sources=plan["sources"] or None,
                            limit=1,
                        )
                        if progress:
                            progress(
                                {
                                    "source_profile": source,
                                    "target_profile": target,
                                    "batch": batches,
                                    "cursor": [batch.cursor_fetched_at, batch.cursor_article_id],
                                    "assigned": batch.assigned,
                                    "remaining": bool(remaining),
                                }
                            )
                        if not remaining:
                            break
                        if batch.fetched == 0:
                            raise RuntimeError("Återbyggnadens cursor gjorde inga framsteg")
                    plan.update(
                        closed=closed,
                        batches=batches,
                        rebuilt_assignments=totals["assigned"],
                        created=totals["created"],
                        matched=totals["matched"],
                        filtered=totals["filtered"],
                        quarantined=totals["quarantined"],
                    )
            finally:
                active_store.release_long_term_lease(source, owner_id)
        return result
    finally:
        close = getattr(active_store, "close", None)
        if owns_store and callable(close):
            close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = execute_rebuild(
            args, progress=lambda item: print(json.dumps(item, ensure_ascii=False), flush=True)
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"Återbyggnad misslyckades: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
