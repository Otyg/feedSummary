"""Read-only aggregate projection over child long-term profiles.

Aggregate profiles reuse child clusters and Map snapshots.  The projection
rewrites only the profile identity presented to Reduce; persisted evidence
keeps its original snapshot and cluster IDs so UI provenance links remain
valid.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Iterable


AGGREGATE_UPDATE_WARNING = "aggregate_profile_update_failed"


def aggregate_profile_ids(
    long_term: dict[str, Any], profile_id: str
) -> tuple[str, ...]:
    """Return and validate the direct children of an aggregate profile."""

    profiles = long_term.get("profiles")
    if not isinstance(profiles, dict):
        raise TypeError("long_term_analysis.profiles måste vara ett YAML-objekt")
    raw_profile = profiles.get(profile_id)
    if not isinstance(raw_profile, dict):
        raise ValueError(f"Okänd långtidsprofil: {profile_id}")
    raw_children = raw_profile.get("aggregate_profiles")
    if raw_children is None:
        return ()
    if not isinstance(raw_children, list) or not raw_children:
        raise ValueError("aggregate_profiles måste vara en lista med profiler")

    children: list[str] = []
    for value in raw_children:
        child_id = str(value or "").strip()
        if not child_id or child_id == profile_id:
            raise ValueError("aggregate_profiles innehåller ett ogiltigt profil-ID")
        child = profiles.get(child_id)
        if not isinstance(child, dict):
            raise ValueError(f"Okänd samlingsprofilmedlem: {child_id}")
        if child.get("enabled") is False:
            raise ValueError(f"Samlingsprofilmedlemmen är avstängd: {child_id}")
        if child.get("aggregate_profiles") is not None:
            raise ValueError("Nästlade samlingsprofiler stöds inte")
        if child_id not in children:
            children.append(child_id)
    return tuple(children)


def normalize_profile_failures(
    failures: Iterable[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Keep a bounded, persistence-safe failure contract for reports and APIs."""

    normalized = []
    seen: set[tuple[str, str, str, str]] = set()
    for raw in failures or ():
        if not isinstance(raw, dict):
            continue
        item = {
            "profile_id": str(raw.get("profile_id") or "").strip(),
            "stage": str(raw.get("stage") or "").strip(),
            "error_type": str(raw.get("error_type") or "Error").strip(),
            "error": " ".join(str(raw.get("error") or "").split())[:1000],
            "observed_at": int(raw.get("observed_at") or 0),
            "last_successful_update_at": int(
                raw.get("last_successful_update_at") or 0
            ),
        }
        if not item["profile_id"] or not item["stage"]:
            continue
        identity = (
            item["profile_id"],
            item["stage"],
            item["error_type"],
            item["error"],
        )
        if identity in seen:
            continue
        seen.add(identity)
        normalized.append(item)
    return sorted(
        normalized,
        key=lambda item: (item["profile_id"], item["stage"], item["error_type"]),
    )


class AggregateReportStore:
    """Project and deduplicate child evidence as one aggregate report profile."""

    def __init__(
        self,
        store: Any,
        *,
        aggregate_profile_id: str,
        child_profile_ids: Iterable[str],
        period_end_ts: int,
        max_memberships_per_cluster: int,
        max_snapshot_records: int,
        profile_update_failures: Iterable[dict[str, Any]] | None = None,
    ) -> None:
        self._store = store
        self.aggregate_profile_id = str(aggregate_profile_id)
        self.child_profile_ids = tuple(str(value) for value in child_profile_ids)
        self.period_end_ts = int(period_end_ts)
        self.max_memberships_per_cluster = max(1, int(max_memberships_per_cluster))
        self.max_snapshot_records = max(1, int(max_snapshot_records))
        self.profile_update_failures = normalize_profile_failures(
            profile_update_failures
        )
        self._loaded = False
        self._clusters: list[dict[str, Any]] = []
        self._memberships: dict[str, list[dict[str, Any]]] = {}
        self._snapshots: list[dict[str, Any]] = []
        self._provenance: dict[str, dict[str, list[str]]] = {}
        self.input_cluster_count = 0
        self.deduplicated_cluster_count = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    @staticmethod
    def _latest_snapshots(rows: Iterable[dict[str, Any]], cutoff: int) -> dict[str, dict[str, Any]]:
        ordered = sorted(
            (
                dict(row)
                for row in rows
                if int(row.get("created_at") or 0) <= cutoff
                and str(row.get("cluster_id") or "")
            ),
            key=lambda row: (
                str(row.get("cluster_id") or ""),
                -int(row.get("membership_revision") or 0),
                -int(row.get("created_at") or 0),
                str(row.get("id") or ""),
            ),
        )
        latest: dict[str, dict[str, Any]] = {}
        for row in ordered:
            latest.setdefault(str(row["cluster_id"]), row)
        return latest

    @staticmethod
    def _deduplication_key(
        cluster: dict[str, Any], memberships: list[dict[str, Any]]
    ) -> tuple[Any, ...]:
        article_ids = tuple(
            sorted(
                {
                    str(row.get("article_id") or "")
                    for row in memberships
                    if str(row.get("article_id") or "")
                }
            )
        )
        if not article_ids:
            return ("empty", str(cluster.get("profile_id") or ""), str(cluster.get("id") or ""))
        tag_scope = str(cluster.get("tag_scope") or "")
        if not tag_scope:
            tag_scope = "|".join(
                sorted(
                    {
                        str(row.get("tag_scope") or "")
                        for row in memberships
                        if str(row.get("tag_scope") or "")
                    }
                )
            )
        return (
            "event",
            str(cluster.get("embedding_model") or ""),
            int(cluster.get("embedding_dimension") or 0),
            str(cluster.get("embedding_instruction") or ""),
            tag_scope,
            article_ids,
        )

    @staticmethod
    def _candidate_rank(candidate: dict[str, Any]) -> tuple[Any, ...]:
        snapshot = candidate.get("snapshot") or {}
        cluster = candidate["cluster"]
        return (
            bool(snapshot),
            int(snapshot.get("membership_revision") or 0),
            int(snapshot.get("created_at") or 0),
            int(cluster.get("last_seen_ts") or 0),
            str(cluster.get("id") or ""),
        )

    def _load(self, query: dict[str, Any] | None = None) -> None:
        if self._loaded:
            return
        query = dict(query or {})
        requested_limit = max(1, int(query.pop("limit", 10000)))
        query.pop("offset", None)
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)

        for child_profile_id in self.child_profile_ids:
            child_snapshots = self._store.list_cluster_snapshots(
                child_profile_id,
                limit=self.max_snapshot_records + 1,
            )
            latest_by_cluster = self._latest_snapshots(
                child_snapshots, self.period_end_ts
            )
            clusters = self._store.list_threat_clusters(
                child_profile_id,
                limit=requested_limit,
                **query,
            )
            for raw_cluster in clusters:
                cluster = dict(raw_cluster)
                cluster_id = str(cluster.get("id") or "")
                if not cluster_id:
                    continue
                memberships = [
                    dict(row)
                    for row in self._store.list_cluster_memberships(
                        cluster_id,
                        limit=self.max_memberships_per_cluster + 1,
                    )[: self.max_memberships_per_cluster]
                ]
                groups[self._deduplication_key(cluster, memberships)].append(
                    {
                        "profile_id": child_profile_id,
                        "cluster": cluster,
                        "memberships": memberships,
                        "snapshot": latest_by_cluster.get(cluster_id),
                    }
                )

        selected: list[dict[str, Any]] = []
        for candidates in groups.values():
            selected.append(max(candidates, key=self._candidate_rank))
        selected.sort(
            key=lambda item: (
                -int(item["cluster"].get("last_seen_ts") or 0),
                str(item["cluster"].get("id") or ""),
            )
        )
        selected = selected[:requested_limit]

        self.input_cluster_count = sum(len(values) for values in groups.values())
        self.deduplicated_cluster_count = self.input_cluster_count - len(groups)
        for candidate in selected:
            original = candidate["cluster"]
            cluster_id = str(original["id"])
            key = self._deduplication_key(original, candidate["memberships"])
            equivalents = groups[key]
            cluster = copy.deepcopy(original)
            cluster["profile_id"] = self.aggregate_profile_id
            cluster["aggregate_source_profiles"] = sorted(
                {str(item["profile_id"]) for item in equivalents}
            )
            cluster["aggregate_source_cluster_ids"] = sorted(
                str(item["cluster"].get("id") or "") for item in equivalents
            )
            self._clusters.append(cluster)
            self._memberships[cluster_id] = candidate["memberships"]
            self._provenance[cluster_id] = {
                "profile_ids": cluster["aggregate_source_profiles"],
                "cluster_ids": cluster["aggregate_source_cluster_ids"],
            }

            raw_snapshot = candidate.get("snapshot")
            if isinstance(raw_snapshot, dict):
                snapshot = copy.deepcopy(raw_snapshot)
                snapshot["profile_id"] = self.aggregate_profile_id
                payload = snapshot.get("payload")
                if isinstance(payload, dict):
                    payload["profile_id"] = self.aggregate_profile_id
                snapshot["aggregate_source_profile_id"] = candidate["profile_id"]
                snapshot["aggregate_source_cluster_ids"] = cluster[
                    "aggregate_source_cluster_ids"
                ]
                self._snapshots.append(snapshot)
        self._loaded = True

    def list_threat_clusters(self, profile_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        if str(profile_id) != self.aggregate_profile_id:
            return self._store.list_threat_clusters(profile_id, **kwargs)
        self._load(kwargs)
        limit = max(1, int(kwargs.get("limit", len(self._clusters) or 1)))
        offset = max(0, int(kwargs.get("offset", 0)))
        return copy.deepcopy(self._clusters[offset : offset + limit])

    def list_cluster_memberships(
        self, cluster_id: str, *, limit: int = 10000
    ) -> list[dict[str, Any]]:
        self._load()
        if cluster_id in self._memberships:
            return copy.deepcopy(self._memberships[cluster_id][: max(1, int(limit))])
        return self._store.list_cluster_memberships(cluster_id, limit=limit)

    def list_cluster_snapshots(
        self,
        profile_id: str,
        *,
        cluster_id: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        if str(profile_id) != self.aggregate_profile_id:
            return self._store.list_cluster_snapshots(
                profile_id, cluster_id=cluster_id, limit=limit
            )
        self._load()
        rows = [
            row
            for row in self._snapshots
            if cluster_id is None or str(row.get("cluster_id") or "") == cluster_id
        ]
        rows.sort(
            key=lambda row: (
                -int(row.get("created_at") or 0),
                str(row.get("cluster_id") or ""),
                str(row.get("id") or ""),
            )
        )
        return copy.deepcopy(rows[: max(1, int(limit))])

    def report_quality(self) -> dict[str, Any]:
        self._load()
        return {
            "completeness": (
                "partial" if self.profile_update_failures else "complete"
            ),
            "aggregate_profile_ids": list(self.child_profile_ids),
            "aggregate_input_cluster_count": self.input_cluster_count,
            "aggregate_deduplicated_cluster_count": self.deduplicated_cluster_count,
            "aggregate_profile_update_failures": copy.deepcopy(
                self.profile_update_failures
            ),
        }

    def save_threat_landscape_report(self, report_doc: dict[str, Any]) -> bool:
        quality = report_doc.setdefault("quality", {})
        if isinstance(quality, dict):
            quality.update(self.report_quality())
        return self._store.save_threat_landscape_report(report_doc)

