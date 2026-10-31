# LICENSE HEADER MANAGED BY add-license-header
#
# BSD 3-Clause License
#
# Copyright (c) 2026, Martin Vesterlund
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Build a bounded hierarchy from long-term cluster tag sets."""

from collections import defaultdict
from typing import Any, Dict, Iterable, List


def normalize_cluster_tag(value: Any) -> str:
    category, separator, name = str(value or "").partition(":")
    category = category.strip().upper()
    name = name.strip().casefold()
    if not separator or not category or not name:
        return ""
    return f"{category}:{name}"


def cluster_scope_tags(value: Any) -> frozenset[str]:
    return frozenset(
        normalized
        for part in str(value or "").split("|")
        if (normalized := normalize_cluster_tag(part))
    )


def _cluster_sort_key(cluster: Dict[str, Any]) -> tuple[int, str]:
    return (-int(cluster.get("last_seen_ts") or 0), str(cluster.get("id") or ""))


def build_cluster_tag_network(
    clusters: Iterable[Dict[str, Any]],
    base_cluster_id: str,
    *,
    up_levels: int,
    down_levels: int,
    max_nodes: int,
    excluded_tags: Iterable[str] = (),
) -> Dict[str, Any]:
    """Return a hierarchy built by successive one-tag removals/additions.

    Signed levels are used in the result: negative levels remove one tag per
    hop, positive levels add one tag per hop, and the selected cluster is zero.
    Upward hops use the closest existing strict subsets and may therefore
    remove several tags. Downward hops are the inverse of that same relation.
    Edges always point from the smaller tag set to the larger tag set.
    """

    base_id = str(base_cluster_id or "").strip()
    if not base_id:
        raise ValueError("base_cluster_id is required")
    if up_levels < 0 or down_levels < 0:
        raise ValueError("levels cannot be negative")
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")

    unique_clusters: Dict[str, Dict[str, Any]] = {}
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("id") or "").strip()
        if cluster_id and cluster_id not in unique_clusters:
            unique_clusters[cluster_id] = cluster
    base = unique_clusters.get(base_id)
    if base is None:
        raise ValueError("base cluster is missing")

    base_profile = str(base.get("profile_id") or "")
    base_original_tags = cluster_scope_tags(base.get("tag_scope"))
    requested_exclusions = {
        normalized
        for value in excluded_tags
        if (normalized := normalize_cluster_tag(value))
    }
    effective_exclusions = frozenset(requested_exclusions & base_original_tags)
    by_tags: Dict[frozenset[str], List[Dict[str, Any]]] = defaultdict(list)
    for cluster in unique_clusters.values():
        if str(cluster.get("profile_id") or "") != base_profile:
            continue
        tags = cluster_scope_tags(cluster.get("tag_scope")) - effective_exclusions
        if tags:
            by_tags[tags].append(cluster)
    for rows in by_tags.values():
        rows.sort(key=_cluster_sort_key)

    base_tags = base_original_tags - effective_exclusions
    level_sets: Dict[int, set[frozenset[str]]] = {0: {base_tags}}
    tag_set_edges: set[tuple[frozenset[str], frozenset[str]]] = set()

    if base_tags:
        available_tag_sets = set(by_tags)
        closest_parent_count: Dict[frozenset[str], int] = {}

        def parent_tag_count(tags: frozenset[str]) -> int:
            cached = closest_parent_count.get(tags)
            if cached is not None:
                return cached
            count = max(
                (len(candidate) for candidate in available_tag_sets if candidate < tags),
                default=0,
            )
            closest_parent_count[tags] = count
            return count

        current = {base_tags}
        visited = {base_tags}
        for distance in range(1, up_levels + 1):
            candidates = {
                candidate
                for tags in current
                for candidate in available_tag_sets
                if candidate not in visited and candidate < tags
            }
            if not candidates:
                break
            next_tag_count = max(len(candidate) for candidate in candidates)
            next_sets = {
                candidate
                for candidate in candidates
                if len(candidate) == next_tag_count
            }
            for tags in current:
                for candidate in next_sets:
                    if candidate < tags:
                        tag_set_edges.add((candidate, tags))
            if not next_sets:
                break
            level_sets[-distance] = next_sets
            visited.update(next_sets)
            current = next_sets

        current = {base_tags}
        visited = {base_tags}
        for distance in range(1, down_levels + 1):
            next_sets: set[frozenset[str]] = set()
            for tags in current:
                for candidate in available_tag_sets:
                    if (
                        candidate not in visited
                        and tags < candidate
                        and len(tags) == parent_tag_count(candidate)
                    ):
                        next_sets.add(candidate)
                        tag_set_edges.add((tags, candidate))
            if not next_sets:
                break
            level_sets[distance] = next_sets
            visited.update(next_sets)
            current = next_sets

    nodes: List[Dict[str, Any]] = []
    included_by_level_and_tags: Dict[
        tuple[int, frozenset[str]], List[str]
    ] = defaultdict(list)
    truncated = False

    def append_node(cluster: Dict[str, Any], level: int, tags: frozenset[str]) -> bool:
        nonlocal truncated
        if len(nodes) >= max_nodes:
            truncated = True
            return False
        cluster_id = str(cluster.get("id") or "")
        nodes.append(
            {
                "id": cluster_id,
                "profile_id": str(cluster.get("profile_id") or ""),
                "status": str(cluster.get("status") or "unknown"),
                "tag_scope": str(cluster.get("tag_scope") or ""),
                "tags": sorted(tags),
                "tag_count": len(tags),
                "level": level,
                "direction": "up" if level < 0 else "down" if level > 0 else "base",
                "member_count": int(cluster.get("member_count") or 0),
                "last_seen_ts": int(cluster.get("last_seen_ts") or 0),
                "latest_snapshot_id": cluster.get("latest_snapshot_id"),
                "is_base": cluster_id == base_id,
            }
        )
        included_by_level_and_tags[(level, tags)].append(cluster_id)
        return True

    append_node(base, 0, base_tags)
    max_distance = max(up_levels, down_levels)
    ordered_levels = [
        level
        for distance in range(1, max_distance + 1)
        for level in (-distance, distance)
        if level in level_sets
    ]
    for level in ordered_levels:
        for tags in sorted(level_sets[level], key=lambda value: tuple(sorted(value))):
            for cluster in by_tags.get(tags, []):
                if not append_node(cluster, level, tags):
                    break
            if truncated:
                break
        if truncated:
            break

    edges: List[Dict[str, Any]] = []
    edge_number = 0
    set_levels = {
        tags: level
        for level, tag_sets in level_sets.items()
        for tags in tag_sets
    }
    for smaller, larger in sorted(
        tag_set_edges,
        key=lambda pair: (tuple(sorted(pair[0])), tuple(sorted(pair[1]))),
    ):
        smaller_level = set_levels.get(smaller)
        larger_level = set_levels.get(larger)
        if smaller_level is None or larger_level is None:
            continue
        added_tags = sorted(larger - smaller)
        for source in included_by_level_and_tags.get((smaller_level, smaller), []):
            for target in included_by_level_and_tags.get((larger_level, larger), []):
                edge_number += 1
                edges.append(
                    {
                        "id": f"edge-{edge_number}",
                        "source": source,
                        "target": target,
                        "added_tags": added_tags,
                    }
                )

    populated_levels = sorted({int(node["level"]) for node in nodes})
    return {
        "base_cluster_id": base_id,
        "profile_id": base_profile,
        "base_tags": sorted(base_tags),
        "base_original_tags": sorted(base_original_tags),
        "excluded_tags": sorted(effective_exclusions),
        "up_levels": up_levels,
        "down_levels": down_levels,
        "nodes": nodes,
        "edges": edges,
        "levels": populated_levels,
        "upward_parent_found": any(int(node["level"]) < 0 for node in nodes),
        "truncated": truncated,
    }
