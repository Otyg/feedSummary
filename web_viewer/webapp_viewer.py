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
#

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from email.utils import format_datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import markdown as md
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.middleware.proxy_fix import ProxyFix
import requests
import yaml

from uicommon import (
    build_source_options,
    build_topic_options,
    format_ts,
    get_store,
    load_config,
    source_to_topics_map,
)
from feedsummary_core.summarizer.main import (
    _build_composed_summary_text,
    _strip_sources_appendix_from_summary,
)
from feedsummary_core.summarizer.tagging import TagManager
from feedsummary_core.long_term import (
    ClusteringSettings,
    ThreatCluster,
    assign_article,
    build_cluster_merge_operation,
    build_cluster_membership_edit,
    build_cluster_review_merge_operation,
    build_cluster_review_resolution,
)
from feedsummary_core.llm_client import create_llm_client
from feedsummary_core.persistence import TagRelationError
from feedsummary_core.prompts.loader import (
    DEFAULT_PROMPTS_PATH,
    list_prompt_packages,
    load_prompt_package,
    resolve_prompt_root,
)
from retrain_tagging_ml import main as retrain_tagging_ml
from uicommon.proofread_rounds import _strip_proofread_feedback_from_summary
from uicommon.tagging_ml_retrain import CoalescingRetrainer
from web_viewer.article_listing import list_article_page, list_filter_tags
from web_viewer.attack_pressure import assess_attack_pressure, valid_cve
from web_viewer.long_term_cluster_graph import (
    build_cluster_tag_network,
    cluster_scope_tags,
    normalize_cluster_tag,
)


logger = logging.getLogger(__name__)

RSS_ATOM_NS = "http://www.w3.org/2005/Atom"
ElementTree.register_namespace("atom", RSS_ATOM_NS)

# Use absolute paths so templates/static work no matter working directory
BASE_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.wsgi_app = ProxyFix(app.wsgi_app, x_host=1, x_port=1, x_proto=1, x_prefix=1)

# Global app state (loaded once)
APP_CONFIG_PATH: str = ""
APP_CFG: Dict[str, Any] = {}
APP_STORE = None


def _run_tagging_ml_retrain() -> None:
    """Force a complete rebuild using a separate store connection."""
    config_path = APP_CONFIG_PATH
    if not config_path:
        raise RuntimeError("Viewer configuration path is unavailable")
    exit_code = retrain_tagging_ml(["--config", config_path])
    if exit_code != 0:
        raise RuntimeError(f"Tagging ML retraining exited with status {exit_code}")


_TAGGING_ML_RETRAINER = CoalescingRetrainer(
    _run_tagging_ml_retrain,
    debounce_seconds=2.0,
    logger=logger,
)


def _schedule_tagging_ml_retrain(*, article_id: str, tag_id: int, action: str) -> str:
    """Schedule retraining after a successful user-initiated tag mutation."""
    tagging = APP_CFG.get("tagging") if isinstance(APP_CFG.get("tagging"), dict) else {}
    ml = tagging.get("ml") if isinstance(tagging.get("ml"), dict) else {}
    if not bool(ml.get("enabled", False)):
        return "disabled"
    if not bool(ml.get("retrain_on_manual_tag_change", False)):
        return "disabled"

    try:
        status = _TAGGING_ML_RETRAINER.request(
            f"{action} tag {int(tag_id)} on article {article_id}"
        )
    except Exception:  # noqa: BLE001 - the tag mutation has already succeeded
        logger.exception(
            "[TagAPI] Could not queue ML retraining after %s of tag %s on article %s",
            action,
            tag_id,
            article_id,
        )
        return "queue_failed"
    logger.info(
        "[TagAPI] ML retraining %s after %s of tag %s on article %s",
        status,
        action,
        tag_id,
        article_id,
    )
    return status


@lru_cache(maxsize=8)
def _load_static_md(filename: str) -> str:
    p = BASE_DIR / "static" / filename
    return p.read_text(encoding="utf-8")


def _resolve_path_from_cwd(p: str) -> str:
    pp = Path(os.path.expandvars(os.path.expanduser(p)))
    if not pp.is_absolute():
        pp = (Path.cwd() / pp).resolve()
    return str(pp)


def _resolve_config_path(cli_path: Optional[str] = None) -> str:
    """
    Priority:
      1) CLI --config (only in __main__)
      2) ENV FEEDSUMMARY_CONFIG
      3) ./config.yaml (cwd), else ./config.yaml.dist
    """
    if cli_path:
        return _resolve_path_from_cwd(cli_path)

    env = os.environ.get("FEEDSUMMARY_CONFIG", "").strip()
    if env:
        return _resolve_path_from_cwd(env)

    p = (Path.cwd() / "config.yaml").resolve()
    if p.exists():
        return str(p)

    pd = (Path.cwd() / "config.yaml.dist").resolve()
    return str(pd) if pd.exists() else str(p)


def _abspath_cfg_paths(cfg: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    """
    Resolve selected cfg paths relative to the directory of config.yaml,
    so running from a different CWD doesn't accidentally use a different DB.
    """
    base = Path(config_path).resolve().parent

    def abs_if_rel(p: str) -> str:
        pp = Path(os.path.expandvars(os.path.expanduser(p)))
        if pp.is_absolute():
            return str(pp)
        return str((base / pp).resolve())

    cfg2 = dict(cfg)

    st = cfg2.get("store")
    if isinstance(st, dict) and st.get("path"):
        st2 = dict(st)
        st2["path"] = abs_if_rel(str(st2["path"]))
        cfg2["store"] = st2

    pr = cfg2.get("prompts")
    if isinstance(pr, dict) and pr.get("path"):
        pr2 = dict(pr)
        pr2["path"] = abs_if_rel(str(pr2["path"]))
        cfg2["prompts"] = pr2

    sch = cfg2.get("scheduler")
    if isinstance(sch, dict) and sch.get("path"):
        sch2 = dict(sch)
        sch2["path"] = abs_if_rel(str(sch2["path"]))
        cfg2["scheduler"] = sch2

    return cfg2


def init_app_state(config_path: str, *, initialize_categories: bool = True) -> None:
    global APP_CONFIG_PATH, APP_CFG, APP_STORE
    APP_CONFIG_PATH = str(Path(config_path).resolve())
    raw = load_config(APP_CONFIG_PATH)
    APP_CFG = _abspath_cfg_paths(raw, APP_CONFIG_PATH)
    APP_STORE = get_store(APP_CFG)

    sp = (APP_CFG.get("store") or {}).get("path")
    logger.info("Viewer config loaded: %s", APP_CONFIG_PATH)
    logger.info("Resolved store path: %s", sp)
    
    if initialize_categories:
        # The regular viewer also serves the tag/category editors and therefore
        # keeps their default metadata initialized. The read-only viewer opts
        # out so its startup path does not deliberately mutate application data.
        try:
            APP_STORE.initialize_default_categories()
            logger.info("Tag categories initialized")
        except Exception as e:
            logger.error("Error initializing categories: %s", e)


def _md_to_html(text: str) -> str:
    return md.markdown(text or "", extensions=["extra"])


def _long_term_ui_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Return the explicit, read-only long-term UI feature flag."""

    root = config if isinstance(config, dict) else APP_CFG
    long_term = root.get("long_term_analysis") or {}
    ui = long_term.get("ui") or {} if isinstance(long_term, dict) else {}
    return isinstance(ui, dict) and ui.get("enabled") is True


def _long_term_review_actions_enabled(
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    root = config if isinstance(config, dict) else APP_CFG
    long_term = root.get("long_term_analysis") or {}
    ui = long_term.get("ui") or {} if isinstance(long_term, dict) else {}
    return (
        isinstance(ui, dict)
        and isinstance(long_term, dict)
        and long_term.get("enabled") is True
        and ui.get("enabled") is True
        and ui.get("review_actions_enabled") is True
    )


def _long_term_profile_ids(config: Optional[Dict[str, Any]] = None) -> List[str]:
    root = config if isinstance(config, dict) else APP_CFG
    long_term = root.get("long_term_analysis") or {}
    profiles = long_term.get("profiles") or {} if isinstance(long_term, dict) else {}
    if not isinstance(profiles, dict):
        return []
    profile_ids = [str(profile_id) for profile_id in profiles if str(profile_id).strip()]

    def sort_key(profile_id: str) -> tuple[bool, bool, str]:
        profile = profiles.get(profile_id)
        archived = (
            not isinstance(profile, dict)
            or profile.get("enabled") is False
            or profile.get("clustering_enabled") is False
        )
        aggregate = isinstance(profile, dict) and bool(profile.get("aggregate_profiles"))
        return archived, not aggregate, profile_id

    return sorted(profile_ids, key=sort_key)


def _long_term_profile_label(profile_id: str) -> str:
    profiles = (APP_CFG.get("long_term_analysis") or {}).get("profiles") or {}
    profile = profiles.get(profile_id) if isinstance(profiles, dict) else None
    if isinstance(profile, dict):
        return str(profile.get("display_name") or profile_id)
    return profile_id


def _long_term_report_list_item(report: Dict[str, Any]) -> Dict[str, Any]:
    analysis = report.get("analysis") or {}
    quality = report.get("quality") or {}
    snapshot_ids = report.get("input_snapshot_ids") or []
    forecasts = report.get("forecast") or []
    return {
        "id": str(report.get("id") or ""),
        "profile_id": str(report.get("profile_id") or ""),
        "period_start_ts": int(report.get("period_start_ts") or 0),
        "period_end_ts": int(report.get("period_end_ts") or 0),
        "created_at": int(report.get("created_at") or 0),
        "status": str(report.get("status") or ""),
        "previous_report_id": report.get("previous_report_id"),
        "executive_summary": (
            str(analysis.get("executive_summary") or "")
            if isinstance(analysis, dict)
            else ""
        ),
        "quality": quality if isinstance(quality, dict) else {},
        "input_snapshot_count": len(snapshot_ids) if isinstance(snapshot_ids, list) else 0,
        "forecast_count": len(forecasts) if isinstance(forecasts, list) else 0,
        "prompt_version": report.get("prompt_version"),
        "segment_prompt_version": report.get("segment_prompt_version"),
        "model": report.get("model"),
    }


def _long_term_run_list_item(run: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(run.get("id") or ""),
        "profile_id": str(run.get("profile_id") or ""),
        "run_type": str(run.get("run_type") or ""),
        "status": str(run.get("status") or "unknown"),
        "mode": str(run.get("mode") or ""),
        "stage": str(run.get("stage") or ""),
        "started_at": int(run.get("started_at") or 0),
        "finished_at": int(run.get("finished_at") or 0),
        "error_type": str(run.get("error_type") or ""),
        "error": str(run.get("error") or "")[:1000],
    }


def _long_term_snapshot_list_item(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    payload = snapshot.get("payload") or {}
    return {
        "id": str(snapshot.get("id") or ""),
        "profile_id": str(snapshot.get("profile_id") or ""),
        "cluster_id": str(snapshot.get("cluster_id") or ""),
        "membership_revision": int(snapshot.get("membership_revision") or 0),
        "prompt_version": snapshot.get("prompt_version"),
        "created_at": int(snapshot.get("created_at") or 0),
        "title": str(payload.get("title") or "") if isinstance(payload, dict) else "",
        "label": _long_term_snapshot_label(snapshot),
        "summary": (
            str(payload.get("summary") or "") if isinstance(payload, dict) else ""
        ),
    }


_LONG_TERM_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.IGNORECASE)


def _long_term_snapshot_label(snapshot: Dict[str, Any]) -> str:
    """Build a concise human label without exposing storage identity as text."""

    payload = snapshot.get("payload") or {}
    payload = payload if isinstance(payload, dict) else {}
    title = " ".join(str(payload.get("title") or "").split())
    summary = " ".join(str(payload.get("summary") or "").split())
    searchable = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    cves = list(
        dict.fromkeys(value.upper() for value in _LONG_TERM_CVE_RE.findall(searchable))
    )
    if title:
        if any(cve.casefold() in title.casefold() for cve in cves):
            return title
        if cves:
            return f"{cves[0]} – {title}"
        return title
    if cves:
        return cves[0]
    if summary:
        return summary if len(summary) <= 120 else f"{summary[:117].rstrip()}…"
    revision = int(snapshot.get("membership_revision") or 0)
    return f"Klustersnapshot, revision {revision}" if revision else "Klustersnapshot"


def _long_term_report_link_labels(
    store: Any, report: Dict[str, Any]
) -> tuple[Dict[str, str], Dict[str, str]]:
    """Resolve report snapshot/cluster IDs to labels from its frozen evidence."""

    get_snapshot = getattr(store, "get_cluster_snapshot", None)
    if not callable(get_snapshot):
        return {}, {}
    snapshot_labels: Dict[str, str] = {}
    cluster_labels: Dict[str, str] = {}
    for raw_snapshot_id in report.get("input_snapshot_ids") or []:
        snapshot_id = str(raw_snapshot_id or "").strip()
        if not snapshot_id:
            continue
        snapshot = get_snapshot(snapshot_id)
        if not isinstance(snapshot, dict):
            continue
        label = _long_term_snapshot_label(snapshot)
        snapshot_labels[snapshot_id] = label
        cluster_id = str(snapshot.get("cluster_id") or "").strip()
        if cluster_id:
            cluster_labels[cluster_id] = label
    return snapshot_labels, cluster_labels


def _long_term_article_labels(
    store: Any, snapshot: Dict[str, Any]
) -> Dict[str, str]:
    """Resolve current or retained article titles for snapshot evidence links."""

    article_ids = [
        str(value).strip()
        for value in snapshot.get("input_article_ids") or []
        if str(value).strip()
    ]
    article_id_set = set(article_ids)
    labels: Dict[str, str] = {}
    get_articles = getattr(store, "get_articles_by_ids", None)
    if callable(get_articles) and article_ids:
        for article in get_articles(article_ids) or []:
            if not isinstance(article, dict):
                continue
            article_id = str(article.get("id") or "").strip()
            title = " ".join(str(article.get("title") or "").split())
            if article_id and title:
                labels[article_id] = title
    list_memberships = getattr(store, "list_cluster_memberships", None)
    cluster_id = str(snapshot.get("cluster_id") or "").strip()
    if callable(list_memberships) and cluster_id:
        rows = list_memberships(cluster_id, limit=max(1000, len(article_ids))) or []
        for membership in rows:
            if not isinstance(membership, dict):
                continue
            article_id = str(membership.get("article_id") or "").strip()
            evidence = membership.get("evidence") or {}
            title = (
                " ".join(str(evidence.get("title") or "").split())
                if isinstance(evidence, dict)
                else ""
            )
            if article_id in article_id_set and title:
                labels.setdefault(article_id, title)
    return labels


def _long_term_cluster_public_item(cluster: Dict[str, Any]) -> Dict[str, Any]:
    """Exclude stored vectors while retaining review and identity metadata."""

    return {
        key: value
        for key, value in cluster.items()
        if key not in {"centroid", "vector_sum"}
    }


_LONG_TERM_CLUSTER_STATUSES = ("active", "dormant", "closed")


def _normalize_long_term_cluster_tag(value: Any) -> str:
    return normalize_cluster_tag(value)


def _long_term_cluster_scope_tags(value: Any) -> frozenset[str]:
    return cluster_scope_tags(value)


def _long_term_tag_network_options() -> tuple[int, int, int, int, int]:
    ui = (APP_CFG.get("long_term_analysis") or {}).get("ui") or {}
    max_levels = max(1, min(int(ui.get("tag_network_max_levels") or 6), 12))
    max_nodes = max(10, min(int(ui.get("tag_network_max_nodes") or 250), 1000))
    candidate_limit = max(
        max_nodes,
        min(int(ui.get("tag_network_candidate_limit") or 10000), 50000),
    )
    default_levels = max(
        0,
        min(int(ui.get("tag_network_default_levels") or 2), max_levels),
    )
    requested_both = request.args.get("levels")
    try:
        up_levels = int(request.args.get("up", requested_both or default_levels))
        down_levels = int(request.args.get("down", requested_both or default_levels))
    except (TypeError, ValueError) as exc:
        raise ValueError("up, down and levels must be integers") from exc
    if not 0 <= up_levels <= max_levels or not 0 <= down_levels <= max_levels:
        raise ValueError(f"up and down must be between 0 and {max_levels}")
    return up_levels, down_levels, max_levels, max_nodes, candidate_limit


def _long_term_cluster_tag_network(
    store: Any,
    cluster_id: str,
    *,
    up_levels: int,
    down_levels: int,
    max_nodes: int,
    candidate_limit: int,
    excluded_tags: List[str],
) -> Optional[Dict[str, Any]]:
    get_cluster = getattr(store, "get_threat_cluster", None)
    list_clusters = getattr(store, "list_threat_clusters", None)
    if not callable(get_cluster) or not callable(list_clusters):
        raise RuntimeError("Long-term cluster storage is unavailable")
    base = get_cluster(cluster_id)
    if not isinstance(base, dict):
        return None
    profile_id = str(base.get("profile_id") or "").strip()
    candidates = list_clusters(
        profile_id,
        statuses=list(_LONG_TERM_CLUSTER_STATUSES),
        limit=candidate_limit,
        include_vectors=False,
    ) or []
    candidate_rows = [row for row in candidates if isinstance(row, dict)]
    if not any(str(row.get("id") or "") == cluster_id for row in candidate_rows):
        candidate_rows.append(base)
    rows_by_id = {
        str(row.get("id") or ""): row
        for row in candidate_rows
        if str(row.get("id") or "").strip()
    }
    graph = build_cluster_tag_network(
        candidate_rows,
        cluster_id,
        up_levels=up_levels,
        down_levels=down_levels,
        max_nodes=max_nodes,
        excluded_tags=excluded_tags,
    )
    graph["source_clusters_truncated"] = len(candidates) >= candidate_limit
    graph["truncated"] = bool(
        graph["truncated"] or graph["source_clusters_truncated"]
    )
    for node in graph["nodes"]:
        row = rows_by_id[node["id"]]
        item = _long_term_cluster_list_item(store, row)
        node["label"] = item["label"]
        node["url"] = url_for("view_long_term_cluster", cluster_id=node["id"])
        node["last_seen_label"] = format_ts(node["last_seen_ts"])
    return graph


def _long_term_article_network_options() -> tuple[float, int, int, int]:
    ui = (APP_CFG.get("long_term_analysis") or {}).get("ui") or {}
    max_neighbors_limit = max(
        1, min(int(ui.get("article_network_max_neighbors") or 200), 1000)
    )
    default_max_neighbors = max(
        1,
        min(
            int(ui.get("article_network_default_max_neighbors") or 40),
            max_neighbors_limit,
        ),
    )
    membership_limit = max(
        100, min(int(ui.get("article_network_membership_limit") or 5000), 20000)
    )
    try:
        min_overlap_percent = float(request.args.get("min_overlap", "1"))
        max_neighbors = int(
            request.args.get("max_neighbors", str(default_max_neighbors))
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("min_overlap must be a number and max_neighbors an integer") from exc
    if not 0 <= min_overlap_percent <= 100:
        raise ValueError("min_overlap must be between 0 and 100")
    if not 1 <= max_neighbors <= max_neighbors_limit:
        raise ValueError(
            f"max_neighbors must be between 1 and {max_neighbors_limit}"
        )
    return (
        min_overlap_percent,
        max_neighbors,
        max_neighbors_limit,
        membership_limit,
    )


def _long_term_cluster_article_network(
    store: Any,
    cluster_id: str,
    *,
    min_overlap_percent: float,
    max_neighbors: int,
    membership_limit: int,
) -> Optional[Dict[str, Any]]:
    get_cluster = getattr(store, "get_threat_cluster", None)
    list_memberships = getattr(store, "list_cluster_memberships", None)
    list_article_memberships = getattr(store, "list_article_cluster_memberships", None)
    if not all(
        callable(method)
        for method in (get_cluster, list_memberships, list_article_memberships)
    ):
        raise RuntimeError("Long-term membership storage is unavailable")
    base = get_cluster(cluster_id)
    if not isinstance(base, dict):
        return None
    profile_id = str(base.get("profile_id") or "").strip()
    membership_rows = list_memberships(cluster_id, limit=membership_limit + 1) or []
    memberships_truncated = len(membership_rows) > membership_limit
    unique_memberships: Dict[str, Dict[str, Any]] = {}
    for membership in membership_rows[:membership_limit]:
        if not isinstance(membership, dict):
            continue
        article_id = str(membership.get("article_id") or "").strip()
        if article_id:
            unique_memberships.setdefault(article_id, membership)

    def article_item(membership: Dict[str, Any]) -> Dict[str, Any]:
        evidence = membership.get("evidence") or {}
        if not isinstance(evidence, dict):
            evidence = {}
        article_id = str(membership.get("article_id") or "")
        return {
            "id": article_id,
            "title": " ".join(str(evidence.get("title") or article_id).split()),
            "source": " ".join(str(evidence.get("source") or "").split()),
            "url": str(evidence.get("url") or ""),
            "published_ts": int(
                evidence.get("published_ts") or membership.get("article_ts") or 0
            ),
        }

    shared_by_cluster: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for article_id, membership in unique_memberships.items():
        article = article_item(membership)
        for related in list_article_memberships(profile_id, article_id) or []:
            if not isinstance(related, dict):
                continue
            related_id = str(related.get("cluster_id") or "").strip()
            if related_id and related_id != cluster_id:
                shared_by_cluster[related_id][article_id] = article

    relations: List[Dict[str, Any]] = []
    for related_id, articles_by_id in shared_by_cluster.items():
        shared_articles = sorted(
            articles_by_id.values(),
            key=lambda item: (-int(item["published_ts"]), item["id"]),
        )
        overlap_ratio = (
            len(shared_articles) / len(unique_memberships)
            if unique_memberships
            else 0.0
        )
        if overlap_ratio * 100 < min_overlap_percent:
            continue
        related_cluster = get_cluster(related_id)
        if (
            not isinstance(related_cluster, dict)
            or str(related_cluster.get("profile_id") or "") != profile_id
        ):
            continue
        item = _long_term_cluster_list_item(store, related_cluster)
        relations.append(
            {
                **item,
                "shared_article_count": len(shared_articles),
                "shared_articles": shared_articles,
                "overlap_ratio": overlap_ratio,
                "url": url_for("view_long_term_cluster", cluster_id=related_id),
            }
        )
    relations.sort(
        key=lambda item: (
            -int(item["shared_article_count"]),
            -int(item["last_seen_ts"]),
            item["id"],
        )
    )
    neighbor_count = len(relations)
    relations = relations[:max_neighbors]
    base_item = _long_term_cluster_list_item(store, base)
    memberships_by_cluster: Dict[str, Dict[str, Dict[str, Any]]] = {
        cluster_id: unique_memberships
    }
    article_items: Dict[str, Dict[str, Any]] = {
        article_id: article_item(membership)
        for article_id, membership in unique_memberships.items()
    }
    truncated_cluster_ids = [cluster_id] if memberships_truncated else []
    for relation in relations:
        related_id = str(relation["id"])
        related_rows = list_memberships(
            related_id, limit=membership_limit + 1
        ) or []
        if len(related_rows) > membership_limit:
            truncated_cluster_ids.append(related_id)
        related_memberships: Dict[str, Dict[str, Any]] = {}
        for membership in related_rows[:membership_limit]:
            if not isinstance(membership, dict):
                continue
            article_id = str(membership.get("article_id") or "").strip()
            if not article_id:
                continue
            related_memberships.setdefault(article_id, membership)
            article_items.setdefault(article_id, article_item(membership))
        memberships_by_cluster[related_id] = related_memberships
        relation["article_count"] = len(related_memberships)

    base_node = {
        **base_item,
        "url": url_for("view_long_term_cluster", cluster_id=cluster_id),
        "article_count": len(unique_memberships),
        "is_base": True,
    }
    nodes = [
        base_node,
        *({**relation, "is_base": False} for relation in relations),
    ]
    edges: List[Dict[str, Any]] = []
    for source_index, source_node in enumerate(nodes):
        source_id = str(source_node["id"])
        source_articles = set(memberships_by_cluster.get(source_id, {}))
        for target_node in nodes[source_index + 1 :]:
            target_id = str(target_node["id"])
            shared_ids = sorted(
                source_articles & set(memberships_by_cluster.get(target_id, {}))
            )
            target_articles = set(memberships_by_cluster.get(target_id, {}))
            connects_base = cluster_id in {source_id, target_id}
            denominator = (
                len(unique_memberships)
                if connects_base
                else len(source_articles | target_articles)
            )
            overlap_ratio = len(shared_ids) / denominator if denominator else 0.0
            if not shared_ids or overlap_ratio * 100 < min_overlap_percent:
                continue
            edges.append(
                {
                    "id": f"edge-{len(edges) + 1}",
                    "source": source_id,
                    "target": target_id,
                    "shared_article_count": len(shared_ids),
                    "shared_article_ids": shared_ids,
                    "shared_articles": [article_items[value] for value in shared_ids],
                    "overlap_ratio": overlap_ratio,
                    "overlap_percent": overlap_ratio * 100,
                    "overlap_basis": "base" if connects_base else "jaccard",
                    "connects_base": connects_base,
                }
            )
    return {
        "base_cluster_id": cluster_id,
        "profile_id": profile_id,
        "base_article_count": len(unique_memberships),
        "min_overlap_percent": min_overlap_percent,
        "max_neighbors": max_neighbors,
        "nodes": nodes,
        "edges": edges,
        "relations": relations,
        "neighbor_count": neighbor_count,
        "neighbors_truncated": neighbor_count > len(relations),
        "memberships_truncated": bool(truncated_cluster_ids),
        "membership_truncated_cluster_ids": truncated_cluster_ids,
    }


def _long_term_cluster_tag_options(
    store: Any, selected_tags: List[str]
) -> List[Dict[str, str]]:
    long_term = APP_CFG.get("long_term_analysis") or {}
    clustering = long_term.get("clustering") or {}
    configured_categories = {
        str(value).strip().upper()
        for value in clustering.get("tag_scope_categories") or []
        if str(value).strip()
    }
    configured_categories -= {"VULNERABILITY", "VULNERABILITIES"}
    get_all_tags = getattr(store, "get_all_tags", None)
    try:
        stored_tags = (get_all_tags() or []) if callable(get_all_tags) else []
    except Exception:
        logger.exception("Could not list tags for the long-term cluster filter")
        stored_tags = []

    options: Dict[str, Dict[str, str]] = {}
    for tag in stored_tags:
        if not isinstance(tag, dict):
            continue
        category = str(tag.get("category") or "").strip().upper()
        name = " ".join(str(tag.get("name") or "").split())
        value = _normalize_long_term_cluster_tag(f"{category}:{name}")
        if (
            not value
            or category in {"VULNERABILITY", "VULNERABILITIES"}
            or (
                configured_categories
                and category not in configured_categories
            )
        ):
            continue
        options[value] = {"value": value, "label": name, "category": category}

    for value in selected_tags:
        if value in options:
            continue
        category, _, name = value.partition(":")
        options[value] = {
            "value": value,
            "label": name,
            "category": category,
        }
    return sorted(
        options.values(),
        key=lambda item: (item["label"].casefold(), item["category"]),
    )


def _long_term_cluster_label(store: Any, cluster: Dict[str, Any]) -> str:
    """Resolve a readable cluster name from its latest retained evidence."""

    latest_snapshot_id = str(cluster.get("latest_snapshot_id") or "").strip()
    get_snapshot = getattr(store, "get_cluster_snapshot", None)
    if latest_snapshot_id and callable(get_snapshot):
        snapshot = get_snapshot(latest_snapshot_id)
        if isinstance(snapshot, dict):
            return _long_term_snapshot_label(snapshot)

    title = ""
    list_memberships = getattr(store, "list_cluster_memberships", None)
    cluster_id = str(cluster.get("id") or "").strip()
    if cluster_id and callable(list_memberships):
        memberships = list_memberships(cluster_id, limit=1) or []
        if memberships and isinstance(memberships[0], dict):
            evidence = memberships[0].get("evidence") or {}
            if isinstance(evidence, dict):
                title = " ".join(str(evidence.get("title") or "").split())

    indicators = [
        " ".join(str(value).split())
        for value in cluster.get("strong_indicators") or []
        if str(value).strip()
    ]
    cves = list(
        dict.fromkeys(
            match.upper()
            for value in [title, *indicators]
            for match in _LONG_TERM_CVE_RE.findall(value)
        )
    )
    if title:
        if cves and not any(cve.casefold() in title.casefold() for cve in cves):
            return f"{cves[0]} – {title}"
        return title
    if cves:
        return cves[0]
    if indicators:
        return indicators[0]
    return "Händelsekluster utan titel"


def _long_term_cluster_list_item(
    store: Any, cluster: Dict[str, Any]
) -> Dict[str, Any]:
    membership_revision = int(cluster.get("membership_revision") or 0)
    summarized_revision = int(cluster.get("summarized_revision") or 0)
    return {
        "id": str(cluster.get("id") or ""),
        "profile_id": str(cluster.get("profile_id") or ""),
        "tag_scope": str(cluster.get("tag_scope") or ""),
        "label": _long_term_cluster_label(store, cluster),
        "status": str(cluster.get("status") or "unknown"),
        "first_seen_ts": int(cluster.get("first_seen_ts") or 0),
        "last_seen_ts": int(cluster.get("last_seen_ts") or 0),
        "member_count": int(cluster.get("member_count") or 0),
        "membership_revision": membership_revision,
        "summarized_revision": summarized_revision,
        "pending_revision_count": max(0, membership_revision - summarized_revision),
        "latest_snapshot_id": cluster.get("latest_snapshot_id"),
        "strong_indicators": [
            str(value)
            for value in cluster.get("strong_indicators") or []
            if str(value).strip()
        ],
        "superseded_by_cluster_id": cluster.get("superseded_by_cluster_id"),
    }


def _long_term_cluster_page(
    store: Any,
    profile_id: str,
    *,
    status: str,
    page: int,
    per_page: int,
    min_members: int,
    has_snapshot: bool = False,
    required_tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    list_clusters = getattr(store, "list_threat_clusters", None)
    if not callable(list_clusters):
        raise RuntimeError("long-term cluster store methods are unavailable")
    page_start = (page - 1) * per_page
    requested_count = page_start + per_page + 1
    batch_size = max(100, per_page + 1)
    source_offset = 0
    matching_rows: List[Dict[str, Any]] = []
    required_tag_set = frozenset(required_tags or [])
    filters_rows_in_memory = has_snapshot or bool(required_tag_set)
    while len(matching_rows) < requested_count:
        rows = list_clusters(
            profile_id,
            statuses=[status] if status else None,
            min_member_count=min_members,
            limit=batch_size if filters_rows_in_memory else per_page + 1,
            offset=source_offset if filters_rows_in_memory else page_start,
            include_vectors=False,
        ) or []
        valid_rows = [
            row
            for row in rows
            if isinstance(row, dict) and str(row.get("id") or "")
        ]
        if filters_rows_in_memory:
            matching_rows.extend(
                row
                for row in valid_rows
                if (
                    not has_snapshot
                    or str(row.get("latest_snapshot_id") or "").strip()
                )
                and required_tag_set.issubset(
                    _long_term_cluster_scope_tags(row.get("tag_scope"))
                )
            )
        else:
            matching_rows = valid_rows
        if not filters_rows_in_memory or len(rows) < batch_size:
            break
        source_offset += len(rows)

    page_rows = (
        matching_rows[page_start : page_start + per_page + 1]
        if filters_rows_in_memory
        else matching_rows
    )
    has_next = len(page_rows) > per_page
    return {
        "items": [
            _long_term_cluster_list_item(store, row)
            for row in page_rows[:per_page]
        ],
        "page": page,
        "per_page": per_page,
        "has_previous": page > 1,
        "has_next": has_next,
    }


def _long_term_cluster_ui_page(
    store: Any,
    profile_id: str,
    *,
    status: str,
    page: int,
    per_page: int,
    min_members: int,
    has_snapshot: bool = False,
    required_tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Keep each cluster distinct even when titles or articles overlap."""

    return _long_term_cluster_page(
        store,
        profile_id,
        status=status,
        page=page,
        per_page=per_page,
        min_members=min_members,
        has_snapshot=has_snapshot,
        required_tags=required_tags,
    )


def _long_term_cluster_list_parameters() -> tuple[
    str, int, int, int, bool, List[str]
]:
    status = str(request.args.get("status") or "").strip()
    if status == "all":
        status = ""
    if status and status not in _LONG_TERM_CLUSTER_STATUSES:
        abort(400)
    try:
        page = max(1, int(request.args.get("page", "1")))
    except (TypeError, ValueError):
        page = 1
    try:
        per_page = max(1, min(100, int(request.args.get("per_page", "50"))))
    except (TypeError, ValueError):
        per_page = 50
    try:
        min_members = max(1, min(100000, int(request.args.get("min_members", "1"))))
    except (TypeError, ValueError):
        min_members = 1
    has_snapshot_value = str(request.args.get("has_snapshot") or "").strip().lower()
    if has_snapshot_value not in {"", "0", "1", "false", "true", "off", "on"}:
        abort(400)
    has_snapshot = has_snapshot_value in {"1", "true", "on"}
    raw_tags = [str(value).strip() for value in request.args.getlist("tag")]
    if len(raw_tags) > 50 or any(len(value) > 256 for value in raw_tags):
        abort(400)
    selected_tags: List[str] = []
    for raw_tag in raw_tags:
        normalized = _normalize_long_term_cluster_tag(raw_tag)
        if not normalized:
            abort(400)
        if normalized not in selected_tags:
            selected_tags.append(normalized)
    return status, page, per_page, min_members, has_snapshot, selected_tags


def _long_term_clustering_settings() -> ClusteringSettings:
    long_term = APP_CFG.get("long_term_analysis") or {}
    clustering = long_term.get("clustering") or {} if isinstance(long_term, dict) else {}
    clustering = clustering if isinstance(clustering, dict) else {}
    return ClusteringSettings(
        similarity_threshold=float(clustering.get("similarity_threshold", 0.80)),
        ambiguity_margin=float(clustering.get("ambiguity_margin", 0.03)),
        candidate_window_days=int(clustering.get("candidate_window_days", 60)),
        min_shared_tags=int(clustering.get("min_shared_tags", 2)),
    )


def _long_term_review_candidate_data(
    store: Any,
    cluster: Dict[str, Any],
    membership: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], str]:
    candidate_ids = [
        str(membership.get(field) or "").strip()
        for field in (
            "best_candidate_cluster_id",
            "second_candidate_cluster_id",
        )
    ]
    scores = [membership.get("similarity"), membership.get("second_similarity")]
    source = "persisted"
    if not any(candidate_ids):
        source = "reconstructed"
        list_clusters = getattr(store, "list_threat_clusters", None)
        if not callable(list_clusters):
            raise RuntimeError("long-term cluster store methods are unavailable")
        assigned_at = int(membership.get("assigned_at") or cluster.get("last_seen_ts") or 0)
        candidate_documents = list_clusters(
            str(cluster.get("profile_id") or ""),
            statuses=["active", "dormant"],
            embedding_model=str(cluster.get("embedding_model") or ""),
            embedding_dimension=int(cluster.get("embedding_dimension") or 0),
            embedding_instruction=str(cluster.get("embedding_instruction") or ""),
            limit=10000,
        ) or []
        candidates = [
            ThreatCluster.from_document(row)
            for row in candidate_documents
            if isinstance(row, dict)
            and str(row.get("id") or "") != str(cluster.get("id") or "")
            and int(row.get("first_seen_ts") or 0) <= assigned_at
        ]
        review = ThreatCluster.from_document(cluster)
        decision = assign_article(
            profile_id=review.profile_id,
            article_ts=review.last_seen_ts,
            embedding=review.centroid,
            signature=review.embedding_signature,
            candidates=candidates,
            strong_indicators=review.strong_indicators,
            strict_cve_identity=review.strict_cve_identity,
            tag_scope=review.tag_scope,
            settings=_long_term_clustering_settings(),
        )
        candidate_ids = [
            str(decision.best_candidate_cluster_id or ""),
            str(decision.second_candidate_cluster_id or ""),
        ]
        scores = [decision.similarity, decision.second_similarity]

    get_cluster = getattr(store, "get_threat_cluster", None)
    if not callable(get_cluster):
        raise RuntimeError("long-term cluster store methods are unavailable")
    result: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for rank, (candidate_id, score) in enumerate(zip(candidate_ids, scores), start=1):
        if not candidate_id or candidate_id in seen:
            continue
        seen.add(candidate_id)
        document = get_cluster(candidate_id)
        if not isinstance(document, dict):
            continue
        bundle = _long_term_cluster_bundle(store, candidate_id, membership_limit=20)
        if bundle is None:
            continue
        item = _long_term_cluster_list_item(store, document)
        item.update(
            {
                "rank": rank,
                "similarity": float(score) if score is not None else None,
                "memberships": bundle["memberships"],
                "memberships_truncated": bundle["memberships_truncated"],
                "latest_snapshot": bundle["snapshots"][0]
                if bundle["snapshots"]
                else None,
                "merge_allowed": str(document.get("status") or "")
                in {"active", "dormant"}
                and str(document.get("tag_scope") or "") == str(cluster.get("tag_scope") or ""),
            }
        )
        result.append(item)
    return result, source


def _long_term_cluster_comparison_data(
    store: Any, cluster: Dict[str, Any], *, limit: int = 6
) -> List[Dict[str, Any]]:
    """Return nearby open clusters that can be compared or merged manually."""

    list_clusters = getattr(store, "list_threat_clusters", None)
    if not callable(list_clusters):
        raise RuntimeError("long-term cluster store methods are unavailable")
    source_id = str(cluster.get("id") or "")
    source_centroid = [float(value) for value in cluster.get("centroid") or []]
    source_norm = sum(value * value for value in source_centroid) ** 0.5
    if not source_norm:
        return []
    source_indicators = {str(value) for value in cluster.get("strong_indicators") or []}
    documents = list_clusters(
        str(cluster.get("profile_id") or ""),
        statuses=["active", "dormant"],
        embedding_model=str(cluster.get("embedding_model") or ""),
        embedding_dimension=int(cluster.get("embedding_dimension") or 0),
        embedding_instruction=str(cluster.get("embedding_instruction") or ""),
        limit=10000,
    ) or []
    candidates: List[Dict[str, Any]] = []
    for document in documents:
        candidate_id = str(document.get("id") or "")
        if (candidate_id == source_id or document.get("superseded_by_cluster_id")
            or str(document.get("tag_scope") or "") != str(cluster.get("tag_scope") or "")):
            continue
        centroid = [float(value) for value in document.get("centroid") or []]
        norm = sum(value * value for value in centroid) ** 0.5
        if len(centroid) != len(source_centroid) or not norm:
            continue
        similarity = sum(left * right for left, right in zip(source_centroid, centroid))
        similarity /= source_norm * norm
        item = _long_term_cluster_list_item(store, document)
        item["similarity"] = round(similarity, 6)
        item["shared_indicators"] = sorted(
            source_indicators
            & {str(value) for value in document.get("strong_indicators") or []}
        )
        bundle = _long_term_cluster_bundle(store, candidate_id, membership_limit=20)
        if bundle is None:
            continue
        item["memberships"] = bundle["memberships"]
        item["latest_snapshot"] = bundle["snapshots"][0] if bundle["snapshots"] else None
        item["merge_allowed"] = True
        candidates.append(item)
    candidates.sort(key=lambda row: (-float(row["similarity"]), str(row["id"])))
    return candidates[:limit]


def _long_term_review_bundle(store: Any, cluster_id: str) -> Optional[Dict[str, Any]]:
    get_cluster = getattr(store, "get_threat_cluster", None)
    if not callable(get_cluster):
        raise RuntimeError("long-term cluster store methods are unavailable")
    cluster = get_cluster(cluster_id)
    if not isinstance(cluster, dict):
        return None
    if str(cluster.get("status") or "") != "needs_review":
        return None
    bundle = _long_term_cluster_bundle(store, cluster_id, membership_limit=20)
    if bundle is None or not bundle["memberships"]:
        raise RuntimeError("review cluster has no retained membership")
    membership = next(
        (
            row
            for row in bundle["memberships"]
            if str(row.get("assignment_action") or "") == "needs_review"
        ),
        bundle["memberships"][0],
    )
    candidates, source = _long_term_review_candidate_data(store, cluster, membership)
    return {
        **bundle,
        "assignment": membership,
        "candidates": candidates,
        "candidate_source": source,
    }


def _long_term_membership_public_item(
    membership: Dict[str, Any], *, article_available: bool
) -> Dict[str, Any]:
    item = dict(membership)
    item["article_available"] = bool(article_available)
    return item


def _long_term_cluster_bundle(
    store: Any, cluster_id: str, *, membership_limit: int
) -> Optional[Dict[str, Any]]:
    get_cluster = getattr(store, "get_threat_cluster", None)
    list_memberships = getattr(store, "list_cluster_memberships", None)
    list_snapshots = getattr(store, "list_cluster_snapshots", None)
    if not all(
        callable(method)
        for method in (get_cluster, list_memberships, list_snapshots)
    ):
        raise RuntimeError("long-term provenance store methods are unavailable")

    cluster = get_cluster(str(cluster_id))
    if not isinstance(cluster, dict):
        return None
    memberships = [
        row
        for row in (list_memberships(str(cluster_id), limit=membership_limit) or [])
        if isinstance(row, dict) and row.get("article_id")
    ]
    article_ids = [str(row["article_id"]) for row in memberships]
    available_ids: set[str] = set()
    get_articles = getattr(store, "get_articles_by_ids", None)
    if callable(get_articles) and article_ids:
        available_ids = {
            str(row.get("id") or "")
            for row in (get_articles(article_ids) or [])
            if isinstance(row, dict) and row.get("id")
        }
    snapshots = [
        _long_term_snapshot_list_item(row)
        for row in (
            list_snapshots(
                str(cluster.get("profile_id") or ""),
                cluster_id=str(cluster_id),
                limit=1000,
            )
            or []
        )
        if isinstance(row, dict) and row.get("id")
    ]
    latest_snapshot = max(
        snapshots,
        key=lambda row: (int(row.get("created_at") or 0), str(row.get("id") or "")),
        default=None,
    )
    superseded_by = None
    superseding_id = str(cluster.get("superseded_by_cluster_id") or "").strip()
    if superseding_id:
        superseding_cluster = get_cluster(superseding_id)
        superseding_snapshots = (
            list_snapshots(
                str(cluster.get("profile_id") or ""),
                cluster_id=superseding_id,
                limit=1000,
            )
            if isinstance(superseding_cluster, dict)
            else []
        )
        superseding_latest = max(
            (
                _long_term_snapshot_list_item(row)
                for row in superseding_snapshots or []
                if isinstance(row, dict) and row.get("id")
            ),
            key=lambda row: (
                int(row.get("created_at") or 0),
                str(row.get("id") or ""),
            ),
            default=None,
        )
        superseded_by = {
            "id": superseding_id,
            "label": (
                str(superseding_latest.get("label") or "Sammanslaget händelsekluster")
                if superseding_latest is not None
                else "Sammanslaget händelsekluster"
            ),
        }
    member_count = int(cluster.get("member_count") or len(memberships))
    return {
        "cluster": _long_term_cluster_public_item(cluster),
        "cluster_label": (
            str(latest_snapshot["label"])
            if latest_snapshot is not None and latest_snapshot.get("label")
            else _long_term_cluster_label(store, cluster)
        ),
        "memberships": [
            _long_term_membership_public_item(
                row, article_available=str(row["article_id"]) in available_ids
            )
            for row in memberships
        ],
        "snapshots": snapshots,
        "superseded_by": superseded_by,
        "memberships_truncated": member_count > len(memberships),
    }


def _require_long_term_ui() -> None:
    if not _long_term_ui_enabled():
        abort(404)


def _require_long_term_review_actions() -> None:
    _require_long_term_ui()
    if not _long_term_review_actions_enabled():
        abort(403)


def _selected_long_term_profile() -> tuple[str, List[str]]:
    profiles = _long_term_profile_ids()
    selected = str(request.args.get("profile") or "").strip()
    if not selected and profiles:
        selected = profiles[0]
    if selected and selected not in profiles:
        abort(404)
    return selected, profiles


def _long_term_replacement_profile(profile_id: str) -> str:
    long_term = APP_CFG.get("long_term_analysis") or {}
    profiles = long_term.get("profiles") or {} if isinstance(long_term, dict) else {}
    if not isinstance(profiles, dict):
        return ""
    selected = profiles.get(profile_id)
    if not isinstance(selected, dict) or selected.get("clustering_enabled") is not False:
        return ""
    return next(
        (
            str(candidate_id)
            for candidate_id, candidate in profiles.items()
            if isinstance(candidate, dict)
            and candidate.get("enabled") is True
            and candidate.get("rebuild_source_profile_id") == profile_id
        ),
        "",
    )


@app.context_processor
def _long_term_template_context() -> Dict[str, Any]:
    long_term = APP_CFG.get("long_term_analysis") or {}
    mode = str(long_term.get("mode") or "") if isinstance(long_term, dict) else ""
    return {
        "long_term_ui_enabled": _long_term_ui_enabled(),
        "long_term_mode": mode,
        "long_term_review_actions_enabled": _long_term_review_actions_enabled(),
    }


_INVALID_XML_CHARS = re.compile(
    "[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD\U00010000-\U0010FFFF]"
)


def _xml_safe_text(value: Any) -> str:
    """Remove characters that XML 1.0 cannot represent."""
    return _INVALID_XML_CHARS.sub("", str(value or ""))


def _rss_pub_date(value: Any) -> str:
    timestamp = _coerce_positive_ts(value)
    if timestamp <= 0:
        return ""
    try:
        published = datetime.fromtimestamp(timestamp, timezone.utc)
        return format_datetime(published, usegmt=True)
    except (OSError, OverflowError, ValueError):
        return ""


def _rss_limit(default: int = 100, maximum: int = 500) -> int:
    try:
        value = int(request.args.get("limit", str(default)))
    except (TypeError, ValueError):
        value = default
    return max(1, min(value, maximum))


def _rss_response(
    *,
    title: str,
    link: str,
    description: str,
    self_url: str,
    items: List[Dict[str, str]],
) -> Response:
    rss = ElementTree.Element(
        "rss",
        {"version": "2.0"},
    )
    channel = ElementTree.SubElement(rss, "channel")
    ElementTree.SubElement(channel, "title").text = _xml_safe_text(title)
    ElementTree.SubElement(channel, "link").text = _xml_safe_text(link)
    ElementTree.SubElement(channel, "description").text = _xml_safe_text(description)
    ElementTree.SubElement(channel, "language").text = "sv"
    ElementTree.SubElement(
        channel,
        f"{{{RSS_ATOM_NS}}}link",
        {"href": self_url, "rel": "self", "type": "application/rss+xml"},
    )

    if items and items[0].get("pub_date"):
        ElementTree.SubElement(channel, "lastBuildDate").text = items[0]["pub_date"]

    for item_data in items:
        item = ElementTree.SubElement(channel, "item")
        ElementTree.SubElement(item, "title").text = _xml_safe_text(
            item_data.get("title")
        )
        ElementTree.SubElement(item, "link").text = _xml_safe_text(
            item_data.get("link")
        )
        ElementTree.SubElement(
            item,
            "guid",
            {"isPermaLink": item_data.get("guid_is_permalink", "true")},
        ).text = _xml_safe_text(item_data.get("guid"))
        if item_data.get("pub_date"):
            ElementTree.SubElement(item, "pubDate").text = item_data["pub_date"]
        ElementTree.SubElement(item, "description").text = _xml_safe_text(
            item_data.get("description")
        )

    payload = ElementTree.tostring(rss, encoding="utf-8", xml_declaration=True)
    return Response(payload, content_type="application/rss+xml; charset=utf-8")


def _has_proofread_audit_data(d: Dict[str, Any]) -> bool:
    if not isinstance(d, dict):
        return False
    direct_keys = (
        "proofread_original_summary",
        "proofread_revised_summary",
        "proofread_published_summary",
    )
    for k in direct_keys:
        if str(d.get(k) or "").strip():
            return True
    if str(d.get("proofread_output") or "").strip():
        return True
    pa = d.get("proofread_audit") or {}
    if isinstance(pa, dict):
        latest = pa.get("latest") or {}
        if isinstance(latest, dict):
            for k in (
                "original_summary",
                "revised_summary",
                "published_summary",
                "proofread_output",
            ):
                if str(latest.get(k) or "").strip():
                    return True
    return False


def _reconstruct_composed_original_summary(store, sdoc: Dict[str, Any]) -> str:
    """
    Best-effort fallback for older composed docs where original proofread text
    was not persisted.
    """
    if not isinstance(sdoc, dict):
        return ""

    sections = sdoc.get("sections") or []
    if not isinstance(sections, list) or not sections:
        return ""

    loaded_sections: List[Dict[str, str]] = []
    for s in sections:
        if not isinstance(s, dict):
            continue
        sec_id = str(s.get("summary_id") or "").strip()
        sec_summary_raw = ""
        if sec_id:
            try:
                sec_doc = store.get_summary_doc(sec_id)
            except Exception:
                sec_doc = None
            if isinstance(sec_doc, dict):
                sec_summary_raw = str(sec_doc.get("summary") or "")
        else:
            # Older composed docs may embed the section summary inline.
            sec_summary_raw = str(s.get("summary") or "")

        sec_summary = _strip_sources_appendix_from_summary(sec_summary_raw)
        if not sec_summary.strip():
            continue
        heading = (
            str(s.get("tag") or "").strip()
            or str(s.get("topic") or "").strip()
            or str(s.get("schedule") or "").strip()
            or str(s.get("promptpackage") or "").strip()
        )
        loaded_sections.append({"tag": heading, "summary": sec_summary})

    if not loaded_sections:
        return ""

    try:
        return str(
            _build_composed_summary_text(
                sections=loaded_sections,
                ingress=None,
            )
            or ""
        ).strip()
    except Exception:
        return ""


def _get_latest_summary(store) -> Optional[Dict[str, Any]]:
    """
    Robust: try get_latest_summary_doc; otherwise pick newest from list and refetch if needed.
    """
    fn = getattr(store, "get_latest_summary_doc", None)
    if callable(fn):
        try:
            d = fn()  # type: ignore
            if isinstance(d, dict) and (d.get("summary") or "").strip():
                return d
            if isinstance(d, dict) and d.get("id"):
                d2 = store.get_summary_doc(str(d.get("id")))
                if isinstance(d2, dict) and (d2.get("summary") or "").strip():
                    return d2
                return d
        except Exception:
            pass

    docs = store.list_summary_docs() or []
    docs = [d for d in docs if isinstance(d, dict)]
    if not docs:
        return None

    docs.sort(key=lambda x: int(x.get("created") or 0), reverse=True)

    for cand in docs[:10]:
        if (cand.get("summary") or "").strip():
            return cand
        sid = str(cand.get("id") or "").strip()
        if not sid:
            continue
        try:
            d = store.get_summary_doc(sid)
            if isinstance(d, dict):
                return d
        except Exception:
            continue

    return docs[0]


def _summary_list_item(d: Dict[str, Any]) -> Dict[str, Any]:
    item = {
        "id": d.get("id"),
        "created": int(d.get("created") or 0),
        "sources_count": len(d.get("sources") or []),
        "title": d.get("title") or "",
    }
    return _enrich_summary_view_model(item)


@lru_cache(maxsize=1)
def _viewer_source_topics_map() -> Dict[str, List[str]]:
    try:
        return source_to_topics_map(APP_CFG)
    except Exception:
        return {}


def _summary_topics(d: Dict[str, Any]) -> List[str]:
    topics = d.get("topics")
    if isinstance(topics, list):
        vals = [str(t).strip() for t in topics if str(t).strip()]
        if vals:
            return sorted(list(dict.fromkeys(vals)), key=lambda x: x.lower())

    src_topics = _viewer_source_topics_map()
    out: List[str] = []
    for snap in d.get("sources_snapshots") or []:
        if not isinstance(snap, dict):
            continue
        source = str(snap.get("source") or "").strip()
        if not source:
            continue
        out.extend(src_topics.get(source, []))
    return sorted(list(dict.fromkeys(out)), key=lambda x: x.lower())


def _enrich_summary_view_model(d: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(d)
    topics = _summary_topics(item)
    item["_viewer_topics"] = topics
    item["_viewer_topics_label"] = ", ".join(topics)
    return item


def _summary_tags_for_view(store, summary_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Resolve stored summary tags against current tag/category metadata."""
    stored_tags = summary_doc.get("tags") or []
    if not isinstance(stored_tags, list):
        return []

    try:
        tags_by_id = {
            tag.get("id"): tag
            for tag in (store.get_all_tags() or [])
            if isinstance(tag, dict) and tag.get("id") is not None
        }
    except Exception:
        tags_by_id = {}

    try:
        categories = {
            str(category.get("name") or "GENERAL"): category
            for category in (store.get_all_categories() or [])
            if isinstance(category, dict)
        }
    except Exception:
        categories = {}

    out: List[Dict[str, Any]] = []
    for stored in stored_tags:
        if not isinstance(stored, dict):
            continue
        current = tags_by_id.get(stored.get("id")) or stored
        name = str(current.get("name") or stored.get("name") or "").strip()
        if not name:
            continue
        category_name = str(
            current.get("category") or stored.get("category") or "GENERAL"
        ).strip() or "GENERAL"
        category = categories.get(category_name) or {}
        out.append(
            {
                "id": current.get("id") or stored.get("id"),
                "name": name,
                "category": category_name,
                "bg_color": str(category.get("bg_color") or "bg-secondary"),
                "text_color": str(category.get("text_color") or "text-dark"),
            }
        )
    return sorted(out, key=lambda tag: str(tag.get("name") or "").lower())


def _list_enriched_summaries(store) -> List[Dict[str, Any]]:
    docs = store.list_summary_docs() or []
    docs = [d for d in docs if isinstance(d, dict)]
    docs.sort(key=lambda d: int(d.get("created") or 0), reverse=True)
    return [_enrich_summary_view_model(d) for d in docs]


def _all_topics_from_summaries(docs: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for d in docs:
        for t in d.get("_viewer_topics") or []:
            tt = str(t).strip()
            if tt:
                out.append(tt)
    return sorted(list(dict.fromkeys(out)), key=lambda x: x.lower())


def _selected_topics_from_request() -> List[str]:
    vals = request.args.getlist("topic")
    out: List[str] = []
    for raw in vals:
        for part in str(raw or "").split(","):
            p = part.strip()
            if p:
                out.append(p)
    return list(dict.fromkeys(out))


def _filter_summaries_by_topics(
    docs: List[Dict[str, Any]], selected_topics: List[str]
) -> List[Dict[str, Any]]:
    if not selected_topics:
        return docs

    selected_lower = {t.lower() for t in selected_topics}
    filtered: List[Dict[str, Any]] = []
    for d in docs:
        # Exclude composed summaries when filtering by topics
        is_composed = d.get("meta", {}).get("composed", False)
        if is_composed:
            continue

        topics = {str(t).strip().lower() for t in d.get("_viewer_topics") or []}
        if topics & selected_lower:
            filtered.append(d)
    return filtered


def _filter_summaries_today(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prefer summaries created today; if none exist, fallback to latest date
    that has summaries.
    """
    today = datetime.now().date()
    by_date: Dict[str, List[Dict[str, Any]]] = {}
    for d in docs:
        created = _coerce_positive_ts(d.get("created"))
        if created <= 0:
            continue
        day_key = datetime.fromtimestamp(int(created)).strftime("%Y-%m-%d")
        by_date.setdefault(day_key, []).append(d)

    if not by_date:
        return []

    today_key = today.strftime("%Y-%m-%d")
    if today_key in by_date:
        return by_date[today_key]

    # docs are already sorted newest-first; first inserted day is latest.
    latest_day = next(iter(by_date.keys()))
    return by_date.get(latest_day, [])


def _normalize_schema_name(v: Any) -> str:
    s = str(v or "").strip().lower()
    s = re.sub(r"[\s_\-]+", " ", s)
    return s.strip()


def _schedule_entries() -> Dict[str, Dict[str, Any]]:
    sch = APP_CFG.get("scheduler") if isinstance(APP_CFG, dict) else None
    path = str((sch or {}).get("path") or "").strip() if isinstance(sch, dict) else ""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in raw.items():
        if isinstance(k, str) and k.strip() and isinstance(v, dict):
            out[k.strip()] = v
    return out


def _schedule_signature_from_entry(entry: Dict[str, Any]) -> tuple:
    freq = str(entry.get("frequency") or "").strip().lower()
    lb = str(entry.get("lookback") or "").strip()
    if not lb:
        if freq == "hourly":
            lb = "1h"
        elif freq == "daily":
            lb = "1d"
        elif freq == "weekly":
            lb = "1w"
        elif freq == "quarterday":
            lb = "6h"
        elif freq == "halfday":
            lb = "12h"
    cats = entry.get("categories") or []
    topics = (
        sorted([str(x).strip() for x in cats if str(x).strip()], key=lambda x: x.lower())
        if isinstance(cats, list)
        else []
    )
    pp = str(entry.get("promptpackage") or "").strip()
    return (_normalize_schema_name(lb), tuple(_normalize_schema_name(t) for t in topics), _normalize_schema_name(pp))


def _schedule_signature_from_overrides(overrides: Dict[str, Any]) -> tuple:
    lb = str(overrides.get("lookback") or "").strip()
    topics = overrides.get("topics") or []
    topics_list = (
        sorted([str(x).strip() for x in topics if str(x).strip()], key=lambda x: x.lower())
        if isinstance(topics, list)
        else []
    )
    pp = str(overrides.get("prompt_package") or "").strip()
    return (_normalize_schema_name(lb), tuple(_normalize_schema_name(t) for t in topics_list), _normalize_schema_name(pp))


def _extract_job_id_from_summary_id(summary_id: str) -> Optional[int]:
    m = re.search(r"_job(\d+)$", str(summary_id or "").strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _infer_schema_name_from_job(summary_id: str, store: Any) -> Optional[str]:
    jid = _extract_job_id_from_summary_id(summary_id)
    if jid is None:
        return None

    job = None
    get_job = getattr(store, "get_job", None)
    if callable(get_job):
        try:
            job = get_job(int(jid))
        except Exception:
            job = None
    if not isinstance(job, dict):
        return None

    fields_raw = job.get("fields_json")
    fields: Dict[str, Any] = {}
    if isinstance(fields_raw, dict):
        fields = fields_raw
    elif isinstance(fields_raw, str) and fields_raw.strip():
        try:
            parsed = json.loads(fields_raw)
            if isinstance(parsed, dict):
                fields = parsed
        except Exception:
            fields = {}
    overrides = fields.get("overrides")
    if not isinstance(overrides, dict):
        return None

    sig = _schedule_signature_from_overrides(overrides)
    sched = _schedule_entries()
    matches: List[str] = []
    for name, entry in sched.items():
        if _schedule_signature_from_entry(entry) == sig:
            matches.append(name)
    if len(matches) == 1:
        return matches[0]
    return None


def _doc_schema_names(doc: Dict[str, Any], store: Any) -> List[str]:
    out: List[str] = []
    sel = doc.get("selection") if isinstance(doc.get("selection"), dict) else {}
    meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
    for cand in (
        sel.get("name"),
        sel.get("schedule"),
        meta.get("schedule_name"),
        meta.get("job_name"),
    ):
        c = str(cand or "").strip()
        if c:
            out.append(c)
    inferred = _infer_schema_name_from_job(str(doc.get("id") or ""), store)
    if inferred:
        out.append(inferred)
    uniq: List[str] = []
    seen: set = set()
    for x in out:
        k = _normalize_schema_name(x)
        if k and k not in seen:
            seen.add(k)
            uniq.append(x)
    return uniq


def _article_list_item(a: Dict[str, Any]) -> Dict[str, Any]:
    text = str(a.get("text") or "")
    preview = (text[:400]).replace("\n", " ")
    
    # Get tags if store is available
    tags = []
    if APP_STORE:
        try:
            article_tags = APP_STORE.get_article_tags(a.get("id")) or []
            for t in article_tags:
                if isinstance(t, dict):
                    tags.append({
                        "id": t.get("id"),
                        "name": t.get("name", ""),
                        "category": t.get("category", "GENERAL")
                    })
        except Exception as e:
            logger.debug(f"Error loading tags for article {a.get('id')}: {e}")
    
    
    # Get tags if store is available
    tags = []
    if APP_STORE:
        try:
            article_tags = APP_STORE.get_article_tags(a.get("id")) or []
            for t in article_tags:
                if isinstance(t, dict):
                    tags.append({
                        "id": t.get("id"),
                        "name": t.get("name", ""),
                        "category": t.get("category", "GENERAL")
                    })
        except Exception as e:
            logger.debug(f"Error loading tags for article {a.get('id')}: {e}")
    
    return {
        "id": a.get("id"),
        "title": a.get("title") or "",
        "source": a.get("source") or "",
        "url": a.get("url") or "",
        "published_ts": int(_coerce_positive_ts(a.get("published_ts"))),
        "fetched_at": int(_coerce_positive_ts(a.get("fetched_at"))),
        "preview": preview,
        "tags": tags,
        "tags": tags,
    }


def _article_tag_filter_tree(
    store: Any,
    articles: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """Return used tags plus ancestors and their transitive, relevant descendants."""
    tags_by_id: Dict[int, Dict[str, Any]] = {}
    descendants_by_id: Dict[int, set[int]] = {}
    pending: List[int] = []

    for article in articles:
        for tag in article.get("tags", []) if isinstance(article, dict) else []:
            if not isinstance(tag, dict):
                continue
            try:
                tag_id = int(tag.get("id"))
            except (TypeError, ValueError):
                continue
            tags_by_id[tag_id] = {
                "id": tag_id,
                "name": str(tag.get("name") or ""),
                "category": str(tag.get("category") or "GENERAL"),
            }
            if tag_id not in descendants_by_id:
                descendants_by_id[tag_id] = {tag_id}
                pending.append(tag_id)

    relation_reader = getattr(store, "get_tag_relations", None)
    if callable(relation_reader):
        relation_cache: Dict[int, List[Dict[str, Any]]] = {}
        queued = set(pending)
        while pending:
            child_id = pending.pop()
            queued.discard(child_id)
            if child_id not in relation_cache:
                try:
                    relations = relation_reader(child_id) or {}
                    relation_cache[child_id] = [
                        parent
                        for parent in relations.get("parents", [])
                        if isinstance(parent, dict)
                    ]
                except Exception as exc:
                    logger.debug(
                        "Error loading tag parents for tag %s: %s", child_id, exc
                    )
                    relation_cache[child_id] = []

            child_descendants = descendants_by_id.setdefault(child_id, {child_id})
            for parent in relation_cache[child_id]:
                try:
                    parent_id = int(parent.get("id"))
                except (TypeError, ValueError):
                    continue
                tags_by_id[parent_id] = {
                    "id": parent_id,
                    "name": str(parent.get("name") or ""),
                    "category": str(parent.get("category") or "GENERAL"),
                }
                parent_descendants = descendants_by_id.setdefault(
                    parent_id, {parent_id}
                )
                previous_size = len(parent_descendants)
                parent_descendants.update(child_descendants)
                if len(parent_descendants) != previous_size and parent_id not in queued:
                    pending.append(parent_id)
                    queued.add(parent_id)

    filter_tags = sorted(
        tags_by_id.values(),
        key=lambda tag: str(tag.get("name") or "").casefold(),
    )
    descendant_ids = {
        str(tag_id): [str(item) for item in sorted(ids)]
        for tag_id, ids in descendants_by_id.items()
    }
    return filter_tags, descendant_ids


def _tag_id_key(value: Any) -> str:
    """Normalize tag identifiers so SQL, MongoDB and TinyDB values match."""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value or "").strip()


def _tag_motivering(tag: Dict[str, Any]) -> str:
    """Return an association explanation using current and legacy field names."""
    for field in ("motivering", "reasoning"):
        value = str(tag.get(field) or "").strip()
        if value:
            return value
    return ""


def _article_tag_motiveringar(store: Any, article_id: str) -> Dict[str, str]:
    """Read tag explanations from the article-tag relation.

    Older feedsummary-core versions persist ``motivering`` on the relation but
    omit it from ``get_article_tags``. These small, read-only backend fallbacks
    keep the viewer useful until all supported store versions expose the field.
    """
    article_id = str(article_id or "").strip()
    if not article_id:
        return {}

    def collect(rows: Any) -> Dict[str, str]:
        found: Dict[str, str] = {}
        for row in rows if rows is not None else ():
            if not isinstance(row, dict):
                try:
                    row = dict(row)
                except (TypeError, ValueError):
                    continue
            tag_id = _tag_id_key(row.get("tag_id") or row.get("id"))
            motivering = _tag_motivering(row)
            if tag_id and motivering:
                found[tag_id] = motivering
        return found

    # MongoDBStore exposes the database as ``db``.
    db = getattr(store, "db", None)
    if db is not None:
        try:
            collection = getattr(db, "article_tags", None)
            if collection is not None:
                return collect(collection.find({"article_id": article_id}))
        except Exception as exc:
            logger.debug(
                "Could not load MongoDB tag explanations for %s: %s",
                article_id,
                exc,
            )

    # SqliteStore exposes its connection factory as ``_connect``.
    connect = getattr(store, "_connect", None)
    if callable(connect):
        con = None
        try:
            con = connect()
            columns = {
                str(row[1])
                for row in con.execute("PRAGMA table_info(article_tags)").fetchall()
            }
            explanation_field = next(
                (field for field in ("motivering", "reasoning") if field in columns),
                None,
            )
            if explanation_field:
                rows = con.execute(
                    f"SELECT tag_id, {explanation_field} AS motivering "
                    "FROM article_tags WHERE article_id = ?",
                    (article_id,),
                ).fetchall()
                return collect(rows)
        except Exception as exc:
            logger.debug(
                "Could not load SQLite tag explanations for %s: %s",
                article_id,
                exc,
            )
        finally:
            if con is not None:
                con.close()

    # TinyDbStore exposes its database factory as ``_db``.
    db_factory = getattr(store, "_db", None)
    if callable(db_factory):
        tiny_db = None
        try:
            tiny_db = db_factory()
            rows = tiny_db.table("article_tags").all()
            return collect(
                row
                for row in rows
                if str(row.get("article_id") or "").strip() == article_id
            )
        except Exception as exc:
            logger.debug(
                "Could not load TinyDB tag explanations for %s: %s",
                article_id,
                exc,
            )
        finally:
            if tiny_db is not None:
                tiny_db.close()

    return {}


def _article_tags_with_motiveringar(store: Any, article_id: str) -> List[Dict[str, Any]]:
    """Return article tags enriched with an optional ``motivering`` field."""
    article_tags = store.get_article_tags(article_id) or []
    motiveringar = _article_tag_motiveringar(store, article_id)
    tags: List[Dict[str, Any]] = []
    for tag in article_tags:
        if not isinstance(tag, dict):
            continue
        item = dict(tag)
        motivering = _tag_motivering(item) or motiveringar.get(
            _tag_id_key(item.get("id")), ""
        )
        if motivering:
            item["motivering"] = motivering
        tags.append(item)
    return tags


def _coerce_positive_ts(v: Any) -> float:
    """
    Coerce timestamps to positive Unix seconds.
    Accept int/float/numeric string, including millisecond epochs.
    """
    if isinstance(v, bool):
        return 0.0
    try:
        fv = float(v)
    except Exception:
        return 0.0
    if fv <= 0:
        return 0.0
    # Millisecond epoch guard (e.g. 1775502985000 -> 1775502985).
    if fv > 10_000_000_000:
        fv = fv / 1000.0
    return fv


def _article_published_ts(a: Dict[str, Any]) -> float:
    return _coerce_positive_ts(a.get("published_ts"))


def _load_yaml_file(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _resolve_prompt_root(cfg: Dict[str, Any]) -> Path:
    prompt_cfg = cfg.get("prompts") or {}
    raw_path = DEFAULT_PROMPTS_PATH
    if isinstance(prompt_cfg, dict) and prompt_cfg.get("path"):
        raw_path = str(prompt_cfg["path"])
    return resolve_prompt_root(raw_path, base_config_path=APP_CONFIG_PATH or "config.yaml")


def _collect_ui_options(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read-only UI options for clients.
    - sources: feed/source names
    - topics: unique topic tags across feeds (if configured)
    - prompt_packages: package filenames in the configured prompt root
    """
    out = {"sources": [], "topics": [], "prompt_packages": []}

    # ---- Feeds: sources + topics ----
    feeds_path = None

    # common patterns in your config variants
    if isinstance(cfg.get("feeds"), dict) and cfg["feeds"].get("path"):
        feeds_path = str(cfg["feeds"]["path"])
    elif isinstance(cfg.get("feeds_path"), str) and cfg.get("feeds_path"):
        feeds_path = str(cfg["feeds_path"])
    else:
        # fallback: same dir as config.yaml
        feeds_path = str(
            (Path(APP_CONFIG_PATH).resolve().parent / "config" / "feeds.yaml").resolve()
        )

    sources = [option["name"] for option in build_source_options(cfg)]
    topics = build_topic_options(cfg)

    # ---- Prompts: directory-based packages (legacy single files also work) ----
    prompts_path = _resolve_prompt_root(cfg)
    try:
        out["prompt_packages"] = list_prompt_packages(prompts_path)
    except (FileNotFoundError, ValueError):
        out["prompt_packages"] = []

    out["sources"] = list(dict.fromkeys(sources))
    out["topics"] = topics

    out["feeds_path"] = feeds_path
    out["prompts_path"] = str(prompts_path)
    return out


def _worker_api_base(cfg: Dict[str, Any]) -> str:
    d = cfg.get("worker_api")
    if not isinstance(d, dict):
        d = {}
    host = str(d.get("host") or "127.0.0.1")
    port = int(d.get("port") or 8799)
    return f"http://{host}:{port}"


@app.route("/api/v1/job/resume", methods=["POST"])
def api_job_resume():
    body = request.get_json(silent=True) or {}
    jid = body.get("job_id")
    try:
        jid_i = int(jid)
    except Exception:
        return jsonify({"error": "missing_or_invalid_job_id"}), 400

    base = _worker_api_base(APP_CFG)
    try:
        r = requests.post(f"{base}/resume", json={"job_id": jid_i}, timeout=5)
        return (
            r.content,
            r.status_code,
            {"Content-Type": r.headers.get("Content-Type", "application/json")},
        )
    except Exception as e:
        return jsonify(
            {"error": "worker_unreachable", "detail": str(e), "worker": base}
        ), 503


@app.route("/api/v1/job/resume/<resume_id>", methods=["GET"])
def api_job_resume_status(resume_id: str):
    base = _worker_api_base(APP_CFG)
    try:
        r = requests.get(f"{base}/resume/{resume_id}", timeout=5)
        return (
            r.content,
            r.status_code,
            {"Content-Type": r.headers.get("Content-Type", "application/json")},
        )
    except Exception as e:
        return jsonify(
            {"error": "worker_unreachable", "detail": str(e), "worker": base}
        ), 503


@app.route("/api/v1/schedule/trigger", methods=["POST"])
def api_schedule_trigger():
    body = request.get_json(silent=True) or {}
    name = str(body.get("name") or "").strip()
    if not name:
        return jsonify({"error": "missing_name"}), 400

    base = _worker_api_base(APP_CFG)
    try:
        r = requests.post(f"{base}/trigger", json={"name": name}, timeout=5)
        return (
            r.content,
            r.status_code,
            {"Content-Type": r.headers.get("Content-Type", "application/json")},
        )
    except Exception as e:
        return jsonify(
            {"error": "worker_unreachable", "detail": str(e), "worker": base}
        ), 502


@app.route("/api/v1/schedule/trigger/<trigger_id>", methods=["GET"])
def api_schedule_trigger_status(trigger_id: str):
    base = _worker_api_base(APP_CFG)
    try:
        r = requests.get(f"{base}/trigger/{trigger_id}", timeout=5)
        return (
            r.content,
            r.status_code,
            {"Content-Type": r.headers.get("Content-Type", "application/json")},
        )
    except Exception as e:
        return jsonify(
            {"error": "worker_unreachable", "detail": str(e), "worker": base}
        ), 502


@app.route("/api/v1/summaries")
def api_summaries():
    """
    List summaries (like sidebar/list). Newest first.
    Query:
      limit= (default 200)
      topic= can be repeated or comma-separated
    """
    store = APP_STORE
    if store is None:
        abort(500)

    try:
        limit = int(request.args.get("limit", "200"))
    except Exception:
        limit = 200
    limit = max(1, min(limit, 2000))

    selected_topics = _selected_topics_from_request()
    docs = _list_enriched_summaries(store)
    docs = _filter_summaries_by_topics(docs, selected_topics)
    docs = docs[:limit]

    return jsonify(
        {
            "items": [_summary_list_item(d) for d in docs],
            "active_topics": selected_topics,
        }
    )


@app.route("/api/v1/summaries/latest")
def api_summaries_latest():
    """
    Latest summary doc (the same as default page redirect target).
    """
    store = APP_STORE
    if store is None:
        abort(500)

    latest = _get_latest_summary(store)
    if not isinstance(latest, dict):
        return jsonify({"item": None}), 404

    # fetch full doc if needed
    sid = str(latest.get("id") or "").strip()
    sdoc = store.get_summary_doc(sid) if sid else latest
    if not isinstance(sdoc, dict):
        return jsonify({"item": None}), 404

    return jsonify({"item": sdoc})


@app.route("/api/v1/summary/<summary_id>")
def api_summary(summary_id: str):
    """
    Full summary doc for reading/rendering.
    """
    store = APP_STORE
    if store is None:
        abort(500)

    sid = str(summary_id).strip()
    sdoc = store.get_summary_doc(sid)
    if not isinstance(sdoc, dict):
        abort(404)

    return jsonify({"item": sdoc})


@app.route("/api/v1/long-term/reports")
def api_long_term_reports():
    """List immutable threat-landscape reports for one configured profile."""

    _require_long_term_ui()
    store = APP_STORE
    if store is None:
        abort(500)

    profile_id, profiles = _selected_long_term_profile()
    try:
        limit = int(request.args.get("limit", "50"))
    except (TypeError, ValueError):
        limit = 50
    limit = max(1, min(limit, 200))

    if not profile_id:
        return jsonify(
            {
                "schema_version": 1,
                "items": [],
                "active_profile": None,
                "available_profiles": profiles,
            }
        )

    list_reports = getattr(store, "list_threat_landscape_reports", None)
    if not callable(list_reports):
        return jsonify({"error": "long_term_store_unavailable"}), 503
    try:
        reports = list_reports(profile_id, limit=limit) or []
    except Exception:
        logger.exception("Could not list long-term reports for %s", profile_id)
        return jsonify({"error": "long_term_store_unavailable"}), 503

    return jsonify(
        {
            "schema_version": 1,
            "items": [
                _long_term_report_list_item(row)
                for row in reports
                if isinstance(row, dict) and row.get("id")
            ],
            "active_profile": profile_id,
            "available_profiles": profiles,
        }
    )


@app.route("/api/v1/long-term/attack-pressure")
def api_long_term_attack_pressure():
    """Return a conservative TEF multiplier for a profile and exact CVE ID."""

    _require_long_term_ui()
    profile_id = str(request.args.get("profile") or "").strip()
    cve_id = valid_cve(request.args.get("cve") or "")
    if not profile_id or not cve_id:
        return jsonify({"error": "profile_and_valid_cve_required"}), 400
    profiles = (APP_CFG.get("long_term_analysis") or {}).get("profiles") or {}
    if profile_id not in profiles:
        return jsonify({"error": "unknown_profile"}), 404
    store = APP_STORE
    if store is None or not all(callable(getattr(store, name, None)) for name in (
        "list_threat_landscape_reports", "list_threat_clusters", "get_cluster_snapshot"
    )):
        return jsonify({"error": "long_term_store_unavailable"}), 503
    try:
        result = assess_attack_pressure(
            store, profile_id, cve_id, profiles[profile_id] or {}
        )
    except Exception:
        logger.exception("Could not assess attack pressure for %s", profile_id)
        return jsonify({"error": "long_term_store_unavailable"}), 503
    return jsonify(result)


@app.route("/api/v1/long-term/reports/<report_id>")
def api_long_term_report(report_id: str):
    """Return one immutable threat-landscape report."""

    _require_long_term_ui()
    store = APP_STORE
    if store is None:
        abort(500)
    get_report = getattr(store, "get_threat_landscape_report", None)
    if not callable(get_report):
        return jsonify({"error": "long_term_store_unavailable"}), 503
    try:
        report = get_report(str(report_id).strip())
    except Exception:
        logger.exception("Could not read long-term report %s", report_id)
        return jsonify({"error": "long_term_store_unavailable"}), 503
    if not isinstance(report, dict):
        abort(404)
    return jsonify({"schema_version": 1, "item": report})


@app.route("/api/v1/long-term/runs")
def api_long_term_runs():
    """List recent long-term worker runs for operational shadow review."""

    _require_long_term_ui()
    store = APP_STORE
    if store is None:
        abort(500)
    profile_id, profiles = _selected_long_term_profile()
    try:
        limit = int(request.args.get("limit", "20"))
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(limit, 200))
    if not profile_id:
        return jsonify(
            {
                "schema_version": 1,
                "items": [],
                "active_profile": None,
                "available_profiles": profiles,
            }
        )
    list_runs = getattr(store, "list_long_term_runs", None)
    if not callable(list_runs):
        return jsonify({"error": "long_term_store_unavailable"}), 503
    try:
        runs = list_runs(profile_id, limit=limit) or []
    except Exception:
        logger.exception("Could not list long-term runs for %s", profile_id)
        return jsonify({"error": "long_term_store_unavailable"}), 503
    return jsonify(
        {
            "schema_version": 1,
            "items": [
                _long_term_run_list_item(row)
                for row in runs
                if isinstance(row, dict) and row.get("id")
            ],
            "active_profile": profile_id,
            "available_profiles": profiles,
        }
    )


@app.route("/api/v1/long-term/snapshots/<snapshot_id>")
def api_long_term_snapshot(snapshot_id: str):
    """Return one immutable cluster snapshot."""

    _require_long_term_ui()
    store = APP_STORE
    if store is None:
        abort(500)
    get_snapshot = getattr(store, "get_cluster_snapshot", None)
    if not callable(get_snapshot):
        return jsonify({"error": "long_term_store_unavailable"}), 503
    try:
        snapshot = get_snapshot(str(snapshot_id).strip())
    except Exception:
        logger.exception("Could not read long-term snapshot %s", snapshot_id)
        return jsonify({"error": "long_term_store_unavailable"}), 503
    if not isinstance(snapshot, dict):
        abort(404)
    return jsonify({"schema_version": 1, "item": snapshot})


@app.route("/api/v1/long-term/clusters")
def api_long_term_clusters():
    """List review-safe threat clusters for one configured profile."""

    _require_long_term_ui()
    store = APP_STORE
    if store is None:
        abort(500)
    profile_id, profiles = _selected_long_term_profile()
    status, page, per_page, min_members, has_snapshot, selected_tags = (
        _long_term_cluster_list_parameters()
    )
    if not profile_id:
        return jsonify(
            {
                "schema_version": 1,
                "items": [],
                "active_profile": None,
                "available_profiles": profiles,
                "available_statuses": list(_LONG_TERM_CLUSTER_STATUSES),
                "active_status": status or "all",
                "page": page,
                "per_page": per_page,
                "min_members": min_members,
                "has_snapshot": has_snapshot,
                "selected_tags": selected_tags,
                "has_previous": False,
                "has_next": False,
            }
        )
    try:
        result = _long_term_cluster_page(
            store,
            profile_id,
            status=status,
            page=page,
            per_page=per_page,
            min_members=min_members,
            has_snapshot=has_snapshot,
            required_tags=selected_tags,
        )
    except Exception:
        logger.exception("Could not list long-term clusters for %s", profile_id)
        return jsonify({"error": "long_term_store_unavailable"}), 503
    return jsonify(
        {
            "schema_version": 1,
            **result,
            "active_profile": profile_id,
            "available_profiles": profiles,
            "available_statuses": list(_LONG_TERM_CLUSTER_STATUSES),
            "active_status": status or "all",
            "min_members": min_members,
            "has_snapshot": has_snapshot,
            "selected_tags": selected_tags,
        }
    )


@app.route("/api/v1/long-term/clusters/<cluster_id>")
def api_long_term_cluster(cluster_id: str):
    """Return a review-safe cluster with snapshots and retained memberships."""

    _require_long_term_ui()
    store = APP_STORE
    if store is None:
        abort(500)
    try:
        membership_limit = int(request.args.get("membership_limit", "500"))
    except (TypeError, ValueError):
        membership_limit = 500
    membership_limit = max(1, min(membership_limit, 2000))
    try:
        bundle = _long_term_cluster_bundle(
            store,
            str(cluster_id).strip(),
            membership_limit=membership_limit,
        )
    except Exception:
        logger.exception("Could not read long-term cluster %s", cluster_id)
        return jsonify({"error": "long_term_store_unavailable"}), 503
    if bundle is None:
        abort(404)
    try:
        get_cluster = getattr(store, "get_threat_cluster", None)
        source_cluster = get_cluster(str(cluster_id).strip()) if callable(get_cluster) else None
        comparison_candidates = _long_term_cluster_comparison_data(
            store, source_cluster if isinstance(source_cluster, dict) else bundle["cluster"]
        )
    except Exception:
        logger.exception("Could not compare long-term cluster %s", cluster_id)
        return jsonify({"error": "long_term_store_unavailable"}), 503
    return jsonify(
        {
            "schema_version": 1,
            "item": bundle["cluster"],
            "memberships": bundle["memberships"],
            "snapshots": bundle["snapshots"],
            "memberships_truncated": bundle["memberships_truncated"],
            "comparison_candidates": comparison_candidates,
        }
    )


@app.route("/api/v1/long-term/clusters/<cluster_id>/tag-network")
def api_long_term_cluster_tag_network(cluster_id: str):
    """Return the reachable one-tag-per-hop hierarchy for one cluster."""

    _require_long_term_ui()
    store = APP_STORE
    if store is None:
        abort(500)
    try:
        up_levels, down_levels, max_levels, max_nodes, candidate_limit = (
            _long_term_tag_network_options()
        )
    except ValueError as exc:
        return jsonify({"error": "invalid_query", "detail": str(exc)}), 400
    try:
        graph = _long_term_cluster_tag_network(
            store,
            str(cluster_id).strip(),
            up_levels=up_levels,
            down_levels=down_levels,
            max_nodes=max_nodes,
            candidate_limit=candidate_limit,
            excluded_tags=request.args.getlist("exclude"),
        )
    except Exception:
        logger.exception("Could not build tag network for cluster %s", cluster_id)
        return jsonify({"error": "long_term_store_unavailable"}), 503
    if graph is None:
        abort(404)
    return jsonify(
        {
            "schema_version": 1,
            "max_levels": max_levels,
            "max_nodes": max_nodes,
            **graph,
        }
    )


@app.route("/api/v1/long-term/clusters/<cluster_id>/article-network")
def api_long_term_cluster_article_network(cluster_id: str):
    """Return clusters connected by shared retained article memberships."""

    _require_long_term_ui()
    store = APP_STORE
    if store is None:
        abort(500)
    try:
        min_overlap_percent, max_neighbors, max_neighbors_limit, membership_limit = (
            _long_term_article_network_options()
        )
    except ValueError as exc:
        return jsonify({"error": "invalid_query", "detail": str(exc)}), 400
    try:
        graph = _long_term_cluster_article_network(
            store,
            str(cluster_id).strip(),
            min_overlap_percent=min_overlap_percent,
            max_neighbors=max_neighbors,
            membership_limit=membership_limit,
        )
    except Exception:
        logger.exception("Could not build article network for cluster %s", cluster_id)
        return jsonify({"error": "long_term_store_unavailable"}), 503
    if graph is None:
        abort(404)
    return jsonify(
        {
            "schema_version": 1,
            "max_neighbors_limit": max_neighbors_limit,
            **graph,
        }
    )


@app.route("/api/v1/long-term/reviews/<cluster_id>")
def api_long_term_cluster_review(cluster_id: str):
    """Return one ambiguous cluster and its two review candidates."""

    _require_long_term_ui()
    store = APP_STORE
    if store is None:
        abort(500)
    try:
        bundle = _long_term_review_bundle(store, str(cluster_id).strip())
    except Exception:
        logger.exception("Could not build long-term review %s", cluster_id)
        return jsonify({"error": "long_term_store_unavailable"}), 503
    if bundle is None:
        abort(404)
    return jsonify(
        {
            "schema_version": 1,
            "item": bundle["cluster"],
            "label": bundle["cluster_label"],
            "assignment": bundle["assignment"],
            "candidates": bundle["candidates"],
            "candidate_source": bundle["candidate_source"],
            "review_actions_enabled": _long_term_review_actions_enabled(),
        }
    )


@app.route("/api/v1/long-term/clusters/<cluster_id>/review", methods=["POST"])
def resolve_long_term_cluster_review(cluster_id: str):
    """Apply one explicit, audited decision to an ambiguous cluster."""

    _require_long_term_review_actions()
    store = APP_STORE
    if store is None:
        abort(500)
    values = request.get_json(silent=True) if request.is_json else request.form
    values = values if isinstance(values, dict) or hasattr(values, "get") else {}
    decision = str(values.get("decision") or "").strip()
    reviewer = " ".join(str(values.get("reviewer") or "").split())
    comment = str(values.get("comment") or "").strip()
    target_cluster_id = str(values.get("target_cluster_id") or "").strip()
    merge_cluster_ids = (
        values.getlist("merge_cluster_ids")
        if hasattr(values, "getlist")
        else values.get("merge_cluster_ids", [])
    )
    if not isinstance(merge_cluster_ids, list):
        return jsonify({"error": "invalid_merge_cluster_ids"}), 400
    merge_cluster_ids = [str(value).strip() for value in merge_cluster_ids]
    try:
        expected_revision = int(values.get("expected_membership_revision"))
    except (TypeError, ValueError):
        return jsonify({"error": "expected_membership_revision_required"}), 400
    if decision not in {"keep_separate", "exclude", "merge"}:
        return jsonify({"error": "invalid_review_decision"}), 400
    if not reviewer:
        return jsonify({"error": "reviewer_required"}), 400

    cluster_id = str(cluster_id).strip()
    try:
        bundle = _long_term_review_bundle(store, cluster_id)
        get_cluster = getattr(store, "get_threat_cluster", None)
        cluster = get_cluster(cluster_id) if callable(get_cluster) else None
        if bundle is None and decision == "merge" and isinstance(cluster, dict):
            comparison_candidates = _long_term_cluster_comparison_data(store, cluster)
            bundle = {
                "cluster": cluster,
                "candidates": comparison_candidates,
            }
    except Exception:
        logger.exception("Could not load long-term review %s", cluster_id)
        return jsonify({"error": "long_term_store_unavailable"}), 503
    if bundle is None or not isinstance(cluster, dict):
        abort(404)
    if int(cluster.get("membership_revision") or 0) != expected_revision:
        return jsonify({"error": "cluster_revision_conflict"}), 409
    reviewed_at = int(time.time())

    try:
        if decision == "merge":
            allowed_ids = {
                str(row["id"])
                for row in bundle["candidates"]
                if row.get("merge_allowed")
            }
            if merge_cluster_ids and target_cluster_id:
                return jsonify({"error": "ambiguous_merge_targets"}), 400
            is_open_cluster = str(cluster.get("status") or "") in {"active", "dormant"}
            if merge_cluster_ids or (not target_cluster_id and is_open_cluster):
                if not is_open_cluster:
                    return jsonify({"error": "cluster_not_open_for_merge"}), 409
                if (
                    not merge_cluster_ids
                    or len(set(merge_cluster_ids)) != len(merge_cluster_ids)
                    or any(
                        candidate_id not in allowed_ids
                        for candidate_id in merge_cluster_ids
                    )
                ):
                    return jsonify({"error": "invalid_merge_cluster_ids"}), 400
                list_memberships = getattr(store, "list_cluster_memberships", None)
                apply_reconciliation = getattr(store, "apply_cluster_reconciliation", None)
                if not callable(list_memberships) or not callable(apply_reconciliation):
                    return jsonify({"error": "long_term_store_unavailable"}), 503
                selected_clusters = [cluster]
                for candidate_id in merge_cluster_ids:
                    candidate = get_cluster(candidate_id)
                    if not isinstance(candidate, dict):
                        return jsonify({"error": "review_target_not_found"}), 404
                    if (
                        str(candidate.get("status") or "") not in {"active", "dormant"}
                        or candidate.get("superseded_by_cluster_id")
                    ):
                        return jsonify({"error": "cluster_not_open_for_merge"}), 409
                    selected_clusters.append(candidate)
                identity = {
                    "clusters": sorted(
                        [str(row["id"]), int(row.get("membership_revision") or 0)]
                        for row in selected_clusters
                    ),
                    "primary_cluster_id": cluster_id,
                    "profile_id": str(cluster.get("profile_id") or ""),
                }
                reconciliation_id = "threat_manual_" + hashlib.sha256(
                    json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()[:32]
                operation = build_cluster_merge_operation(
                    selected_clusters,
                    {
                        str(row["id"]): list_memberships(str(row["id"]))
                        for row in selected_clusters
                    },
                    primary_cluster_id=cluster_id,
                    reconciliation_id=reconciliation_id,
                    reconciled_at=reviewed_at,
                )
                for tombstone in operation["superseded_clusters"]:
                    tombstone.update(
                        {
                            "review_decision": "merge",
                            "reviewed_at": reviewed_at,
                            "reviewed_by": reviewer,
                            "review_comment": " ".join(comment.split()) or None,
                            "reviewed_target_cluster_id": cluster_id,
                        }
                    )
                operation["manual_merge"] = {
                    "primary_cluster_id": cluster_id,
                    "merged_cluster_ids": sorted(merge_cluster_ids),
                    "reviewed_at": reviewed_at,
                    "reviewed_by": reviewer,
                    "review_comment": " ".join(comment.split()) or None,
                }
                if not apply_reconciliation(operation):
                    return jsonify({"error": "cluster_revision_conflict"}), 409
                result = {
                    "decision": decision,
                    "cluster_id": cluster_id,
                    "merged_cluster_ids": sorted(merge_cluster_ids),
                    "reconciliation_id": operation["id"],
                }
                redirect_cluster_id = cluster_id
            else:
                if target_cluster_id not in allowed_ids:
                    return jsonify({"error": "invalid_review_target"}), 400
                target = get_cluster(target_cluster_id)
                if not isinstance(target, dict):
                    return jsonify({"error": "review_target_not_found"}), 404
                list_memberships = getattr(store, "list_cluster_memberships", None)
                apply_reconciliation = getattr(store, "apply_cluster_reconciliation", None)
                if not callable(list_memberships) or not callable(apply_reconciliation):
                    return jsonify({"error": "long_term_store_unavailable"}), 503
                if str(cluster.get("status") or "") == "needs_review":
                    operation = build_cluster_review_merge_operation(
                        cluster,
                        target,
                        {
                            cluster_id: list_memberships(cluster_id),
                            target_cluster_id: list_memberships(target_cluster_id),
                        },
                        allowed_candidate_cluster_ids=allowed_ids,
                        reviewed_at=reviewed_at,
                        reviewed_by=reviewer,
                        comment=comment,
                    )
                else:
                    if str(cluster.get("status") or "") not in {"active", "dormant"}:
                        return jsonify({"error": "cluster_not_open_for_merge"}), 409
                    identity = {
                        "source": [cluster_id, int(cluster.get("membership_revision") or 0)],
                        "target": [target_cluster_id, int(target.get("membership_revision") or 0)],
                        "profile_id": str(cluster.get("profile_id") or ""),
                    }
                    reconciliation_id = "threat_manual_" + hashlib.sha256(
                        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest()[:32]
                    operation = build_cluster_merge_operation(
                        [cluster, target],
                        {
                            cluster_id: list_memberships(cluster_id),
                            target_cluster_id: list_memberships(target_cluster_id),
                        },
                        primary_cluster_id=target_cluster_id,
                        reconciliation_id=reconciliation_id,
                        reconciled_at=reviewed_at,
                    )
                    for tombstone in operation["superseded_clusters"]:
                        if str(tombstone.get("id") or "") == cluster_id:
                            tombstone.update(
                                {
                                    "review_decision": "merge",
                                    "reviewed_at": reviewed_at,
                                    "reviewed_by": reviewer,
                                    "review_comment": " ".join(comment.split()) or None,
                                    "reviewed_target_cluster_id": target_cluster_id,
                                }
                            )
                    operation["review_resolution"] = {
                        "review_cluster_id": cluster_id,
                        "target_cluster_id": target_cluster_id,
                        "review_decision": "merge",
                        "reviewed_at": reviewed_at,
                        "reviewed_by": reviewer,
                        "review_comment": " ".join(comment.split()) or None,
                        "reviewed_target_cluster_id": target_cluster_id,
                    }
                if not apply_reconciliation(operation):
                    return jsonify({"error": "cluster_revision_conflict"}), 409
                result = {
                    "decision": decision,
                    "cluster_id": cluster_id,
                    "target_cluster_id": target_cluster_id,
                    "reconciliation_id": operation["id"],
                }
                redirect_cluster_id = cluster_id
        else:
            resolve_review = getattr(store, "resolve_threat_cluster_review", None)
            if not callable(resolve_review):
                return jsonify({"error": "long_term_store_unavailable"}), 503
            resolution = build_cluster_review_resolution(
                cluster,
                decision=decision,
                reviewed_at=reviewed_at,
                reviewed_by=reviewer,
                comment=comment,
            )
            if not resolve_review(
                resolution,
                expected_membership_revision=expected_revision,
            ):
                return jsonify({"error": "cluster_revision_conflict"}), 409
            result = {
                "decision": decision,
                "cluster_id": cluster_id,
                "status": resolution["status"],
            }
            redirect_cluster_id = cluster_id
    except (TypeError, ValueError) as exc:
        return jsonify({"error": "invalid_review", "detail": str(exc)}), 400
    except Exception:
        logger.exception("Could not resolve long-term review %s", cluster_id)
        return jsonify({"error": "long_term_store_unavailable"}), 503

    if request.is_json:
        return jsonify({"schema_version": 1, "item": result})
    return redirect(url_for("view_long_term_cluster", cluster_id=redirect_cluster_id))


@app.route("/api/v1/long-term/clusters/<cluster_id>/members", methods=["POST"])
def edit_long_term_cluster_membership(cluster_id: str):
    """Move an article here or split one member into a separate cluster."""

    _require_long_term_review_actions()
    store = APP_STORE
    if store is None:
        abort(500)
    values = request.get_json(silent=True) if request.is_json else request.form
    values = values if isinstance(values, dict) or hasattr(values, "get") else {}
    action = str(values.get("action") or "").strip()
    article_id = str(values.get("article_id") or "").strip()
    reviewer = " ".join(str(values.get("reviewer") or "").split())
    comment = str(values.get("comment") or "").strip()
    try:
        expected_revision = int(values.get("expected_membership_revision"))
    except (TypeError, ValueError):
        return jsonify({"error": "expected_membership_revision_required"}), 400
    if action not in {"move_here", "split_out"}:
        return jsonify({"error": "invalid_membership_action"}), 400
    if not article_id:
        return jsonify({"error": "article_id_required"}), 400
    if not reviewer:
        return jsonify({"error": "reviewer_required"}), 400

    cluster_id = str(cluster_id).strip()
    try:
        get_cluster = getattr(store, "get_threat_cluster", None)
        get_membership = getattr(store, "get_cluster_membership", None)
        list_memberships = getattr(store, "list_cluster_memberships", None)
        get_articles = getattr(store, "get_articles_by_ids", None)
        apply_edit = getattr(store, "apply_cluster_membership_edit", None)
        if not all(
            callable(method)
            for method in (
                get_cluster,
                get_membership,
                list_memberships,
                get_articles,
                apply_edit,
            )
        ):
            return jsonify({"error": "long_term_store_unavailable"}), 503
        selected = get_cluster(cluster_id)
        if not isinstance(selected, dict):
            abort(404)
        if int(selected.get("membership_revision") or 0) != expected_revision:
            return jsonify({"error": "cluster_revision_conflict"}), 409
        profile_id = str(selected.get("profile_id") or "")
        list_article_memberships = getattr(store, "list_article_cluster_memberships", None)
        if callable(list_article_memberships):
            article_memberships = list_article_memberships(profile_id, article_id)
            selected_scope = str(selected.get("tag_scope") or "")
            current_membership = next(
                (
                    row for row in article_memberships
                    if (str(row.get("cluster_id") or "") == cluster_id
                        if action == "split_out"
                        else str(row.get("tag_scope") or "") == selected_scope)
                ), None,
            )
        else:
            current_membership = get_membership(profile_id, article_id)
        if not isinstance(current_membership, dict):
            return jsonify({"error": "article_has_no_cluster_membership"}), 404
        source_id = str(current_membership.get("cluster_id") or "")
        source = get_cluster(source_id)
        if not isinstance(source, dict):
            return jsonify({"error": "source_cluster_not_found"}), 404
        articles = get_articles([article_id]) or []
        article = next(
            (
                row
                for row in articles
                if isinstance(row, dict) and str(row.get("id") or "") == article_id
            ),
            None,
        )
        if article is None:
            return jsonify({"error": "article_embedding_unavailable"}), 409
        if action == "move_here":
            target = selected
            if source_id == cluster_id:
                return jsonify({"error": "article_already_in_target_cluster"}), 409
        else:
            if source_id != cluster_id:
                return jsonify({"error": "article_not_in_source_cluster"}), 409
            target = None
        operation = build_cluster_membership_edit(
            source,
            list_memberships(source_id, limit=100000),
            target_cluster=target,
            target_memberships=(
                list_memberships(cluster_id, limit=100000) if target is not None else ()
            ),
            article=article,
            article_id=article_id,
            edited_at=int(time.time()),
            edited_by=reviewer,
            comment=comment,
        )
        if not apply_edit(operation):
            return jsonify({"error": "cluster_revision_conflict"}), 409
    except (TypeError, ValueError) as exc:
        return jsonify({"error": "invalid_membership_edit", "detail": str(exc)}), 400
    except Exception:
        logger.exception("Could not edit membership %s for cluster %s", article_id, cluster_id)
        return jsonify({"error": "long_term_store_unavailable"}), 503

    result = {
        "edit_id": operation["id"],
        "action": operation["action"],
        "article_id": article_id,
        "source_cluster_id": operation["source_cluster_id"],
        "target_cluster_id": operation["target_cluster_id"],
    }
    if request.is_json:
        return jsonify({"schema_version": 1, "item": result})
    return redirect(
        url_for("view_long_term_cluster", cluster_id=operation["target_cluster_id"])
    )


@app.route("/api/v1/articles")
def api_articles():
    """
    List articles (like /articles page).
    Query:
      limit= (default 300, max 5000)
    """
    store = APP_STORE
    if store is None:
        abort(500)

    try:
        limit = int(request.args.get("limit", "300"))
    except Exception:
        limit = 300
    limit = max(1, min(limit, 5000))

    raw = store.list_articles() or []
    articles = [a for a in raw if isinstance(a, dict) and a.get("id")]
    articles.sort(key=_article_published_ts, reverse=True)
    return jsonify({"items": [_article_list_item(a) for a in articles[:limit]]})


@app.route("/api/v1/article/<article_id>")
def api_article(article_id: str):
    """
    Full article doc for reading.
    """
    store = APP_STORE
    if store is None:
        abort(500)

    a = store.get_article(str(article_id))
    if not isinstance(a, dict):
        abort(404)

    # Add tags to the response
    tags = []
    try:
        article_tags = _article_tags_with_motiveringar(store, article_id)
        for t in article_tags:
            if isinstance(t, dict):
                tags.append({
                    "id": t.get("id"),
                    "name": t.get("name", ""),
                    "category": t.get("category", "GENERAL"),
                    "motivering": _tag_motivering(t),
                })
    except Exception as e:
        logger.debug(f"Error loading tags for article {article_id}: {e}")
    
    a["tags"] = tags
    return jsonify({"item": a})


@app.route("/rss/summaries.xml")
def rss_summaries():
    """RSS 2.0 feed containing the latest published summaries."""
    store = APP_STORE
    if store is None:
        abort(500)

    docs = store.list_summary_docs() or []
    docs = [doc for doc in docs if isinstance(doc, dict) and doc.get("id")]
    docs.sort(key=lambda doc: _coerce_positive_ts(doc.get("created")), reverse=True)

    items: List[Dict[str, str]] = []
    for listed_doc in docs[: _rss_limit()]:
        summary_id = str(listed_doc.get("id") or "").strip()
        try:
            full_doc = store.get_summary_doc(summary_id)
        except Exception:
            full_doc = None
        doc = full_doc if isinstance(full_doc, dict) else listed_doc
        item_url = url_for(
            "view_summary", summary_id=summary_id, _external=True
        )
        created = _coerce_positive_ts(doc.get("created"))
        title = str(doc.get("title") or "").strip()
        if not title:
            title = (
                f"Sammanfattning {format_ts(int(created))}"
                if created
                else summary_id
            )
        summary_text = str(
            doc.get("summary") or doc.get("proofread_published_summary") or ""
        ).strip()
        items.append(
            {
                "title": title,
                "link": item_url,
                "guid": item_url,
                "pub_date": _rss_pub_date(created),
                "description": _md_to_html(summary_text),
            }
        )

    return _rss_response(
        title="FeedSummary – sammanfattningar",
        link=url_for("index", _external=True),
        description="De senaste publicerade sammanfattningarna från FeedSummary.",
        self_url=url_for("rss_summaries", _external=True),
        items=items,
    )


@app.route("/rss/articles.xml")
def rss_articles():
    """RSS 2.0 feed containing the latest collected articles."""
    store = APP_STORE
    if store is None:
        abort(500)

    limit = _rss_limit()
    try:
        raw_articles = store.list_articles(limit=limit) or []
    except TypeError:
        raw_articles = (store.list_articles() or [])[:limit]
    articles = [
        article
        for article in raw_articles
        if isinstance(article, dict) and article.get("id")
    ]
    articles.sort(key=_article_published_ts, reverse=True)

    items: List[Dict[str, str]] = []
    for article in articles[:limit]:
        article_id = str(article.get("id") or "").strip()
        viewer_url = url_for("view_article", article_id=article_id, _external=True)
        original_url = str(article.get("url") or "").strip()
        published = _article_published_ts(article) or _coerce_positive_ts(
            article.get("fetched_at")
        )
        text = str(article.get("text") or article.get("preview") or "").strip()
        source = str(article.get("source") or "").strip()
        description = f"Källa: {source}\n\n{text}" if source else text
        items.append(
            {
                "title": str(article.get("title") or article_id),
                "link": original_url or viewer_url,
                "guid": viewer_url,
                "pub_date": _rss_pub_date(published),
                "description": description,
            }
        )

    return _rss_response(
        title="FeedSummary – artiklar",
        link=url_for("list_articles", _external=True),
        description="De senaste insamlade artiklarna från FeedSummary.",
        self_url=url_for("rss_articles", _external=True),
        items=items,
    )


@app.route("/api/v1/pages/source")
def api_page_source():
    return jsonify({"markdown": _load_static_md("source.md")})


@app.route("/api/v1/pages/license")
def api_page_license():
    return jsonify({"markdown": _load_static_md("license.md")})


@app.route("/api/v1/prompt/<name>")
def api_prompt_package(name: str):
    """
    Return the YAML content for one package from the configured prompt root.
    Read-only.
    """
    pkg = str(name or "").strip()
    if not pkg:
        abort(404)

    prompts_path = _resolve_prompt_root(APP_CFG)
    if not prompts_path.exists():
        return jsonify(
            {"error": "prompt_path_not_found", "prompts_path": str(prompts_path)}
        ), 404

    try:
        item = load_prompt_package(prompts_path, pkg)
    except KeyError:
        return jsonify(
            {"error": "prompt_not_found", "name": pkg, "prompts_path": str(prompts_path)}
        ), 404

    # Dump only that package as YAML (nice for display/copy)
    one = {pkg: item}
    yaml_text = yaml.safe_dump(one, sort_keys=False, allow_unicode=True)

    return jsonify(
        {
            "name": pkg,
            "prompts_path": str(prompts_path),
            "yaml": yaml_text,
            "item": item,  # also return as JSON if you want programmatic use
        }
    )


@app.route("/api/v1/ui_options")
def api_ui_options():
    """
    Read-only options used by clients (Qt remote):
      {
        sources: [...],
        topics: [...],
        prompt_packages: [...],
        feeds_path: "...",
        prompts_path: "..."
      }
    """
    return jsonify(_collect_ui_options(APP_CFG))


@app.route("/")
def index():
    store = APP_STORE
    if store is None:
        abort(500)

    selected_topics = _selected_topics_from_request()
    docs = _list_enriched_summaries(store)
    all_topics = _all_topics_from_summaries(docs)
    filtered_docs = _filter_summaries_by_topics(docs, selected_topics)
    sidebar_docs = _filter_summaries_today(filtered_docs)

    latest = sidebar_docs[0] if sidebar_docs else None
    if not isinstance(latest, dict):
        msg = (
            "<p>Inga summaries matchar valt topic-filter.</p>"
            if selected_topics
            else "<p>Inga summaries ännu. Använd knappen Lista för historik.</p>"
        )
        return render_template(
            "index.html",
            summary=None,
            html=msg,
            summaries=sidebar_docs,
            default_selected=None,
            available_topics=all_topics,
            active_topics=selected_topics,
            format_ts=format_ts,
        )

    sid = str(latest.get("id") or "")
    return redirect(url_for("view_summary", summary_id=sid, topic=selected_topics))


@app.route("/summaries")
def list_summaries():
    store = APP_STORE
    if store is None:
        abort(500)

    selected_topics = _selected_topics_from_request()
    docs = _list_enriched_summaries(store)
    all_topics = _all_topics_from_summaries(docs)
    filtered_docs = _filter_summaries_by_topics(docs, selected_topics)
    return render_template(
        "summaries.html",
        summaries=filtered_docs,
        available_topics=all_topics,
        active_topics=selected_topics,
        format_ts=format_ts,
    )


@app.route("/long-term")
def list_long_term_reports():
    _require_long_term_ui()
    store = APP_STORE
    if store is None:
        abort(500)

    selected = str(request.args.get("profile") or "").strip()
    profiles = _long_term_profile_ids()
    if selected and selected not in profiles:
        abort(404)
    visible_profiles = [selected] if selected else profiles
    list_reports = getattr(store, "list_threat_landscape_reports", None)
    if visible_profiles and not callable(list_reports):
        abort(503)
    list_runs = getattr(store, "list_long_term_runs", None)
    groups: List[Dict[str, Any]] = []
    for profile_id in visible_profiles:
        try:
            reports = [
                _long_term_report_list_item(row)
                for row in (list_reports(profile_id, limit=100) or [])
                if isinstance(row, dict) and row.get("id")
            ]
        except Exception:
            logger.exception("Could not list long-term reports for %s", profile_id)
            abort(503)
        runs: List[Dict[str, Any]] = []
        if callable(list_runs):
            try:
                runs = [
                    _long_term_run_list_item(row)
                    for row in (list_runs(profile_id, limit=10) or [])
                    if isinstance(row, dict) and row.get("id")
                ]
            except Exception:
                logger.exception("Could not list long-term runs for %s", profile_id)
        groups.append({
            "profile_id": profile_id,
            "label": _long_term_profile_label(profile_id),
            "is_aggregate": bool(
                (APP_CFG.get("long_term_analysis") or {})
                .get("profiles", {}).get(profile_id, {}).get("aggregate_profiles")
            ),
            "reports": reports,
            "runs": runs,
        })

    return render_template(
        "long_term_reports.html",
        groups=groups,
        available_profiles=profiles,
        format_ts=format_ts,
    )


@app.route("/long-term/report/<report_id>")
def view_long_term_report(report_id: str):
    _require_long_term_ui()
    store = APP_STORE
    if store is None:
        abort(500)
    get_report = getattr(store, "get_threat_landscape_report", None)
    if not callable(get_report):
        abort(503)
    try:
        report = get_report(str(report_id).strip())
    except Exception:
        logger.exception("Could not read long-term report %s", report_id)
        abort(503)
    if not isinstance(report, dict):
        abort(404)
    try:
        snapshot_labels, cluster_labels = _long_term_report_link_labels(store, report)
    except Exception:
        logger.exception("Could not resolve long-term report labels %s", report_id)
        snapshot_labels, cluster_labels = {}, {}

    return render_template(
        "long_term_report.html",
        report=report,
        profile_label=_long_term_profile_label(str(report.get("profile_id") or "")),
        report_item=_long_term_report_list_item(report),
        analysis=(report.get("analysis") or {}),
        forecasts=(report.get("forecast") or []),
        quality=(report.get("quality") or {}),
        snapshot_labels=snapshot_labels,
        cluster_labels=cluster_labels,
        format_ts=format_ts,
    )


@app.route("/long-term/snapshot/<snapshot_id>")
def view_long_term_snapshot(snapshot_id: str):
    _require_long_term_ui()
    store = APP_STORE
    if store is None:
        abort(500)
    get_snapshot = getattr(store, "get_cluster_snapshot", None)
    if not callable(get_snapshot):
        abort(503)
    try:
        snapshot = get_snapshot(str(snapshot_id).strip())
    except Exception:
        logger.exception("Could not read long-term snapshot %s", snapshot_id)
        abort(503)
    if not isinstance(snapshot, dict):
        abort(404)
    payload = snapshot.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {}
    try:
        article_labels = _long_term_article_labels(store, snapshot)
    except Exception:
        logger.exception("Could not resolve long-term article labels %s", snapshot_id)
        article_labels = {}
    return render_template(
        "long_term_snapshot.html",
        snapshot=snapshot,
        payload=payload,
        snapshot_label=_long_term_snapshot_label(snapshot),
        cluster_label=_long_term_snapshot_label(snapshot),
        article_labels=article_labels,
        format_ts=format_ts,
    )


@app.route("/long-term/clusters")
def list_long_term_clusters():
    _require_long_term_ui()
    store = APP_STORE
    if store is None:
        abort(500)
    selected = str(request.args.get("profile") or "").strip()
    profiles = _long_term_profile_ids()
    if selected and selected not in profiles:
        abort(404)
    visible_profiles = [selected] if selected else profiles
    status, page, per_page, min_members, has_snapshot, selected_tags = (
        _long_term_cluster_list_parameters()
    )
    available_cluster_tags = _long_term_cluster_tag_options(store, selected_tags)
    groups: List[Dict[str, Any]] = []
    for profile_id in visible_profiles:
        try:
            result = _long_term_cluster_ui_page(
                store,
                profile_id,
                status=status,
                page=page,
                per_page=per_page,
                min_members=min_members,
                has_snapshot=has_snapshot,
                required_tags=selected_tags,
            )
        except Exception:
            logger.exception("Could not list long-term clusters for %s", profile_id)
            abort(503)
        groups.append({
            "profile_id": profile_id,
            "label": _long_term_profile_label(profile_id),
            "clusters": result["items"],
            "has_next": result["has_next"],
            "replacement_profile": _long_term_replacement_profile(profile_id),
        })
    return render_template(
        "long_term_clusters.html",
        groups=groups,
        selected_profile=selected,
        available_profiles=profiles,
        available_statuses=_LONG_TERM_CLUSTER_STATUSES,
        active_status=status or "all",
        page=page,
        per_page=per_page,
        min_members=min_members,
        has_snapshot=has_snapshot,
        selected_cluster_tags=selected_tags,
        available_cluster_tags=available_cluster_tags,
        has_previous=page > 1,
        has_next=any(group["has_next"] for group in groups),
        format_ts=format_ts,
    )


@app.route("/long-term/review/<cluster_id>")
def review_long_term_cluster(cluster_id: str):
    _require_long_term_ui()
    store = APP_STORE
    if store is None:
        abort(500)
    try:
        bundle = _long_term_review_bundle(store, str(cluster_id).strip())
    except Exception:
        logger.exception("Could not build long-term review %s", cluster_id)
        abort(503)
    if bundle is None:
        abort(404)
    return render_template(
        "long_term_review.html",
        **bundle,
        format_ts=format_ts,
    )


@app.route("/long-term/cluster/<cluster_id>")
def view_long_term_cluster(cluster_id: str):
    _require_long_term_ui()
    store = APP_STORE
    if store is None:
        abort(500)
    try:
        bundle = _long_term_cluster_bundle(
            store,
            str(cluster_id).strip(),
            membership_limit=500,
        )
    except Exception:
        logger.exception("Could not read long-term cluster %s", cluster_id)
        abort(503)
    if bundle is None:
        abort(404)
    try:
        get_cluster = getattr(store, "get_threat_cluster", None)
        source_cluster = get_cluster(str(cluster_id).strip()) if callable(get_cluster) else None
        bundle["comparison_candidates"] = _long_term_cluster_comparison_data(
            store, source_cluster if isinstance(source_cluster, dict) else bundle["cluster"]
        )
        list_article_memberships = getattr(store, "list_article_cluster_memberships", None)
        related_ids: set[str] = set()
        if callable(list_article_memberships):
            for membership in bundle["memberships"]:
                for related_membership in list_article_memberships(
                    bundle["cluster"]["profile_id"], membership["article_id"]
                ) or []:
                    related_id = str(related_membership.get("cluster_id") or "")
                    if related_id and related_id != cluster_id:
                        related_ids.add(related_id)
        bundle["related_clusters"] = [
            _long_term_cluster_list_item(store, row)
            for related_id in sorted(related_ids)
            if isinstance((row := get_cluster(related_id)), dict)
        ] if callable(get_cluster) else []
    except Exception:
        logger.exception("Could not compare long-term cluster %s", cluster_id)
        abort(503)
    return render_template(
        "long_term_cluster.html",
        **bundle,
        format_ts=format_ts,
    )


@app.route("/long-term/cluster/<cluster_id>/tag-network")
def view_long_term_cluster_tag_network(cluster_id: str):
    _require_long_term_ui()
    store = APP_STORE
    if store is None:
        abort(500)
    try:
        up_levels, down_levels, max_levels, max_nodes, candidate_limit = (
            _long_term_tag_network_options()
        )
    except ValueError:
        abort(400)
    try:
        graph = _long_term_cluster_tag_network(
            store,
            str(cluster_id).strip(),
            up_levels=up_levels,
            down_levels=down_levels,
            max_nodes=max_nodes,
            candidate_limit=candidate_limit,
            excluded_tags=request.args.getlist("exclude"),
        )
    except Exception:
        logger.exception("Could not build tag network for cluster %s", cluster_id)
        abort(503)
    if graph is None:
        abort(404)
    base_node = next(node for node in graph["nodes"] if node["is_base"])
    return render_template(
        "long_term_cluster_tag_network.html",
        graph=graph,
        base_node=base_node,
        max_levels=max_levels,
        max_nodes=max_nodes,
    )


@app.route("/long-term/cluster/<cluster_id>/article-network")
def view_long_term_cluster_article_network(cluster_id: str):
    _require_long_term_ui()
    store = APP_STORE
    if store is None:
        abort(500)
    try:
        min_overlap_percent, max_neighbors, max_neighbors_limit, membership_limit = (
            _long_term_article_network_options()
        )
    except ValueError:
        abort(400)
    try:
        graph = _long_term_cluster_article_network(
            store,
            str(cluster_id).strip(),
            min_overlap_percent=min_overlap_percent,
            max_neighbors=max_neighbors,
            membership_limit=membership_limit,
        )
    except Exception:
        logger.exception("Could not build article network for cluster %s", cluster_id)
        abort(503)
    if graph is None:
        abort(404)
    return render_template(
        "long_term_cluster_article_network.html",
        graph=graph,
        base_node=graph["nodes"][0],
        max_neighbors_limit=max_neighbors_limit,
    )


@app.route("/summary/<summary_id>")
def view_summary(summary_id: str):
    store = APP_STORE
    if store is None:
        abort(500)

    selected_topics = _selected_topics_from_request()
    all_docs = _list_enriched_summaries(store)
    all_topics = _all_topics_from_summaries(all_docs)
    docs = _filter_summaries_by_topics(all_docs, selected_topics)
    sidebar_docs = _filter_summaries_today(docs)

    sid = str(summary_id).strip()
    sdoc = None
    try:
        sdoc = store.get_summary_doc(sid)
    except Exception:
        sdoc = None

    if not sdoc:
        for d in docs:
            if str(d.get("id") or "") == sid:
                sdoc = d
                break

    if not isinstance(sdoc, dict):
        abort(404)

    # Only render the published summary here. Proofread snapshots and reports
    # belong to the separate audit page and must never leak into the article.
    summary_text = str(
        sdoc.get("summary") or sdoc.get("proofread_published_summary") or ""
    ).strip()
    summary_text = _strip_proofread_feedback_from_summary(
        summary_text,
        {
            "proofread_output": str(sdoc.get("proofread_output") or ""),
            "proofread_last_feedback": str(
                ((sdoc.get("proofread_audit") or {}).get("latest") or {}).get(
                    "proofread_last_feedback"
                )
                if isinstance(sdoc.get("proofread_audit"), dict)
                else ""
            ),
        },
    )
    if not summary_text:
        keys = ", ".join(sorted(list(sdoc.keys())))
        summary_text = (
            "*(Ingen summary-text hittades i dokumentet.)*\n\n"
            f"- requested id: `{sid}`\n"
            f"- doc id: `{sdoc.get('id')}`\n"
            f"- created: `{sdoc.get('created')}`\n"
            f"- keys: `{keys}`\n"
        )

    html = _md_to_html(summary_text)
    sdoc = _enrich_summary_view_model(sdoc)
    sdoc["_viewer_tags"] = _summary_tags_for_view(store, sdoc)

    return render_template(
        "index.html",
        summary=sdoc,
        html=html,
        has_proofread_audit=_has_proofread_audit_data(sdoc),
        summaries=sidebar_docs,
        default_selected=sid,
        available_topics=all_topics,
        active_topics=selected_topics,
        format_ts=format_ts,
    )


@app.route("/summary/<summary_id>/proofread-audit")
def view_summary_proofread_audit(summary_id: str):
    store = APP_STORE
    if store is None:
        abort(500)

    selected_topics = _selected_topics_from_request()
    sid = str(summary_id).strip()
    sdoc = None
    try:
        sdoc = store.get_summary_doc(sid)
    except Exception:
        sdoc = None

    if not isinstance(sdoc, dict):
        abort(404)

    pa = sdoc.get("proofread_audit") or {}
    latest = pa.get("latest") if isinstance(pa, dict) else {}
    latest = latest if isinstance(latest, dict) else {}

    original_text = str(
        sdoc.get("proofread_original_summary")
        or latest.get("original_summary")
        or ""
    ).strip()
    if not original_text:
        original_text = _reconstruct_composed_original_summary(store, sdoc)
    revised_text = str(
        sdoc.get("proofread_revised_summary")
        or latest.get("revised_summary")
        or ""
    ).strip()
    published_text = str(
        sdoc.get("proofread_published_summary")
        or latest.get("published_summary")
        or sdoc.get("summary")
        or ""
    ).strip()

    history = pa.get("history") if isinstance(pa, dict) else []
    history = history if isinstance(history, list) else []
    proofread_output_text = str(
        latest.get("proofread_output")
        or sdoc.get("proofread_output")
        or ""
    ).strip()
    if not proofread_output_text:
        for h in reversed(history):
            if isinstance(h, dict):
                cand = str(h.get("proofread_output") or "").strip()
                if cand:
                    proofread_output_text = cand
                    break

    proofread_trace: List[Dict[str, Any]] = []
    trace_candidate = latest.get("proofread_trace")
    if isinstance(trace_candidate, list):
        proofread_trace = [t for t in trace_candidate if isinstance(t, dict)]
    if not proofread_trace:
        for h in reversed(history):
            if isinstance(h, dict) and isinstance(h.get("proofread_trace"), list):
                proofread_trace = [
                    t for t in (h.get("proofread_trace") or []) if isinstance(t, dict)
                ]
                if proofread_trace:
                    break

    return render_template(
        "summary_proofread_audit.html",
        summary=sdoc,
        summary_id=sid,
        active_topics=selected_topics,
        original_text=original_text,
        revised_text=revised_text,
        proofread_output_text=proofread_output_text,
        proofread_trace=proofread_trace,
        published_text=published_text,
        history=history,
        format_ts=format_ts,
    )


_CATEGORY_COLOR_TOKENS = {
    "bg-primary": "#0d6efd",
    "bg-secondary": "#6c757d",
    "bg-success": "#198754",
    "bg-danger": "#dc3545",
    "bg-warning": "#ffc107",
    "bg-info": "#0dcaf0",
    "bg-light": "#f8f9fa",
    "bg-dark": "#212529",
    "text-dark": "#212529",
    "text-light": "#f8f9fa",
    "text-white": "#fff",
}


def _category_css_color(value: Any, fallback: str) -> str:
    color = str(value or "").strip()
    if color in _CATEGORY_COLOR_TOKENS:
        return _CATEGORY_COLOR_TOKENS[color]
    if re.fullmatch(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})", color):
        return color
    return fallback


@app.route("/articles")
def list_articles():
    store = APP_STORE
    if store is None:
        abort(500)

    from datetime import timedelta

    now = datetime.now().replace(microsecond=0)
    if "range" in request.args:
        start_value = str(request.args.get("start") or "").strip()
        end_value = str(request.args.get("end") or "").strip()
    else:
        start_value = (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
        end_value = now.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        start_ts = int(datetime.fromisoformat(start_value).timestamp()) if start_value else None
        end_ts = int(datetime.fromisoformat(end_value).timestamp()) if end_value else None
    except ValueError:
        abort(400, "Ogiltigt datumfilter")
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        abort(400, "Startdatum måste vara före slutdatum")

    try:
        per_page = int(request.args.get("per_page", "25"))
        page = int(request.args.get("page", "1"))
    except ValueError:
        abort(400, "Ogiltig sidstorlek eller sidnummer")
    if per_page not in (25, 50, 100) or page < 1:
        abort(400, "Ogiltig sidstorlek eller sidnummer")

    categories = [item for item in (store.get_all_categories() or []) if isinstance(item, dict)]
    category_colors = {
        str(item.get("name") or ""): {
            "bg_color": _category_css_color(item.get("bg_color"), "#6c757d"),
            "text_color": _category_css_color(item.get("text_color"), "#212529"),
        }
        for item in categories
    }
    category_names = {str(item.get("name") or "") for item in categories}
    selected_categories = list(dict.fromkeys(
        value for value in request.args.getlist("category") if value in category_names
    ))
    sources = list(dict.fromkeys(
        option["name"] for option in build_source_options(APP_CFG)
    ))
    selected_sources = list(dict.fromkeys(
        value for value in request.args.getlist("source") if value in sources
    ))
    try:
        selected_tags = list(dict.fromkeys(
            int(value) for value in request.args.getlist("tag")
        ))
    except ValueError:
        abort(400, "Ogiltigt taggfilter")
    selected_tags = [value for value in selected_tags if value > 0]

    result = list_article_page(
        store, start_ts=start_ts, end_ts=end_ts, sources=selected_sources,
        categories=selected_categories, selected_tags=selected_tags,
        page=page, per_page=per_page,
    )
    selected_tags = result["selected_tags"]

    def page_url(number: int) -> str:
        return url_for(
            "list_articles", range="custom", start=start_value, end=end_value,
            source=selected_sources, category=selected_categories,
            tag=selected_tags, per_page=per_page, page=number,
        )

    return render_template(
        "articles.html",
        articles=result["articles"], total=result["total"],
        page=result["page"], pages=result["pages"], per_page=per_page,
        start_value=start_value, end_value=end_value,
        sources=sources, selected_sources=selected_sources,
        categories=categories, selected_categories=selected_categories,
        category_colors=category_colors,
        selected_tags=selected_tags, filter_tags=result["filter_tags"], page_url=page_url,
        format_ts=format_ts,
        error=None,
    )


@app.route("/api/v1/article-filter-tags")
def api_article_filter_tags():
    store = APP_STORE
    if store is None:
        abort(500)
    category_names = {
        str(item.get("name") or "") for item in (store.get_all_categories() or [])
        if isinstance(item, dict)
    }
    selected = list(dict.fromkeys(
        value for value in request.args.getlist("category") if value in category_names
    ))
    return jsonify({"tags": list_filter_tags(store, selected)})


@app.route("/article/<article_id>")
def view_article(article_id: str):
    store = APP_STORE
    if store is None:
        abort(500)

    a = None
    try:
        a = store.get_article(str(article_id))
    except Exception as e:
        return render_template(
            "article.html",
            a={
                "title": "(Kunde inte läsa artikel)",
                "source": "",
                "published_ts": 0,
                "fetched_at": 0,
                "url": "",
                "text": f"Fel vid get_article({article_id}): {e}",
                "tags": [],
                "tags": [],
            },
            format_ts=format_ts,
        ), 500

    if not isinstance(a, dict):
        abort(404)

    for k in ("title", "source", "url", "text"):
        if a.get(k) is None:
            a[k] = ""

    if not str(a.get("text") or "").strip():
        keys = ", ".join(sorted(a.keys()))
        a["text"] = (
            "⚠️ Ingen artikeltext hittades i posten.\n\n"
            f"id: {a.get('id')}\n"
            f"keys: {keys}\n"
        )

    # Add tags to the view
    tags = []
    try:
        article_tags = _article_tags_with_motiveringar(store, article_id)
        logger.info(f"[Article] Loaded {len(article_tags)} tags for article {article_id[:20]}...")
        for t in article_tags:
            if isinstance(t, dict):
                tags.append({
                    "id": t.get("id"),
                    "name": t.get("name", ""),
                    "category": t.get("category", "GENERAL"),
                    "motivering": _tag_motivering(t),
                })
    except Exception as e:
        logger.error(f"Error loading tags for article {article_id}: {e}")
    
    a["tags"] = tags
    logger.info(f"[Article] Final tags list: {tags}")

    cluster_memberships: List[Dict[str, Any]] = []
    list_article_memberships = getattr(store, "list_article_cluster_memberships", None)
    get_cluster = getattr(store, "get_threat_cluster", None)
    if _long_term_ui_enabled() and callable(list_article_memberships) and callable(get_cluster):
        seen_clusters: set[tuple[str, str]] = set()
        for profile_id in _long_term_profile_ids():
            for membership in list_article_memberships(profile_id, str(article_id)) or []:
                cluster_id = str(membership.get("cluster_id") or "")
                key = (profile_id, cluster_id)
                if not cluster_id or key in seen_clusters:
                    continue
                seen_clusters.add(key)
                cluster = get_cluster(cluster_id)
                if not isinstance(cluster, dict):
                    continue
                cluster_memberships.append({
                    "id": cluster_id,
                    "profile": _long_term_profile_label(profile_id),
                    "label": _long_term_cluster_label(store, cluster),
                    "tag_scope": str(membership.get("tag_scope") or "").split("#cluster:", 1)[0],
                    "status": str(cluster.get("status") or ""),
                })
    return render_template(
        "article.html", a=a, format_ts=format_ts,
        cluster_memberships=cluster_memberships,
    )


@app.route("/status")
def status():
    viewer_info = {
        "ok": True,
        "config": APP_CONFIG_PATH,
        "store_path": (
            (APP_CFG.get("store") or {}).get("path")
            if isinstance(APP_CFG, dict)
            else None
        ),
    }

    base = _worker_api_base(APP_CFG)
    try:
        r = requests.get(f"{base}/status", timeout=3)
        r.raise_for_status()
        worker_payload = r.json()
        return jsonify({"worker": worker_payload, "viewer": viewer_info}), 200
    except Exception as e:
        # Mirror intent: status depends on worker; if worker is down, service is unavailable
        return (
            jsonify(
                {
                    "worker": None,
                    "viewer": viewer_info,
                    "error": "worker_unavailable",
                    "worker_url": base,
                    "detail": str(e),
                }
            ),
            503,
        )


# ---- Tag management API endpoints ----


@app.route("/api/v1/tags", methods=["GET"])
def api_get_all_tags():
    """Get all tags."""
    store = APP_STORE
    if store is None:
        abort(500)
    
    try:
        tags = store.get_all_tags() or []
        return jsonify({"tags": tags}), 200
    except Exception as e:
        logger.error(f"Error getting all tags: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/tags/categories", methods=["PUT"])
def api_update_tag_categories():
    """Update the category for multiple tags in one validated request."""
    store = APP_STORE
    if store is None:
        abort(500)

    try:
        data = request.get_json(silent=True) or {}
        changes = data.get("changes")
        if not isinstance(changes, list):
            return jsonify({"error": "changes must be a list"}), 400

        categories = store.get_all_categories() or []
        category_names = {
            str(category.get("name") or "").strip()
            for category in categories
            if isinstance(category, dict) and str(category.get("name") or "").strip()
        }
        tags_by_id = {
            int(tag["id"]): tag
            for tag in (store.get_all_tags() or [])
            if isinstance(tag, dict) and tag.get("id") is not None
        }

        validated = []
        seen_tag_ids = set()
        for index, change in enumerate(changes):
            if not isinstance(change, dict):
                return jsonify({"error": f"changes[{index}] must be an object"}), 400

            try:
                tag_id = int(change.get("tag_id"))
            except (TypeError, ValueError):
                return jsonify({"error": f"changes[{index}].tag_id is invalid"}), 400

            category = str(change.get("category") or "").strip()
            if tag_id in seen_tag_ids:
                return jsonify({"error": f"duplicate tag_id: {tag_id}"}), 400
            if tag_id not in tags_by_id:
                return jsonify({"error": f"tag not found: {tag_id}"}), 404
            if category not in category_names:
                return jsonify({"error": f"category not found: {category}"}), 400

            seen_tag_ids.add(tag_id)
            if str(tags_by_id[tag_id].get("category") or "GENERAL") != category:
                validated.append((tag_id, category))

        updated_tags = []
        for tag_id, category in validated:
            updated = store.update_tag(tag_id, category=category)
            if not updated:
                logger.error(
                    "[TagCategoryAPI] Failed after %s updates while updating tag %s",
                    len(updated_tags),
                    tag_id,
                )
                return jsonify(
                    {
                        "error": f"could not update tag: {tag_id}",
                        "updated_tag_ids": [tag["id"] for tag in updated_tags],
                    }
                ), 500
            updated_tags.append(updated)

        logger.info("[TagCategoryAPI] Updated categories for %s tags", len(updated_tags))
        return jsonify(
            {
                "success": True,
                "updated_count": len(updated_tags),
                "tags": updated_tags,
            }
        ), 200
    except Exception as e:
        logger.error(f"Error updating tag categories: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/tags", methods=["POST"])
def api_create_tag():
    """Create a new tag."""
    store = APP_STORE
    if store is None:
        abort(500)
    
    try:
        data = request.get_json() or {}
        name = data.get("name", "").strip()
        category = data.get("category", "GENERAL").strip()
        description = data.get("description", "").strip()
        synonyms = data.get("synonyms", []) or []
        parent_ids = data.get("parent_ids") if "parent_ids" in data else None
        child_ids = data.get("child_ids") if "child_ids" in data else None
        
        if not name:
            return jsonify({"error": "name is required"}), 400
        
        # Try to create tag
        tag = store.create_tag(name, category, description, synonyms)
        if tag is None:
            return jsonify({"error": "tag already exists"}), 409

        if parent_ids is not None or child_ids is not None:
            try:
                relations = store.set_tag_relations(
                    int(tag["id"]), parent_ids=parent_ids, child_ids=child_ids
                )
                tag["relations"] = relations
            except Exception:
                store.delete_tag(int(tag["id"]))
                raise
        
        logger.info(f"[TagAPI] Created tag: {tag.get('name')} ({tag.get('id')}) with {len(synonyms)} synonyms")
        return jsonify({"tag": tag}), 201
    except TagRelationError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error creating tag: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/article/<article_id>/tags", methods=["GET"])
def api_get_article_tags(article_id: str):
    """Get all tags for an article."""
    store = APP_STORE
    if store is None:
        abort(500)
    
    try:
        tags = _article_tags_with_motiveringar(store, article_id)
        return jsonify({"tags": tags}), 200
    except Exception as e:
        logger.error(f"Error getting tags for article {article_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/article/<article_id>/tags", methods=["POST"])
def api_add_tag_to_article(article_id: str):
    """Add a tag to an article."""
    store = APP_STORE
    if store is None:
        abort(500)
    
    try:
        data = request.get_json() or {}
        tag_id = data.get("tag_id")
        
        if not tag_id:
            return jsonify({"error": "tag_id is required"}), 400
        
        added = store.add_tag_to_article(article_id, int(tag_id))
        if not added:
            return jsonify({"error": "tag already associated with article"}), 409
        
        logger.info(f"[TagAPI] Added tag {tag_id} to article {article_id}")
        retrain_status = _schedule_tagging_ml_retrain(
            article_id=article_id,
            tag_id=int(tag_id),
            action="addition",
        )
        return jsonify({"success": True, "ml_retrain": retrain_status}), 200
    except Exception as e:
        logger.error(f"Error adding tag to article {article_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/article/<article_id>/tags/<int:tag_id>", methods=["DELETE"])
def api_remove_tag_from_article(article_id: str, tag_id: int):
    """Remove a tag from an article."""
    store = APP_STORE
    if store is None:
        abort(500)
    
    try:
        removed = store.remove_article_tag(article_id, tag_id)
        if not removed:
            return jsonify({"error": "tag not associated with article"}), 404
        
        logger.info(f"[TagAPI] Removed tag {tag_id} from article {article_id}")
        retrain_status = _schedule_tagging_ml_retrain(
            article_id=article_id,
            tag_id=tag_id,
            action="removal",
        )
        return jsonify({"success": True, "ml_retrain": retrain_status}), 200
    except Exception as e:
        logger.error(f"Error removing tag from article {article_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/article/<article_id>/reclassify", methods=["POST"])
def api_reclassify_article_with_existing_tags(article_id: str):
    """
    Reclassify an article using only existing tags from the database.
    
    No new tags are created - the LLM is asked to suggest only from existing tags.
    The LLM is instructed that it's OK to suggest no tags if none are relevant.
    
    Request body:
        max_tags: Maximum number of tags to suggest (default 20)
    
    Response:
        suggested_tags: List of suggested tags with 'id', 'name', 'category', 'reasoning'
    """
    store = APP_STORE
    if store is None:
        abort(500)
    
    try:
        data = request.get_json() or {}
        max_tags = int(data.get("max_tags", 20))
        max_tags = max(1, min(max_tags, 20))
        
        # Get article
        article = store.get_article(article_id)
        if not article:
            return jsonify({"error": "article not found"}), 404
        
        # Get current tags on article
        current_tags = store.get_article_tags(article_id) or []
        
        # Create LLM client
        if not APP_CFG:
            return jsonify({"error": "App configuration not available"}), 503
        
        llm_client = create_llm_client(APP_CFG)
        if not llm_client:
            return jsonify({"error": "LLM client not available"}), 503
        
        # Create tagger instance with the store
        tagger = TagManager(store)
        
        # Run reclassification
        import asyncio
        suggested_tags = asyncio.run(
            tagger.reclassify_article_with_existing_tags(
                llm_client=llm_client,
                article=article,
                current_article_tags=current_tags,
                max_tags=max_tags,
            )
        )
        
        logger.info(
            f"[ReclassifyAPI] Article {article_id}: suggested {len(suggested_tags)} tags "
            f"(max_tags={max_tags})"
        )
        
        return jsonify({
            "article_id": article_id,
            "article_title": article.get("title"),
            "current_tags": current_tags,
            "suggested_tags": suggested_tags,
            "suggestion_count": len(suggested_tags),
        }), 200
    except Exception as e:
        logger.error(f"Error reclassifying article {article_id}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/tags/<int:tag_id>", methods=["GET"])
def api_get_tag(tag_id: int):
    """Get a specific tag with usage count."""
    store = APP_STORE
    if store is None:
        abort(500)
    
    try:
        # Get the tag
        all_tags = store.get_all_tags() or []
        tag = next((t for t in all_tags if t.get("id") == tag_id), None)
        
        if not tag:
            return jsonify({"error": "tag not found"}), 404
        
        # Count how many articles use this tag
        usage_count = 0
        try:
            all_articles = store.list_articles(limit=10000) or []
            for article in all_articles:
                if isinstance(article, dict):
                    article_tags = store.get_article_tags(article.get("id")) or []
                    if any(t.get("id") == tag_id for t in article_tags if isinstance(t, dict)):
                        usage_count += 1
        except Exception:
            pass
        
        tag["usage_count"] = usage_count
        relation_reader = getattr(store, "get_tag_relations", None)
        tag["relations"] = (
            relation_reader(tag_id)
            if callable(relation_reader)
            else {"parents": [], "children": []}
        )
        return jsonify({"tag": tag}), 200
    except Exception as e:
        logger.error(f"Error getting tag {tag_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/tags/<int:tag_id>", methods=["PUT"])
def api_update_tag(tag_id: int):
    """Update a tag and handle synonym migrations."""
    store = APP_STORE
    if store is None:
        abort(500)
    
    try:
        data = request.get_json() or {}
        name = data.get("name")
        category = data.get("category")
        description = data.get("description")
        new_synonyms = data.get("synonyms") if "synonyms" in data else None
        parent_ids = data.get("parent_ids") if "parent_ids" in data else None
        child_ids = data.get("child_ids") if "child_ids" in data else None
        
        # Get current tag to check for old synonyms
        all_tags = store.get_all_tags() or []
        old_tag = next((t for t in all_tags if t.get("id") == tag_id), None)
        old_synonyms = old_tag.get("synonyms", []) if old_tag else []
        
        # Find newly added synonyms
        old_synonyms_set = set(s.lower() if isinstance(s, str) else str(s).lower() for s in old_synonyms)
        new_synonyms_set = (
            set(
                s.lower() if isinstance(s, str) else str(s).lower()
                for s in new_synonyms
            )
            if new_synonyms is not None
            else old_synonyms_set
        )
        added_synonyms = new_synonyms_set - old_synonyms_set
        
        # Update the tag
        updated_tag = store.update_tag(tag_id, name, category, description, new_synonyms)
        if not updated_tag:
            return jsonify({"error": "tag not found"}), 404

        if parent_ids is not None or child_ids is not None:
            updated_tag["relations"] = store.set_tag_relations(
                tag_id, parent_ids=parent_ids, child_ids=child_ids
            )
        
        # Handle synonym migrations
        if added_synonyms:
            logger.info(f"[TagAPI] Processing {len(added_synonyms)} new synonyms for tag {tag_id}")
            
            # Find tags that match the new synonym names
            synonym_tag_ids = []
            for synonym_name in added_synonyms:
                # Search for a tag with this name (case-insensitive)
                matching_tag = next(
                    (t for t in all_tags if t.get("name", "").lower() == synonym_name),
                    None
                )
                if matching_tag:
                    synonym_tag_ids.append(matching_tag.get("id"))
                    logger.debug(f"[TagAPI] Found synonym tag: {matching_tag.get('name')} (ID: {matching_tag.get('id')})")
            
            # Migrate synonyms to main tag
            if synonym_tag_ids:
                articles_migrated, synonyms_deleted = store.migrate_synonym_to_main_tag(
                    tag_id, synonym_tag_ids
                )
                logger.info(f"[TagAPI] Synonym migration complete: {articles_migrated} articles updated, {synonyms_deleted} tags deleted")
        
        logger.info(f"[TagAPI] Updated tag {tag_id}: {updated_tag.get('name')}")
        return jsonify({"tag": updated_tag}), 200
    except TagRelationError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error updating tag {tag_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/tags/<int:tag_id>", methods=["DELETE"])
def api_delete_tag(tag_id: int):
    """Delete a tag."""
    store = APP_STORE
    if store is None:
        abort(500)
    
    try:
        deleted = store.delete_tag(tag_id)
        if not deleted:
            return jsonify({"error": "tag not found"}), 404
        
        logger.info(f"[TagAPI] Deleted tag {tag_id}")
        return jsonify({"success": True}), 200
    except Exception as e:
        logger.error(f"Error deleting tag {tag_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/tags/<int:tag_id>/relations", methods=["GET"])
def api_get_tag_relations(tag_id: int):
    """Get the parents and children of a tag."""
    store = APP_STORE
    if store is None:
        abort(500)
    if not any(tag.get("id") == tag_id for tag in (store.get_all_tags() or [])):
        return jsonify({"error": "tag not found"}), 404
    return jsonify({"relations": store.get_tag_relations(tag_id)}), 200


@app.route("/api/v1/tags/<int:tag_id>/relations", methods=["PUT"])
def api_set_tag_relations(tag_id: int):
    """Replace a tag's parent relations, child relations, or both."""
    store = APP_STORE
    if store is None:
        abort(500)
    try:
        data = request.get_json(silent=True) or {}
        has_parents = "parent_ids" in data
        has_children = "child_ids" in data
        if not has_parents and not has_children:
            return jsonify({"error": "parent_ids or child_ids is required"}), 400
        relations = store.set_tag_relations(
            tag_id,
            parent_ids=data.get("parent_ids") if has_parents else None,
            child_ids=data.get("child_ids") if has_children else None,
        )
        return jsonify({"relations": relations}), 200
    except TagRelationError as e:
        message = str(e)
        status = 404 if message.startswith("tag not found:") else 400
        return jsonify({"error": message}), status
    except Exception as e:
        logger.error(f"Error updating relations for tag {tag_id}: {e}")
        return jsonify({"error": str(e)}), 500


# ---- Category management API ----


@app.route("/api/v1/categories", methods=["GET"])
def api_get_all_categories():
    """Get all tag categories."""
    store = APP_STORE
    if store is None:
        abort(500)
    
    try:
        categories = store.get_all_categories() or []
        return jsonify({"categories": categories}), 200
    except Exception as e:
        logger.error(f"Error getting all categories: {e}")
        return jsonify({"error": str(e)}), 500


def _assign_tags_to_category(store, tag_names, category_name):
    """Assign a list of tags to a category by updating their category field."""
    try:
        all_tags = store.get_all_tags() or []
        tag_names_set = set(tag_names)
        tags_updated = 0
        tags_removed = 0
        
        for tag in all_tags:
            tag_name = tag.get("name")
            tag_id = tag.get("id")
            current_category = tag.get("category", "GENERAL")
            
            if tag_name in tag_names_set:
                # This tag should be in the new category
                if current_category != category_name:
                    store.update_tag(tag_id, category=category_name)
                    tags_updated += 1
            else:
                # This tag should NOT be in this category
                # If it was previously in this category, move it to GENERAL
                if current_category == category_name:
                    store.update_tag(tag_id, category="GENERAL")
                    tags_removed += 1
        
        logger.info(f"[CategoryAPI] Assigned tags to {category_name}: {tags_updated} added, {tags_removed} removed")
    except Exception as e:
        logger.error(f"Error assigning tags to category {category_name}: {e}")


@app.route("/api/v1/categories", methods=["POST"])
def api_create_category():
    """Create a new tag category."""
    store = APP_STORE
    if store is None:
        abort(500)
    
    try:
        data = request.get_json() or {}
        name = data.get("name", "").strip().upper()
        label = data.get("label", "").strip()
        bg_color = data.get("bg_color", "bg-secondary").strip()
        text_color = data.get("text_color", "text-dark").strip()
        description = data.get("description", "").strip()
        tags_to_assign = data.get("tags", [])
        
        if not name or not label:
            return jsonify({"error": "name and label are required"}), 400
        
        category = store.create_category(name, label, bg_color, text_color, description)
        if category is None:
            return jsonify({"error": "category already exists"}), 409
        
        # Assign tags to this category
        if tags_to_assign:
            _assign_tags_to_category(store, tags_to_assign, name)
        
        logger.info(f"[CategoryAPI] Created category: {category.get('name')} ({category.get('id')}) with {len(tags_to_assign)} tags")
        return jsonify({"category": category}), 201
    except Exception as e:
        logger.error(f"Error creating category: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/categories/<int:category_id>", methods=["GET"])
def api_get_category(category_id: int):
    """Get a category by ID."""
    store = APP_STORE
    if store is None:
        abort(500)
    
    try:
        category = store.get_category(category_id)
        if not category:
            return jsonify({"error": "category not found"}), 404
        
        return jsonify({"category": category}), 200
    except Exception as e:
        logger.error(f"Error getting category {category_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/categories/<int:category_id>", methods=["PUT"])
def api_update_category(category_id: int):
    """Update a category."""
    store = APP_STORE
    if store is None:
        abort(500)
    
    try:
        data = request.get_json() or {}
        label = data.get("label")
        bg_color = data.get("bg_color")
        text_color = data.get("text_color")
        description = data.get("description")
        tags_to_assign = data.get("tags", [])
        
        updated = store.update_category(category_id, label, bg_color, text_color, description)
        if not updated:
            return jsonify({"error": "category not found"}), 404
        
        # Assign tags to this category
        if tags_to_assign is not None:
            category = store.get_category(category_id)
            if category:
                category_name = category.get("name")
                _assign_tags_to_category(store, tags_to_assign, category_name)
        
        logger.info(f"[CategoryAPI] Updated category {category_id}")
        return jsonify({"success": True}), 200
    except Exception as e:
        logger.error(f"Error updating category {category_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/categories/<int:category_id>", methods=["DELETE"])
def api_delete_category(category_id: int):
    """Delete a category."""
    store = APP_STORE
    if store is None:
        abort(500)
    
    try:
        deleted = store.delete_category(category_id)
        if not deleted:
            return jsonify({"error": "category not found"}), 404
        
        logger.info(f"[CategoryAPI] Deleted category {category_id}")
        return jsonify({"success": True}), 200
    except Exception as e:
        logger.error(f"Error deleting category {category_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/categories/<category_name>/tags", methods=["GET"])
def api_get_tags_for_category(category_name: str):
    """Get all tags for a specific category."""
    store = APP_STORE
    if store is None:
        abort(500)
    
    try:
        all_tags = store.get_all_tags() or []
        # Filter tags by category
        category_tags = [tag for tag in all_tags if tag.get("category") == category_name]
        # Sort by name
        category_tags.sort(key=lambda t: t.get("name", "").lower())
        return jsonify({"category": category_name, "tags": category_tags}), 200
    except Exception as e:
        logger.error(f"Error getting tags for category {category_name}: {e}")
        return jsonify({"error": str(e)}), 500


# ---- Tag management UI ----


@app.route("/tags")
def manage_tags():
    """Global tag management page."""
    store = APP_STORE
    if store is None:
        abort(500)
    
    try:
        all_tags = store.get_all_tags() or []
        categories = store.get_all_categories() or []
        categories.sort(
            key=lambda category: (
                str(category.get("label") or "").lower(),
                str(category.get("name") or "").lower(),
            )
        )

        category_counts = {}
        for tag in all_tags:
            category = str(tag.get("category") or "GENERAL")
            category_counts[category] = category_counts.get(category, 0) + 1

        requested_category = str(request.args.get("category") or "").strip()
        category_names = {
            str(category.get("name") or "") for category in categories
        }
        selected_category = (
            requested_category if requested_category in category_names else ""
        )
        tags = [
            tag
            for tag in all_tags
            if not selected_category
            or str(tag.get("category") or "GENERAL") == selected_category
        ]
        
        # Count usage for each tag
        all_articles = store.list_articles(limit=10000) or []
        usage_counts = {}
        for article in all_articles:
            if isinstance(article, dict):
                article_tags = store.get_article_tags(article.get("id")) or []
                for tag in article_tags:
                    if isinstance(tag, dict):
                        tag_id = tag.get("id")
                        usage_counts[tag_id] = usage_counts.get(tag_id, 0) + 1
        
        # Add usage count to each tag
        relation_reader = getattr(store, "get_tag_relations", None)
        for tag in tags:
            tag["usage_count"] = usage_counts.get(tag.get("id"), 0)
            tag["relations"] = (
                relation_reader(int(tag["id"]))
                if callable(relation_reader) and tag.get("id") is not None
                else {"parents": [], "children": []}
            )
        
        # Sort by name
        tags.sort(key=lambda t: t.get("name", "").lower())
        
        return render_template(
            "tags.html",
            tags=tags,
            categories=categories,
            category_counts=category_counts,
            selected_category=selected_category,
            total_tag_count=len(all_tags),
            format_ts=format_ts,
        )
    except Exception as e:
        logger.error(f"Error loading tags: {e}")
        abort(500)


@app.route("/categories")
def manage_categories():
    """Global tag category management page."""
    store = APP_STORE
    if store is None:
        abort(500)
    
    try:
        categories = store.get_all_categories() or []
        
        # Count tags in each category
        all_tags = store.get_all_tags() or []
        category_tag_counts = {}
        for tag in all_tags:
            category = tag.get("category", "GENERAL")
            category_tag_counts[category] = category_tag_counts.get(category, 0) + 1
        
        # Add tag count to each category
        for cat in categories:
            cat_name = cat.get("name")
            cat["tag_count"] = category_tag_counts.get(cat_name, 0)
        
        # Sort by name
        categories.sort(key=lambda c: c.get("name", "").lower())
        
        logger.info(f"[CategoryUI] Loaded {len(categories)} categories with tag counts")
        return render_template("categories.html", categories=categories, format_ts=format_ts)
    except Exception as e:
        logger.error(f"Error loading categories: {e}")
        abort(500)


@app.route("/tag-categories")
def categorize_tags():
    """Bulk editor for assigning tags to categories."""
    store = APP_STORE
    if store is None:
        abort(500)

    try:
        categories = store.get_all_categories() or []
        categories.sort(
            key=lambda category: (
                str(category.get("label") or "").lower(),
                str(category.get("name") or "").lower(),
            )
        )

        requested_category = str(request.args.get("category") or "").strip()
        category_names = {str(category.get("name") or "") for category in categories}
        if requested_category in category_names:
            selected_category = requested_category
        elif "GENERAL" in category_names:
            selected_category = "GENERAL"
        else:
            selected_category = (
                str(categories[0].get("name") or "") if categories else ""
            )

        tags = [
            tag
            for tag in (store.get_all_tags() or [])
            if str(tag.get("category") or "GENERAL") == selected_category
        ]
        tags.sort(key=lambda tag: str(tag.get("name") or "").lower())

        return render_template(
            "tag_categories.html",
            categories=categories,
            selected_category=selected_category,
            tags=tags,
        )
    except Exception as e:
        logger.error(f"Error loading tag category editor: {e}")
        abort(500)


@app.route("/license")
def view_license():
    store = APP_STORE
    if store is None:
        abort(500)

    selected_topics = _selected_topics_from_request()
    all_docs = _list_enriched_summaries(store)
    all_topics = _all_topics_from_summaries(all_docs)
    docs = _filter_summaries_by_topics(all_docs, selected_topics)
    sidebar_docs = _filter_summaries_today(docs)

    html = _md_to_html(_load_static_md("license.md"))
    return render_template(
        "index.html",
        summary={},
        html=html,
        summaries=sidebar_docs,
        default_selected="__license__",
        available_topics=all_topics,
        active_topics=selected_topics,
        format_ts=format_ts,
    )


@app.route("/source")
def view_source():
    store = APP_STORE
    if store is None:
        abort(500)

    selected_topics = _selected_topics_from_request()
    all_docs = _list_enriched_summaries(store)
    all_topics = _all_topics_from_summaries(all_docs)
    docs = _filter_summaries_by_topics(all_docs, selected_topics)
    sidebar_docs = _filter_summaries_today(docs)

    html = _md_to_html(_load_static_md("source.md"))
    return render_template(
        "index.html",
        summary={},
        html=html,
        summaries=sidebar_docs,
        default_selected="__source__",
        available_topics=all_topics,
        active_topics=selected_topics,
        format_ts=format_ts,
    )


@app.route("/<schema_name>")
def view_latest_for_schema(schema_name: str):
    store = APP_STORE
    if store is None:
        abort(500)

    target = _normalize_schema_name(schema_name)
    if not target:
        abort(404)

    selected_topics = _selected_topics_from_request()
    docs = _list_enriched_summaries(store)

    for d in docs:
        names = _doc_schema_names(d, store)
        if any(_normalize_schema_name(n) == target for n in names):
            sid = str(d.get("id") or "").strip()
            if sid:
                return redirect(
                    url_for("view_summary", summary_id=sid, topic=selected_topics)
                )

    abort(404)


# ---- WSGI init (gunicorn/waitress): initialize from env/cwd ----
def _wsgi_init_once() -> None:
    global APP_STORE
    if APP_STORE is not None:
        return
    cfg_path = _resolve_config_path(None)
    init_app_state(cfg_path)


if os.environ.get("FEEDSUMMARY_SKIP_VIEWER_AUTO_INIT") != "1":
    _wsgi_init_once()


# ---- Dev entrypoint (optional) ----
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="FeedSummary Viewer WebApp")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "5000")))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    init_app_state(_resolve_config_path(args.config))
    app.run(host=args.host, port=args.port, debug=bool(args.debug))
