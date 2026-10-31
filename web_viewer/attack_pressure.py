"""Evidence-bound, three-part TEF context from long-term reports and clusters.

Geographic and sector signals are explicitly labeled proxies until independent
facet-level event series exist.
"""

from __future__ import annotations

import re
import time
from typing import Any


CVE_RE = re.compile(r"CVE-\d{4}-\d{4,}", re.IGNORECASE)
ACTIVE_RE = re.compile(
    r"\b(?:(?:attackers?|threat\s+actors?)\s+(?:are\s+)?(?:actively\s+)?exploit(?:ing|ed|s)?|"
    r"actively\s+exploited|exploited\s+in\s+the\s+wild|"
    r"active\s+exploitation|ongoing\s+exploitation|"
    r"angripare\s+utnyttjar|utnyttjas\s+aktivt|aktivt\s+utnyttjande|"
    r"pågående\s+utnyttjande)\b",
    re.IGNORECASE,
)
NEGATED_RE = re.compile(
    r"\b(?:not\s+(?:being\s+)?(?:actively\s+)?exploited|"
    r"no\s+evidence\s+of\s+(?:active\s+)?exploitation|"
    r"unconfirmed\s+exploitation|not\s+confirmed|"
    r"inte\s+(?:aktivt\s+)?utnyttjad|ej\s+bekräftad|"
    r"inga\s+belägg\s+för\s+utnyttjande)\b",
    re.IGNORECASE,
)
CHAIN_RE = re.compile(r"\b(?:chain|vulnerabilit(?:y|ies)|sårbarhet(?:er|erna)?)\b", re.IGNORECASE)
CHAIN_DEFINITION_RE = re.compile(
    r"\b(?:the\s+chain\s+consists\s+of|these\s+vulnerabilities|"
    r"the\s+vulnerabilities|kedjan\s+består\s+av|sårbarheterna\s+är)\b",
    re.IGNORECASE,
)
DAY_SECONDS = 86400
REPORT_MAX_AGE_SECONDS = 14 * DAY_SECONDS
SIGNAL_WINDOW_SECONDS = 30 * DAY_SECONDS
CLUSTER_PAGE_SIZE = 250
MAX_CLUSTER_SCAN = 10000
WEIGHTS = {"region": 0.25, "sector": 0.25, "cve": 0.50}


def valid_cve(value: str) -> str | None:
    candidate = str(value or "").strip().upper()
    return candidate if CVE_RE.fullmatch(candidate) else None


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _active_cve_claim(payload: dict[str, Any], cve_id: str) -> bool:
    """Require an affirmative exploitation claim about this exact CVE."""

    summary_sentences = re.split(
        r"(?<=[.!?])\s+", str(payload.get("summary") or "")
    )
    statements = list(summary_sentences)
    statements.extend(
        str(fact.get("statement") or "")
        for fact in payload.get("facts") or []
        if isinstance(fact, dict) and fact.get("status") == "active"
    )
    statements.extend(
        str(entry.get("statement") or "")
        for entry in payload.get("timeline") or []
        if isinstance(entry, dict)
    )
    for statement in statements:
        for sentence in re.split(r"(?<=[.!?])\s+", statement):
            if (cve_id in {match.group(0).upper() for match in CVE_RE.finditer(sentence)}
                    and ACTIVE_RE.search(sentence) and not NEGATED_RE.search(sentence)):
                return True
    # A summary may state that a vulnerability chain is being exploited, then
    # identify its CVEs in the immediately following definition sentence.
    for claim, definition in zip(summary_sentences, summary_sentences[1:]):
        if (ACTIVE_RE.search(claim) and CHAIN_RE.search(claim)
                and not NEGATED_RE.search(claim)
                and CHAIN_DEFINITION_RE.search(definition)
                and not NEGATED_RE.search(definition)
                and cve_id in {m.group(0).upper() for m in CVE_RE.finditer(definition)}):
            return True
    return False


def _report_context(store: Any, profile_id: str, now_ts: int) -> dict[str, Any]:
    context = {
        "report_id": None,
        "period_end_ts": None,
        "window_days": 30,
        "profile_event_count": None,
        "comparison_event_count": None,
        "trend_eligible": False,
        "event_cluster_ids": [],
        "reason": "no_recent_report",
    }
    reports = store.list_threat_landscape_reports(profile_id, limit=10) or []
    report = next(
        (row for row in reports if isinstance(row, dict)
         and row.get("status") == "published"),
        None,
    )
    if report is None:
        return context
    context["report_id"] = str(report.get("id") or "") or None
    end_ts = _int(report.get("period_end_ts"))
    context["period_end_ts"] = end_ts or None
    if end_ts < 1 or end_ts > now_ts or now_ts - end_ts > REPORT_MAX_AGE_SECONDS:
        context["reason"] = "stale_report"
        return context
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        context["reason"] = "incomplete_report_metrics"
        return context
    quality = metrics.get("quality") or {}
    if isinstance(quality, dict) and quality.get("input_truncated"):
        context["reason"] = "truncated_report_inputs"
        return context
    windows = metrics.get("windows") or []
    window = next(
        (row for row in windows if isinstance(row, dict) and row.get("days") == 30),
        None,
    )
    if not isinstance(window, dict):
        context["reason"] = "missing_30_day_metrics"
        return context
    comparison = window.get("comparison") or {}
    coverage = window.get("coverage") or {}
    current = window.get("independent_event_count")
    previous = comparison.get("independent_event_count") if isinstance(comparison, dict) else None
    if (not isinstance(current, int) or isinstance(current, bool) or current < 0
            or not isinstance(previous, int) or isinstance(previous, bool) or previous < 0
            or not isinstance(coverage, dict)):
        context["reason"] = "incomplete_report_metrics"
        return context
    context.update({
        "profile_event_count": current,
        "comparison_event_count": previous,
        "trend_eligible": coverage.get("trend_eligible") is True,
        "event_cluster_ids": [
            str(cluster_id)
            for cluster_id in window.get("event_cluster_ids") or []
            if str(cluster_id)
        ],
        "reason": "report_trend_eligible" if coverage.get("trend_eligible") is True
                  else "report_trend_ineligible",
    })
    return context


def _recent_cve_clusters(
    store: Any, profile_id: str, cve_id: str, now_ts: int
) -> tuple[list[dict[str, Any]], bool]:
    """Scan bounded pages of recent clusters; never depend on report size."""

    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    target = f"cve:{cve_id.casefold()}"
    for offset in range(0, MAX_CLUSTER_SCAN, CLUSTER_PAGE_SIZE):
        page = store.list_threat_clusters(
            profile_id,
            statuses=["active", "dormant"],
            min_last_seen_ts=now_ts - SIGNAL_WINDOW_SECONDS,
            limit=CLUSTER_PAGE_SIZE,
            offset=offset,
            include_vectors=False,
        ) or []
        for cluster in page:
            if not isinstance(cluster, dict):
                continue
            cluster_id = str(cluster.get("id") or "")
            if not cluster_id or cluster_id in seen:
                continue
            seen.add(cluster_id)
            indicators = cluster.get("strong_indicators") or []
            if (cluster.get("profile_id") == profile_id
                    and target in {str(item).casefold() for item in indicators}):
                matches.append(cluster)
        if len(page) < CLUSTER_PAGE_SIZE:
            return matches, False
    return matches, True


def _report_cve_clusters(
    store: Any, report: dict[str, Any], cve_id: str
) -> list[dict[str, Any]]:
    """Return CVE-bearing clusters explicitly included in the latest report."""

    get_cluster = getattr(store, "get_threat_cluster", None)
    if not callable(get_cluster):
        return []
    target = f"cve:{cve_id.casefold()}"
    matches: list[dict[str, Any]] = []
    for cluster_id in report.get("event_cluster_ids") or []:
        cluster = get_cluster(str(cluster_id))
        if (isinstance(cluster, dict) and cluster.get("profile_id") == report["profile_id"]
                and target in {str(item).casefold()
                               for item in cluster.get("strong_indicators") or []}):
            matches.append(cluster)
    return matches


def _cve_component(
    store: Any,
    clusters: list[dict[str, Any]],
    report_clusters: list[dict[str, Any]],
    cve_id: str,
    now_ts: int,
) -> dict[str, Any]:
    cluster_ids = sorted(str(row["id"]) for row in clusters)
    active_snapshot_ids: list[str] = []
    article_ids: set[str] = set()
    for cluster in clusters:
        snapshot_id = str(cluster.get("latest_snapshot_id") or "")
        if not snapshot_id:
            continue
        snapshot = store.get_cluster_snapshot(snapshot_id)
        if (not isinstance(snapshot, dict)
                or snapshot.get("cluster_id") != cluster.get("id")
                or snapshot.get("profile_id") != cluster.get("profile_id")):
            continue
        created_at = _int(snapshot.get("created_at"))
        if created_at < now_ts - SIGNAL_WINDOW_SECONDS or created_at > now_ts:
            continue
        payload = snapshot.get("payload") or {}
        if (not isinstance(payload, dict) or payload.get("insufficient_evidence") is True
                or payload.get("confidence") not in {"medium", "high"}
                or not _active_cve_claim(payload, cve_id)):
            continue
        active_snapshot_ids.append(snapshot_id)
        for fact in payload.get("facts") or []:
            if (isinstance(fact, dict) and fact.get("status") == "active"
                    and cve_id in {m.group(0).upper() for m in CVE_RE.finditer(
                        str(fact.get("statement") or "")
                    )}):
                article_ids.update(str(value) for value in fact.get("evidence_article_ids") or [])
        for entry in payload.get("timeline") or []:
            if (isinstance(entry, dict)
                    and cve_id in {m.group(0).upper() for m in CVE_RE.finditer(
                        str(entry.get("statement") or "")
                    )}
                    and ACTIVE_RE.search(str(entry.get("statement") or ""))
                    and not NEGATED_RE.search(str(entry.get("statement") or ""))):
                article_ids.update(str(value) for value in entry.get("evidence_article_ids") or [])

    report_cluster_ids = sorted(str(row["id"]) for row in report_clusters)
    if active_snapshot_ids:
        level, factor = "active_exploitation_reported", 1.4
    elif len(cluster_ids) >= 3:
        level, factor = "reported", 1.2
    elif cluster_ids:
        level, factor = "reported", 1.15 if report_cluster_ids else 1.1
    else:
        level, factor = "not_observed", 1.0
    return {
        "level": level,
        "factor": factor,
        "basis": "recent_cluster_indicator_and_snapshot",
        "active_exploitation_reported": bool(active_snapshot_ids),
        "independent_event_count": len(cluster_ids),
        "evidence_cluster_ids": cluster_ids,
        "report_evidence_cluster_ids": report_cluster_ids,
        "evidence_snapshot_ids": sorted(active_snapshot_ids),
        "evidence_article_ids": sorted(article_ids)[:20],
    }


def assess_attack_pressure(
    store: Any,
    profile_id: str,
    cve_id: str,
    profile: dict[str, Any],
    *,
    now_ts: int | None = None,
) -> dict[str, Any]:
    """Return three visible component factors and their bounded TEF scalar."""

    now_ts = int(time.time() if now_ts is None else now_ts)
    region_name = str(profile.get("region") or "").strip() or None
    sector_name = str(profile.get("sector") or "").strip() or None
    report = _report_context(store, profile_id, now_ts)
    cve_clusters, scan_truncated = _recent_cve_clusters(store, profile_id, cve_id, now_ts)
    report_for_clusters = {**report, "profile_id": profile_id}
    report_cve_clusters = _report_cve_clusters(store, report_for_clusters, cve_id)
    clusters_by_id = {
        str(cluster.get("id")): cluster
        for cluster in [*cve_clusters, *report_cve_clusters]
        if isinstance(cluster, dict) and cluster.get("id")
    }
    cve = _cve_component(
        store, list(clusters_by_id.values()), report_cve_clusters, cve_id, now_ts
    )

    current = report["profile_event_count"]
    previous = report["comparison_event_count"]
    rising = (report["trend_eligible"] and isinstance(current, int)
              and isinstance(previous, int) and current > previous)
    # The same CVE events cannot also supply the regional/sectorial uplift.
    independent_rise = (
        rising and not scan_truncated and current - len(cve_clusters) > previous
    )
    if report["trend_eligible"]:
        level = (
            "elevated" if independent_rise else
            "overlaps_cve_signal" if rising else "no_increase_detected"
        )
    else:
        level = "unknown"

    region = {
        "name": region_name,
        "level": level if region_name else "unknown",
        "factor": 1.1 if region_name and (independent_rise or report_cve_clusters) else 1.0,
        "basis": "latest_report_cluster_and_profile_trend_proxy",
        "independent_geographic_evidence": False,
    }
    sector = {
        "name": sector_name,
        "level": level if sector_name else "unknown",
        "factor": 1.1 if sector_name and (independent_rise or report_cve_clusters) else 1.0,
        "basis": "latest_report_cluster_and_profile_trend_proxy",
        "independent_sector_evidence": False,
    }
    components = {"region": region["factor"], "sector": sector["factor"], "cve": cve["factor"]}
    scalar = round(1.0 + sum(
        WEIGHTS[name] * (component - 1.0)
        for name, component in components.items()
    ), 3)
    has_report_assessment = report["trend_eligible"]
    has_cve_assessment = bool(cve_clusters)
    status = "assessed" if has_report_assessment or has_cve_assessment else "insufficient_data"
    return {
        "schema_version": 1,
        "profile_id": profile_id,
        "cve_id": cve_id,
        "status": status,
        "reason": "current_active_exploitation_reported" if cve["active_exploitation_reported"]
                  else "current_cve_reporting" if cve_clusters else report["reason"],
        "region": region,
        "sector": sector,
        "vulnerability": cve,
        "evidence": {**report, "cluster_scan_truncated": scan_truncated},
        "tef_adjustment": {
            "method": "multiply",
            "factor": scalar,
            "combination": "weighted_additive_uplift",
            "weights": dict(WEIGHTS),
            "components": components,
            "formula": "1 + 0.25*(region-1) + 0.25*(sector-1) + 0.50*(cve-1)",
            "calibration": "heuristic_v2",
        },
    }
