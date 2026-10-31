#!/usr/bin/env python3
"""Koppla needs_review-kluster till det föreslagna klustret istället för att lämna dem isolerade.

Artiklar kan tillhöra flera kluster (olika tag_scope), så en tvetydig
klustertillhörighet ska inte behöva vänta på manuell granskning: skriptet
slår samman varje needs_review-kluster (en singel-artikel) med det
bäst rankade kandidatklustret som klustringen redan föreslog.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from feedsummary_core.long_term import (
    build_cluster_review_merge_operation,
    build_cluster_review_resolution,
)
from feedsummary_core.persistence import NewsStore, create_store

log = logging.getLogger("resolve_needs_review_clusters")
DEFAULT_REVIEWER = "resolve_needs_review_clusters.py"
OPEN_TARGET_STATUSES = {"active", "dormant"}


def _resolve_config_path(raw_path: Optional[str]) -> Path:
    candidate = raw_path or os.environ.get("FEEDSUMMARY_CONFIG") or "config.yaml"
    return Path(os.path.expandvars(os.path.expanduser(candidate))).resolve()


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Konfigurationsfilen saknas: {path}")

    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Konfigurationen måste vara ett YAML-objekt: {path}")

    store_config = config.get("store")
    if isinstance(store_config, dict) and store_config.get("path"):
        resolved_store = dict(store_config)
        store_path = Path(
            os.path.expandvars(os.path.expanduser(str(store_config["path"])))
        )
        if not store_path.is_absolute():
            store_path = path.parent / store_path
        resolved_store["path"] = str(store_path.resolve())
        config["store"] = resolved_store

    return config


def _profile_ids(config: Dict[str, Any], requested: Sequence[str]) -> List[str]:
    if requested:
        return list(dict.fromkeys(requested))
    long_term = config.get("long_term_analysis") or {}
    profiles = long_term.get("profiles") if isinstance(long_term, dict) else {}
    if not isinstance(profiles, dict):
        return []
    return sorted(str(profile_id) for profile_id in profiles if str(profile_id).strip())


def _candidate_ids(membership: Dict[str, Any]) -> List[str]:
    ids = []
    for key in ("best_candidate_cluster_id", "second_candidate_cluster_id"):
        value = str(membership.get(key) or "").strip()
        if value and value not in ids:
            ids.append(value)
    return ids


def resolve_profile_needs_review(
    store: NewsStore,
    profile_id: str,
    *,
    apply: bool,
    reviewer: str,
    comment: str,
) -> Dict[str, Any]:
    """Merge every unresolved needs_review cluster into its suggested candidate."""

    result: Dict[str, Any] = {
        "profile_id": profile_id,
        "needs_review": 0,
        "merged": 0,
        "kept_separate": 0,
        "no_candidate": 0,
        "candidate_unavailable": 0,
        "failed": 0,
        "errors": [],
    }

    review_clusters = [
        cluster
        for cluster in store.list_threat_clusters(profile_id, statuses=["needs_review"])
        if not cluster.get("review_decision")
    ]
    result["needs_review"] = len(review_clusters)

    for review_doc in review_clusters:
        review_id = str(review_doc.get("id") or "")
        memberships = store.list_cluster_memberships(review_id)
        if not memberships:
            result["no_candidate"] += 1
            result["errors"].append({"cluster_id": review_id, "error": "ingen medlemskapspost hittades"})
            continue

        candidate_ids = _candidate_ids(memberships[0])
        if not candidate_ids:
            result["no_candidate"] += 1
            result["errors"].append({"cluster_id": review_id, "error": "inga föreslagna kluster i medlemskapet"})
            continue

        open_candidates = []
        for candidate_id in candidate_ids:
            candidate = store.get_threat_cluster(candidate_id)
            if candidate and str(candidate.get("status") or "") in OPEN_TARGET_STATUSES:
                open_candidates.append(candidate)
        if not open_candidates:
            result["candidate_unavailable"] += 1
            result["errors"].append(
                {"cluster_id": review_id, "error": f"inget föreslaget kluster är öppet: {candidate_ids}"}
            )
            continue

        # En sammanslagning kräver identisk tag_scope; föreslagna kluster kan ha
        # matchat på delvis överlappande scope, så föredra ett exakt scope-matchande förslag.
        review_tag_scope = str(review_doc.get("tag_scope") or "")
        target_doc = next(
            (c for c in open_candidates if str(c.get("tag_scope") or "") == review_tag_scope),
            None,
        )

        try:
            if target_doc is not None:
                target_id = str(target_doc.get("id") or "")
                operation = build_cluster_review_merge_operation(
                    review_doc,
                    target_doc,
                    {
                        review_id: memberships,
                        target_id: store.list_cluster_memberships(target_id),
                    },
                    allowed_candidate_cluster_ids=candidate_ids,
                    reviewed_at=int(time.time()),
                    reviewed_by=reviewer,
                    comment=comment,
                )
                if not apply:
                    result["merged"] += 1
                    continue
                if store.apply_cluster_reconciliation(operation):
                    result["merged"] += 1
                else:
                    result["failed"] += 1
                    result["errors"].append(
                        {"cluster_id": review_id, "error": "apply_cluster_reconciliation avvisade operationen (revision ändrad)"}
                    )
            else:
                # Inget förslag delar exakt tag_scope, så en sammanslagning skulle
                # bryta mot invarianten att ett kluster har en enda tag_scope.
                # Artiklar får tillhöra flera kluster, så kopplingen till förslaget
                # behålls via best_candidate_cluster_id och klustret lämnar needs_review.
                resolved_doc = build_cluster_review_resolution(
                    review_doc,
                    decision="keep_separate",
                    reviewed_at=int(time.time()),
                    reviewed_by=reviewer,
                    comment=comment,
                )
                if not apply:
                    result["kept_separate"] += 1
                    continue
                if store.resolve_threat_cluster_review(
                    resolved_doc, expected_membership_revision=review_doc["membership_revision"]
                ):
                    result["kept_separate"] += 1
                else:
                    result["failed"] += 1
                    result["errors"].append(
                        {"cluster_id": review_id, "error": "resolve_threat_cluster_review avvisade operationen (revision ändrad)"}
                    )
        except Exception as error:
            result["failed"] += 1
            result["errors"].append({"cluster_id": review_id, "error": str(error)})
            continue

    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Koppla needs_review-kluster till det föreslagna klustret. "
            "Kör utan --apply för förhandsgranskning."
        )
    )
    parser.add_argument("--config", help="Sökväg till config.yaml (standard: config.yaml)")
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help="Långtidsprofil att behandla (kan upprepas). Standard: alla profiler i konfigurationen.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Utför sammanslagningarna; utan flaggan görs endast en förhandsgranskning",
    )
    parser.add_argument("--reviewer", default=DEFAULT_REVIEWER, help="Namn som sparas som granskare")
    parser.add_argument(
        "--comment",
        default="Automatisk koppling till föreslaget kluster (needs_review borttaget)",
        help="Kommentar som sparas på sammanslagningen",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    try:
        config_path = _resolve_config_path(args.config)
        config = _load_config(config_path)
        store = create_store(config.get("store", {}))
        profile_ids = _profile_ids(config, args.profile)
        if not profile_ids:
            print("Inga långtidsprofiler att behandla.")
            return 0

        mode = "ÄNDRING" if args.apply else "FÖRHANDSGRANSKNING"
        totals = {
            "needs_review": 0,
            "merged": 0,
            "kept_separate": 0,
            "no_candidate": 0,
            "candidate_unavailable": 0,
            "failed": 0,
        }
        for profile_id in profile_ids:
            result = resolve_profile_needs_review(
                store,
                profile_id,
                apply=args.apply,
                reviewer=args.reviewer,
                comment=args.comment,
            )
            for key in totals:
                totals[key] += result[key]
            print(
                f"{mode} [{profile_id}]: {result['needs_review']} needs_review, "
                f"{result['merged']} sammanslagna, {result['kept_separate']} behållna separat, "
                f"{result['no_candidate']} utan förslag, "
                f"{result['candidate_unavailable']} förslag ej tillgängliga, "
                f"{result['failed']} misslyckade."
            )
            for error in result["errors"]:
                log.warning("%s: %s", error["cluster_id"], error["error"])

        print(
            f"TOTALT: {totals['needs_review']} needs_review, {totals['merged']} sammanslagna, "
            f"{totals['kept_separate']} behållna separat, {totals['no_candidate']} utan förslag, "
            f"{totals['candidate_unavailable']} förslag ej tillgängliga, "
            f"{totals['failed']} misslyckade."
        )
        if not args.apply and totals["needs_review"]:
            print("Kör med --apply för att faktiskt slå samman klustren.")
        return 0
    except Exception as error:
        log.error("Skriptet misslyckades: %s", error, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
