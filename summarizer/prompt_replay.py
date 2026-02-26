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

from __future__ import annotations

import os
import time
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from llmClient import create_llm_client
from persistence import NewsStore
from summarizer.summarizer import (
    summarize_batches_then_meta_with_stats,
    super_meta_from_topic_sections_with_stats,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptSet:
    batch_system: str
    batch_user_template: str
    meta_system: str
    meta_user_template: str
    super_meta_system: str
    super_meta_user_template: str


def _published_ts(a: dict) -> int:
    ts = a.get("published_ts")
    if isinstance(ts, int) and ts > 0:
        return ts
    fa = a.get("fetched_at")
    if isinstance(fa, int) and fa > 0:
        return fa
    return 0


def _fmt_dt_hm(ts: int) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")


def _build_sources_appendix_markdown(snapshots: List[Dict[str, Any]]) -> str:
    if not snapshots:
        return ""

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in snapshots:
        src = str(s.get("source") or "").strip() or "Okänd källa"
        groups[src].append(s)

    out: List[str] = []
    out.append("## Källor")
    out.append("")

    for src in sorted(groups.keys(), key=lambda x: x.lower()):
        items = sorted(
            groups[src], key=lambda x: int(x.get("published_ts") or 0), reverse=True
        )

        out.append(f"### {src}")
        out.append("")
        for it in items:
            title = str(it.get("title") or "").strip() or "(utan titel)"
            url = str(it.get("url") or "").strip()
            pts = int(it.get("published_ts") or 0)
            dt = _fmt_dt_hm(pts) if pts else ""
            line = f"{title} — {dt}" if dt else title
            out.append(f"- {line}")
            if url:
                out.append(f"  {url}")
        out.append("")

    return "\n".join(out).strip() + "\n"


def _extract_promptset_from_summary_doc(summary_doc: Dict[str, Any]) -> PromptSet:
    p = summary_doc.get("prompts") or {}
    return PromptSet(
        batch_system=str(p.get("batch_system") or ""),
        batch_user_template=str(p.get("batch_user_template") or ""),
        meta_system=str(p.get("meta_system") or ""),
        meta_user_template=str(p.get("meta_user_template") or ""),
        super_meta_system=str(p.get("super_meta_system") or ""),
        super_meta_user_template=str(p.get("super_meta_user_template") or ""),
    )


def _apply_promptset_to_config(cfg: Dict[str, Any], ps: PromptSet) -> Dict[str, Any]:
    """
    IMPORTANT:
    summarizer.helpers.load_prompts() detects "embedded prompts" if the base keys
    exist directly under cfg["prompts"].

    The old code wrote cfg["prompts"]["inline"] which load_prompts() does NOT read.
    """
    cfg2 = dict(cfg)
    cfg2["prompts"] = dict(cfg2.get("prompts") or {})
    cfg2["prompts"]["batch_system"] = ps.batch_system
    cfg2["prompts"]["batch_user_template"] = ps.batch_user_template
    cfg2["prompts"]["meta_system"] = ps.meta_system
    cfg2["prompts"]["meta_user_template"] = ps.meta_user_template
    # optional super-meta keys (used by super_meta_from_topic_sections_with_stats)
    cfg2["prompts"]["super_meta_system"] = ps.super_meta_system
    cfg2["prompts"]["super_meta_user_template"] = ps.super_meta_user_template
    cfg2["prompts"]["_package"] = "promptlab_inline"
    return cfg2


def _load_prompts_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    return obj if isinstance(obj, dict) else {}


def _save_prompts_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def resolve_prompts_path(cfg: Dict[str, Any], *, config_path: str) -> Path:
    raw = "config/prompts.yaml"
    p = cfg.get("prompts")
    if isinstance(p, dict) and p.get("path"):
        raw = str(p.get("path"))

    base_dir = Path(os.path.abspath(config_path)).parent
    raw2 = os.path.expanduser(os.path.expandvars(raw))
    if os.path.isabs(raw2):
        return Path(raw2)
    return (base_dir / raw2).resolve()


def list_prompt_packages(cfg: Dict[str, Any], *, config_path: str) -> List[str]:
    path = resolve_prompts_path(cfg, config_path=config_path)
    data = _load_prompts_yaml(path)
    return sorted([str(k) for k in data.keys()])


def load_prompt_package(
    cfg: Dict[str, Any], *, config_path: str, package_name: str
) -> Optional[PromptSet]:
    path = resolve_prompts_path(cfg, config_path=config_path)
    data = _load_prompts_yaml(path)
    pkg = data.get(package_name)
    if not isinstance(pkg, dict):
        return None
    return PromptSet(
        batch_system=str(pkg.get("batch_system") or ""),
        batch_user_template=str(pkg.get("batch_user_template") or ""),
        meta_system=str(pkg.get("meta_system") or ""),
        meta_user_template=str(pkg.get("meta_user_template") or ""),
        super_meta_system=str(pkg.get("super_meta_system") or ""),
        super_meta_user_template=str(pkg.get("super_meta_user_template") or ""),
    )


def save_prompt_package(
    cfg: Dict[str, Any], *, config_path: str, package_name: str, promptset: PromptSet
) -> Path:
    path = resolve_prompts_path(cfg, config_path=config_path)
    data = _load_prompts_yaml(path)
    data[str(package_name)] = {
        "batch_system": promptset.batch_system,
        "batch_user_template": promptset.batch_user_template,
        "meta_system": promptset.meta_system,
        "meta_user_template": promptset.meta_user_template,
        # NEW: persist super-meta fields too
        "super_meta_system": promptset.super_meta_system,
        "super_meta_user_template": promptset.super_meta_user_template,
    }
    _save_prompts_yaml(path, data)
    return path


def _topic_from_snapshot(s: Dict[str, Any]) -> str:
    t = str(s.get("topic") or "").strip()
    return t if t else "Okategoriserat"


def _topic_order(groups: Dict[str, List[dict]]) -> List[str]:
    def key(t: str) -> Tuple[int, int, str]:
        if t == "Okategoriserat":
            return (999999, 1, t.lower())
        return (len(groups.get(t) or []) * -1, 0, t.lower())

    return sorted(list(groups.keys()), key=key)


def _extract_lookback_from_orig(orig: Dict[str, Any]) -> str:
    sel = orig.get("selection") or {}
    if isinstance(sel, dict):
        return str(sel.get("lookback") or "").strip()
    return ""


async def rerun_summary_from_existing(
    *,
    config_path: str,
    cfg: Dict[str, Any],
    store: NewsStore,
    summary_id: str,
    new_prompts: PromptSet,
) -> Dict[str, Any]:
    """
    Re-run summarization using ONLY the articles referenced by summary_doc["sources"].

    NEW BEHAVIOR:
      - If original summary has sections/topics -> replay will produce the SAME structure:
          overview (super-meta) + stitched per-topic sections + sources appendix
      - Returns an ephemeral result dict:
          {
            "replay_of": <orig_id>,
            "created": <ts>,
            "from": <ts>,
            "to": <ts>,
            "sources": [ids...],
            "sources_snapshots": [...],
            "overview": <string>,
            "sections": [...],
            "summary_markdown": <string>,
            "meta": {...},
            "selection": {...}
          }
    """
    orig = store.get_summary_doc(str(summary_id))
    if not orig:
        raise RuntimeError(f"Summary not found: {summary_id}")

    sources = orig.get("sources") or []
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("Selected summary has no sources[] list.")

    get_by_ids = getattr(store, "get_articles_by_ids", None)
    if not callable(get_by_ids):
        raise RuntimeError("Store saknar get_articles_by_ids(article_ids).")

    articles = get_by_ids(sources)
    if not articles:
        raise RuntimeError("Kunde inte ladda artiklar för summary.sources.")

    by_id = {str(a.get("id")): a for a in articles if a.get("id") is not None}
    ordered_all = [by_id[str(i)] for i in sources if str(i) in by_id]

    # Apply promptset correctly (embedded prompts)
    cfg2 = _apply_promptset_to_config(cfg, new_prompts)
    llm = create_llm_client(cfg2)

    # Determine topic grouping based on original summary data
    # Prefer:
    #  1) orig.sections (each section has topic + sources)
    #  2) orig.sources_snapshots with "topic" field
    sections_orig = orig.get("sections")
    snaps_orig = orig.get("sources_snapshots") or []

    # Build snapshots for ordered_all (we may add topic if we can infer it)
    # If original has topic per snapshot, use that mapping.
    topic_by_article_id: Dict[str, str] = {}
    if isinstance(snaps_orig, list):
        for s in snaps_orig:
            if not isinstance(s, dict):
                continue
            aid = str(s.get("id") or "").strip()
            if not aid:
                continue
            topic_by_article_id[aid] = _topic_from_snapshot(s)

    # Helper: create snapshots for a list of articles, with inferred topic if available
    def make_snaps(
        items: List[dict], topic: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for a in items:
            aid = str(a.get("id") or "")
            t = topic or topic_by_article_id.get(aid, "")
            snap: Dict[str, Any] = {
                "id": a.get("id"),
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "source": a.get("source", ""),
                "published_ts": _published_ts(a),
                "content_hash": a.get("content_hash", ""),
            }
            if t:
                snap["topic"] = t
            out.append(snap)
        return out

    # Selection passthrough (nice to have)
    selection = orig.get("selection") if isinstance(orig.get("selection"), dict) else {}
    lookback_str = _extract_lookback_from_orig(orig)

    # MULTI-TOPIC replay path
    multi_topic_groups: Dict[str, List[dict]] = {}
    topic_sequence: List[str] = []

    if isinstance(sections_orig, list) and sections_orig:
        # Build groups using original sections sources list (stable and matches original grouping)
        for sec in sections_orig:
            if not isinstance(sec, dict):
                continue
            topic = str(sec.get("topic") or "").strip() or "Okategoriserat"
            sec_sources = sec.get("sources") or []
            if not isinstance(sec_sources, list) or not sec_sources:
                continue
            items = [by_id[str(i)] for i in sec_sources if str(i) in by_id]
            if not items:
                continue
            multi_topic_groups[topic] = items
            topic_sequence.append(topic)

    elif topic_by_article_id:
        # Group by inferred topic from snapshots
        for a in ordered_all:
            aid = str(a.get("id") or "")
            t = topic_by_article_id.get(aid) or "Okategoriserat"
            multi_topic_groups.setdefault(t, []).append(a)
        topic_sequence = _topic_order(multi_topic_groups)

    # If no multi-topic info, fall back to single-summary replay
    if not multi_topic_groups or len(multi_topic_groups.keys()) <= 1:
        meta_text, stats = await summarize_batches_then_meta_with_stats(
            cfg2, ordered_all, llm=llm, store=store, job_id=None
        )

        snapshots = make_snaps(ordered_all)
        appendix = _build_sources_appendix_markdown(snapshots)
        if appendix:
            meta_text = (meta_text or "").rstrip() + "\n\n" + appendix

        pts = [_published_ts(a) for a in ordered_all]
        pts2 = [p for p in pts if p > 0]
        from_ts = min(pts2) if pts2 else int(orig.get("from") or 0)
        to_ts = max(pts2) if pts2 else int(orig.get("to") or 0)

        return {
            "replay_of": str(orig.get("id")),
            "created": int(time.time()),
            "from": int(from_ts or 0),
            "to": int(to_ts or 0),
            "sources": sources,
            "sources_snapshots": snapshots,
            "overview": "",
            "sections": [],
            "summary_markdown": meta_text or "",
            "meta": {
                "batch_total": int(stats.get("batch_total") or 0),
                "trims": int(stats.get("trims") or 0),
                "drops": int(stats.get("drops") or 0),
                "meta_budget_tokens": int(stats.get("meta_budget_tokens") or 0),
            },
            "selection": dict(selection or {}),
        }

    # Else: full multi-topic replay, same style as normal summary
    stitched_parts: List[str] = []
    stitched_parts.append("# Sammanfattning per ämnesområde")
    if lookback_str:
        stitched_parts.append(f"_Tidsfönster: {lookback_str}_")
    stitched_parts.append("")

    sections_out: List[Dict[str, Any]] = []
    all_snaps: List[Dict[str, Any]] = []
    all_ids: List[str] = []

    pts_all = [_published_ts(a) for a in ordered_all]
    pts_all2 = [p for p in pts_all if p > 0]
    overall_from = min(pts_all2) if pts_all2 else int(orig.get("from") or 0)
    overall_to = max(pts_all2) if pts_all2 else int(orig.get("to") or 0)

    for i, topic in enumerate(topic_sequence, start=1):
        items = multi_topic_groups.get(topic) or []
        if not items:
            continue

        topic_meta, topic_stats = await summarize_batches_then_meta_with_stats(
            cfg2, items, llm=llm, store=store, job_id=None
        )

        ids = [a.get("id") for a in items if a.get("id")]
        all_ids.extend([str(x) for x in ids if x])

        snaps = make_snaps(items, topic=topic)
        all_snaps.extend(snaps)

        pts = [_published_ts(a) for a in items]
        pts2 = [p for p in pts if p > 0]
        from_ts = min(pts2) if pts2 else 0
        to_ts = max(pts2) if pts2 else 0

        sections_out.append(
            {
                "topic": topic,
                "from": int(from_ts or 0),
                "to": int(to_ts or 0),
                "sources": [a.get("id") for a in items if a.get("id")],
                "sources_snapshots": snaps,
                "summary": topic_meta,
                "meta": {
                    "batch_total": int(topic_stats.get("batch_total") or 0),
                    "trims": int(topic_stats.get("trims") or 0),
                    "drops": int(topic_stats.get("drops") or 0),
                    "meta_budget_tokens": int(
                        topic_stats.get("meta_budget_tokens") or 0
                    ),
                },
            }
        )

        stitched_parts.append(f"## {topic}")
        stitched_parts.append("")
        stitched_parts.append((topic_meta or "").strip())
        stitched_parts.append("")

    stitched_summary = "\n".join(stitched_parts).strip() + "\n"

    # Optional super-meta overview (uses super_meta_* prompts if present)
    overview_text = ""
    overview_stats: Dict[str, Any] = {
        "super_meta_budget_tokens": 0,
        "super_meta_enabled": 0,
    }
    try:
        overview_text, overview_stats = await super_meta_from_topic_sections_with_stats(
            config=cfg2, sections=sections_out, llm=llm, store=store, job_id=None
        )
    except KeyError:
        overview_text = ""
        overview_stats = {"super_meta_budget_tokens": 0, "super_meta_enabled": 0}

    final_summary = stitched_summary
    if overview_text.strip():
        final_summary = overview_text.strip() + "\n\n" + stitched_summary.strip() + "\n"

    appendix = _build_sources_appendix_markdown(all_snaps)
    if appendix:
        final_summary = final_summary.rstrip() + "\n\n" + appendix

    # Dedupe source IDs (keep order)
    dedup_ids = list(dict.fromkeys([x for x in all_ids if x]))

    return {
        "replay_of": str(orig.get("id")),
        "created": int(time.time()),
        "from": int(overall_from or 0),
        "to": int(overall_to or 0),
        "sources": sources,  # original stable corpus
        "sources_snapshots": all_snaps,
        "overview": overview_text,
        "sections": sections_out,
        "summary_markdown": final_summary or "",
        "meta": {
            "super_meta_enabled": int(overview_stats.get("super_meta_enabled") or 0),
            "super_meta_budget_tokens": int(
                overview_stats.get("super_meta_budget_tokens") or 0
            ),
        },
        "selection": dict(selection or {}),
        "replay_sources_dedup": dedup_ids,
    }


def get_promptset_for_summary(store: NewsStore, summary_id: str) -> PromptSet:
    sdoc = store.get_summary_doc(str(summary_id))
    if not sdoc:
        raise RuntimeError(f"Summary not found: {summary_id}")
    return _extract_promptset_from_summary_doc(sdoc)
