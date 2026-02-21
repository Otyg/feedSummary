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

import asyncio
import copy
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yaml

from llmClient import create_llm_client
from persistence import NewsStore, create_store
from summarizer.helpers import (
    setup_logging,
    load_prompts,
    parse_lookback_to_seconds,
    load_feeds_into_config,
)
from summarizer.ingest import gather_articles_to_store
from summarizer.summarizer import summarize_batches_then_meta_with_stats

setup_logging()
logger = logging.getLogger(__name__)


def _published_ts(a: dict) -> int:
    ts = a.get("published_ts")
    if isinstance(ts, int) and ts > 0:
        return ts
    fa = a.get("fetched_at")
    if isinstance(fa, int) and fa > 0:
        return fa
    return 0


def _summary_doc_id(created_ts: int, job_id: Optional[int]) -> str:
    dt = datetime.fromtimestamp(created_ts)
    base = dt.strftime("sum_%Y%m%d_%H%M")
    return f"{base}_job{job_id}" if job_id is not None else base


def _persist_summary_doc(store: NewsStore, doc: Dict[str, Any]) -> Any:
    fn = getattr(store, "save_summary_doc", None)
    if not callable(fn):
        raise RuntimeError("Store saknar save_summary_doc() för summary_docs.")
    return fn(doc)


def _get_config_sources(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    feeds = config.get("feeds")
    if isinstance(feeds, list) and all(isinstance(x, dict) for x in feeds):
        return feeds  # type: ignore[return-value]
    return []


def _set_config_sources(config: Dict[str, Any], sources: List[Dict[str, Any]]) -> None:
    config["feeds"] = sources


def _name_of(s: Dict[str, Any]) -> str:
    return str(s.get("name") or s.get("title") or s.get("label") or "").strip()


def _topics_of(s: Dict[str, Any]) -> List[str]:
    """
    Read topics from feed/source dict. Normalizes to a list[str].
    Supports:
      topics: ["Cyber", "Sverige"]
      topic: "Cyber"
    """
    t = s.get("topics")
    if isinstance(t, list):
        out = [str(x).strip() for x in t if str(x).strip()]
        return out
    if isinstance(t, str) and t.strip():
        return [t.strip()]

    t2 = s.get("topic")
    if isinstance(t2, str) and t2.strip():
        return [t2.strip()]

    return []


def _source_topics_map(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Map from source/feed name -> topics list.
    """
    out: Dict[str, List[str]] = {}
    for s in _get_config_sources(config):
        n = _name_of(s)
        if not n:
            continue
        out[n] = _topics_of(s)
    return out


def _apply_overrides(
    config: Dict[str, Any], overrides: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    if not overrides:
        return config

    cfg = copy.deepcopy(config)

    lookback = overrides.get("lookback")
    if isinstance(lookback, str) and lookback.strip():
        ingest = cfg.setdefault("ingest", {})
        if isinstance(ingest, dict):
            ingest["lookback"] = lookback.strip()

    selected = overrides.get("sources")
    if isinstance(selected, list) and selected:
        selected_set = {str(x) for x in selected if str(x).strip()}
        all_sources = _get_config_sources(cfg)
        filtered = [s for s in all_sources if _name_of(s) in selected_set]
        _set_config_sources(cfg, filtered)

    selected_topics = overrides.get("topics")
    if (
        (not (isinstance(selected, list) and selected))
        and isinstance(selected_topics, list)
        and selected_topics
    ):
        wanted = {str(t).strip() for t in selected_topics if str(t).strip()}
        if wanted:
            all_sources = _get_config_sources(cfg)

            def has_topic(s: Dict[str, Any]) -> bool:
                ts = set(_topics_of(s))
                return bool(ts.intersection(wanted))

            filtered = [s for s in all_sources if has_topic(s)]
            _set_config_sources(cfg, filtered)

    prompt_pkg = overrides.get("prompt_package")
    if isinstance(prompt_pkg, str) and prompt_pkg.strip():
        p = cfg.setdefault("prompts", {})
        if isinstance(p, dict):
            p["selected"] = prompt_pkg.strip()

    return cfg


def _selected_source_names(config: Dict[str, Any]) -> List[str]:
    srcs = _get_config_sources(config)
    out: List[str] = []
    for s in srcs:
        n = _name_of(s)
        if n:
            out.append(n)
    return out


def _selected_topics_from_config(config: Dict[str, Any]) -> List[str]:
    """
    After overrides, the feeds list already reflects selected sources/topics.
    We compute the union of topics on the selected feeds to store in selection metadata.
    """
    topics: List[str] = []
    seen = set()
    for s in _get_config_sources(config):
        for t in _topics_of(s):
            if t not in seen:
                seen.add(t)
                topics.append(t)
    topics.sort(key=lambda x: x.lower())
    return topics


def _select_articles_for_summary(
    config: Dict[str, Any], store: NewsStore, *, limit: int = 2000
) -> List[dict]:
    ingest = config.get("ingest") or {}
    lookback = str(ingest.get("lookback") or "").strip()
    sources = _selected_source_names(config)

    now = int(time.time())
    since_ts = 0
    if lookback:
        since_ts = now - parse_lookback_to_seconds(lookback)

    list_by_filter = getattr(store, "list_articles_by_filter", None)
    if callable(list_by_filter) and since_ts > 0 and sources:
        rows = list_by_filter(
            sources=sources, since_ts=since_ts, until_ts=now, limit=limit
        )
        rows.sort(key=_published_ts)  # type: ignore
        return rows  # type: ignore

    list_articles = getattr(store, "list_articles", None)
    if callable(list_articles):
        rows = list_articles(limit=limit)
    else:
        rows = store.list_unsummarized_articles(limit=limit)  # sista utväg

    if sources:
        srcset = set(sources)
        rows = [a for a in rows if a.get("source") in srcset]  # type: ignore
    if since_ts > 0:
        rows = [a for a in rows if _published_ts(a) >= since_ts]  # type: ignore

    rows.sort(key=_published_ts)  # type: ignore
    return rows[:limit]  # type: ignore


def _primary_topic_for_article(
    a: Dict[str, Any], topic_map: Dict[str, List[str]]
) -> str:
    src = str(a.get("source") or "").strip()
    ts = topic_map.get(src) or []
    if ts:
        return ts[0]
    return "Okategoriserat"


def _group_articles_by_primary_topic(
    articles: List[dict],
    topic_map: Dict[str, List[str]],
) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}
    for a in articles:
        t = _primary_topic_for_article(a, topic_map)
        groups.setdefault(t, []).append(a)
    for t, items in groups.items():
        items.sort(key=_published_ts)
        groups[t] = items
    return groups


def _topic_order(groups: Dict[str, List[dict]]) -> List[str]:
    """
    Order topics so smaller topics don't drown:
    - Put "Okategoriserat" last.
    - Otherwise sort by (count desc, name asc).
    """

    def key(t: str) -> Tuple[int, int, str]:
        if t == "Okategoriserat":
            return (999999, 1, t.lower())
        return (len(groups.get(t) or []) * -1, 0, t.lower())

    return sorted(list(groups.keys()), key=key)


async def run_pipeline(
    config_path: str = "config.yaml",
    job_id: Optional[int] = None,
    overrides: Optional[Dict[str, Any]] = None,
    config_dict: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    if config_dict is None:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = config_dict

    config = load_feeds_into_config(config, base_config_path=config_path)

    config = _apply_overrides(config, overrides)

    store = create_store(config.get("store", {}))
    llm = create_llm_client(config)

    if job_id is not None:
        store.update_job(
            job_id,
            status="running",
            started_at=int(time.time()),
            message="Startar ingest...",
        )
        if overrides:
            store.update_job(
                job_id, message=f"Startar ingest... (overrides: {overrides})"
            )

    ins, upd = await gather_articles_to_store(config, store, job_id=job_id)

    if job_id is not None:
        store.update_job(
            job_id,
            message=f"Ingest klart. Inserted={ins}, Updated={upd}. Förbereder summering...",
        )

    to_sum = _select_articles_for_summary(config, store, limit=2000)
    if not to_sum:
        if job_id is not None:
            store.update_job(
                job_id,
                status="done",
                finished_at=int(time.time()),
                message="Klart: inga artiklar matchade urvalet (lookback/källor/ämnen).",
            )
        return None

    topic_map = _source_topics_map(config)
    groups = _group_articles_by_primary_topic(to_sum, topic_map)
    topics = _topic_order(groups)

    created_ts = int(time.time())

    if len(topics) <= 1:
        meta_text, stats = await summarize_batches_then_meta_with_stats(
            config, to_sum, llm=llm, store=store, job_id=job_id
        )

        ids = [a.get("id") for a in to_sum if a.get("id")]

        pts = [_published_ts(a) for a in to_sum]
        pts2 = [p for p in pts if p > 0]
        from_ts = min(pts2) if pts2 else 0
        to_ts = max(pts2) if pts2 else 0

        summary_doc: Dict[str, Any] = {
            "id": _summary_doc_id(created_ts, job_id),
            "created": created_ts,
            "kind": "summary",
            "llm": {
                "provider": (config.get("llm") or {}).get("provider", "unknown"),
                "model": (config.get("llm") or {}).get("model", "unknown"),
                "temperature": 0.2,
                "max_output_tokens": int(
                    (config.get("llm") or {}).get("max_output_tokens") or 0
                ),
            },
            "prompts": load_prompts(config),
            "batching": config.get("batching", {}) or {},
            "sources": ids,
            "sources_snapshots": [
                {
                    "id": a.get("id"),
                    "title": a.get("title", ""),
                    "url": a.get("url", ""),
                    "source": a.get("source", ""),
                    "published_ts": _published_ts(a),
                    "content_hash": a.get("content_hash", ""),
                }
                for a in to_sum
            ],
            "from": from_ts,
            "to": to_ts,
            "summary": meta_text,
            "meta": {
                "batch_total": int(stats.get("batch_total") or 0),
                "trims": int(stats.get("trims") or 0),
                "drops": int(stats.get("drops") or 0),
                "meta_budget_tokens": int(stats.get("meta_budget_tokens") or 0),
            },
            "selection": {
                "lookback": str((config.get("ingest") or {}).get("lookback") or ""),
                "sources": _selected_source_names(config),
                "topics": _selected_topics_from_config(config),
                "prompt_package": str(
                    (
                        (config.get("prompts") or {})
                        if isinstance(config.get("prompts"), dict)
                        else {}
                    ).get("selected")
                    or ""
                ),
            },
        }

        summary_doc_id = _persist_summary_doc(store, summary_doc)

        if job_id is not None:
            store.update_job(
                job_id,
                status="done",
                finished_at=int(time.time()),
                message=f"Klart: summerade {len(ids)} artiklar (urval: lookback/källor/ämnen).",
                summary_id=str(summary_doc_id),
            )

        return summary_doc_id
    sections: List[Dict[str, Any]] = []
    stitched_parts: List[str] = []
    lookback_str = str((config.get("ingest") or {}).get("lookback") or "").strip()

    stitched_parts.append("# Sammanfattning per ämnesområde")
    if lookback_str:
        stitched_parts.append(f"_Tidsfönster: {lookback_str}_")
    stitched_parts.append("")

    all_ids: List[str] = []
    all_snaps: List[Dict[str, Any]] = []

    pts_all = [_published_ts(a) for a in to_sum]
    pts_all2 = [p for p in pts_all if p > 0]
    overall_from = min(pts_all2) if pts_all2 else 0
    overall_to = max(pts_all2) if pts_all2 else 0

    for i, topic in enumerate(topics, start=1):
        items = groups.get(topic) or []
        if not items:
            continue

        if job_id is not None:
            store.update_job(
                job_id,
                message=f"Summerar ämnesområde {i}/{len(topics)}: {topic} ({len(items)} artiklar)...",
            )

        topic_meta, topic_stats = await summarize_batches_then_meta_with_stats(
            config, items, llm=llm, store=store, job_id=job_id
        )

        ids = [a.get("id") for a in items if a.get("id")]
        all_ids.extend(ids)

        snaps = [
            {
                "id": a.get("id"),
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "source": a.get("source", ""),
                "published_ts": _published_ts(a),
                "content_hash": a.get("content_hash", ""),
                "topic": topic,
            }
            for a in items
        ]
        all_snaps.extend(snaps)

        pts = [_published_ts(a) for a in items]
        pts2 = [p for p in pts if p > 0]
        from_ts = min(pts2) if pts2 else 0
        to_ts = max(pts2) if pts2 else 0

        sections.append(
            {
                "topic": topic,
                "from": from_ts,
                "to": to_ts,
                "sources": ids,
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

    summary_doc = {
        "id": _summary_doc_id(created_ts, job_id),
        "created": created_ts,
        "kind": "summary",
        "llm": {
            "provider": (config.get("llm") or {}).get("provider", "unknown"),
            "model": (config.get("llm") or {}).get("model", "unknown"),
            "temperature": 0.2,
            "max_output_tokens": int(
                (config.get("llm") or {}).get("max_output_tokens") or 0
            ),
        },
        "prompts": load_prompts(config),
        "batching": config.get("batching", {}) or {},
        "sources": list(dict.fromkeys([x for x in all_ids if x])),  # dedupe keep order
        "sources_snapshots": all_snaps,
        "from": overall_from,
        "to": overall_to,
        "summary": stitched_summary,
        "sections": sections,
        "selection": {
            "lookback": str((config.get("ingest") or {}).get("lookback") or ""),
            "sources": _selected_source_names(config),
            "topics": _selected_topics_from_config(config),
            "prompt_package": str(
                (
                    (config.get("prompts") or {})
                    if isinstance(config.get("prompts"), dict)
                    else {}
                ).get("selected")
            )
            or "",  # Here be dragons...
        },
    }

    summary_doc_id = _persist_summary_doc(store, summary_doc)

    if job_id is not None:
        store.update_job(
            job_id,
            status="done",
            finished_at=int(time.time()),
            message=f"Klart: summerade {len(to_sum)} artiklar i {len(sections)} ämnesområden.",
            summary_id=str(summary_doc_id),
        )

    return summary_doc_id


if __name__ == "__main__":
    asyncio.run(run_pipeline("config.yaml"))
