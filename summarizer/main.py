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
from collections import defaultdict
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
    _checkpoint_key,
    _checkpoint_path,
    _meta_ckpt_path,
    _load_checkpoint,
)
from summarizer.ingest import gather_articles_to_store
from summarizer.summarizer import (
    summarize_batches_then_meta_with_stats,
    super_meta_from_topic_sections_with_stats,
)

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


def _selected_prompt_package(config: Dict[str, Any]) -> str:
    p = config.get("prompts") or {}
    if isinstance(p, dict):
        return str(p.get("selected") or "").strip()
    return ""


def _selection_doc(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "lookback": str((config.get("ingest") or {}).get("lookback") or ""),
        "sources": _selected_source_names(config),
        "topics": _selected_topics_from_config(config),
        "prompt_package": _selected_prompt_package(config),
    }


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


def _fmt_dt_hm(ts: int) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")


def _build_sources_appendix_markdown(snapshots: List[Dict[str, Any]]) -> str:
    """
    Builds a markdown appendix grouped by source.
    Output:
      ## Källor
      ### <Source>
      - <Title> — <YYYY-MM-DD HH:MM>
        <URL>
    """
    if not snapshots:
        return ""

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in snapshots:
        src = str(s.get("source") or "").strip() or "Okänd källa"
        groups[src].append(s)

    if not groups:
        return ""

    out: List[str] = []
    out.append("## Källor")
    out.append("")

    for src in sorted(groups.keys(), key=lambda x: x.lower()):
        items = groups[src]
        items = sorted(
            items, key=lambda x: int(x.get("published_ts") or 0), reverse=True
        )

        out.append(f"### {src}")
        out.append("")

        for it in items:
            title = str(it.get("title") or "").strip() or "(utan titel)"
            url = str(it.get("url") or "").strip()
            pts = int(it.get("published_ts") or 0)
            dt = _fmt_dt_hm(pts) if pts else ""

            line = title
            if dt:
                line = f"{line} — {dt}"

            out.append(f"- {line}")
            if url:
                out.append(f"  {url}")

        out.append("")

    return "\n".join(out).strip() + "\n"


def _snapshot_topic_map_for_job(
    store: NewsStore,
    job_id: int,
    *,
    topic_map: Dict[str, List[str]],
    selection: Dict[str, Any],
    overrides: Optional[Dict[str, Any]],
) -> None:
    """
    Sparar job-kontext i store så resume blir stabil även om config ändras efteråt.
    SqliteStore lägger detta i fields_json; TinyDB lägger som vanliga keys.
    """
    try:
        store.update_job(
            job_id,
            selection=selection,
            source_topics_map=topic_map,
            overrides=overrides or {},
        )
    except Exception as e:
        logger.warning(
            "Kunde inte spara job context (selection/topic_map) i store: %s", e
        )


def _load_job_context(store: NewsStore, job_id: int) -> Dict[str, Any]:
    try:
        j = store.get_job(job_id) or {}
        if isinstance(j, dict):
            return j
    except Exception:
        pass
    return {}


def _load_ordered_articles_from_job_checkpoint(
    config: Dict[str, Any], store: NewsStore, job_id: int
) -> Tuple[List[str], List[dict]]:
    """
    Ladda article_ids från checkpoint (job_<id>.json eller job_<id>.meta.json),
    hämta artiklar från store och returnera i samma stabila ordning.
    """
    key = _checkpoint_key(job_id, [])
    cp_path = _checkpoint_path(config, key)
    meta_path = _meta_ckpt_path(config, key)

    cp = None
    if meta_path.exists():
        cp = _load_checkpoint(meta_path)
    if not cp and cp_path.exists():
        cp = _load_checkpoint(cp_path)

    if not cp:
        raise RuntimeError(
            f"Ingen checkpoint hittades för job {job_id} ({cp_path} / {meta_path})"
        )

    article_ids = cp.get("article_ids") or []
    if not article_ids:
        raise RuntimeError(f"Checkpoint saknar article_ids för job {job_id}")

    articles = store.get_articles_by_ids(article_ids)
    if not articles:
        raise RuntimeError(
            "Kunde inte ladda artiklar från store för checkpointens article_ids"
        )

    by_id = {str(a.get("id")): a for a in articles if a.get("id")}
    ordered = [by_id[i] for i in article_ids if i in by_id]
    return article_ids, ordered


def _default_summary_title(*, lookback: str, from_ts: int, to_ts: int) -> str:
    lb = (lookback or "").strip()
    if from_ts and to_ts:
        a = datetime.fromtimestamp(int(from_ts)).strftime("%Y-%m-%d")
        b = datetime.fromtimestamp(int(to_ts)).strftime("%Y-%m-%d")
        span = a if a == b else f"{a}–{b}"
        return f"Nyhetssammanfattning {span}" + (f" ({lb})" if lb else "")
    return "Nyhetssammanfattning" + (f" ({lb})" if lb else "")


async def _generate_summary_title(
    *,
    config: Dict[str, Any],
    llm,
    summary_text: str,
    from_ts: int,
    to_ts: int,
    selection: Dict[str, Any],
) -> str:
    """
    Uses prompt keys (if present in selected prompt package):
      - title_system
      - title_user_template  (expects at least {summary}; may also use {lookback}, {from_date}, {to_date})
    Falls back to a deterministic title if prompts missing or LLM fails.
    """
    lookback = str((selection or {}).get("lookback") or "").strip()
    fallback = _default_summary_title(lookback=lookback, from_ts=from_ts, to_ts=to_ts)

    prompts = load_prompts(config)
    sys_p = str(prompts.get("title_system") or "").strip()
    user_t = str(prompts.get("title_user_template") or "").strip()
    if not sys_p or not user_t:
        return fallback

    from_date = (
        datetime.fromtimestamp(int(from_ts)).strftime("%Y-%m-%d") if from_ts else ""
    )
    to_date = datetime.fromtimestamp(int(to_ts)).strftime("%Y-%m-%d") if to_ts else ""

    try:
        user = user_t.format(
            summary=(summary_text or "").strip(),
            lookback=lookback,
            from_date=from_date,
            to_date=to_date,
        )
    except Exception:
        # template formatting mismatch -> fallback
        return fallback

    try:
        out = await llm.chat(
            [{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
            temperature=0.2,
        )
        title = str(out or "").strip()
    except Exception:
        return fallback

    if not title:
        return fallback

    # normalize: take first line, strip quotes, cap length
    title = title.splitlines()[0].strip().strip('"').strip("'").strip()
    if not title:
        return fallback

    if len(title) > 120:
        title = title[:120].rstrip() + "…"
    return title


async def _summarize_and_persist_like_refresh(
    *,
    config: Dict[str, Any],
    store: NewsStore,
    llm,
    job_id: Optional[int],
    articles: List[dict],
    topic_map: Dict[str, List[str]],
    selection: Dict[str, Any],
) -> Any:
    """
    Bygger summary_doc IDENTISKT med refresh-flödet:
      - single-topic => meta + källappendix + selection
      - multi-topic  => sections + overview (super-meta) + appendix + selection
    """
    groups = _group_articles_by_primary_topic(articles, topic_map)
    topics = _topic_order(groups)
    created_ts = int(time.time())

    # ----------------------------
    # Single summary (no topic split)
    # ----------------------------
    if len(topics) <= 1:
        meta_text, stats = await summarize_batches_then_meta_with_stats(
            config, articles, llm=llm, store=store, job_id=job_id
        )

        ids = [a.get("id") for a in articles if a.get("id")]

        pts = [_published_ts(a) for a in articles]
        pts2 = [p for p in pts if p > 0]
        from_ts = min(pts2) if pts2 else 0
        to_ts = max(pts2) if pts2 else 0

        snapshots = [
            {
                "id": a.get("id"),
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "source": a.get("source", ""),
                "published_ts": _published_ts(a),
                "content_hash": a.get("content_hash", ""),
            }
            for a in articles
        ]

        appendix = _build_sources_appendix_markdown(snapshots)
        if appendix:
            meta_text = (meta_text or "").rstrip() + "\n\n" + appendix

        title = await _generate_summary_title(
            config=config,
            llm=llm,
            summary_text=meta_text or "",
            from_ts=from_ts,
            to_ts=to_ts,
            selection=selection,
        )

        summary_doc: Dict[str, Any] = {
            "id": _summary_doc_id(created_ts, job_id),
            "title": title,
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
            "sources_snapshots": snapshots,
            "from": from_ts,
            "to": to_ts,
            "summary": meta_text,
            "meta": {
                "batch_total": int(stats.get("batch_total") or 0),
                "trims": int(stats.get("trims") or 0),
                "drops": int(stats.get("drops") or 0),
                "meta_budget_tokens": int(stats.get("meta_budget_tokens") or 0),
            },
            "selection": dict(selection or {}),
        }

        return _persist_summary_doc(store, summary_doc)

    # ----------------------------
    # Multi-topic summary
    # ----------------------------
    sections: List[Dict[str, Any]] = []
    stitched_parts: List[str] = []
    lookback_str = str((config.get("ingest") or {}).get("lookback") or "").strip()

    stitched_parts.append("# Sammanfattning per ämnesområde")
    if lookback_str:
        stitched_parts.append(f"_Tidsfönster: {lookback_str}_")
    stitched_parts.append("")

    all_ids: List[str] = []
    all_snaps: List[Dict[str, Any]] = []

    pts_all = [_published_ts(a) for a in articles]
    pts_all2 = [p for p in pts_all if p > 0]
    overall_from = min(pts_all2) if pts_all2 else 0
    overall_to = max(pts_all2) if pts_all2 else 0

    topics = _topic_order(groups)

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

    # Super-meta overview (optional, prompt-driven)
    overview_text = ""
    overview_stats: Dict[str, Any] = {
        "super_meta_budget_tokens": 0,
        "super_meta_enabled": 0,
    }
    try:
        overview_text, overview_stats = await super_meta_from_topic_sections_with_stats(
            config=config, sections=sections, llm=llm, store=store, job_id=job_id
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

    title = await _generate_summary_title(
        config=config,
        llm=llm,
        summary_text=final_summary or "",
        from_ts=overall_from,
        to_ts=overall_to,
        selection=selection,
    )

    summary_doc = {
        "id": _summary_doc_id(created_ts, job_id),
        "title": title,
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
        "sources": list(dict.fromkeys([x for x in all_ids if x])),
        "sources_snapshots": all_snaps,
        "from": overall_from,
        "to": overall_to,
        "overview": overview_text,
        "summary": final_summary,
        "sections": sections,
        "meta": {
            "super_meta_enabled": int(overview_stats.get("super_meta_enabled") or 0),
            "super_meta_budget_tokens": int(
                overview_stats.get("super_meta_budget_tokens") or 0
            ),
        },
        "selection": dict(selection or {}),
    }

    return _persist_summary_doc(store, summary_doc)


async def run_pipeline(
    config_path: str = "config.yaml",
    job_id: Optional[int] = None,
    overrides: Optional[Dict[str, Any]] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    llm=None,
) -> Optional[Any]:
    """
    Normal refresh pipeline.

    NEW:
      - status="failed" + finished_at vid exception
      - sparar selection/topic_map/overrides i job record för stabil resume
    """
    try:
        if config_dict is None:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        else:
            config = config_dict

        config = load_feeds_into_config(config, base_config_path=config_path)
        config = _apply_overrides(config, overrides)

        store = create_store(config.get("store", {}))
        if llm is None:
            llm = create_llm_client(config)

        if job_id is not None:
            store.update_job(
                job_id,
                status="running",
                started_at=int(time.time()),
                message="Startar ingest...",
            )

        # Ingest
        ins, upd = await gather_articles_to_store(config, store, job_id=job_id)

        if job_id is not None:
            store.update_job(
                job_id,
                message=f"Ingest klart. Inserted={ins}, Updated={upd}. Förbereder summering...",
            )

        # Selection
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
        selection = _selection_doc(config)

        # Save job context for stable resume
        if job_id is not None:
            _snapshot_topic_map_for_job(
                store,
                job_id,
                topic_map=topic_map,
                selection=selection,
                overrides=overrides,
            )

        # Summarize + persist (same structure as refresh)
        summary_doc_id = await _summarize_and_persist_like_refresh(
            config=config,
            store=store,
            llm=llm,
            job_id=job_id,
            articles=to_sum,
            topic_map=topic_map,
            selection=selection,
        )

        if job_id is not None:
            store.update_job(
                job_id,
                status="done",
                finished_at=int(time.time()),
                message=f"Klart: summerade {len(to_sum)} artiklar.",
                summary_id=str(summary_doc_id),
            )

        return summary_doc_id

    except Exception as e:
        # NEW: mark job failed on exception
        if job_id is not None:
            try:
                store.update_job(
                    job_id,
                    status="failed",
                    finished_at=int(time.time()),
                    message=f"Fel: {e}",
                )
            except Exception:
                pass
        raise


async def run_resume_job(
    *,
    config: Dict[str, Any],
    store: NewsStore,
    llm,
    job_id: int,
) -> str:
    """
    Resume som producerar samma summary_doc-struktur som en vanlig refresh.

    - Läser artiklarna från checkpoint (stabil corpus)
    - Försöker läsa selection/topic_map från job-recorden (stabilt även om config ändrats)
    - Bygger sections + appendix + selection osv
    - Sätter job status failed vid exception
    """
    jid = int(job_id)
    try:
        # Try load job context (saved during original run)
        ctx = _load_job_context(store, jid)

        selection = ctx.get("selection")
        if not isinstance(selection, dict):
            selection = _selection_doc(config)

        topic_map = ctx.get("source_topics_map")
        if not isinstance(topic_map, dict):
            topic_map = _source_topics_map(config)

        # Load stable ordered articles from checkpoint/meta-checkpoint
        _article_ids, ordered_articles = _load_ordered_articles_from_job_checkpoint(
            config, store, jid
        )

        # Summarize + persist like refresh
        summary_doc_id = await _summarize_and_persist_like_refresh(
            config=config,
            store=store,
            llm=llm,
            job_id=jid,
            articles=ordered_articles,
            topic_map=topic_map,
            selection=selection,
        )

        # Mark job done
        try:
            store.update_job(
                jid,
                status="done",
                finished_at=int(time.time()),
                message=f"Resume klart: summerade {len(ordered_articles)} artiklar.",
                summary_id=str(summary_doc_id),
            )
        except Exception:
            pass

        return str(summary_doc_id)

    except Exception as e:
        try:
            store.update_job(
                jid,
                status="failed",
                finished_at=int(time.time()),
                message=f"Resume fel: {e}",
            )
        except Exception:
            pass
        raise


if __name__ == "__main__":
    asyncio.run(run_pipeline("config.yaml"))
