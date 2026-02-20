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
from typing import Any, Dict, List, Optional

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
    # Feeds är nu alltid listan vi använder
    feeds = config.get("feeds")
    if isinstance(feeds, list) and all(isinstance(x, dict) for x in feeds):
        return feeds  # type: ignore[return-value]
    return []


def _set_config_sources(config: Dict[str, Any], sources: List[Dict[str, Any]]) -> None:
    config["feeds"] = sources


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

        def _name_of(s: Dict[str, Any]) -> str:
            return str(s.get("name") or s.get("title") or s.get("label") or "").strip()

        filtered = [s for s in all_sources if _name_of(s) in selected_set]
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

    def _name_of(s: Dict[str, Any]) -> str:
        return str(s.get("name") or s.get("title") or s.get("label") or "").strip()

    for s in srcs:
        n = _name_of(s)
        if n:
            out.append(n)
    return out


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
        rows.sort(key=_published_ts)  # äldsta först
        return rows

    list_articles = getattr(store, "list_articles", None)
    if callable(list_articles):
        rows = list_articles(limit=limit)
    else:
        rows = store.list_unsummarized_articles(limit=limit)  # sista utväg

    if sources:
        srcset = set(sources)
        rows = [a for a in rows if a.get("source") in srcset]
    if since_ts > 0:
        rows = [a for a in rows if _published_ts(a) >= since_ts]

    rows.sort(key=_published_ts)
    return rows[:limit]


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

    # ✅ Load feeds from config/feeds.yaml (or feeds.path / feeds_path)
    config = load_feeds_into_config(config, base_config_path=config_path)

    # ✅ Apply UI overrides AFTER feeds are loaded (so source filtering works)
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
                message="Klart: inga artiklar matchade urvalet (lookback/källor).",
            )
        return None

    meta_text, stats = await summarize_batches_then_meta_with_stats(
        config, to_sum, llm=llm, store=store, job_id=job_id
    )

    created_ts = int(time.time())
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
            message=f"Klart: summerade {len(ids)} artiklar (urval: lookback/källor).",
            summary_id=str(summary_doc_id),
        )

    return summary_doc_id


if __name__ == "__main__":
    asyncio.run(run_pipeline("config.yaml"))
