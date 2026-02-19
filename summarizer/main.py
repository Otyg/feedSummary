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
from summarizer.helpers import setup_logging, load_prompts, parse_lookback_to_seconds
from summarizer.ingest import gather_articles_to_store
from summarizer.summarizer import summarize_batches_then_meta_with_stats

setup_logging()
logger = logging.getLogger(__name__)


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or isinstance(v, bool):
            return default
        return int(v)
    except Exception:
        return default


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


def _extract_llm_doc(
    config: Dict[str, Any], llm: Any, temperature: float
) -> Dict[str, Any]:
    llm_cfg = config.get("llm") or {}
    provider = str(
        llm_cfg.get("provider") or llm_cfg.get("type") or llm_cfg.get("client") or ""
    )
    model = str(llm_cfg.get("model") or llm_cfg.get("name") or "")

    if not provider:
        provider = str(
            getattr(getattr(llm, "cfg", None), "provider", "")
            or getattr(llm, "provider", "")
            or ""
        )
    if not model:
        model = str(
            getattr(getattr(llm, "cfg", None), "model", "")
            or getattr(llm, "model", "")
            or ""
        )

    return {
        "provider": provider or "unknown",
        "model": model or "unknown",
        "temperature": temperature,
        "max_output_tokens": _safe_int(llm_cfg.get("max_output_tokens"), 0),
    }


def _extract_batching_doc(config: Dict[str, Any]) -> Dict[str, Any]:
    batching = config.get("batching", {}) or {}
    return {
        "ordering": str(batching.get("ordering") or "source_interleave_oldest_first"),
        "max_articles_per_batch": _safe_int(batching.get("max_articles_per_batch"), 0),
        "max_chars_per_batch": _safe_int(batching.get("max_chars_per_batch"), 0),
        "article_clip_chars": _safe_int(batching.get("article_clip_chars"), 0),
    }


def _sources_snapshots(articles: List[dict]) -> List[dict]:
    snaps: List[dict] = []
    for a in articles:
        snaps.append(
            {
                "id": a.get("id"),
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "source": a.get("source", ""),
                "published_ts": _published_ts(a),
                "content_hash": a.get("content_hash", ""),
            }
        )
    return snaps


def _persist_summary_doc(store: NewsStore, doc: Dict[str, Any]) -> Any:
    for name in (
        "save_summary_doc",
        "save_summary_document",
        "put_summary_doc",
        "insert_summary_doc",
    ):
        fn = getattr(store, name, None)
        if callable(fn):
            return fn(doc)
    raise RuntimeError(
        "Store saknar metod för att spara summary-dokument (summary_docs)."
    )


def _get_config_sources(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Försök hitta sources-lista i vanliga nycklar.
    Vi förväntar oss listor av dictar med åtminstone name+url (eller title+url).
    """
    candidates = [
        config.get("sources"),
        config.get("feeds"),
        config.get("rss_sources"),
        (config.get("ingest") or {}).get("sources"),
        (config.get("ingest") or {}).get("feeds"),
    ]
    for c in candidates:
        if isinstance(c, list) and c and all(isinstance(x, dict) for x in c):
            return c  # type: ignore[return-value]
    return []


def _set_config_sources(config: Dict[str, Any], sources: List[Dict[str, Any]]) -> None:
    """
    Sätt sources tillbaka där de redan låg (prioritet), annars i config["sources"].
    """
    if isinstance(config.get("sources"), list):
        config["sources"] = sources
        return
    if isinstance(config.get("feeds"), list):
        config["feeds"] = sources
        return
    if isinstance(config.get("rss_sources"), list):
        config["rss_sources"] = sources
        return
    ingest = config.setdefault("ingest", {})
    if isinstance(ingest, dict):
        if isinstance(ingest.get("sources"), list):
            ingest["sources"] = sources
            return
        if isinstance(ingest.get("feeds"), list):
            ingest["feeds"] = sources
            return
    config["sources"] = sources


def _apply_overrides(
    config: Dict[str, Any], overrides: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    overrides:
      - lookback: str   ex: "24h", "3d"
      - sources: List[str]  list of source names to include
      - prompt_package: str
    """
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
        if all_sources:

            def _name_of(s: Dict[str, Any]) -> str:
                return str(
                    s.get("name") or s.get("title") or s.get("label") or ""
                ).strip()

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
    """
    Välj artiklar baserat på:
      - ingest.lookback (default-värde, men överstyrbart från UI via overrides)
      - valda källor (config feeds/sources filtreras av overrides från UI)

    Viktigt: urvalet ignorerar 'summarized'-flagga helt.
    """
    ingest = config.get("ingest") or {}
    lookback = str(ingest.get("lookback") or "").strip()
    sources = _selected_source_names(config)

    now = int(time.time())
    since_ts = 0
    if lookback:
        since_ts = now - parse_lookback_to_seconds(lookback)

    # Föredra store-filter om det finns
    list_by_filter = getattr(store, "list_articles_by_filter", None)
    if callable(list_by_filter) and since_ts > 0 and sources:
        rows = list_by_filter(
            sources=sources, since_ts=since_ts, until_ts=now, limit=limit
        )
        # säkra ordning: äldsta först
        rows.sort(key=_published_ts)
        return rows

    # fallback: hämta allt och filtrera i minnet
    list_articles = getattr(store, "list_articles", None)
    if callable(list_articles):
        rows = list_articles(limit=limit)
    else:
        # sista utväg
        rows = store.list_unsummarized_articles(limit=limit)

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
    """
    Orchestrator.
    - config.yaml values are defaults
    - overrides (from webapp) can set ingest.lookback + chosen sources + prompt_package
    - article store contains only articles; summary selection is based on lookback/sources (NOT summarized flag)
    """
    if config_dict is None:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = config_dict

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

    # ✅ NYTT: urval styrs av lookback + valda källor (från UI overrides)
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

    temperature = 0.2
    llm_doc = _extract_llm_doc(config, llm, temperature=temperature)
    batching_doc = _extract_batching_doc(config)

    summary_doc: Dict[str, Any] = {
        "id": _summary_doc_id(created_ts, job_id),
        "created": created_ts,
        "kind": "summary",
        "llm": llm_doc,
        "prompts": load_prompts(config),
        "batching": batching_doc,
        "sources": ids,
        "sources_snapshots": _sources_snapshots(to_sum),
        "from": from_ts,
        "to": to_ts,
        "summary": meta_text,
        "meta": {
            "batch_total": int(stats.get("batch_total") or 0),
            "trims": int(stats.get("trims") or 0),
            "drops": int(stats.get("drops") or 0),
            "meta_budget_tokens": int(stats.get("meta_budget_tokens") or 0),
        },
    }

    summary_doc_id = _persist_summary_doc(store, summary_doc)

    legacy_summary_id = None
    try:
        legacy_summary_id = store.save_summary(meta_text, ids)
    except Exception:
        legacy_summary_id = None

    # ✅ VIKTIGT: vi markerar INTE artiklar som "summarized" längre.
    # Det är sammanfattningslogikens urval (lookback/källor) som avgör.

    if job_id is not None:
        store.update_job(
            job_id,
            status="done",
            finished_at=int(time.time()),
            message=f"Klart: summerade {len(ids)} artiklar (urval: lookback/källor).",
            summary_id=summary_doc_id,
            legacy_summary_id=legacy_summary_id,
        )

    return summary_doc_id


if __name__ == "__main__":
    asyncio.run(run_pipeline("config.yaml"))
