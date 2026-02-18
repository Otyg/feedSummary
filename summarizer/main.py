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

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml

from llmClient import create_llm_client
from persistence import NewsStore, create_store
from summarizer.batching import batch_articles
from summarizer.helpers import (
    interleave_by_source_oldest_first,
    load_prompts,
    setup_logging,
)
from summarizer.ingest import gather_articles_to_store
from summarizer.summarizer import summarize_batches_then_meta

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


def _extract_llm_doc(config: Dict[str, Any], llm: Any, temperature: float) -> Dict[str, Any]:
    llm_cfg = config.get("llm") or {}
    provider = str(llm_cfg.get("provider") or llm_cfg.get("type") or llm_cfg.get("client") or "")
    model = str(llm_cfg.get("model") or llm_cfg.get("name") or "")

    # fallback: om klienten exponerar cfg/model/provider
    if not provider:
        provider = str(getattr(getattr(llm, "cfg", None), "provider", "") or getattr(llm, "provider", "") or "")
    if not model:
        model = str(getattr(getattr(llm, "cfg", None), "model", "") or getattr(llm, "model", "") or "")

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
    """
    Försök spara summary-dokumentet via ny API (om den finns).
    Annars: fallback till save_summary(summary_text, ids) för bakåtkomp.
    """
    for name in ("save_summary_doc", "save_summary_document", "put_summary_doc", "insert_summary_doc"):
        fn = getattr(store, name, None)
        if callable(fn):
            return fn(doc)

    fn2 = getattr(store, "save_summary", None)
    if callable(fn2):
        # håll web/UI kompatibelt (förväntar ofta text + ids)
        return fn2(doc.get("summary", ""), doc.get("sources") or [])

    raise RuntimeError("Store saknar metod för att spara summary-dokument.")


def _estimate_batch_total(config: Dict[str, Any], articles: List[dict]) -> int:
    """
    summarize_batches_then_meta kan trimma/move/droppa och därmed ändra batchform under körning.
    Men för metadata i summary_doc räcker oftast att räkna 'default batch_total' deterministiskt.
    """
    batching = config.get("batching", {}) or {}
    max_chars = int(batching.get("max_chars_per_batch", 18000))
    max_n = int(batching.get("max_articles_per_batch", 10))
    clip_chars = int(batching.get("article_clip_chars", 6000))

    ordered = interleave_by_source_oldest_first(articles)
    batches = batch_articles(ordered, max_chars, max_n, article_clip_chars=clip_chars)
    return len(batches)


# ----------------------------
# Pipeline (orchestrates only)
# ----------------------------
async def run_pipeline(
    config_path: str = "config.yaml", job_id: Optional[int] = None
) -> Optional[Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    store = create_store(config.get("store", {}))
    llm = create_llm_client(config)

    if job_id is not None:
        store.update_job(
            job_id,
            status="running",
            started_at=int(time.time()),
            message="Startar ingest...",
        )
        logger.info("Startar ingest job %s", job_id)

    ins, upd = await gather_articles_to_store(config, store, job_id=job_id)

    if job_id is not None:
        store.update_job(
            job_id,
            message=f"Ingest klart. Inserted={ins}, Updated={upd}. Förbereder summering...",
        )

    to_sum = store.list_unsummarized_articles(limit=200)
    if not to_sum:
        if job_id is not None:
            store.update_job(
                job_id,
                status="done",
                finished_at=int(time.time()),
                message="Klart: inga nya/ändrade artiklar att summera.",
            )
        return None

    # 1) Summarize (batch+meta) – returns markdown text
    meta_text = await summarize_batches_then_meta(
        config, to_sum, llm=llm, store=store, job_id=job_id
    )

    # 2) Build summary document (your schema)
    created_ts = int(time.time())
    ids = [a.get("id") for a in to_sum if a.get("id")]

    pts = [_published_ts(a) for a in to_sum]
    pts2 = [p for p in pts if p > 0]
    from_ts = min(pts2) if pts2 else 0
    to_ts = max(pts2) if pts2 else 0

    temperature = 0.2  # matchar summarize_batches_then_meta i summarizer.py
    llm_doc = _extract_llm_doc(config, llm, temperature=temperature)
    batching_doc = _extract_batching_doc(config)

    # Best-effort meta stats (summarizer.py returnerar ej stats idag)
    batch_total = _estimate_batch_total(config, to_sum)

    llm_cfg = config.get("llm") or {}
    max_ctx = int(llm_cfg.get("context_window_tokens", 32768))
    max_out = int(llm_cfg.get("max_output_tokens", 700))
    margin = int(llm_cfg.get("prompt_safety_margin", 1024))
    # "budget" här är startbudget, inte adaptiv slutbudget (best-effort)
    meta_budget_tokens = max(0, max_ctx - max_out - margin)

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
            "batch_total": batch_total,
            "trims": 0,  # kräver att summarizer.py returnerar stats
            "drops": 0,  # kräver att summarizer.py returnerar stats
            "meta_budget_tokens": meta_budget_tokens,  # best-effort
        },
    }

    # 3) Persist summary doc (prefers new API if present)
    summary_id = _persist_summary_doc(store, summary_doc)

    # 4) Mark summarized in global corpus
    store.mark_articles_summarized(ids) # type: ignore

    if job_id is not None:
        store.update_job(
            job_id,
            status="done",
            finished_at=int(time.time()),
            message=f"Klart: summerade {len(ids)} artiklar.",
            summary_id=summary_id,
        )

    return summary_id


if __name__ == "__main__":
    asyncio.run(run_pipeline("config.yaml"))
