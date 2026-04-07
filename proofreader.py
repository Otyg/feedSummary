#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Proofread + revise an EXISTING summary_doc using the articles that summary_doc references.

Commands:
  - run  : proofread+revise a summary_doc, caching batch_output in a local run DB
  - list : list runs in the local run DB (optionally filtered by summary id)
  - show : show a single run by run-doc-id, or latest run for a summary id

Local run DB (TinyDB JSON):
- original_summary_id
- original_summary
- lookback_label
- batch_output (cached; reused across reruns)
- sources_text
- corrected_summary
- proofread_output
- prompts (proofread/revise prompt texts + hashes; optional batch prompt texts)
- versions (schema_version + run_revision)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from tinydb import Query, TinyDB

from feedsummary_core.llm_client import create_llm_client, LLMClient
from feedsummary_core.persistence import create_store, NewsStore
from feedsummary_core.summarizer.helpers import (
    clip_text,
    interleave_by_source_oldest_first,
    load_prompts,
    lookback_label_from_range,
    set_job,
)
from feedsummary_core.summarizer.batching import (
    PromptTooLongStructural,
    _choose_trim_action,
    _estimate_article_chars,
    _move_article_to_tail_batch,
    batch_articles,
    build_messages_for_batch,
    trim_text_tail_by_words,
)
from feedsummary_core.summarizer.chat import chat_guarded
from feedsummary_core.summarizer.summarizer import _proofread_and_revise_meta_with_stats

from uicommon import primary_llm_config
from uicommon.bootstrap_ui import _setup_logging_if_needed

log = logging.getLogger(__name__)
_setup_logging_if_needed()

SCHEMA_VERSION = 1


class _DryRunLLM:
    """
    Test-double for LLM calls that validates payload presence without sending
    network requests.
    """

    def __init__(
        self,
        *,
        batch_system: str,
        proofread_system: str,
        revise_system: str,
    ) -> None:
        self._batch_system = (batch_system or "").strip()
        self._proofread_system = (proofread_system or "").strip()
        self._revise_system = (revise_system or "").strip()
        self.batch_calls = 0
        self.proofread_calls = 0
        self.revise_calls = 0

    async def chat(
        self, messages: List[Dict[str, str]], *, temperature: float = 0.2
    ) -> str:
        if not messages:
            raise RuntimeError("Dry-run validation failed: messages list is empty")

        nonempty_contents = [str(m.get("content") or "").strip() for m in messages]
        if not any(nonempty_contents):
            raise RuntimeError(
                "Dry-run validation failed: no message in request had non-empty content"
            )

        system_content = str((messages[0] or {}).get("content") or "").strip()
        user_content = str((messages[-1] or {}).get("content") or "").strip()
        if not user_content:
            raise RuntimeError("Dry-run validation failed: user message is empty")

        if system_content and system_content == self._batch_system:
            self.batch_calls += 1
            return f"[DRY-RUN] batch_ok_{self.batch_calls}"

        if system_content and system_content == self._proofread_system:
            self.proofread_calls += 1
            # First proofread call returns FAIL so revise-step is exercised.
            if self.proofread_calls == 1:
                return "FAIL\nISSUES:\n- Dry-run: verifierar revise-steget."
            return "PASS"

        if system_content and system_content == self._revise_system:
            self.revise_calls += 1
            return "[DRY-RUN] revised_summary_ok"

        # Unknown phase still returns a non-empty string after payload validation.
        return "[DRY-RUN] generic_ok"


def _missing_prompt_keys(prompts: Dict[str, Any], keys: List[str]) -> List[str]:
    missing: List[str] = []
    for k in keys:
        if not str(prompts.get(k) or "").strip():
            missing.append(k)
    return missing


def _read_optional_text_file(path: str) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    return p.read_text(encoding="utf-8").strip()


def _first_nonempty_str(*values: Any) -> str:
    for v in values:
        s = str(v or "").strip()
        if s:
            return s
    return ""


def _extract_prompt_package_from_summary_doc(doc: Dict[str, Any]) -> str:
    # Newer summary-doc shape
    selection = doc.get("selection") or {}
    prompts = doc.get("prompts") or {}
    if isinstance(selection, dict) and isinstance(prompts, dict):
        s = _first_nonempty_str(
            selection.get("prompt_package"),
            prompts.get("_package"),
            prompts.get("prompt_package"),
        )
        if s:
            return s

    # Older/alternate shapes
    top = _first_nonempty_str(doc.get("prompt_package"))
    if top:
        return top

    # Legacy compatibility: doc_json may be dict or JSON string
    dj = doc.get("doc_json")
    dj_obj: Dict[str, Any] = {}
    if isinstance(dj, dict):
        dj_obj = dj
    elif isinstance(dj, str):
        try:
            parsed = json.loads(dj)
            if isinstance(parsed, dict):
                dj_obj = parsed
        except Exception:
            dj_obj = {}

    if dj_obj:
        dj_sel = dj_obj.get("selection") or {}
        dj_prompts = dj_obj.get("prompts") or {}
        return _first_nonempty_str(
            dj_obj.get("prompt_package"),
            (dj_sel.get("prompt_package") if isinstance(dj_sel, dict) else ""),
            (dj_prompts.get("_package") if isinstance(dj_prompts, dict) else ""),
            (dj_prompts.get("prompt_package") if isinstance(dj_prompts, dict) else ""),
        )

    return ""


def _load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _get_store_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    for k in ("store", "db", "database", "persistence"):
        v = cfg.get(k)
        if isinstance(v, dict):
            return v
    return {
        "provider": cfg.get("store_provider", "tinydb"),
        "path": cfg.get("store_path", "news_docs.json"),
    }


def _derive_lookback_label(cfg: Dict[str, Any], summary_doc: Dict[str, Any]) -> str:
    ingest = cfg.get("ingest") or {}
    lookback_raw = str(ingest.get("lookback") or "").strip()

    from_ts = int(summary_doc.get("from") or 0)
    to_ts = int(summary_doc.get("to") or 0)

    if from_ts and to_ts:
        return lookback_label_from_range(lookback_raw, from_ts, to_ts)

    return lookback_raw


def _hash_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def _hash_prompts_subset(prompts: Dict[str, str], keys: List[str]) -> str:
    blob = "\n\n".join([f"=={k}==\n{prompts.get(k, '')}" for k in keys])
    return _hash_text(blob)


def _update_summary_doc_audit(
    *,
    summary_doc: Dict[str, Any],
    original_summary: str,
    revised_summary: str,
    proofread_output: str,
    proofread_trace: List[Dict[str, Any]],
    proofread_rounds: int,
    proofread_last_feedback: str,
    prompt_package: str,
    run_revision: int,
) -> Dict[str, Any]:
    """
    Store original vs revised snapshots in summary_doc so they can be compared
    later in the primary DB.
    """
    out = dict(summary_doc or {})
    now_ts = int(time.time())

    # Simple top-level fields for easy ad-hoc checks.
    out["proofread_original_summary"] = str(original_summary or "")
    out["proofread_revised_summary"] = str(revised_summary or "")

    entry = {
        "created_at": now_ts,
        "run_revision": int(run_revision),
        "prompt_package": str(prompt_package or ""),
        "original_summary": str(original_summary or ""),
        "revised_summary": str(revised_summary or ""),
        "proofread_output": str(proofread_output or ""),
        "proofread_rounds": int(proofread_rounds or 0),
        "proofread_last_feedback": str(proofread_last_feedback or ""),
        "proofread_trace": list(proofread_trace or []),
    }

    pa = out.get("proofread_audit") or {}
    if not isinstance(pa, dict):
        pa = {}
    history = pa.get("history") or []
    if not isinstance(history, list):
        history = []
    history.append(entry)
    history = history[-20:]

    out["proofread_audit"] = {
        "latest": entry,
        "history": history,
    }
    return out


def _open_run_db(path: str) -> TinyDB:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return TinyDB(str(p))


def _get_latest_run_for_summary(
    db: TinyDB, original_summary_id: str
) -> Optional[Dict[str, Any]]:
    R = Query()
    rows = db.table("runs").search(R.original_summary_id == original_summary_id)
    if not rows:
        return None
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            int((r.get("versions") or {}).get("run_revision") or 0),
            int(r.get("created_at") or 0),
        ),
        reverse=True,
    )
    return dict(rows_sorted[0])


def _next_run_revision(db: TinyDB, original_summary_id: str) -> int:
    latest = _get_latest_run_for_summary(db, original_summary_id)
    if not latest:
        return 1
    return int((latest.get("versions") or {}).get("run_revision") or 0) + 1


def _batch_output_to_tuples(
    batch_output: List[Dict[str, Any]],
) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for item in batch_output or []:
        try:
            idx = int(item.get("batch_index") or 0)
        except Exception:
            idx = 0
        txt = str(item.get("text") or "").strip()
        if idx > 0 and txt:
            out.append((idx, txt))
    out.sort(key=lambda x: x[0])
    return out


async def _build_batch_output_from_articles(
    *,
    cfg: Dict[str, Any],
    prompts: Dict[str, str],
    llm: LLMClient,
    articles: List[dict],
    store: NewsStore,
) -> List[Dict[str, Any]]:
    batching = cfg.get("batching", {}) or {}
    max_chars = int(batching.get("max_chars_per_batch", 18000))
    max_n = int(batching.get("max_articles_per_batch", 10))
    article_clip_chars = int(batching.get("article_clip_chars", 6000))

    llm_cfg = primary_llm_config(cfg)
    max_ctx = int(llm_cfg.get("context_window_tokens", 32768))
    max_out = int(llm_cfg.get("max_output_tokens", 700))
    margin = int(llm_cfg.get("prompt_safety_margin", 1024))
    chars_per_token = float(llm_cfg.get("token_chars_per_token", 2.4))
    max_attempts = int(llm_cfg.get("prompt_too_long_max_attempts", 6))
    structural_threshold = int(
        llm_cfg.get("prompt_too_long_structural_threshold_tokens", 1200)
    )

    ordered = interleave_by_source_oldest_first(articles)
    batches = batch_articles(
        ordered, max_chars, max_n, article_clip_chars=article_clip_chars
    )

    out: List[Dict[str, Any]] = []
    idx = 1
    while idx <= len(batches):
        batch = batches[idx - 1]
        set_job(
            f"[offline] Summerar batch {idx}/{len(batches)}...",
            job_id=None,
            store=store,
        )
        log.info(
            f"Building batch summary {idx}/{len(batches)} with {len(batch)} articles..."
        )

        while True:
            try:
                summary = await chat_guarded(
                    llm=llm,
                    messages=build_messages_for_batch(
                        prompts=prompts,
                        batch_index=idx,
                        batch_total=len(batches),
                        batch_items=batch,
                    ),
                    temperature=0.2,
                    max_ctx=max_ctx,
                    max_out=max_out,
                    margin=margin,
                    chars_per_token=chars_per_token,
                    max_attempts=max_attempts,
                    structural_threshold=structural_threshold,
                )
                break
            except PromptTooLongStructural as e:
                overflow = int(getattr(e, "overflow_tokens", 0) or 0)
                action = _choose_trim_action(overflow, structural_threshold)
                log.warning(
                    f"Batch {idx}/{len(batches)} prompt too long by {overflow} tokens. Action: {action}"
                )

                if len(batch) <= 1:
                    a0 = batch[0]
                    remove_tokens = (overflow + 2048) if overflow else 4096
                    a0["text"] = trim_text_tail_by_words(
                        a0.get("text", "") or "",
                        remove_tokens,
                        chars_per_token=chars_per_token,
                    )
                    continue

                target_remove_tokens = overflow + 1024
                target_remove_chars = int(target_remove_tokens * chars_per_token)

                removed_chars = 0
                if action == "drop_one_article":
                    a = batch.pop()
                    removed_chars = _estimate_article_chars(a)
                    _move_article_to_tail_batch(
                        batches,
                        a,
                        max_chars_per_batch=max_chars,
                        max_articles_per_batch=max_n,
                        avoid_batch=batch,
                    )
                else:
                    while len(batch) > 1 and removed_chars < target_remove_chars:
                        a = batch.pop()
                        removed_chars += _estimate_article_chars(a)
                        _move_article_to_tail_batch(
                            batches,
                            a,
                            max_chars_per_batch=max_chars,
                            max_articles_per_batch=max_n,
                            avoid_batch=batch,
                        )
                continue

        out.append({"batch_index": idx, "text": str(summary or "").strip()})
        idx += 1

    return out


def _sources_text_from_articles(articles: List[dict], clip_chars: int = 140) -> str:
    return "\n".join(
        [
            f"- {clip_text(a.get('title', ''), clip_chars)} — {(a.get('url') or '').strip()}"
            for a in (articles or [])
        ]
    ).strip()


def _fmt_ts(ts: int) -> str:
    if not ts:
        return ""
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(ts)))
    except Exception:
        return str(ts)


def _list_runs(run_db: TinyDB, *, summary_id: str = "", limit: int = 50) -> int:
    t = run_db.table("runs")
    rows = list(t.all())

    if summary_id:
        rows = [
            r for r in rows if str(r.get("original_summary_id") or "") == summary_id
        ]

    rows.sort(
        key=lambda r: (
            str(r.get("original_summary_id") or ""),
            int((r.get("versions") or {}).get("run_revision") or 0),
            int(r.get("created_at") or 0),
        ),
        reverse=True,
    )

    rows = rows[: max(1, int(limit))]

    if not rows:
        log.info("No runs found.")
        return 0

    log.info("Runs:")
    for r in rows:
        doc_id = getattr(r, "doc_id", None)  # TinyDB Document
        # If tinydb returns plain dicts, doc_id is not present; try stored field fallback
        rid = doc_id if doc_id is not None else r.get("_doc_id") or "?"
        sid = str(r.get("original_summary_id") or "")
        rev = int((r.get("versions") or {}).get("run_revision") or 0)
        created = _fmt_ts(int(r.get("created_at") or 0))
        pkg = str((r.get("prompts") or {}).get("prompt_package") or "")
        bh = str((r.get("prompts") or {}).get("batch_prompts_hash") or "")[:10]
        ph = str((r.get("prompts") or {}).get("proofread_revise_prompts_hash") or "")[
            :10
        ]
        log.info(
            f"- run_doc_id={rid} summary_id={sid} run_revision={rev} created={created} pkg={pkg} batch_hash={bh} pr_hash={ph}"
        )

    return 0


def _show_run(
    run_db: TinyDB,
    *,
    run_doc_id: int = 0,
    summary_id: str = "",
    show_batch: bool = False,
) -> int:
    t = run_db.table("runs")

    doc: Optional[Dict[str, Any]] = None

    if run_doc_id:
        d = t.get(doc_id=int(run_doc_id))
        if d:
            doc = {"_doc_id": int(run_doc_id), **dict(d)}
    else:
        if not summary_id:
            log.error("show requires --run-doc-id or --summary-id")
            return 2
        latest = _get_latest_run_for_summary(run_db, summary_id)
        if latest:
            doc = dict(latest)
        else:
            log.error(f"No runs found for summary_id={summary_id}")
            return 2

    if not doc:
        log.error("Run not found.")
        return 2

    rid = doc.get("_doc_id") or "?"
    sid = str(doc.get("original_summary_id") or "")
    rev = int((doc.get("versions") or {}).get("run_revision") or 0)
    created = _fmt_ts(int(doc.get("created_at") or 0))
    pkg = str((doc.get("prompts") or {}).get("prompt_package") or "")

    log.info("\n" + "=" * 90)
    log.info(f"RUN_DOC_ID: {rid}")
    log.info(f"ORIGINAL_SUMMARY_ID: {sid}")
    log.info(f"RUN_REVISION: {rev}")
    log.info(f"CREATED: {created}")
    log.info(f"PROMPT_PACKAGE: {pkg}")
    log.info("=" * 90 + "\n")

    log.info("ORIGINAL SUMMARY (clip 2000):\n")
    log.info(clip_text(str(doc.get("original_summary") or ""), 2000))
    log.info("\n" + "-" * 90 + "\n")

    log.info("PROOFREAD OUTPUT:\n")
    log.info(str(doc.get("proofread_output") or "").strip() or "(no proofread output)")
    log.info("\n" + "-" * 90 + "\n")

    log.info("CORRECTED SUMMARY:\n")
    log.info(str(doc.get("corrected_summary") or "").strip())
    log.info("\n" + "-" * 90 + "\n")

    prompts = doc.get("prompts") or {}
    log.info("PROMPTS (proofread/revise):")
    log.info(
        f"- proofread_revise_prompts_hash: {str(prompts.get('proofread_revise_prompts_hash') or '')}"
    )
    log.info("  proofread_system:\n" + (prompts.get("proofread_system") or ""))
    log.info(
        "  proofread_user_template:\n" + (prompts.get("proofread_user_template") or "")
    )
    log.info("  revise_system:\n" + (prompts.get("revise_system") or ""))
    log.info("  revise_user_template:\n" + (prompts.get("revise_user_template") or ""))

    if prompts.get("batch_system") or prompts.get("batch_user_template"):
        log.info("\nPROMPTS (batch):")
        log.info(
            f"- batch_prompts_hash: {str(prompts.get('batch_prompts_hash') or '')}"
        )
        if prompts.get("batch_system"):
            log.info("  batch_system:\n" + (prompts.get("batch_system") or ""))
        if prompts.get("batch_user_template"):
            log.info(
                "  batch_user_template:\n" + (prompts.get("batch_user_template") or "")
            )

    if show_batch:
        batch_output = doc.get("batch_output") or []
        log.info("\n" + "-" * 90)
        log.info(f"BATCH_OUTPUT: {len(batch_output)} batches")
        for b in batch_output:
            bi = b.get("batch_index")
            bt = str(b.get("text") or "")
            log.info(f"\n--- BATCH {bi} ---\n{bt}\n")

    log.info("\n" + "=" * 90 + "\n")
    return 0


async def _run_command(args: argparse.Namespace) -> int:
    cfg = _load_yaml(args.config)

    store_cfg = _get_store_cfg(cfg)
    if args.sqlite_db:
        sqlite_db = str(Path(args.sqlite_db).expanduser())
        store_cfg["provider"] = "sqlite"
        store_cfg["path"] = sqlite_db
        log.info(f"Using SQLite DB override from CLI: {sqlite_db}")

    store: NewsStore = create_store(store_cfg)

    run_db = _open_run_db(args.run_db)
    runs_table = run_db.table("runs")

    # Load summary_doc
    if args.latest:
        doc = store.get_latest_summary_doc()
        if not doc:
            log.error("No summary_docs found in DB.")
            return 2
    else:
        if not args.summary_id:
            log.error("Need --summary-id or --latest")
            return 2
        doc = store.get_summary_doc(args.summary_id)
        if not doc:
            log.error(f"summary_doc not found: {args.summary_id}")
            return 2

    original_summary_id = str(doc.get("id") or args.summary_id or "latest")
    original_summary = str(doc.get("summary") or "").strip()
    if not original_summary:
        log.error(f"summary_doc {original_summary_id} has empty 'summary'")
        return 3

    sources: List[str] = [str(x) for x in (doc.get("sources") or []) if str(x).strip()]
    if not sources:
        log.error(f"summary_doc {original_summary_id} has no 'sources' list")
        return 3

    # Resolve effective prompt package:
    # 1) --prompt-package (explicit override)
    # 2) package persisted in summary_doc
    # 3) config default behavior
    selected_pkg = str(args.prompt_package or "").strip()
    if not selected_pkg:
        selected_pkg = _extract_prompt_package_from_summary_doc(doc)
        if selected_pkg:
            log.info(
                "Using prompt package from summary_doc %s: %s",
                original_summary_id,
                selected_pkg,
            )

    if selected_pkg:
        p = cfg.get("prompts")
        if not isinstance(p, dict):
            p = {}
            cfg["prompts"] = p
        p["selected"] = selected_pkg

    prompts = load_prompts(cfg)
    proofread_keys = [
        "proofread_system",
        "proofread_user_template",
        "revise_system",
        "revise_user_template",
    ]

    # Optional fallback package for proofread/revise prompts only.
    if args.proofread_package:
        fallback_prompts = load_prompts(cfg, package=str(args.proofread_package))
        filled = []
        for k in proofread_keys:
            if not str(prompts.get(k) or "").strip():
                v = str(fallback_prompts.get(k) or "").strip()
                if v:
                    prompts[k] = v
                    filled.append(k)
        if filled:
            log.info(
                "Filled missing proofread/revise prompts from --proofread-package '%s': %s",
                str(args.proofread_package),
                ", ".join(filled),
            )

    # Optional per-key file overrides (highest precedence).
    file_overrides = {
        "proofread_system": str(args.proofread_system_file or "").strip(),
        "proofread_user_template": str(args.proofread_user_template_file or "").strip(),
        "revise_system": str(args.revise_system_file or "").strip(),
        "revise_user_template": str(args.revise_user_template_file or "").strip(),
    }
    for key, file_path in file_overrides.items():
        if file_path:
            prompts[key] = _read_optional_text_file(file_path)
            log.info("Using %s override from file: %s", key, file_path)

    if args.dry_run:
        llm: LLMClient = _DryRunLLM(
            batch_system=str(prompts.get("batch_system") or ""),
            proofread_system=str(prompts.get("proofread_system") or ""),
            revise_system=str(prompts.get("revise_system") or ""),
        )  # type: ignore[assignment]
        log.info("Dry-run mode enabled: no external LLM requests will be made.")
    else:
        llm = create_llm_client(cfg)

    lookback_label = _derive_lookback_label(cfg, doc)
    log.info(
        f"Selected summary_doc {original_summary_id} (lookback label: {lookback_label})"
    )

    latest_run = _get_latest_run_for_summary(run_db, original_summary_id)
    reuse_batch = (latest_run is not None) and (not args.rebuild_batch)

    # Proofread/revise prompts are always required.
    missing_pr = _missing_prompt_keys(prompts, proofread_keys)
    if missing_pr:
        log.error(
            "Selected prompt package '%s' is missing required proofread/revise prompts: %s",
            str(prompts.get("_package") or ""),
            ", ".join(missing_pr),
        )
        log.error(
            "Choose a package that includes proofread+revise prompts (or add includes for _includes/proofread.yaml and _includes/revise.yaml)."
        )
        return 2

    batch_output: List[Dict[str, Any]] = []
    if reuse_batch:
        batch_output = list(latest_run.get("batch_output") or [])
        if not batch_output:
            reuse_batch = False
            log.warning(
                "Cached run exists but batch_output was empty; rebuilding batch_output instead."
            )
        else:
            log.info(
                f"Reusing cached batch_output from run_revision="
                f"{int((latest_run.get('versions') or {}).get('run_revision') or 0)}"
            )

    if not reuse_batch:
        missing_batch = _missing_prompt_keys(
            prompts, ["batch_system", "batch_user_template"]
        )
        if missing_batch:
            log.error(
                "Selected prompt package '%s' is missing required batch prompts for rebuild: %s",
                str(prompts.get("_package") or ""),
                ", ".join(missing_batch),
            )
            log.error(
                "Either run without --rebuild-batch to reuse cached batch_output, or use a package with batch prompts."
            )
            return 2

        articles = store.get_articles_by_ids(sources)
        if not articles:
            log.error("Could not load any articles for summary_doc sources.")
            return 4

        log.info(
            f"Building batch_output from {len(articles)} articles (expensive step)..."
        )
        batch_output = await _build_batch_output_from_articles(
            cfg=cfg,
            prompts=prompts,
            llm=llm,
            articles=articles,
            store=store,
        )

        if args.dry_run:
            empty_batches = [
                str(b.get("batch_index") or "?")
                for b in (batch_output or [])
                if not str(b.get("text") or "").strip()
            ]
            if empty_batches:
                log.error(
                    "Dry-run validation failed: empty batch output text for batch indices: %s",
                    ", ".join(empty_batches),
                )
                return 2

    sources_text = ""
    if (
        reuse_batch
        and args.skip_sources_text_refresh
        and latest_run
        and latest_run.get("sources_text")
    ):
        sources_text = str(latest_run.get("sources_text") or "").strip()
        log.info("Reusing cached sources_text (skip refresh).")
    else:
        articles2 = store.get_articles_by_ids(sources)
        if articles2:
            batching_cfg = cfg.get("batching", {}) or {}
            meta_sources_clip_chars = int(
                batching_cfg.get("meta_sources_clip_chars", 140)
            )
            sources_text = _sources_text_from_articles(
                articles2, clip_chars=meta_sources_clip_chars
            )
        else:
            log.warning("Could not refresh sources_text (no articles found by ids).")

    batch_summaries: List[Tuple[int, str]] = _batch_output_to_tuples(batch_output)

    log.info(
        f"Proofreading summary_doc {original_summary_id} (max_rounds={int(args.max_rounds)})"
    )
    proof_sys = str(prompts.get("proofread_system") or "").strip()
    revise_sys = str(prompts.get("revise_system") or "").strip()
    trace_rows: List[Dict[str, Any]] = []
    proof_round = 0
    revise_round = 0

    class _TraceLLM:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        async def chat(self, messages: List[Dict[str, str]], *, temperature: float = 0.2) -> str:
            nonlocal proof_round, revise_round
            out = await self._inner.chat(messages, temperature=temperature)

            sys_msg = str((messages[0] or {}).get("content") or "").strip() if messages else ""
            step = ""
            round_no = 0
            if proof_sys and sys_msg == proof_sys:
                step = "proofread"
                proof_round += 1
                round_no = proof_round
            elif revise_sys and sys_msg == revise_sys:
                step = "revise"
                revise_round += 1
                round_no = revise_round

            if step:
                trace_rows.append(
                    {
                        "at": int(time.time()),
                        "step": step,
                        "round": int(round_no),
                        "output": clip_text(str(out or ""), 16000),
                    }
                )
            return out

    llm_for_proofread = _TraceLLM(llm)
    revised, pr_stats = await _proofread_and_revise_meta_with_stats(
        config=cfg,
        llm=llm_for_proofread,
        store=store,
        job_id=None,
        prompts=prompts,
        lookback=lookback_label,
        meta_text=original_summary,
        batch_summaries=batch_summaries,
        sources_text=sources_text,
        max_rounds=int(args.max_rounds),
    )
    pr_stats = dict(pr_stats or {})
    if trace_rows and not isinstance(pr_stats.get("proofread_trace"), list):
        pr_stats["proofread_trace"] = trace_rows

    proof_out = (
        pr_stats.get("proofread_output")
        or pr_stats.get("proofread_last_feedback")
        or ""
    ).strip()
    corrected_summary = (revised or "").strip()

    proof_hash_current = _hash_prompts_subset(
        prompts,
        [
            "proofread_system",
            "proofread_user_template",
            "revise_system",
            "revise_user_template",
        ],
    )
    batch_hash_current = _hash_prompts_subset(
        prompts, ["batch_system", "batch_user_template"]
    )
    run_revision = _next_run_revision(run_db, original_summary_id)

    run_doc: Dict[str, Any] = {
        "created_at": int(time.time()),
        "original_summary_id": original_summary_id,
        "original_summary": original_summary,
        "lookback_label": lookback_label,
        "batch_output": batch_output,
        "sources_text": sources_text,
        "corrected_summary": corrected_summary,
        "proofread_output": proof_out,
        "proofread_rounds": int(pr_stats.get("proofread_rounds") or 0),
        "proofread_last_feedback": str(pr_stats.get("proofread_last_feedback") or ""),
        "proofread_trace": list(pr_stats.get("proofread_trace") or []),
        "prompts": {
            "prompt_package": str(prompts.get("_package") or ""),
            "proofread_system": str(prompts.get("proofread_system") or ""),
            "proofread_user_template": str(
                prompts.get("proofread_user_template") or ""
            ),
            "revise_system": str(prompts.get("revise_system") or ""),
            "revise_user_template": str(prompts.get("revise_user_template") or ""),
            "proofread_revise_prompts_hash": proof_hash_current,
            "batch_prompts_hash": batch_hash_current,
        },
        "versions": {
            "schema_version": SCHEMA_VERSION,
            "run_revision": run_revision,
        },
    }

    if args.store_batch_prompts:
        run_doc["prompts"]["batch_system"] = str(prompts.get("batch_system") or "")
        run_doc["prompts"]["batch_user_template"] = str(
            prompts.get("batch_user_template") or ""
        )

    if args.dry_run:
        dry_llm = llm  # keep type narrow for runtime attribute checks
        batch_calls = int(getattr(dry_llm, "batch_calls", 0))
        proofread_calls = int(getattr(dry_llm, "proofread_calls", 0))
        revise_calls = int(getattr(dry_llm, "revise_calls", 0))

        if not reuse_batch and batch_calls < 1:
            log.error("Dry-run validation failed: batch step sent no LLM request.")
            return 2
        if int(args.max_rounds) > 0 and proofread_calls < 1:
            log.error("Dry-run validation failed: proofread step sent no LLM request.")
            return 2
        if int(args.max_rounds) > 0 and revise_calls < 1:
            log.error("Dry-run validation failed: revise step sent no LLM request.")
            return 2

        log.info("\n" + "=" * 90)
        log.info("DRY-RUN OK")
        log.info(f"LLM batch calls: {batch_calls}")
        log.info(f"LLM proofread calls: {proofread_calls}")
        log.info(f"LLM revise calls: {revise_calls}")
        log.info(f"BATCH_OUTPUT mode: {'REUSED' if reuse_batch else 'REBUILT'}")
        log.info("=" * 90 + "\n")
        run_db.close()
        return 0

    inserted_id = runs_table.insert(run_doc)

    if args.store_original_in_summary_doc:
        try:
            doc_to_save = _update_summary_doc_audit(
                summary_doc=doc,
                original_summary=original_summary,
                revised_summary=corrected_summary,
                proofread_output=proof_out,
                proofread_trace=list(pr_stats.get("proofread_trace") or []),
                proofread_rounds=int(pr_stats.get("proofread_rounds") or 0),
                proofread_last_feedback=str(
                    pr_stats.get("proofread_last_feedback") or ""
                ),
                prompt_package=str(prompts.get("_package") or ""),
                run_revision=run_revision,
            )
            saved_sid = store.save_summary_doc(doc_to_save)
            log.info(
                "Updated summary_doc %s with proofread original/revised snapshots.",
                str(saved_sid),
            )
        except Exception as e:
            log.warning(
                "Could not update summary_doc with proofread snapshots: %s",
                str(e),
            )

    log.info("\n" + "=" * 90)
    log.info(f"RUN_DB: {args.run_db}")
    log.info(f"RUN_DOC_ID: {inserted_id}")
    log.info(f"ORIGINAL_SUMMARY_ID: {original_summary_id}")
    log.info(f"RUN_REVISION: {run_revision}")
    log.info(f"USED_PROMPT_PACKAGE: {prompts.get('_package')}")
    log.info(
        f"BATCH_OUTPUT: {'REUSED' if reuse_batch else 'REBUILT'} (hash={batch_hash_current[:10]})"
    )
    log.info("=" * 90 + "\n")

    log.info("ORIGINAL SUMMARY (clip 2000):\n")
    log.info(clip_text(original_summary, 2000))
    log.info("\n" + "-" * 90 + "\n")

    log.info("PROOFREAD OUTPUT:\n")
    log.info(proof_out if proof_out else "(no proofread output)")
    log.info("\n" + "-" * 90 + "\n")

    log.info("CORRECTED SUMMARY:\n")
    log.info(corrected_summary)
    log.info("\n" + "=" * 90 + "\n")

    if args.out:
        Path(args.out).write_text(corrected_summary + "\n", encoding="utf-8")
        log.info(f"Wrote corrected summary to: {args.out}")

    run_db.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-db",
        default="./.proofreader_runs.json",
        help="Path to local run DB (TinyDB JSON)",
    )

    sub = ap.add_subparsers(dest="cmd", required=True)

    # run
    runp = sub.add_parser(
        "run",
        help="Run proofread+revise and store a run record (reuses batch_output on reruns)",
    )
    runp.add_argument("--config", required=True, help="Path to config.yaml")
    runp.add_argument(
        "--sqlite-db",
        default="",
        help="Override source DB for this run (forces store.provider=sqlite and sets store.path)",
    )

    grp = runp.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--summary-id", default="", help="summary_doc id to proofread/revise"
    )
    grp.add_argument(
        "--latest",
        action="store_true",
        help="Use latest summary_doc instead of --summary-id",
    )

    runp.add_argument(
        "--prompt-package", default="", help="Override prompts.selected for this run"
    )
    runp.add_argument(
        "--proofread-package",
        default="",
        help="Fallback prompt package for missing proofread/revise prompts",
    )
    runp.add_argument(
        "--proofread-system-file",
        default="",
        help="Read proofread_system prompt from file (overrides package)",
    )
    runp.add_argument(
        "--proofread-user-template-file",
        default="",
        help="Read proofread_user_template prompt from file (overrides package)",
    )
    runp.add_argument(
        "--revise-system-file",
        default="",
        help="Read revise_system prompt from file (overrides package)",
    )
    runp.add_argument(
        "--revise-user-template-file",
        default="",
        help="Read revise_user_template prompt from file (overrides package)",
    )
    runp.add_argument(
        "--max-rounds", type=int, default=4, help="Max proofread critique rounds"
    )
    runp.add_argument("--out", default="", help="Write corrected summary to file")

    runp.add_argument(
        "--rebuild-batch",
        action="store_true",
        help="Force rebuild of batch_output even if cached exists",
    )
    runp.add_argument(
        "--skip-sources-text-refresh",
        action="store_true",
        help="When reusing cached batch_output, also reuse cached sources_text if available",
    )
    runp.add_argument(
        "--store-batch-prompts",
        action="store_true",
        help="Also store batch_system + batch_user_template in run DB (can be large).",
    )
    runp.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate that each executed LLM step has non-empty payloads without sending external requests",
    )
    runp.add_argument(
        "--store-original-in-summary-doc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store original + revised summary snapshots in summary_doc for DB-side comparisons",
    )

    # list
    listp = sub.add_parser("list", help="List runs in run DB")
    listp.add_argument("--summary-id", default="", help="Filter by original_summary_id")
    listp.add_argument("--limit", type=int, default=50, help="Max rows to show")

    # show
    showp = sub.add_parser(
        "show", help="Show one run (by run-doc-id) or latest run for a summary-id"
    )
    showp.add_argument(
        "--run-doc-id", type=int, default=0, help="TinyDB doc_id of the run"
    )
    showp.add_argument(
        "--summary-id", default="", help="Show latest run for original_summary_id"
    )
    showp.add_argument(
        "--show-batch", action="store_true", help="Also print batch_output"
    )

    args = ap.parse_args()

    run_db = _open_run_db(args.run_db)
    try:
        if args.cmd == "list":
            return _list_runs(
                run_db,
                summary_id=getattr(args, "summary_id", ""),
                limit=getattr(args, "limit", 50),
            )
        if args.cmd == "show":
            return _show_run(
                run_db,
                run_doc_id=int(getattr(args, "run_doc_id", 0) or 0),
                summary_id=str(getattr(args, "summary_id", "") or ""),
                show_batch=bool(getattr(args, "show_batch", False)),
            )
        if args.cmd == "run":
            return asyncio.run(_run_command(args))
        log.error("Unknown command.")
        return 2
    finally:
        run_db.close()


if __name__ == "__main__":
    raise SystemExit(main())
