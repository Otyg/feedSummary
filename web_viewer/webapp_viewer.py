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
import json
import logging
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from functools import lru_cache
import markdown as md
from flask import Flask, abort, redirect, render_template, request, url_for, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix
import requests
import yaml

from uicommon import (
    format_ts,
    get_store,
    load_config,
    parse_ymd_to_range,
    source_to_topics_map,
)
from feedsummary_core.summarizer.main import (
    _build_composed_summary_text,
    _strip_sources_appendix_from_summary,
)

logger = logging.getLogger(__name__)

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


def init_app_state(config_path: str) -> None:
    global APP_CONFIG_PATH, APP_CFG, APP_STORE
    APP_CONFIG_PATH = str(Path(config_path).resolve())
    raw = load_config(APP_CONFIG_PATH)
    APP_CFG = _abspath_cfg_paths(raw, APP_CONFIG_PATH)
    APP_STORE = get_store(APP_CFG)

    sp = (APP_CFG.get("store") or {}).get("path")
    logger.info("Viewer config loaded: %s", APP_CONFIG_PATH)
    logger.info("Resolved store path: %s", sp)


def _md_to_html(text: str) -> str:
    return md.markdown(text or "", extensions=["extra"])


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
        if freq == "daily":
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
    return {
        "id": a.get("id"),
        "title": a.get("title") or "",
        "source": a.get("source") or "",
        "url": a.get("url") or "",
        "published_ts": int(_coerce_positive_ts(a.get("published_ts"))),
        "fetched_at": int(_coerce_positive_ts(a.get("fetched_at"))),
        "preview": preview,
    }


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


def _sqlite_store_path_from_cfg() -> Optional[str]:
    st = APP_CFG.get("store") if isinstance(APP_CFG, dict) else None
    if not isinstance(st, dict):
        return None
    provider = str(st.get("provider") or "").strip().lower()
    path = str(st.get("path") or "").strip()
    if provider != "sqlite" or not path:
        return None
    return path


def _list_article_dates_fast(store, *, max_days: int) -> List[Dict[str, Any]]:
    """
    Return [{date: 'YYYY-MM-DD', count: N}, ...] ordered desc.
    Uses fast SQL path for sqlite store; falls back to in-memory grouping.
    """
    sqlite_path = _sqlite_store_path_from_cfg()
    if sqlite_path:
        try:
            con = sqlite3.connect(sqlite_path)
            try:
                cur = con.execute(
                    """
                    SELECT
                      DATE(COALESCE(NULLIF(published_ts, 0), fetched_at), 'unixepoch', 'localtime') AS day_key,
                      COUNT(*) AS cnt
                    FROM articles
                    WHERE COALESCE(NULLIF(published_ts, 0), fetched_at, 0) > 0
                    GROUP BY day_key
                    ORDER BY day_key DESC
                    LIMIT ?
                    """,
                    (max_days,),
                )
                rows = cur.fetchall()
            finally:
                con.close()
            out: List[Dict[str, Any]] = []
            for day_key, cnt in rows:
                if day_key:
                    out.append({"date": str(day_key), "count": int(cnt or 0)})
            return out
        except Exception:
            pass

    # Fallback for non-sqlite stores.
    raw = store.list_articles(limit=50000) or []
    counts: Dict[str, int] = {}
    for a in raw:
        if not isinstance(a, dict):
            continue
        tsv = int(_article_published_ts(a))
        if tsv <= 0:
            continue
        day_key = datetime.fromtimestamp(tsv).strftime("%Y-%m-%d")
        counts[day_key] = counts.get(day_key, 0) + 1
    days = sorted(counts.keys(), reverse=True)[:max_days]
    return [{"date": d, "count": counts[d]} for d in days]


def _list_articles_for_day_fast(
    store, *, date_ymd: str, limit: int
) -> List[Dict[str, Any]]:
    """
    Return lightweight article rows for one date only.
    """
    dr = parse_ymd_to_range(date_ymd)
    if not dr:
        return []
    start_ts, end_ts = dr

    sqlite_path = _sqlite_store_path_from_cfg()
    if sqlite_path:
        try:
            con = sqlite3.connect(sqlite_path)
            try:
                cur = con.execute(
                    """
                    SELECT id, title, source, url, published_ts, fetched_at
                    FROM articles
                    WHERE COALESCE(NULLIF(published_ts, 0), fetched_at, 0) BETWEEN ? AND ?
                    ORDER BY COALESCE(NULLIF(published_ts, 0), fetched_at, 0) DESC
                    LIMIT ?
                    """,
                    (int(start_ts), int(end_ts), int(limit)),
                )
                rows = cur.fetchall()
            finally:
                con.close()
            out: List[Dict[str, Any]] = []
            for rid, title, source, url, published_ts, fetched_at in rows:
                out.append(
                    {
                        "id": rid,
                        "title": title or "",
                        "source": source or "",
                        "url": url or "",
                        "published_ts": int(_coerce_positive_ts(published_ts)),
                        "fetched_at": int(_coerce_positive_ts(fetched_at)),
                    }
                )
            return out
        except Exception:
            pass

    # Fallback for non-sqlite stores.
    src_map = source_to_topics_map(APP_CFG)
    sources = sorted(list(src_map.keys()))
    rows: List[Dict[str, Any]] = []
    if sources:
        try:
            rows = (
                store.list_articles_by_filter(
                    sources=sources,
                    since_ts=int(start_ts),
                    until_ts=int(end_ts),
                    limit=int(limit),
                )
                or []
            )
        except Exception:
            rows = []
    if not rows:
        raw = store.list_articles(limit=50000) or []
        rows = []
        for a in raw:
            if not isinstance(a, dict):
                continue
            tsv = int(_article_published_ts(a))
            if start_ts <= tsv <= end_ts:
                rows.append(a)
        rows.sort(key=_article_published_ts, reverse=True)
        rows = rows[:limit]

    out: List[Dict[str, Any]] = []
    for a in rows:
        if isinstance(a, dict) and a.get("id"):
            out.append(
                {
                    "id": a.get("id"),
                    "title": a.get("title") or "",
                    "source": a.get("source") or "",
                    "url": a.get("url") or "",
                    "published_ts": int(_article_published_ts(a)),
                    "fetched_at": int(_coerce_positive_ts(a.get("fetched_at"))),
                }
            )
    return out


def _load_yaml_file(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _collect_ui_options(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read-only UI options for clients.
    - sources: feed/source names
    - topics: unique topic tags across feeds (if configured)
    - prompt_packages: keys in prompts.yaml
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

    feeds = _load_yaml_file(feeds_path)
    sources: List[str] = []
    topics: List[str] = []

    if isinstance(feeds, dict):
        for name, f in feeds.items():
            if isinstance(name, str) and name.strip():
                sources.append(name.strip())
            if isinstance(f, dict):
                ts = f.get("topics") or f.get("topic") or []
                if isinstance(ts, str) and ts.strip():
                    topics.append(ts.strip())
                elif isinstance(ts, list):
                    for t in ts:
                        t = str(t).strip()
                        if t:
                            topics.append(t)

    # ---- Prompts: prompt packages ----
    prompts_path = None
    if isinstance(cfg.get("prompts"), dict) and cfg["prompts"].get("path"):
        prompts_path = str(cfg["prompts"]["path"])
    else:
        prompts_path = str(
            (
                Path(APP_CONFIG_PATH).resolve().parent / "config" / "prompts.yaml"
            ).resolve()
        )

    prompts = _load_yaml_file(prompts_path)
    if isinstance(prompts, dict):
        out["prompt_packages"] = sorted(
            [k for k in prompts.keys() if isinstance(k, str)]
        )

    out["sources"] = sorted(list(dict.fromkeys(sources)))
    out["topics"] = sorted(list(dict.fromkeys(topics)))

    out["feeds_path"] = feeds_path
    out["prompts_path"] = prompts_path
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

    return jsonify({"item": a})


@app.route("/api/v1/pages/source")
def api_page_source():
    return jsonify({"markdown": _load_static_md("source.md")})


@app.route("/api/v1/pages/license")
def api_page_license():
    return jsonify({"markdown": _load_static_md("license.md")})


@app.route("/api/v1/prompt/<name>")
def api_prompt_package(name: str):
    """
    Return the YAML content for a single prompt package from prompts.yaml.
    Read-only.
    """
    pkg = str(name or "").strip()
    if not pkg:
        abort(404)

    # Reuse the same prompts path logic as /api/ui_options
    prompts_path = None
    if isinstance(APP_CFG.get("prompts"), dict) and APP_CFG["prompts"].get("path"):
        prompts_path = str(APP_CFG["prompts"]["path"])
    else:
        prompts_path = str(
            (
                Path(APP_CONFIG_PATH).resolve().parent / "config" / "prompts.yaml"
            ).resolve()
        )

    p = Path(prompts_path)
    if not p.exists():
        return jsonify(
            {"error": "prompts_yaml_not_found", "prompts_path": prompts_path}
        ), 404

    prompts = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(prompts, dict) or pkg not in prompts:
        return jsonify(
            {"error": "prompt_not_found", "name": pkg, "prompts_path": prompts_path}
        ), 404

    # Dump only that package as YAML (nice for display/copy)
    one = {pkg: prompts[pkg]}
    yaml_text = yaml.safe_dump(one, sort_keys=False, allow_unicode=True)

    return jsonify(
        {
            "name": pkg,
            "prompts_path": prompts_path,
            "yaml": yaml_text,
            "item": prompts[pkg],  # also return as JSON if you want programmatic use
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

    summary_text = str(
        sdoc.get("proofread_revised_summary")
        or sdoc.get("proofread_published_summary")
        or sdoc.get("summary")
        or ""
    ).strip()
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


@app.route("/articles")
def list_articles():
    store = APP_STORE
    if store is None:
        abort(500)

    try:
        limit = int(request.args.get("limit", "2000"))
    except Exception:
        limit = 2000
    limit = max(1, min(limit, 50000))

    try:
        max_days = int(request.args.get("days", "3650"))
    except Exception:
        max_days = 3650
    max_days = max(1, min(max_days, 10000))

    date_rows = _list_article_dates_fast(store, max_days=max_days)
    date_tabs = [str(r.get("date") or "") for r in date_rows if r.get("date")]
    date_counts: Dict[str, int] = {
        str(r.get("date")): int(r.get("count") or 0) for r in date_rows if r.get("date")
    }

    def _format_published_ts_iso(a: Dict[str, Any]) -> str:
        tsv = _article_published_ts(a)
        if tsv <= 0:
            return ""
        whole = int(tsv)
        frac2 = int(round((tsv - whole) * 100))
        if frac2 >= 100:
            whole += 1
            frac2 = 0
        return f"{datetime.fromtimestamp(whole).strftime('%Y-%m-%dT%H:%M:%S')}.{frac2:02d}"

    requested_date = str(request.args.get("date") or "").strip()
    active_date = requested_date if requested_date in date_counts else ""
    if not active_date and date_tabs:
        active_date = date_tabs[0]

    active_articles: List[Dict[str, Any]] = []
    if active_date:
        active_articles = _list_articles_for_day_fast(
            store, date_ymd=active_date, limit=limit
        )

    return render_template(
        "articles.html",
        active_articles=active_articles,
        date_tabs=date_tabs,
        date_counts=date_counts,
        active_date=active_date,
        format_published_ts_iso=_format_published_ts_iso,
        format_ts=format_ts,
        error=None,
    )


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

    return render_template("article.html", a=a, format_ts=format_ts)


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
