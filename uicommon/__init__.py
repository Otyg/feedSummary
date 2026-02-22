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

# app_shared.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from persistence import NewsStore, create_store
from summarizer.helpers import load_feeds_into_config


# ----------------------------
# Config + store
# ----------------------------
def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load config.yaml and merge feeds from feeds.path (config/feeds.yaml).
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = load_feeds_into_config(cfg, base_config_path=config_path)
    return cfg


def get_store(cfg: Dict[str, Any]) -> NewsStore:
    return create_store(cfg.get("store", {}))


# ----------------------------
# Common formatting / parsing
# ----------------------------
def format_ts(ts: Optional[int]) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")


def published_ts(a: Dict[str, Any]) -> int:
    ts = a.get("published_ts")
    if isinstance(ts, int) and ts > 0:
        return ts
    fa = a.get("fetched_at")
    if isinstance(fa, int) and fa > 0:
        return fa
    return 0


def parse_ymd_to_range(date_str: str) -> Optional[Tuple[int, int]]:
    """
    Parse YYYY-MM-DD to (start_ts, end_ts) for that day in local time.
    """
    s = (date_str or "").strip()
    if not s:
        return None
    try:
        d = datetime.strptime(s, "%Y-%m-%d")
        start = int(d.timestamp())
        end = int((d + timedelta(days=1) - timedelta(seconds=1)).timestamp())
        return start, end
    except Exception:
        return None


def split_lookback(
    lb: str, default_value: int = 24, default_unit: str = "h"
) -> Tuple[int, str]:
    """
    "24h" -> (24, "h")
    """
    lb = (lb or "").strip()
    if not lb:
        return default_value, default_unit
    digits = "".join([c for c in lb if c.isdigit()])
    unit = "".join([c for c in lb if not c.isdigit()]).strip().lower()
    return int(digits or default_value), (unit or default_unit)


def resolve_path(config_path: str, p: str) -> str:
    """
    Expand env + ~ and resolve relative paths relative to config.yaml location.
    """
    cfg_dir = os.path.dirname(os.path.abspath(config_path)) or "."
    p2 = os.path.expanduser(os.path.expandvars(p))
    if not os.path.isabs(p2):
        p2 = os.path.join(cfg_dir, p2)
    return p2


# ----------------------------
# Feeds -> sources/topics
# ----------------------------
def get_config_sources(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    feeds = cfg.get("feeds")
    if isinstance(feeds, list) and all(isinstance(x, dict) for x in feeds):
        return feeds  # type: ignore[return-value]

    candidates = [
        cfg.get("sources"),
        cfg.get("rss_sources"),
        (cfg.get("ingest") or {}).get("sources"),
        (cfg.get("ingest") or {}).get("feeds"),
    ]
    for c in candidates:
        if isinstance(c, list) and c and all(isinstance(x, dict) for x in c):
            return c  # type: ignore[return-value]
    return []


def source_name(s: Dict[str, Any]) -> str:
    return str(s.get("name") or s.get("title") or s.get("label") or "").strip()


def source_topics(s: Dict[str, Any]) -> List[str]:
    t = s.get("topics")
    if isinstance(t, list):
        return [str(x).strip() for x in t if str(x).strip()]
    if isinstance(t, str) and t.strip():
        return [t.strip()]
    t2 = s.get("topic")
    if isinstance(t2, str) and t2.strip():
        return [t2.strip()]
    return []


def build_source_options(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in get_config_sources(cfg):
        name = source_name(s)
        if not name:
            continue
        out.append(
            {
                "name": name,
                "url": str(s.get("url") or s.get("href") or s.get("rss") or ""),
                "default_checked": bool(s.get("enabled", True)),
                "topics": source_topics(s),
            }
        )
    out.sort(key=lambda x: x["name"].lower())
    return out


def build_topic_options(cfg: Dict[str, Any]) -> List[str]:
    seen: Set[str] = set()
    topics: List[str] = []
    for s in get_config_sources(cfg):
        for t in source_topics(s):
            if t not in seen:
                seen.add(t)
                topics.append(t)
    topics.sort(key=lambda x: x.lower())
    return topics


def source_to_topics_map(cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for s in get_config_sources(cfg):
        n = source_name(s)
        if not n:
            continue
        out[n] = source_topics(s)
    return out


def sources_for_topics(cfg: Dict[str, Any], topics: List[str]) -> List[str]:
    wanted = {t.strip() for t in topics if t.strip()}
    if not wanted:
        return []
    m = source_to_topics_map(cfg)
    out: List[str] = []
    for src, ts in m.items():
        if set(ts).intersection(wanted):
            out.append(src)
    out.sort(key=lambda x: x.lower())
    return out


# ----------------------------
# prompts.yaml packages
# ----------------------------
def load_prompt_packages(cfg: Dict[str, Any], *, config_path: str) -> List[str]:
    p_cfg = cfg.get("prompts") or {}
    if not isinstance(p_cfg, dict):
        return []
    raw_path = str(p_cfg.get("path") or "config/prompts.yaml")
    path = resolve_path(config_path, raw_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            all_pkgs = yaml.safe_load(f) or {}
        if isinstance(all_pkgs, dict):
            return sorted([str(k) for k in all_pkgs.keys()])
    except Exception:
        return []
    return []


def default_prompt_package(cfg: Dict[str, Any], packages: List[str]) -> str:
    p_cfg = cfg.get("prompts") or {}
    if isinstance(p_cfg, dict):
        sel = str(p_cfg.get("selected") or "").strip()
        if sel:
            return sel
        d = str(p_cfg.get("default_package") or "").strip()
        if d:
            return d
    return packages[0] if packages else ""


def default_lookback_parts(cfg: Dict[str, Any]) -> Tuple[int, str]:
    ingest = cfg.get("ingest") or {}
    default_lb = str(ingest.get("lookback") or "24h")
    return split_lookback(default_lb, 24, "h")


# ----------------------------
# Article filtering (shared)
# ----------------------------
@dataclass(frozen=True)
class ArticleFilters:
    sources: List[str]
    topics: List[str]
    from_ymd: str
    to_ymd: str


def filter_articles(
    articles: List[Dict[str, Any]],
    *,
    cfg: Dict[str, Any],
    filters: ArticleFilters,
) -> List[Dict[str, Any]]:
    selected_sources = [s.strip() for s in (filters.sources or []) if s.strip()]
    selected_topics = [t.strip() for t in (filters.topics or []) if t.strip()]

    allowed_sources: Optional[Set[str]] = None
    if selected_sources:
        allowed_sources = set(selected_sources)
    elif selected_topics:
        allowed_sources = set(sources_for_topics(cfg, selected_topics))

    from_ts = None
    to_ts = None
    if filters.from_ymd:
        r = parse_ymd_to_range(filters.from_ymd)
        from_ts = r[0] if r else None
    if filters.to_ymd:
        r = parse_ymd_to_range(filters.to_ymd)
        to_ts = r[1] if r else None

    def keep(a: Dict[str, Any]) -> bool:
        src = str(a.get("source") or "").strip()
        if allowed_sources is not None and src not in allowed_sources:
            return False

        ts = published_ts(a)
        if from_ts is not None and ts and ts < from_ts:
            return False
        if to_ts is not None and ts and ts > to_ts:
            return False
        if ts == 0 and (from_ts is not None or to_ts is not None):
            return False
        return True

    out = [a for a in articles if keep(a)]
    out.sort(key=published_ts, reverse=True)
    return out


# ----------------------------
# Unified UI options object
# ----------------------------
@dataclass(frozen=True)
class UIOptions:
    prompt_packages: List[str]
    default_prompt_package: str
    source_options: List[Dict[str, Any]]
    topic_options: List[str]
    default_lookback_value: int
    default_lookback_unit: str


def get_ui_options(cfg: Dict[str, Any], *, config_path: str) -> UIOptions:
    pkgs = load_prompt_packages(cfg, config_path=config_path)
    default_pkg = default_prompt_package(cfg, pkgs)
    src_opts = build_source_options(cfg)
    topic_opts = build_topic_options(cfg)
    lb_val, lb_unit = default_lookback_parts(cfg)

    return UIOptions(
        prompt_packages=pkgs,
        default_prompt_package=default_pkg,
        source_options=src_opts,
        topic_options=topic_opts,
        default_lookback_value=int(lb_val),
        default_lookback_unit=str(lb_unit),
    )


# ----------------------------
# NEW: common refresh overrides builder
# ----------------------------
def build_refresh_overrides(
    *,
    lookback_value: int,
    lookback_unit: str,
    prompt_package: str,
    selected_sources: List[str],
    selected_topics: List[str],
) -> Dict[str, Any]:
    """
    Common semantics for both webapp + qt:
      - lookback required (value+unit)
      - prompt_package optional
      - if selected_sources non-empty -> overrides["sources"] = ...
        else if selected_topics non-empty -> overrides["topics"] = ...
    """
    ov: Dict[str, Any] = {}

    unit = (lookback_unit or "").strip().lower()
    if unit not in {"h", "d", "w", "m", "y"}:
        unit = "h"
    lv = int(lookback_value) if lookback_value and int(lookback_value) > 0 else 24
    ov["lookback"] = f"{lv}{unit}"

    pp = (prompt_package or "").strip()
    if pp:
        ov["prompt_package"] = pp

    srcs = [str(s).strip() for s in (selected_sources or []) if str(s).strip()]
    tops = [str(t).strip() for t in (selected_topics or []) if str(t).strip()]

    if srcs:
        ov["sources"] = srcs
    elif tops:
        ov["topics"] = tops

    return ov
