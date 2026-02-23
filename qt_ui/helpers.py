from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

def _fmt_dt_hm(ts: int) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")


def _safe_list_str(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        s = v.strip()
        return [s] if s else []
    return []


def _parse_csv(s: str) -> List[str]:
    # comma-separated, allow spaces
    parts = [p.strip() for p in (s or "").split(",")]
    return [p for p in parts if p]


def _resolve_relative_to_config(config_path: str, rel: str) -> Path:
    base = Path(os.path.abspath(config_path)).parent
    rel2 = os.path.expanduser(os.path.expandvars(rel))
    p = Path(rel2)
    return p if p.is_absolute() else (base / p).resolve()


def resolve_feeds_path(cfg: Dict[str, Any], *, config_path: str) -> Path:
    # config: feeds: { path: "config/feeds.yaml" }
    feeds_cfg = cfg.get("feeds")
    raw = "config/feeds.yaml"
    if isinstance(feeds_cfg, dict) and feeds_cfg.get("path"):
        raw = str(feeds_cfg.get("path"))
    return _resolve_relative_to_config(config_path, raw)
