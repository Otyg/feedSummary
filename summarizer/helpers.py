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

import datetime
from email.utils import parsedate_to_datetime
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Deque, Callable

import logging
import sys
from collections import defaultdict, deque

import feedparser
import yaml


def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)

    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.INFO)
    formatter = logging.Formatter(
        " %(asctime)s - %(name)s - %(levelname)s:   %(message)s"
    )
    h.setFormatter(formatter)
    root.addHandler(h)


logger = logging.getLogger(__name__)


def _expand_path(p: str) -> str:
    return os.path.expandvars(os.path.expanduser(p))


def _checkpoint_dir(config: Dict[str, Any]) -> Path:
    cp = config.get("checkpointing") or {}
    enabled = bool(cp.get("enabled", True))
    if not enabled:
        return Path()  # unused
    d = cp.get("dir", "./.checkpoints")
    return Path(_expand_path(str(d))).resolve()


def _checkpoint_key(job_id: Optional[int], articles: List[dict]) -> str:
    # Om job_id finns: använd den (bäst). Annars: stabil hash på artikel-id:n.
    if job_id is not None:
        return f"job_{job_id}"
    ids = [a.get("id", "") for a in articles]
    ids_join = "|".join(ids)
    return hashlib.sha256(ids_join.encode("utf-8")).hexdigest()[:16]


def _checkpoint_path(config: Dict[str, Any], key: str) -> Path:
    d = _checkpoint_dir(config)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_checkpoint(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _meta_ckpt_path(config: Dict[str, Any], key: str) -> Path:
    d = _checkpoint_dir(config)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.meta.json"


# ----------------------------
# Hash helpers
# ----------------------------
def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def compute_content_hash(title: str, url: str, text: str) -> str:
    base = f"{(title or '').strip()}|{(url or '').strip()}|{normalize_text(text)}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def stable_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def text_clip(s: str, max_chars: int) -> str:
    return clip_text(s=s, n=max_chars)


def clip_text(s: str, n: int = 5000) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


def clip_line(s: str, n: int = 200) -> str:
    return clip_text(s=s, n=n)


def trim_text_tail_by_words(
    text: str, remove_tokens: int, *, chars_per_token: float
) -> str:
    return trim_last_user_word_boundary(
        messages=[{"foo": text}],
        remove_tokens=remove_tokens,
        chars_per_token=chars_per_token,
    )[0]["foo"]


def trim_last_user_word_boundary(
    messages: List[Dict[str, str]],
    remove_tokens: int,
    *,
    chars_per_token: float,
) -> List[Dict[str, str]]:
    out = [dict(m) for m in messages]
    idx = None
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            idx = i
            break
    if idx is None:
        return out

    content = out[idx].get("content") or ""
    if not content:
        return out

    remove_chars = int(max(1, remove_tokens) * chars_per_token)
    if remove_chars >= len(content):
        out[idx]["content"] = "[TRUNCATED FOR CONTEXT WINDOW]\n"
        return out

    target = len(content) - remove_chars
    if target < 0:
        target = 0

    cut_space = content.rfind(" ", 0, target)
    cut_nl = content.rfind("\n", 0, target)
    cut_tab = content.rfind("\t", 0, target)
    cut = max(cut_space, cut_nl, cut_tab)
    if cut <= 0:
        cut = target

    out[idx]["content"] = (
        content[:cut].rstrip() + "\n\n[TRUNCATED FOR CONTEXT WINDOW]\n"
    )
    return out


def _published_ts(a: dict) -> int:
    ts = a.get("published_ts")
    if isinstance(ts, int) and ts > 0:
        return ts
    return 0


def interleave_by_source_oldest_first(
    articles: List[dict],
    *,
    source_key: str = "source",
    ts_key_fn: Callable[[dict], int] = _published_ts,
) -> List[dict]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for a in articles:
        src = str(a.get(source_key) or "unknown")
        groups[src].append(a)

    queues: Dict[str, Deque[dict]] = {}
    for src, items in groups.items():
        items_sorted = sorted(items, key=ts_key_fn)  # äldst först
        queues[src] = deque(items_sorted)

    out: List[dict] = []

    while True:
        active = [(src, q) for src, q in queues.items() if q]
        if not active:
            break

        active.sort(key=lambda sq: ts_key_fn(sq[1][0]))

        for src, q in active:
            if q:
                out.append(q.popleft())

    return out


class RateLimitError(Exception):
    def __init__(
        self, status: int, retry_after: Optional[float] = None, body: str = ""
    ):
        super().__init__(f"HTTP {status} rate-limited")
        self.status = status
        self.retry_after = retry_after
        self.body = body


_PROMPT_TOO_LONG_RE = re.compile(
    r"exceeded max context length by\s+(\d+)\s+tokens", re.IGNORECASE
)
_DURATION_RE = re.compile(r"^\s*(\d+)\s*([mhdw])\s*$", re.IGNORECASE)


def _extract_overflow_tokens(err: Exception) -> Optional[int]:
    m = _PROMPT_TOO_LONG_RE.search(str(err))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def parse_lookback_to_seconds(s: str) -> int:
    if not s:
        raise ValueError("lookback är tom")
    m = _DURATION_RE.match(s)
    if not m:
        raise ValueError(
            f"Ogiltigt lookback-format: {s!r} (förväntar t.ex. 90m, 24h, 3d, 2w)"
        )
    n = int(m.group(1))
    unit = m.group(2).lower()
    HOUR = 60 * 60
    DAY = HOUR * 24
    WEEK = DAY * 7
    MONTH = WEEK * 4
    if unit == "h":
        return n * HOUR
    if unit == "d":
        return n * DAY
    if unit == "w":
        return n * WEEK
    if unit == "m":
        return n * MONTH
    raise ValueError(f"Okänd enhet: {unit}")


def entry_published_ts(entry: feedparser.FeedParserDict) -> Optional[int]:
    for attr in ("published_parsed", "updated_parsed"):
        st = getattr(entry, attr, None)
        if st:
            try:
                return int(time.mktime(st))
            except Exception:
                pass

    for attr in ("published", "updated"):
        s = getattr(entry, attr, None)
        if s:
            try:
                dt = parsedate_to_datetime(s)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.timezone.utc)
                return int(dt.timestamp())
            except Exception:
                pass

    return None


# ----------------------------
# Prompt loader (from config)
# ----------------------------
def load_prompts(
    config: Dict[str, Any], package: Optional[str] = None
) -> Dict[str, str]:
    """
    1) NYTT: läsa promptpaket från config/prompts.yaml
       config.yaml:
         prompts:
           path: "config/prompts.yaml"
           default_package: "standard_sv"
           selected: ""  (kan sättas av webappen)
    2) BAKÅTKOMP: om config["prompts"] redan innehåller batch_system/meta_system etc,
       använd det direkt.

    Viktigt:
    - Vi returnerar alltid basnycklarna.
    - Vi tar även med valfria nycklar om de finns i paketet:
        super_meta_system
        super_meta_user_template
        title_system
        title_user_template
    """

    p_cfg = config.get("prompts") or {}

    base_keys = (
        "batch_system",
        "batch_user_template",
        "meta_system",
        "meta_user_template",
    )
    optional_keys = (
        "super_meta_system",
        "super_meta_user_template",
        "title_system",
        "title_user_template",
    )

    # Backward compat: prompts directly embedded in config.yaml
    if isinstance(p_cfg, dict) and any(k in p_cfg for k in base_keys):
        out: Dict[str, str] = {k: str(p_cfg.get(k, "")) for k in base_keys}
        for k in optional_keys:
            if k in p_cfg:
                out[k] = str(p_cfg.get(k, ""))
        out["_package"] = str(p_cfg.get("_package") or "embedded")
        return out

    # New: prompts.yaml packages
    path = "config/prompts.yaml"
    default_pkg = "default"

    if isinstance(p_cfg, dict):
        path = str(p_cfg.get("path") or path)
        default_pkg = str(p_cfg.get("default_package") or default_pkg)
        selected = str(p_cfg.get("selected") or "").strip()
    else:
        selected = ""

    pkg = (package or selected or default_pkg).strip()
    if not pkg:
        pkg = default_pkg

    path = os.path.expanduser(os.path.expandvars(path))

    with open(path, "r", encoding="utf-8") as f:
        all_pkgs = yaml.safe_load(f) or {}

    if pkg not in all_pkgs:
        if isinstance(all_pkgs, dict) and all_pkgs:
            pkg = next(iter(all_pkgs.keys()))
        else:
            raise RuntimeError(f"Inga prompt-paket hittades i {path}")

    blob = all_pkgs.get(pkg) or {}
    if not isinstance(blob, dict):
        raise RuntimeError(f"Prompt-paket '{pkg}' i {path} är inte ett dict-objekt")

    out: Dict[str, str] = {}
    for k in base_keys:
        out[k] = str(blob.get(k, ""))

    # include optional keys if present
    for k in optional_keys:
        if k in blob:
            out[k] = str(blob.get(k, ""))

    out["_package"] = pkg
    return out


def set_job(msg: str, job_id, store):
    if job_id is not None:
        store.update_job(job_id, message=msg)


def _resolve_path(base_config_path: str, p: str) -> str:
    p2 = os.path.expanduser(os.path.expandvars(p))
    if os.path.isabs(p2):
        return p2
    base_dir = os.path.dirname(os.path.abspath(base_config_path)) or "."
    return os.path.join(base_dir, p2)


def load_feeds_into_config(
    config: Dict[str, Any], *, base_config_path: str = "config.yaml"
) -> Dict[str, Any]:
    logger.info("Reading feed-configs")
    feeds = config.get("feeds")
    if isinstance(feeds, list):
        return config

    feeds_path: Optional[str] = None
    if isinstance(feeds, dict) and isinstance(feeds.get("path"), str):
        feeds_path = feeds["path"]
    elif isinstance(config.get("feeds_path"), str):
        feeds_path = str(config["feeds_path"])
    else:
        feeds_path = "config/feeds.yaml"

    path = _resolve_path(base_config_path, feeds_path)

    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or None
        if not isinstance(loaded, list):
            raise ValueError(f"feeds.yaml must be a list, got {type(loaded)}")

        for i, item in enumerate(loaded):
            if not isinstance(item, dict):
                raise ValueError(
                    f"feeds.yaml item {i} must be a dict, got {type(item)}"
                )

        config["feeds"] = loaded
        return config
    except FileNotFoundError as e:
        logger.error(
            "feeds.yaml not found: %s (configure feeds.path or feeds_path)", path
        )
        config["feeds"] = []
        raise e
    except Exception as e:
        logger.error("failed to read feeds.yaml: %s -> %s", path, e)
        config["feeds"] = []
        raise e

def lookback_label_from_range(lookback_raw: str, from_ts: int, to_ts: int) -> str:
    """
    Builds a human-readable label for the time window.

    If lookback_raw is set (e.g. "24h", "1w") it is included.
    If from_ts/to_ts are present, we add the date span.
    """
    lb = (lookback_raw or "").strip()

    span = ""
    if from_ts and to_ts:
        a = datetime.datetime.fromtimestamp(int(from_ts)).strftime("%Y-%m-%d")
        b = datetime.datetime.fromtimestamp(int(to_ts)).strftime("%Y-%m-%d")
        span = a if a == b else f"{a}–{b}"

    if lb and span:
        return f"{lb} ({span})"
    if lb:
        return lb
    if span:
        return span
    return ""


def lookback_label_from_articles(lookback_raw: str, articles: List[dict]) -> str:
    """
    Derives a label from article timestamps when available.

    - Uses published_ts (fallback fetched_at) to compute min/max.
    - If lookback_raw is set, returns "lookback_raw (YYYY-MM-DD–YYYY-MM-DD)".
    - If not, returns just the date span.
    """
    if not articles:
        return (lookback_raw or "").strip()

    ts_list: List[int] = []
    for a in articles:
        ts = a.get("published_ts")
        if not (isinstance(ts, int) and ts > 0):
            ts = a.get("fetched_at")
        if isinstance(ts, int) and ts > 0:
            ts_list.append(ts)

    if not ts_list:
        return (lookback_raw or "").strip()

    return lookback_label_from_range(lookback_raw, min(ts_list), max(ts_list))