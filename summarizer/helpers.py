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

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import logging
import sys


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


class RateLimitError(Exception):
    def __init__(
        self, status: int, retry_after: Optional[float] = None, body: str = ""
    ):
        super().__init__(f"HTTP {status} rate-limited")
        self.status = status
        self.retry_after = retry_after
        self.body = body
