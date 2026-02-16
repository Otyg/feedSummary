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

from typing import Any, Dict, List, Optional, Protocol

from persistence.TinyDbStore import TinyDBStore
import os
from pathlib import Path


class StoreError(Exception):
    pass


class NewsStore(Protocol):
    # ---- Articles
    def get_article(self, article_id: str) -> Optional[Dict[str, Any]]: ...

    def upsert_article(self, article_doc: Dict[str, Any]) -> None: ...

    def list_unsummarized_articles(self, limit: int = 200) -> List[Dict[str, Any]]: ...

    def mark_articles_summarized(self, article_ids: List[str]) -> None: ...

    # ---- Summaries
    def save_summary(self, summary_text: str, article_ids: List[str]) -> int: ...

    def get_latest_summary(self) -> Optional[Dict[str, Any]]: ...

    def list_summaries(self) -> List[Dict[str, Any]]: ...

    def get_summary(self, summary_id: int) -> Optional[Dict[str, Any]]: ...

    # ---- Jobs
    def create_job(self) -> int: ...

    def update_job(self, job_id: int, **fields) -> None: ...

    def get_job(self, job_id: int) -> Optional[Dict[str, Any]]: ...

    # ---- Utility
    def get_articles_by_ids(self, article_ids: List[str]) -> List[Dict[str, Any]]: ...


def _expand_path(p: str) -> str:
    # Expand ~ och miljövariabler som $HOME eller ${HOME}
    expanded = os.path.expandvars(os.path.expanduser(p))
    # Normalisera till absolut path (valfritt men ofta skönt)
    return str(Path(expanded).resolve())


def create_store(cfg: Dict[str, Any]) -> NewsStore:
    provider = (cfg.get("provider") or "tinydb").lower()

    if provider == "tinydb":
        raw_path = cfg.get("path", "news_docs.json")
        path = _expand_path(raw_path)

        # Se till att katalogen finns
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        return TinyDBStore(path=path)

    raise ValueError(f"Unsupported store provider: {provider}")
