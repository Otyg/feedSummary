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
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging
from ollama import AsyncClient

logger = logging.getLogger(__name__)


class LLMRateLimitError(RuntimeError):
    def __init__(self, message: str, retry_after_seconds: Optional[int] = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class LLMAuthError(RuntimeError):
    pass


class LLMUnavailableError(RuntimeError):
    pass


@dataclass
class OllamaCloudConfig:
    host: str = "https://ollama.com"
    model: str = "gemma3:270m"
    api_key: str = ""
    # quota/preflight
    preflight: bool = True
    min_interval_seconds: float = 1.0
    timeout_seconds: float = 180.0


def _resolve_env(value: str) -> str:
    """
    Stöd för ${VAR} och $VAR i config.yaml.
    """
    if not isinstance(value, str):
        return str(value)
    return os.path.expandvars(value)


def _extract_retry_after_seconds(exc: Exception) -> Optional[int]:
    """
    Försök läsa Retry-After från exception/response om biblioteket exponerar det.
    (Ollama-python’s exceptions kan variera mellan versioner.)
    """
    # ResponseError i ollama-python brukar ha status_code + error, men inte alltid headers.
    # Vi gör därför "best effort".
    for attr in ("retry_after", "retry_after_seconds"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    return None


def _is_status(exc: Exception, code: int) -> bool:
    sc = getattr(exc, "status_code", None)
    if isinstance(sc, int) and sc == code:
        return True
    # vissa varianter använder .status
    st = getattr(exc, "status", None)
    return isinstance(st, int) and st == code


class OllamaCloudClient:
    """
    LLMClient-implementation för Ollama Cloud API med gemma3:270m.

    Använder officiella 'ollama' AsyncClient och kör mot host=https://ollama.com
    med Authorization Bearer API key. :contentReference[oaicite:3]{index=3}

    "Quota check" görs via preflight call till /api/tags (client.list()).
    Det ger tidig signal för 401/429 innan vi drar igång stora prompts.
    """

    def __init__(self, cfg: Dict[str, Any]):
        llm_cfg = cfg or {}
        quota_cfg = llm_cfg.get("quota") or {}

        self.cfg = OllamaCloudConfig(
            host=str(llm_cfg.get("host", "https://ollama.com")),
            model=str(llm_cfg.get("model", "gemma3:270m")),
            api_key=_resolve_env(str(llm_cfg.get("api_key", ""))),
            preflight=bool(quota_cfg.get("preflight", True)),
            min_interval_seconds=float(quota_cfg.get("min_interval_seconds", 1.0)),
            timeout_seconds=float(llm_cfg.get("timeout_seconds", 360.0)),
        )

        if not self.cfg.api_key:
            raise LLMAuthError(
                "Ollama Cloud kräver api_key i config.yaml (eller via env-var)."
            )

        self._client = AsyncClient(
            host=self.cfg.host,
            headers={"Authorization": f"Bearer {self.cfg.api_key}"},
            timeout=self.cfg.timeout_seconds,
        )
        self.log = logging.getLogger(__name__)
        self._last_call_ts: float = 0.0
        self._preflight_ok_ts: float = 0.0

    async def _throttle(self) -> None:
        now = time.time()
        dt = now - self._last_call_ts
        if dt < self.cfg.min_interval_seconds:
            await asyncio.sleep(self.cfg.min_interval_seconds - dt)
        self._last_call_ts = time.time()

    async def _preflight_quota_check(self) -> None:
        if not self.cfg.preflight:
            return

        # Kör inte preflight onödigt ofta
        now = time.time()
        if (now - self._preflight_ok_ts) < 10.0:
            return

        try:
            # list() motsvarar /api/tags och kräver auth på ollama.com
            # :contentReference[oaicite:4]{index=4}
            await self._client.list()
            self._preflight_ok_ts = time.time()
        except Exception as e:
            if _is_status(e, 401) or _is_status(e, 403):
                raise LLMAuthError(
                    "Ollama Cloud auth misslyckades (api_key fel/nekad)."
                ) from e
            if _is_status(e, 429):
                ra = _extract_retry_after_seconds(e)
                raise LLMRateLimitError(
                    "Ollama Cloud: rate limit/quota uppnådd (preflight).", ra
                ) from e
            raise LLMUnavailableError(
                f"Ollama Cloud preflight misslyckades: {e}"
            ) from e

    async def chat(
        self, messages: List[Dict[str, str]], temperature: float = 0.2
    ) -> str:
        """
        Returnerar en enda textsträng (sammanfogad content).
        Hanterar preflight quota-check och 429/timeout tydligt.
        """
        await self._throttle()
        await self._preflight_quota_check()
        self.log.info("LLM request start (model=%s)", self.cfg.model)
        # Ollama API använder "options" för parametrar.
        # (temperature är ett standard-alternativ i Ollama.)
        payload_options = {"temperature": temperature}

        try:
            resp = await self._client.chat(
                model=self.cfg.model,
                messages=messages,
                stream=False,
                options=payload_options,
            )
            # resp kan vara dict-lik eller typed; vi stödjer båda:
            if isinstance(resp, dict):
                return (resp.get("message") or {}).get("content", "") or ""
            return getattr(resp.message, "content", "") or ""
        except Exception as e:
            if _is_status(e, 401) or _is_status(e, 403):
                raise LLMAuthError(
                    "Ollama Cloud: auth misslyckades (api_key fel/nekad)."
                ) from e
            if _is_status(e, 429):
                ra = _extract_retry_after_seconds(e)
                raise LLMRateLimitError(
                    "Ollama Cloud: rate limit/quota uppnådd.", ra
                ) from e
            # timeouts kan komma som httpx.TimeoutException under huven
            name = e.__class__.__name__.lower()
            if "timeout" in name:
                raise LLMUnavailableError(f"Ollama Cloud: timeout: {e}") from e
            raise LLMUnavailableError(f"Ollama Cloud: chat misslyckades: {e}") from e
