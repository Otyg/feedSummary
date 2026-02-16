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

# llmClient/ollama.py
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import aiohttp
from aiolimiter import AsyncLimiter
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:latest"
    max_rps: float = 1.0

    # Hur länge vi kan vänta på att Ollama börjar svara (första bytes/headers).
    # Sätt högt om din maskin är långsam eller modellen “tänker” länge.
    first_byte_timeout_s: int = 900  # 15 min

    # Max “tystnad” mellan bytes när streaming redan är igång.
    sock_read_timeout_s: int = 300  # 5 min

    # Heartbeat-logg när data faktiskt kommer in (inte true progress)
    progress_log_every_s: float = 2.0

    # Retry
    max_retries: int = 3


class OllamaClient:
    """
    Ollama chat client using /api/chat with NDJSON streaming.

    Notera:
    - Ollama skickar typiskt NDJSON-rader. Vi läser med readline().
    - Vi sätter aiohttp timeout så att vi kan vänta länge på första byte.
    """

    def __init__(
        self, cfg: OllamaConfig, logger_override: Optional[logging.Logger] = None
    ):
        self.cfg = cfg
        self.log = logger_override or logger

        self._limiter = AsyncLimiter(max_rate=max(1, int(cfg.max_rps)), time_period=1)
        self._min_interval = 1.0 / cfg.max_rps if cfg.max_rps > 0 else 0.0
        self._last_call = 0.0
        self._gate_lock = asyncio.Lock()

        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

    async def _rate_gate(self):
        if self._min_interval <= 0:
            return
        async with self._gate_lock:
            now = asyncio.get_event_loop().time()
            wait_for = (self._last_call + self._min_interval) - now
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            self._last_call = asyncio.get_event_loop().time()

    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Återanvänd en enda ClientSession för stabilare sockets + lägre overhead.
        """
        async with self._session_lock:
            if self._session is not None and not self._session.closed:
                return self._session

            # Viktigt: tillåt lång total tid, men styr “socket read” (tystnad).
            # Vi sätter sock_read till max(...) för att inte dö vid first byte.
            sock_read = int(
                max(self.cfg.first_byte_timeout_s, self.cfg.sock_read_timeout_s)
            )

            timeout = aiohttp.ClientTimeout(
                total=None,
                sock_connect=30,
                sock_read=sock_read,
            )
            connector = aiohttp.TCPConnector(
                limit=1,  # en socket åt gången (du kör ändå sekventiellt)
                keepalive_timeout=30,
                ttl_dns_cache=300,
            )
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            return self._session

    async def close(self):
        """
        Valfritt: kalla vid shutdown om du vill stänga sessionen snyggt.
        """
        async with self._session_lock:
            if self._session and not self._session.closed:
                await self._session.close()
            self._session = None

    @retry(
        wait=wait_exponential_jitter(initial=2, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(
            (
                aiohttp.SocketTimeoutError,
                aiohttp.ClientError,
                asyncio.TimeoutError,
            )
        ),
        reraise=True,
    )
    async def chat(
        self, messages: List[Dict[str, str]], *, temperature: float = 0.2
    ) -> str:
        await self._rate_gate()
        session = await self._get_session()

        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }

        url = f"{self.cfg.base_url.rstrip('/')}/api/chat"
        self.log.info("LLM request start (model=%s)", self.cfg.model)

        chunks: List[str] = []
        total_chars = 0

        # Här kan vi sätta en *extra* “first byte timeout” på requesten,
        # men vi håller oss till session timeout + max(...) för enkelhet.
        # Om du vill hård-skilja dem: gör sock_read väldigt hög och
        # använd asyncio.wait_for runt första readline().
        async with self._limiter:
            self.log.info("Pre-request")
            async with session.post(url, json=payload) as resp:
                if resp.status >= 400:
                    text = await resp.text(errors="ignore")
                    self.log.error("Ollama error %s: %s", resp.status, text[:400])
                    raise RuntimeError(f"Ollama error {resp.status}: {text[:400]}")
                self.log.info("Request sent")
                last_log = asyncio.get_event_loop().time()

                # Läs NDJSON rad för rad
                while True:
                    raw = await resp.content.readline()
                    if not raw:
                        break

                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        # Ibland kan det komma partials – ignorera tysta fel
                        continue

                    piece = ((data.get("message") or {}).get("content")) or ""
                    if piece:
                        chunks.append(piece)
                        total_chars += len(piece)

                    now = asyncio.get_event_loop().time()
                    if (now - last_log) >= float(self.cfg.progress_log_every_s):
                        self.log.info(
                            "LLM still running... received_chars=%d", total_chars
                        )
                        last_log = now

                    if data.get("done") is True:
                        break

        text = "".join(chunks).strip()
        self.log.info("LLM request done (chars=%d)", len(text))
        return text
