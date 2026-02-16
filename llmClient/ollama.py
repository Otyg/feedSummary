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

from dataclasses import dataclass
import asyncio
from typing import Dict, List

import aiohttp
from aiolimiter import AsyncLimiter
from llmClient import LLMError


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1"
    max_rps: float = 1.0  # request throttle (local still benefits)


class OllamaClient:
    """
    Minimal Ollama chat client using /api/chat (local).
    """

    def __init__(self, cfg: OllamaConfig):
        self.cfg = cfg
        # AsyncLimiter takes ints; approximate RPS with period=1 and max_rate=ceil(rps).
        # For rps < 1, we implement a sleep gate.
        self._limiter = AsyncLimiter(max_rate=max(1, int(cfg.max_rps)), time_period=1)
        self._min_interval = 1.0 / cfg.max_rps if cfg.max_rps > 0 else 0.0
        self._last_call = 0.0
        self._gate_lock = asyncio.Lock()

    async def _rate_gate(self):
        if self._min_interval <= 0:
            return
        async with self._gate_lock:
            now = asyncio.get_event_loop().time()
            wait_for = (self._last_call + self._min_interval) - now
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            self._last_call = asyncio.get_event_loop().time()

    async def chat(
        self, messages: List[Dict[str, str]], *, temperature: float = 0.2
    ) -> str:
        await self._rate_gate()

        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        async with aiohttp.ClientSession() as session:
            async with self._limiter:
                async with session.post(
                    f"{self.cfg.base_url.rstrip('/')}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=180),
                ) as resp:
                    if resp.status >= 400:
                        text = await resp.text(errors="ignore")
                        raise LLMError(f"Ollama error {resp.status}: {text[:400]}")
                    data = await resp.json()
                    return (data.get("message", {}) or {}).get("content", "").strip()
