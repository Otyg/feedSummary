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
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

from llmClient.ollama_cloud import LLMRateLimitError

log = logging.getLogger(__name__)


class LLMClient(Protocol):
    async def chat(
        self, messages: List[Dict[str, str]], *, temperature: float = 0.2
    ) -> str: ...


@dataclass
class FallbackPolicy:
    max_quota_retries: int = (
        1  # "vid retry" -> om det slår i igen efter 1 retry: fallback
    )
    default_wait_s: int = 30  # om retry_after saknas
    jitter_s: float = 0.0  # håll 0 om du vill vara deterministisk


class FallbackLLMClient:
    """
    Wrapper som kör primary, men om quota/rate-limit kvarstår efter retry -> fallback.

    Beteende:
      - Får vi LLMRateLimitError:
          * vänta retry_after (eller default_wait_s)
          * försök igen upp till max_quota_retries
      - Om fortfarande rate limited:
          * växla permanent till fallback (för resten av processen)
    """

    def __init__(
        self,
        primary: LLMClient,
        fallback: Optional[LLMClient],
        policy: Optional[FallbackPolicy] = None,
    ):
        self.primary = primary
        self.fallback = fallback
        self.policy = policy or FallbackPolicy()
        self._force_fallback = False

    async def chat(
        self, messages: List[Dict[str, str]], *, temperature: float = 0.2
    ) -> str:
        # Om vi redan växlat till fallback
        if self._force_fallback:
            if not self.fallback:
                raise RuntimeError(
                    "Fallback aktiverad men llm_fallback saknas i config."
                )
            return await self.fallback.chat(messages, temperature=temperature)

        # Försök primary med quota-retries
        attempt = 0
        while True:
            try:
                return await self.primary.chat(messages, temperature=temperature)
            except LLMRateLimitError as e:
                attempt += 1
                wait_s = int(e.retry_after_seconds or self.policy.default_wait_s)

                # Om vi redan testat retry och slår i igen -> fallback
                if attempt > self.policy.max_quota_retries:
                    if not self.fallback:
                        # ingen fallback konfigurerad
                        raise
                    log.warning(
                        "LLM quota/rate-limit kvarstår efter %d retry(s). Växlar till fallback.",
                        self.policy.max_quota_retries,
                    )
                    self._force_fallback = True
                    return await self.fallback.chat(messages, temperature=temperature)

                log.warning(
                    "LLM rate-limited (attempt %d/%d). Väntar %ss och retry...",
                    attempt,
                    self.policy.max_quota_retries,
                    wait_s,
                )
                await asyncio.sleep(wait_s)
