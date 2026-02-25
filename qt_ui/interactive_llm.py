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

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class LLMClientProto(Protocol):
    async def chat(
        self, messages: List[Dict[str, str]], *, temperature: float = 0.2
    ) -> str: ...


@dataclass(frozen=True)
class LLMFailureContext:
    provider: str
    model: str
    temperature: float
    attempt: int
    exception_type: str
    exception_message: str
    response_body: str
    message_roles: str  # compact: "system,user"
    user_chars: int
    total_chars: int


DecisionProvider = Callable[[LLMFailureContext], str]
# expected decisions: "retry" | "skip" | "abort"


def _clip(s: str, n: int = 6000) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "\n…(truncated)…"


def _extract_response_body_best_effort(err: BaseException) -> str:
    """
    Walk exception chain and try to extract a response body.
    Works with:
      - aiohttp: err has .status / sometimes .message (but body usually not here)
      - requests/httpx: err.response.text / err.response.content
      - ollama-python: ResponseError often has .error or .message
      - our own RuntimeError messages that include body snippet
    """
    seen = set()
    cur: Optional[BaseException] = err
    bodies: List[str] = []

    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))

        for attr in ("body", "text", "error", "message"):
            v = getattr(cur, attr, None)
            if isinstance(v, str) and v.strip():
                bodies.append(f"{cur.__class__.__name__}.{attr}: {v.strip()}")

        resp = getattr(cur, "response", None)
        if resp is not None:
            # requests/httpx style
            try:
                txt = getattr(resp, "text", None)
                if isinstance(txt, str) and txt.strip():
                    bodies.append(f"response.text: {txt.strip()}")
            except Exception:
                pass
            try:
                content = getattr(resp, "content", None)
                if isinstance(content, (bytes, bytearray)) and content:
                    bodies.append(f"response.content(bytes): {content[:8000]!r}")
            except Exception:
                pass

        # move to cause/context
        nxt = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
        cur = nxt if isinstance(nxt, BaseException) else None

    # If the exception string itself looks like it contains a body snippet, include it.
    s = str(err).strip()
    if s:
        bodies.append(f"exception.str: {s}")

    # Deduplicate while keeping order
    out: List[str] = []
    used = set()
    for b in bodies:
        if b in used:
            continue
        used.add(b)
        out.append(b)

    return _clip("\n\n".join(out), 12000)


class InteractiveLLMClient:
    """
    Wrapper that:
      - logs response body on failure (best effort)
      - asks user (through DecisionProvider) whether to retry / skip / abort
      - if skip -> returns a placeholder summary so pipeline can continue
    """

    def __init__(
        self,
        *,
        inner: LLMClientProto,
        decision_provider: DecisionProvider,
        provider_name: str = "",
        model_name: str = "",
        max_user_prompt_chars_in_log: int = 0,  # set >0 if you want more logging
    ):
        self._inner = inner
        self._decide = decision_provider
        self._provider = provider_name or "unknown"
        self._model = model_name or "unknown"
        self._max_user_chars = int(max_user_prompt_chars_in_log or 0)

    async def chat(
        self, messages: List[Dict[str, str]], *, temperature: float = 0.2
    ) -> str:
        attempt = 1
        while True:
            try:
                return await self._inner.chat(messages, temperature=temperature)
            except Exception as e:
                body = _extract_response_body_best_effort(e)

                total_chars = sum(
                    len((m.get("content") or "")) for m in (messages or [])
                )
                user_chars = sum(
                    len((m.get("content") or ""))
                    for m in (messages or [])
                    if (m.get("role") == "user")
                )
                roles = ",".join([str(m.get("role") or "?") for m in (messages or [])])

                ctx = LLMFailureContext(
                    provider=self._provider,
                    model=self._model,
                    temperature=float(temperature),
                    attempt=int(attempt),
                    exception_type=e.__class__.__name__,
                    exception_message=str(e),
                    response_body=body,
                    message_roles=roles,
                    user_chars=int(user_chars),
                    total_chars=int(total_chars),
                )

                # Log response body (best effort). Keep at INFO/WARNING levels.
                logger.error(
                    "LLM call failed (provider=%s model=%s attempt=%s type=%s). Response/body:\n%s",
                    ctx.provider,
                    ctx.model,
                    ctx.attempt,
                    ctx.exception_type,
                    ctx.response_body,
                )

                decision = "abort"
                try:
                    decision = str(self._decide(ctx) or "abort").strip().lower()
                except Exception as decide_err:
                    logger.warning("Decision provider failed: %s", decide_err)
                    decision = "abort"

                if decision == "retry":
                    attempt += 1
                    continue

                if decision == "skip":
                    # Continue pipeline with a marker text (meta will still work)
                    return (
                        f"⚠️ *LLM-anrop hoppades över efter fel*.\n\n"
                        f"- Provider: `{ctx.provider}`\n"
                        f"- Model: `{ctx.model}`\n"
                        f"- Fel: `{ctx.exception_type}`: {ctx.exception_message}\n"
                    )

                # abort
                raise
