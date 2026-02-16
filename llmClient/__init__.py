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

from typing import Any, Dict, List, Protocol

from llmClient.ollama import OllamaClient, OllamaConfig


class LLMError(Exception):
    pass


class LLMClient(Protocol):
    async def chat(
        self, messages: List[Dict[str, str]], *, temperature: float = 0.2
    ) -> str: ...


def create_llm_client(llm_cfg: Dict[str, Any]) -> LLMClient:
    provider = (llm_cfg.get("provider") or "ollama").lower()

    if provider == "ollama":
        cfg = OllamaConfig(
            base_url=llm_cfg.get("base_url", "http://localhost:11434"),
            model=llm_cfg.get("model", "llama3.1"),
            max_rps=float(llm_cfg.get("max_rps", 1.0)),
            progress_log_every_s=float(llm_cfg.get("progress_log_every_s", 5.0)),
        )
        return OllamaClient(cfg)

    raise ValueError(f"Unsupported LLM provider: {provider}")
