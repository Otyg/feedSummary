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

from llmClient.fallback_client import FallbackLLMClient, FallbackPolicy

from typing import Any, Dict, List, Optional, Protocol


class LLMError(Exception):
    pass


class LLMClient(Protocol):
    async def chat(
        self, messages: List[Dict[str, str]], *, temperature: float = 0.2
    ) -> str: ...


def _create_single_llm(llm_cfg: Dict[str, Any]):
    provider = (llm_cfg.get("provider") or "ollama").lower()

    if provider == "ollama_cloud_gemma3_270m":
        from llmClient.ollama_cloud import OllamaCloudGemmaClient

        return OllamaCloudGemmaClient(llm_cfg)

    if provider == "ollama":
        from llmClient.ollama_local import OllamaClient, OllamaConfig

        cfg = OllamaConfig(
            base_url=str(llm_cfg.get("base_url", "http://localhost:11434")),
            model=str(llm_cfg.get("model", "llama3.1:latest")),
            max_rps=float(llm_cfg.get("max_rps", 1.0)),
            first_byte_timeout_s=int(llm_cfg.get("first_byte_timeout_s", 900)),
            sock_read_timeout_s=int(llm_cfg.get("sock_read_timeout_s", 300)),
            progress_log_every_s=float(llm_cfg.get("progress_log_every_s", 2.0)),
            max_retries=int(llm_cfg.get("max_retries", 3)),
        )
        return OllamaClient(cfg)

    raise ValueError(f"Unsupported LLM provider: {provider}")


def create_llm_client(config: Dict[str, Any]):
    """
    Bygger primary från config['llm'] och (om finns) fallback från config['llm_fallback'].

    Fallback triggas när cloud slår i quota igen efter retry.
    """
    llm_cfg = config.get("llm") or {}
    primary = _create_single_llm(llm_cfg)

    fallback_cfg: Optional[Dict[str, Any]] = config.get("llm_fallback")
    fallback = (
        _create_single_llm(fallback_cfg) if isinstance(fallback_cfg, dict) else None
    )

    # policy kan ligga under llm.quota eller llm_fallback_policy – välj det du gillar
    quota_cfg = llm_cfg.get("quota") or {}
    policy = FallbackPolicy(
        max_quota_retries=int(quota_cfg.get("max_quota_retries", 1)),
        default_wait_s=int(quota_cfg.get("default_wait_s", 30)),
    )

    if fallback:
        return FallbackLLMClient(primary=primary, fallback=fallback, policy=policy)

    return primary
