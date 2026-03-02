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
import threading
from typing import Any, Dict, Optional

from PySide6.QtCore import QThread, Signal

from feedsummary_core.summarizer.main import run_pipeline, run_resume_job
from feedsummary_core.summarizer.prompt_replay import PromptSet, rerun_summary_from_existing

from uicommon.bootstrap_ui import resolve_config_path
from qt_ui.interactive_llm import InteractiveLLMClient, LLMFailureContext
from feedsummary_core.llm_client import create_llm_client

RUNTIME = resolve_config_path()
CONFIG_PATH = str(RUNTIME.config_path)


class _DecisionBox:
    def __init__(self) -> None:
        self._ev = threading.Event()
        self._decision: str = "abort"

    def set(self, decision: str) -> None:
        self._decision = str(decision or "abort").strip().lower()
        self._ev.set()

    def wait(self) -> str:
        self._ev.wait()
        return self._decision

    def reset(self) -> None:
        self._decision = "abort"
        self._ev.clear()


class PipelineWorker(QThread):
    status = Signal(str)
    done = Signal(object)  # (summary_id, job_id)
    failed = Signal(str)

    # NEW: ask UI for retry/skip/abort on LLM failure
    llm_decision_requested = Signal(object)  # payload dict

    def __init__(
        self, cfg: Dict[str, Any], overrides: Dict[str, Any], job_id: Optional[int]
    ):
        super().__init__()
        self.cfg = cfg
        self.overrides = overrides
        self.job_id = job_id
        self._decision_box = _DecisionBox()

    def set_llm_decision(self, decision: str) -> None:
        self._decision_box.set(decision)

    def _decision_provider(self, ctx: LLMFailureContext) -> str:
        self._decision_box.reset()
        self.llm_decision_requested.emit(
            {
                "kind": "llm_failure",
                "provider": ctx.provider,
                "model": ctx.model,
                "temperature": ctx.temperature,
                "attempt": ctx.attempt,
                "exception_type": ctx.exception_type,
                "exception_message": ctx.exception_message,
                "response_body": ctx.response_body,
                "message_roles": ctx.message_roles,
                "user_chars": ctx.user_chars,
                "total_chars": ctx.total_chars,
            }
        )
        return self._decision_box.wait()

    def run(self) -> None:
        try:
            self.status.emit("Kör pipeline…")

            base_llm = create_llm_client(self.cfg)
            llm_cfg = self.cfg.get("llm") or {}
            provider = str(llm_cfg.get("provider") or llm_cfg.get("type") or "unknown")
            model = str(llm_cfg.get("model") or llm_cfg.get("name") or "unknown")

            llm = InteractiveLLMClient(
                inner=base_llm,
                decision_provider=self._decision_provider,
                provider_name=provider,
                model_name=model,
            )

            summary_id = asyncio.run(
                run_pipeline(
                    CONFIG_PATH,
                    job_id=self.job_id,
                    overrides=self.overrides,
                    config_dict=self.cfg,
                    llm=llm,
                    # NOTE: run_pipeline already creates its own llm in your code.
                    # For minimal change, we rely on llmClient wrapping inside create_llm_client usage
                    # in run_pipeline; if you want full injection, we can add an optional llm param.
                )
            )
            self.done.emit((summary_id, self.job_id))
        except Exception as e:
            self.failed.emit(str(e))


class ResumeWorker(QThread):
    """
    Resume-from-checkpoint and persist a summary_doc.
    Produces the SAME summary_doc structure as a normal refresh.
    """

    status = Signal(str)
    done = Signal(object)  # summary_id (str)
    failed = Signal(str)

    # NEW: ask UI for retry/skip/abort on LLM failure
    llm_decision_requested = Signal(object)

    def __init__(self, *, cfg: Dict[str, Any], store, llm, job_id: int):
        super().__init__()
        self.cfg = cfg
        self.store = store
        self.llm = llm
        self.job_id = int(job_id)
        self._decision_box = _DecisionBox()

    def set_llm_decision(self, decision: str) -> None:
        self._decision_box.set(decision)

    def _decision_provider(self, ctx: LLMFailureContext) -> str:
        self._decision_box.reset()
        self.llm_decision_requested.emit(
            {
                "kind": "llm_failure",
                "provider": ctx.provider,
                "model": ctx.model,
                "temperature": ctx.temperature,
                "attempt": ctx.attempt,
                "exception_type": ctx.exception_type,
                "exception_message": ctx.exception_message,
                "response_body": ctx.response_body,
                "message_roles": ctx.message_roles,
                "user_chars": ctx.user_chars,
                "total_chars": ctx.total_chars,
            }
        )
        return self._decision_box.wait()

    def run(self) -> None:
        try:
            self.status.emit(f"Återupptar från checkpoint (job {self.job_id})…")

            # Wrap the provided llm too (in case caller passed a plain client)
            llm_cfg = self.cfg.get("llm") or {}
            provider = str(llm_cfg.get("provider") or llm_cfg.get("type") or "unknown")
            model = str(llm_cfg.get("model") or llm_cfg.get("name") or "unknown")

            llm = InteractiveLLMClient(
                inner=self.llm,
                decision_provider=self._decision_provider,
                provider_name=provider,
                model_name=model,
            )

            summary_id = asyncio.run(
                run_resume_job(
                    config=self.cfg,
                    store=self.store,
                    llm=llm,
                    job_id=self.job_id,
                )
            )
            self.done.emit(summary_id)
        except Exception as e:
            self.failed.emit(str(e))


class PromptReplayWorker(QThread):
    """
    Ephemeral replay: returns computed summary text as dict; does NOT persist.
    """

    status = Signal(str)
    done = Signal(object)  # result dict
    failed = Signal(str)

    def __init__(
        self, *, cfg: Dict[str, Any], store, summary_id: str, prompts: PromptSet
    ):
        super().__init__()
        self.cfg = cfg
        self.store = store
        self.summary_id = summary_id
        self.prompts = prompts

    def run(self) -> None:
        try:
            self.status.emit("Kör om summary med ändrade prompts…")
            result = asyncio.run(
                rerun_summary_from_existing(
                    config_path=CONFIG_PATH,
                    cfg=self.cfg,
                    store=self.store,
                    summary_id=self.summary_id,
                    new_prompts=self.prompts,
                )
            )
            self.done.emit(result)
        except Exception as e:
            self.failed.emit(str(e))
