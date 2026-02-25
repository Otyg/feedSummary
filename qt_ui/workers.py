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
from typing import Any, Dict, Optional

from PySide6.QtCore import QThread, Signal

from summarizer.main import run_pipeline, run_resume_job
from summarizer.prompt_replay import PromptSet, rerun_summary_from_existing

from uicommon.bootstrap_ui import resolve_config_path

RUNTIME = resolve_config_path()
CONFIG_PATH = str(RUNTIME.config_path)


class PipelineWorker(QThread):
    status = Signal(str)
    done = Signal(object)  # (summary_id, job_id)
    failed = Signal(str)

    def __init__(
        self, cfg: Dict[str, Any], overrides: Dict[str, Any], job_id: Optional[int]
    ):
        super().__init__()
        self.cfg = cfg
        self.overrides = overrides
        self.job_id = job_id

    def run(self) -> None:
        try:
            self.status.emit("Kör pipeline…")
            summary_id = asyncio.run(
                run_pipeline(
                    CONFIG_PATH,
                    job_id=self.job_id,
                    overrides=self.overrides,
                    config_dict=self.cfg,
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

    def __init__(self, *, cfg: Dict[str, Any], store, llm, job_id: int):
        super().__init__()
        self.cfg = cfg
        self.store = store
        self.llm = llm
        self.job_id = int(job_id)

    def run(self) -> None:
        try:
            self.status.emit(f"Återupptar från checkpoint (job {self.job_id})…")
            summary_id = asyncio.run(
                run_resume_job(
                    config=self.cfg,
                    store=self.store,
                    llm=self.llm,
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
