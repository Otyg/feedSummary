from __future__ import annotations
import asyncio
from typing import Any, Dict
from PySide6.QtCore import QThread, Signal


from qt_gui import CONFIG_PATH
from summarizer.main import run_pipeline
from summarizer.prompt_replay import PromptSet, rerun_summary_from_existing


class PipelineWorker(QThread):
    status = Signal(str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, cfg: Dict[str, Any], overrides: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.overrides = overrides

    def run(self) -> None:
        try:
            self.status.emit("Kör pipeline…")
            summary_id = asyncio.run(
                run_pipeline(
                    CONFIG_PATH,
                    job_id=None,
                    overrides=self.overrides,
                    config_dict=self.cfg,
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
