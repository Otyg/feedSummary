from __future__ import annotations
import logging
import os
import threading
from PySide6.QtCore import QObject, Signal

def _env_log_level(default: int = logging.INFO) -> int:
    lvl = (os.environ.get("FEEDSUMMARY_LOG_LEVEL") or "").strip().upper()
    if not lvl:
        return default
    return getattr(logging, lvl, default)


class QtLogEmitter(QObject):
    text = Signal(str)


class QtStream:
    def __init__(self, emitter: QtLogEmitter):
        self.emitter = emitter
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        if not s:
            return 0
        with self._lock:
            self.emitter.text.emit(str(s))
        return len(s)

    def flush(self) -> None:
        return


class QtLoggingHandler(logging.Handler):
    def __init__(self, emitter: QtLogEmitter):
        super().__init__()
        self.emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.emitter.text.emit(msg + "\n")
        except Exception:
            pass
