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
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Coalesce model-retraining requests and execute them outside HTTP requests."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable


class CoalescingRetrainer:
    """Run one retraining job at a time and merge bursts of requests.

    A request received while training is running schedules exactly one follow-up
    pass. The short debounce also lets a rapid remove/add sequence become one
    training pass instead of two.
    """

    def __init__(
        self,
        retrain: Callable[[], None],
        *,
        debounce_seconds: float = 2.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._retrain = retrain
        self._debounce_seconds = max(0.0, float(debounce_seconds))
        self._logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._changed = threading.Event()
        self._generation = 0
        self._running = False
        self._last_reason = ""

    def request(self, reason: str = "") -> str:
        """Queue retraining and return ``started`` or ``queued``."""
        with self._lock:
            self._generation += 1
            self._last_reason = str(reason or "")
            already_running = self._running
            self._running = True
            self._changed.set()
            if not already_running:
                threading.Thread(
                    target=self._run,
                    name="tagging-ml-retrain",
                    daemon=True,
                ).start()
        return "queued" if already_running else "started"

    def _wait_for_quiet_generation(self) -> tuple[int, str]:
        while True:
            self._changed.clear()
            with self._lock:
                generation = self._generation
                reason = self._last_reason
            if not self._changed.wait(self._debounce_seconds):
                with self._lock:
                    if generation == self._generation:
                        return generation, reason

    def _run(self) -> None:
        while True:
            generation, reason = self._wait_for_quiet_generation()
            try:
                self._logger.info("Starting ML tag retraining (%s)", reason)
                self._retrain()
                self._logger.info("ML tag retraining completed")
            except Exception:  # noqa: BLE001 - background boundary must stay alive
                self._logger.exception("ML tag retraining failed; previous model retained")

            with self._lock:
                if generation == self._generation:
                    self._running = False
                    return
                self._logger.info(
                    "Tag data changed during retraining; scheduling one follow-up pass"
                )
