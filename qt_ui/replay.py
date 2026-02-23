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


import markdown as md
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


class ReplayResultWindow(QDialog):
    def __init__(self, parent: QWidget, *, original_md: str, new_md: str, title: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1200, 820)

        outer = QVBoxLayout(self)

        top = QHBoxLayout()
        self.btn_compare = QPushButton("Visa jämförelse")
        self.btn_compare.setCheckable(True)
        self.btn_close = QPushButton("Stäng")
        top.addWidget(self.btn_compare)
        top.addStretch(1)
        top.addWidget(self.btn_close)
        outer.addLayout(top)

        self.split = QSplitter(Qt.Horizontal)  # type: ignore
        self.new_view = QTextBrowser()
        self.orig_view = QTextBrowser()

        self.new_view.setHtml(md.markdown(new_md or "", extensions=["extra"]))
        self.orig_view.setHtml(md.markdown(original_md or "", extensions=["extra"]))

        self.split.addWidget(self.new_view)
        self.split.addWidget(self.orig_view)
        self.split.setSizes([750, 450])
        self.orig_view.hide()

        outer.addWidget(self.split, 1)

        self.btn_close.clicked.connect(self.accept)
        self.btn_compare.toggled.connect(self._toggle_compare)

    def _toggle_compare(self, enabled: bool) -> None:
        self.orig_view.setVisible(bool(enabled))
