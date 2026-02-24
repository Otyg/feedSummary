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
from typing import Any, Dict
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QTextDocument
from PySide6.QtPrintSupport import QPrinter, QPrintDialog
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from qt_ui.helpers import _fmt_dt_hm
from qt_ui.richtexteditor import RichTextEditorDialog
from uicommon import published_ts


class ArticleReaderDialog(QDialog):
    def __init__(self, parent: QWidget, article: Dict[str, Any]):
        super().__init__(parent)
        self.article = article or {}
        self.setWindowTitle("Artikel")
        self.resize(980, 760)

        outer = QVBoxLayout(self)

        meta_box = QGroupBox("Metadata")
        meta_layout = QFormLayout(meta_box)

        def _val(key: str) -> str:
            v = self.article.get(key)
            if v is None:
                return ""
            if isinstance(v, (dict, list)):
                return str(v)
            return str(v)

        title = _val("title")
        source = _val("source")
        url = _val("url")
        published = _val("published") or _fmt_dt_hm(published_ts(self.article))

        meta_layout.addRow("Titel", QLabel(title))
        meta_layout.addRow("Källa", QLabel(source))
        meta_layout.addRow("Publicerad", QLabel(published))
        meta_layout.addRow("ID", QLabel(_val("id")))
        meta_layout.addRow("URL", QLabel(url))
        outer.addWidget(meta_box)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.btn_print = QPushButton("Skriv ut")
        self.btn_open = QPushButton("Öppna i webbläsare")
        self.btn_close = QPushButton("Stäng")
        actions.addWidget(self.btn_print)
        actions.addWidget(self.btn_open)
        actions.addWidget(self.btn_close)
        outer.addLayout(actions)

        self.btn_close.clicked.connect(self.accept)
        self.btn_open.clicked.connect(self._open_in_browser)
        self.btn_print.clicked.connect(self._print_article)

        self.text_view = QTextBrowser()
        self.text_view.setOpenExternalLinks(True)
        outer.addWidget(self.text_view, 1)

        full_text = (
            self.article.get("text") or ""
        ).strip() or "(Ingen text lagrad för artikeln.)"
        safe = full_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.text_view.setHtml(
            f"<pre style='white-space: pre-wrap; font-family: system-ui;'>{safe}</pre>"
        )

    def _open_in_browser(self) -> None:
        url = str(self.article.get("url") or "").strip()
        if not url:
            QMessageBox.information(self, "Ingen URL", "Artikeln saknar URL.")
            return
        QDesktopServices.openUrl(QUrl(url))

    def _print_article(self) -> None:
        title = str(self.article.get("title") or "").strip()
        source = str(self.article.get("source") or "").strip()
        url = str(self.article.get("url") or "").strip()
        pub = str(self.article.get("published") or "").strip() or _fmt_dt_hm(published_ts(self.article))
        full_text = (self.article.get("text") or "").strip() or "(Ingen text lagrad för artikeln.)"

        # Build initial HTML with a nice header
        def esc(s: str) -> str:
            return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        html = f"""
        <h1>{esc(title)}</h1>
        <p><b>Källa:</b> {esc(source)}<br/>
            <b>Publicerad:</b> {esc(pub)}<br/>
            <b>URL:</b> {esc(url)}</p>
        <hr/>
        <p style="white-space: pre-wrap;">{esc(full_text).replace("\\n", "<br/>")}</p>
        """

        dlg = RichTextEditorDialog(self, title="Artikel – redigera före utskrift", initial_html=f"<html><body>{html}</body></html>")
        dlg.exec()