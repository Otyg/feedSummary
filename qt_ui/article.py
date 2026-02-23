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
        printer = QPrinter(QPrinter.HighResolution) # type: ignore
        printer.setDocName("Artikel")

        dlg = QPrintDialog(printer, self)
        dlg.setWindowTitle("Skriv ut artikel")
        if dlg.exec() != QDialog.Accepted: # type: ignore
            return

        title = str(self.article.get("title") or "").strip()
        source = str(self.article.get("source") or "").strip()
        url = str(self.article.get("url") or "").strip()
        pub = str(self.article.get("published") or "").strip() or _fmt_dt_hm(
            published_ts(self.article)
        )

        full_text = (
            self.article.get("text") or ""
        ).strip() or "(Ingen text lagrad för artikeln.)"

        safe_text = (
            full_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        safe_title = (
            title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        safe_source = (
            source.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        safe_url = url.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        html = f"""
        <h2>{safe_title}</h2>
        <p><b>Källa:</b> {safe_source}<br/>
           <b>Publicerad:</b> {pub}<br/>
           <b>URL:</b> {safe_url}</p>
        <hr/>
        <pre style="white-space: pre-wrap; font-family: system-ui;">{safe_text}</pre>
        """
        doc = QTextDocument()
        doc.setHtml(html)
        doc.print_(printer)
