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

        self.split = QSplitter(Qt.Horizontal) # type: ignore
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
