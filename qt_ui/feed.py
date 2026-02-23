from __future__ import annotations
from typing import Any, Dict, Optional

from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from qt_ui.helpers import _safe_list_str, _parse_csv


class FeedEditDialog(QDialog):
    def __init__(self, parent: QWidget, feed: Optional[Dict[str, Any]] = None):
        super().__init__(parent)
        self.setWindowTitle("Feed")
        self.resize(640, 420)
        self._feed_in = feed or {}

        outer = QVBoxLayout(self)
        form = QFormLayout()

        self.name = QLineEdit(str(self._feed_in.get("name") or ""))
        self.url = QLineEdit(str(self._feed_in.get("url") or ""))

        topics = _safe_list_str(self._feed_in.get("topics"))
        self.topics = QLineEdit(", ".join(topics))

        cat_inc = _safe_list_str(self._feed_in.get("category_include"))
        self.category_include = QLineEdit(", ".join(cat_inc))

        form.addRow("Name", self.name)
        form.addRow("URL", self.url)
        form.addRow("Topics (comma)", self.topics)
        form.addRow("Category include (comma, optional)", self.category_include)

        outer.addLayout(form)

        hint = QLabel(
            "Tips: Topics används för grouping/urval i UI. Category include är valfritt."
        )
        hint.setWordWrap(True)
        outer.addWidget(hint)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.btn_cancel = QPushButton("Avbryt")
        self.btn_ok = QPushButton("Spara")
        actions.addWidget(self.btn_cancel)
        actions.addWidget(self.btn_ok)
        outer.addLayout(actions)

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self.accept)

    def value(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        out["name"] = self.name.text().strip()
        out["url"] = self.url.text().strip()

        topics = _parse_csv(self.topics.text())
        if topics:
            out["topics"] = topics

        cat_inc = _parse_csv(self.category_include.text())
        if cat_inc:
            out["category_include"] = cat_inc

        return out
