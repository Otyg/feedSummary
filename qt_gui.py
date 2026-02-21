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
import os
import sys
from typing import Any, Dict, List

import markdown as md
from PySide6.QtCore import Qt, QThread, Signal, QDate
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QTabWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QTextBrowser,
    QSplitter,
    QDialog,
    QFormLayout,
    QSpinBox,
    QComboBox,
    QCheckBox,
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QDateEdit,
    QMessageBox,
)
from PySide6.QtCore import QUrl

from summarizer.main import run_pipeline

from app_shared import (
    load_config,
    get_store,
    format_ts,
    published_ts,
    filter_articles,
    ArticleFilters,
    get_ui_options,
)

CONFIG_PATH = os.environ.get("FEEDSUMMARY_CONFIG", "config.yaml")


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


class RefreshDialog(QDialog):
    def __init__(self, parent: QWidget, cfg: Dict[str, Any], ui_opts):
        super().__init__(parent)
        self.setWindowTitle("Refresh – inställningar")
        self.cfg = cfg
        self.ui_opts = ui_opts

        root = QVBoxLayout(self)

        form = QFormLayout()

        self.lb_value = QSpinBox()
        self.lb_value.setMinimum(1)
        self.lb_value.setMaximum(9999)
        self.lb_value.setValue(int(self.ui_opts.default_lookback_value))

        self.lb_unit = QComboBox()
        self.lb_unit.addItems(["h", "d", "w", "m", "y"])
        self.lb_unit.setCurrentText(str(self.ui_opts.default_lookback_unit))

        row = QHBoxLayout()
        row.addWidget(self.lb_value)
        row.addWidget(self.lb_unit)
        row_wrap = QWidget()
        row_wrap.setLayout(row)
        form.addRow("Lookback", row_wrap)

        self.prompt_pkg = QComboBox()
        self.prompt_pkg.addItems(self.ui_opts.prompt_packages)
        if self.ui_opts.default_prompt_package:
            self.prompt_pkg.setCurrentText(self.ui_opts.default_prompt_package)
        form.addRow("Prompt-paket", self.prompt_pkg)

        root.addLayout(form)

        # Topics
        self.topic_checks: Dict[str, QCheckBox] = {}
        if self.ui_opts.topic_options:
            gb = QGroupBox("Ämnen")
            v = QVBoxLayout(gb)
            for t in self.ui_opts.topic_options:
                cb = QCheckBox(t)
                self.topic_checks[t] = cb
                v.addWidget(cb)
            root.addWidget(gb)

        # Sources
        self.source_checks: Dict[str, QCheckBox] = {}
        self.source_topics: Dict[str, List[str]] = {}
        gb = QGroupBox("Källor")
        v = QVBoxLayout(gb)
        for s in self.ui_opts.source_options:
            name = s["name"]
            topics = s.get("topics") or []
            self.source_topics[name] = topics
            label = name + (f" · {', '.join(topics)}" if topics else "")
            cb = QCheckBox(label)
            cb.setChecked(bool(s.get("default_checked", True)))
            self.source_checks[name] = cb
            v.addWidget(cb)
        root.addWidget(gb)

        # topic -> sources (same semantics as web UI)
        for t, tcb in self.topic_checks.items():
            tcb.stateChanged.connect(lambda _=None, topic=t: self._apply_topic(topic))

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.btn_cancel = QPushButton("Avbryt")
        self.btn_ok = QPushButton("Kör refresh")
        actions.addWidget(self.btn_cancel)
        actions.addWidget(self.btn_ok)
        root.addLayout(actions)

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self.accept)

    def _apply_topic(self, topic: str) -> None:
        checked = self.topic_checks[topic].isChecked()
        for src, cb in self.source_checks.items():
            if topic in (self.source_topics.get(src) or []):
                cb.setChecked(checked)

    def overrides(self) -> Dict[str, Any]:
        lookback = f"{self.lb_value.value()}{self.lb_unit.currentText()}"
        prompt_pkg = self.prompt_pkg.currentText().strip()

        sources = [s for s, cb in self.source_checks.items() if cb.isChecked()]
        topics = [t for t, cb in self.topic_checks.items() if cb.isChecked()]

        ov: Dict[str, Any] = {"lookback": lookback}
        if prompt_pkg:
            ov["prompt_package"] = prompt_pkg

        # sources wins; else topics
        if sources:
            ov["sources"] = sources
        elif topics:
            ov["topics"] = topics

        return ov


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FeedSummary (Qt)")
        self.resize(1200, 800)

        self.cfg = load_config(CONFIG_PATH)
        self.store = get_store(self.cfg)
        self.ui_opts = get_ui_options(self.cfg, config_path=CONFIG_PATH)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self._build_summaries_tab()
        self._build_articles_tab()

        self.reload_summaries()
        self.reload_articles()

    # ---- Summaries tab ----
    def _build_summaries_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        top = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh…")
        self.lbl_status = QLabel("idle")
        top.addWidget(self.btn_refresh)
        top.addWidget(self.lbl_status, 1)
        layout.addLayout(top)

        split = QSplitter()
        self.summary_list = QListWidget()
        self.summary_view = QTextBrowser()
        split.addWidget(self.summary_list)
        split.addWidget(self.summary_view)
        split.setStretchFactor(1, 1)
        layout.addWidget(split, 1)

        self.btn_refresh.clicked.connect(self.open_refresh)
        self.summary_list.currentItemChanged.connect(self.on_summary_selected)

        self.tabs.addTab(w, "Sammanfattningar")

    def reload_summaries(self):
        docs = self.store.list_summary_docs() or []
        docs.sort(key=lambda d: int(d.get("created") or 0), reverse=True)

        self.summary_list.blockSignals(True)
        self.summary_list.clear()

        for d in docs:
            sid = str(d.get("id") or "")
            created = int(d.get("created") or 0)
            n = len(d.get("sources") or [])
            item = QListWidgetItem(f"{format_ts(created)} · Artiklar: {n}")
            item.setData(Qt.UserRole, sid)
            self.summary_list.addItem(item)

        self.summary_list.blockSignals(False)

        if self.summary_list.count():
            self.summary_list.setCurrentRow(0)
        else:
            self.summary_view.setHtml("<p>Inga sammanfattningar ännu.</p>")

    def on_summary_selected(self, current: QListWidgetItem, _prev: QListWidgetItem):
        if not current:
            return
        sid = current.data(Qt.UserRole)
        doc = self.store.get_summary_doc(str(sid))
        if not doc:
            self.summary_view.setHtml("<p>Kunde inte läsa sammanfattning.</p>")
            return
        text = doc.get("summary", "") or ""
        html = md.markdown(text, extensions=["extra"])
        self.summary_view.setHtml(html)

    def open_refresh(self):
        dlg = RefreshDialog(self, self.cfg, self.ui_opts)
        if dlg.exec() != QDialog.Accepted:
            return

        overrides = dlg.overrides()
        self.btn_refresh.setEnabled(False)
        self.lbl_status.setText("running…")

        self.worker = PipelineWorker(self.cfg, overrides)
        self.worker.status.connect(self.lbl_status.setText)
        self.worker.done.connect(self.on_pipeline_done)
        self.worker.failed.connect(self.on_pipeline_failed)
        self.worker.start()

    def on_pipeline_done(self, _sid: object):
        self.lbl_status.setText("done")
        self.btn_refresh.setEnabled(True)

        # reload from disk (config/store may change)
        self.cfg = load_config(CONFIG_PATH)
        self.store = get_store(self.cfg)
        self.ui_opts = get_ui_options(self.cfg, config_path=CONFIG_PATH)

        self.reload_summaries()
        self.reload_articles()

    def on_pipeline_failed(self, err: str):
        self.lbl_status.setText("error")
        self.btn_refresh.setEnabled(True)
        QMessageBox.critical(self, "Refresh misslyckades", err)

    # ---- Articles tab ----
    def _build_articles_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        frow = QHBoxLayout()
        self.from_date = QDateEdit()
        self.from_date.setCalendarPopup(True)
        self.from_date.setDisplayFormat("yyyy-MM-dd")

        self.to_date = QDateEdit()
        self.to_date.setCalendarPopup(True)
        self.to_date.setDisplayFormat("yyyy-MM-dd")

        self.from_date.setDate(QDate.currentDate().addDays(-7))
        self.to_date.setDate(QDate.currentDate())

        self.btn_apply = QPushButton("Filtrera")
        self.btn_clear = QPushButton("Rensa")

        frow.addWidget(QLabel("Från"))
        frow.addWidget(self.from_date)
        frow.addWidget(QLabel("Till"))
        frow.addWidget(self.to_date)
        frow.addWidget(self.btn_apply)
        frow.addWidget(self.btn_clear)
        frow.addStretch(1)
        layout.addLayout(frow)

        # topics + sources lists
        boxes = QHBoxLayout()

        self.topic_checks: Dict[str, QCheckBox] = {}
        self.source_checks: Dict[str, QCheckBox] = {}
        self.source_topics: Dict[str, List[str]] = {}

        gb_t = QGroupBox("Ämnen")
        vb_t = QVBoxLayout(gb_t)
        for t in self.ui_opts.topic_options:
            cb = QCheckBox(t)
            self.topic_checks[t] = cb
            vb_t.addWidget(cb)
            cb.stateChanged.connect(
                lambda _=None, topic=t: self._apply_topic_to_sources(topic)
            )
        boxes.addWidget(gb_t)

        gb_s = QGroupBox("Källor")
        vb_s = QVBoxLayout(gb_s)
        for s in self.ui_opts.source_options:
            name = s["name"]
            self.source_topics[name] = s.get("topics") or []
            cb = QCheckBox(name)
            self.source_checks[name] = cb
            vb_s.addWidget(cb)
        boxes.addWidget(gb_s)

        layout.addLayout(boxes)

        self.btn_apply.clicked.connect(self.reload_articles)
        self.btn_clear.clicked.connect(self.clear_article_filters)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Titel", "Källa", "Publicerad", "Preview"]
        )
        self.table.cellDoubleClicked.connect(self.open_article)
        layout.addWidget(self.table, 1)

        self.tabs.addTab(w, "Artiklar")

    def _apply_topic_to_sources(self, topic: str) -> None:
        checked = self.topic_checks[topic].isChecked()
        for src, cb in self.source_checks.items():
            if topic in (self.source_topics.get(src) or []):
                cb.setChecked(checked)

    def clear_article_filters(self):
        for cb in self.topic_checks.values():
            cb.setChecked(False)
        for cb in self.source_checks.values():
            cb.setChecked(False)
        self.from_date.setDate(QDate.currentDate().addDays(-7))
        self.to_date.setDate(QDate.currentDate())
        self.reload_articles()

    def reload_articles(self):
        articles = self.store.list_articles() or []

        sources = [s for s, cb in self.source_checks.items() if cb.isChecked()]
        topics = [t for t, cb in self.topic_checks.items() if cb.isChecked()]

        from_ymd = self.from_date.date().toString("yyyy-MM-dd")
        to_ymd = self.to_date.date().toString("yyyy-MM-dd")

        filtered = filter_articles(
            articles,
            cfg=self.cfg,
            filters=ArticleFilters(
                sources=sources,
                topics=topics,
                from_ymd=from_ymd,
                to_ymd=to_ymd,
            ),
        )

        self.table.setRowCount(len(filtered))
        for r, a in enumerate(filtered):
            title = a.get("title", "")
            url = a.get("url", "")
            src = a.get("source", "")
            pub = a.get("published", "") or format_ts(published_ts(a))
            prev = (a.get("text", "") or "")[:300].replace("\n", " ")

            it0 = QTableWidgetItem(title)
            it0.setData(Qt.UserRole, url)
            it0.setToolTip(url)

            it1 = QTableWidgetItem(src)
            it2 = QTableWidgetItem(pub)
            it3 = QTableWidgetItem(prev)

            self.table.setItem(r, 0, it0)
            self.table.setItem(r, 1, it1)
            self.table.setItem(r, 2, it2)
            self.table.setItem(r, 3, it3)

        self.table.resizeColumnsToContents()

    def open_article(self, row: int, col: int):
        it = self.table.item(row, 0)
        if not it:
            return
        url = it.data(Qt.UserRole)
        if url:
            QDesktopServices.openUrl(QUrl(str(url)))


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
