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

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import markdown as md
import yaml
from PySide6.QtCore import Qt, QDate
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from qt_ui.article import ArticleReaderDialog
from qt_ui.feed import FeedEditDialog
from qt_ui.helpers import _safe_list_str, resolve_feeds_path
from qt_ui.log import _env_log_level, QtLogEmitter, QtStream, QtLoggingHandler
from qt_ui.refresh import RefreshDialog
from qt_ui.replay import ReplayResultWindow
from qt_ui.workers import PipelineWorker, PromptReplayWorker, ResumeWorker
from qt_ui.richtexteditor import RichTextEditorDialog

from uicommon import (
    load_config,
    get_store,
    format_ts,
    published_ts,
    filter_articles,
    ArticleFilters,
    get_ui_options,
)
from uicommon.bootstrap_ui import resolve_config_path

from summarizer.prompt_replay import (
    PromptSet,
    get_promptset_for_summary,
    list_prompt_packages,
    load_prompt_package,
    save_prompt_package,
)

from llmClient import create_llm_client  # NEW: for resume

RUNTIME = resolve_config_path()
CONFIG_PATH = str(RUNTIME.config_path)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FeedSummary (Qt)")
        self.resize(1280, 900)

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

        self.cfg = load_config(CONFIG_PATH)
        self.store = get_store(self.cfg)
        self.ui_opts = get_ui_options(self.cfg, config_path=CONFIG_PATH)

        self._last_replay_window: Optional[ReplayResultWindow] = None
        self.pl_selected_summary_id: Optional[str] = None

        # NEW: remember job_id of last refresh so we can offer resume
        self._last_job_id: Optional[int] = None

        splitter = QSplitter(Qt.Vertical)  # type: ignore

        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)

        # Log panel
        log_container = QWidget()
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(6, 6, 6, 6)

        log_toolbar = QHBoxLayout()
        log_toolbar.addWidget(QLabel("Logg (logging + stdout/stderr)"))
        log_toolbar.addStretch(1)
        self.btn_clear_log = QPushButton("Rensa")
        log_toolbar.addWidget(self.btn_clear_log)
        log_layout.addLayout(log_toolbar)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setPlaceholderText("Logg…")
        log_layout.addWidget(self.log_view, 1)

        splitter.addWidget(log_container)
        splitter.setSizes([680, 220])
        splitter.setCollapsible(1, True)

        self.setCentralWidget(splitter)
        self.btn_clear_log.clicked.connect(self.log_view.clear)

        # logging bridge
        self._log_emitter = QtLogEmitter()
        self._log_emitter.text.connect(self._append_log)

        sys.stdout = QtStream(self._log_emitter)
        sys.stderr = QtStream(self._log_emitter)
        self._install_logging_bridge()

        # Tabs
        self._build_summaries_tab()
        self._build_articles_tab()
        self._build_promptlab_tab()
        self._build_feeds_tab()

        # Initial load
        self.reload_summaries()
        self.reload_articles()
        self._refresh_promptlab_lists()
        self._feeds_reload()

        logging.getLogger("feedsum.qt").info("Bootstrap config_path=%s", CONFIG_PATH)

    def _on_llm_decision_requested(self, payload: object) -> None:
        try:
            p = payload if isinstance(payload, dict) else {}
            provider = str(p.get("provider") or "unknown")
            model = str(p.get("model") or "unknown")
            attempt = int(p.get("attempt") or 1)
            et = str(p.get("exception_type") or "Error")
            em = str(p.get("exception_message") or "")
            body = str(p.get("response_body") or "")
        except Exception:
            provider, model, attempt, et, em, body = (
                "unknown",
                "unknown",
                1,
                "Error",
                "",
                "",
            )

        msg = (
            f"LLM-anrop misslyckades.\n\n"
            f"Provider: {provider}\n"
            f"Model: {model}\n"
            f"Försök: {attempt}\n"
            f"Fel: {et}\n"
            f"Meddelande: {em}\n\n"
            f"Response/body har loggats i loggpanelen."
        )

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("LLM-fel")
        box.setText(msg)

        btn_retry = box.addButton("Försök igen", QMessageBox.AcceptRole)
        btn_skip = box.addButton("Gå vidare (hoppa över)", QMessageBox.DestructiveRole)
        btn_abort = box.addButton("Avbryt", QMessageBox.RejectRole)
        box.setDefaultButton(btn_retry)

        if body.strip():
            box.setDetailedText(body)

        box.exec()

        clicked = box.clickedButton()
        decision = "abort"
        if clicked == btn_retry:
            decision = "retry"
        elif clicked == btn_skip:
            decision = "skip"
        else:
            decision = "abort"

        try:
            w = getattr(self, "worker", None)
            if w and getattr(w, "isRunning", lambda: False)():
                w.set_llm_decision(decision)
                return
        except Exception:
            pass
        try:
            rw = getattr(self, "resume_worker", None)
            if rw and getattr(rw, "isRunning", lambda: False)():
                rw.set_llm_decision(decision)
                return
        except Exception:
            pass

    def _reload_all_config_dependent_ui(self) -> None:
        self.cfg = load_config(CONFIG_PATH)
        self.store = get_store(self.cfg)
        self.ui_opts = get_ui_options(self.cfg, config_path=CONFIG_PATH)

        self.reload_summaries()
        self._rebuild_article_filters_from_ui_opts()
        self.reload_articles()
        self._refresh_promptlab_lists()

    # -------- logging bridge --------
    def _install_logging_bridge(self) -> None:
        level = _env_log_level(logging.INFO)
        root = logging.getLogger()
        root.setLevel(level)

        self._qt_log_handler = QtLoggingHandler(self._log_emitter)
        self._qt_log_handler.setLevel(level)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        self._qt_log_handler.setFormatter(fmt)

        for h in root.handlers:
            if isinstance(h, QtLoggingHandler):
                return
        root.addHandler(self._qt_log_handler)
        logging.getLogger("asyncio").setLevel(logging.WARNING)

    def _append_log(self, text: str) -> None:
        if text == "\n":
            self.log_view.appendPlainText("")
            return
        self.log_view.appendPlainText(text.rstrip("\n"))

    def closeEvent(self, event) -> None:
        try:
            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr
        except Exception:
            pass
        try:
            logging.getLogger().removeHandler(self._qt_log_handler)
        except Exception:
            pass
        super().closeEvent(event)

    # -------- timeout/resume helpers --------
    def _looks_like_timeout(self, err: str) -> bool:
        s = (err or "").lower()
        needles = [
            "timeout",
            "timed out",
            "readtimeout",
            "connecttimeout",
            "deadline exceeded",
            "gateway timeout",
            "504",
        ]
        return any(n in s for n in needles)

    def _maybe_create_job(self) -> Optional[int]:
        fn = getattr(self.store, "create_job", None)
        if not callable(fn):
            return None
        try:
            jid = fn()
            return int(jid)  # type: ignore
        except Exception:
            return None

    def _start_resume(self, job_id: int) -> None:
        self.lbl_status.setText("resuming…")
        self.btn_refresh.setEnabled(False)

        # Create fresh store+llm for resume (avoid stale handles)
        cfg = self.cfg
        store = get_store(cfg)
        llm = create_llm_client(cfg)

        self.resume_worker = ResumeWorker(
            cfg=cfg, store=store, llm=llm, job_id=int(job_id)
        )
        self.resume_worker.llm_decision_requested.connect(
            self._on_llm_decision_requested
        )
        self.resume_worker.status.connect(self.lbl_status.setText)
        self.resume_worker.done.connect(self._on_resume_done)
        self.resume_worker.failed.connect(self._on_resume_failed)
        self.resume_worker.start()

    def _on_resume_done(self, summary_id: object) -> None:
        self.lbl_status.setText("resume done")
        self.btn_refresh.setEnabled(True)
        self._reload_all_config_dependent_ui()
        self._feeds_reload()
        QMessageBox.information(self, "Klart", f"Resume klar. Summary id: {summary_id}")

    def _on_resume_failed(self, err: str) -> None:
        self.lbl_status.setText("resume error")
        self.btn_refresh.setEnabled(True)
        QMessageBox.critical(self, "Resume misslyckades", err)

    # -------- printing --------
    def print_current_summary(self) -> None:
        current = self.summary_list.currentItem()
        if not current:
            QMessageBox.information(
                self, "Ingen sammanfattning", "Välj en sammanfattning först."
            )
            return

        sid = str(current.data(Qt.UserRole))  # type: ignore
        sdoc = self.store.get_summary_doc(sid)
        if not sdoc:
            QMessageBox.warning(self, "Saknas", "Kunde inte läsa sammanfattningen.")
            return

        md_text = sdoc.get("summary", "") or ""
        html = md.markdown(md_text, extensions=["extra"])
        title = f"Sammanfattning – {sid}"
        dlg = RichTextEditorDialog(
            self, title=title, initial_html=f"<html><body>{html}</body></html>"
        )
        dlg.exec()

    # ---- Summaries tab ----
    def _build_summaries_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        top = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh…")
        self.btn_jobs = QPushButton("Återuppta jobb…")  # NEW
        self.btn_print_summary = QPushButton("Skriv ut")
        self.lbl_status = QLabel("idle")

        top.addWidget(self.btn_refresh)
        top.addWidget(self.btn_jobs)  # NEW
        top.addWidget(self.btn_print_summary)
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
        self.btn_jobs.clicked.connect(self.open_jobs_dialog)  # NEW
        self.btn_print_summary.clicked.connect(self.print_current_summary)
        self.summary_list.currentItemChanged.connect(self.on_summary_selected)

        self.tabs.addTab(w, "Sammanfattningar")

    def open_jobs_dialog(self) -> None:
        from qt_ui.jobs import JobsDialog

        dlg = JobsDialog(self, cfg=self.cfg, store=self.store)
        if dlg.exec() != QDialog.Accepted:  # type: ignore
            return
        jid = dlg.property("selected_job_id")
        try:
            jid_i = int(jid)
        except Exception:
            return
        self._start_resume(jid_i)

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
            item.setData(Qt.UserRole, sid)  # type: ignore
            self.summary_list.addItem(item)
        self.summary_list.blockSignals(False)

        if self.summary_list.count():
            self.summary_list.setCurrentRow(0)
        else:
            self.summary_view.setHtml("<p>Inga sammanfattningar ännu.</p>")

    def on_summary_selected(self, current: QListWidgetItem, _prev: QListWidgetItem):
        if not current:
            return
        sid = current.data(Qt.UserRole)  # type: ignore
        doc = self.store.get_summary_doc(str(sid))
        if not doc:
            self.summary_view.setHtml("<p>Kunde inte läsa sammanfattning.</p>")
            return
        text = doc.get("summary", "") or ""
        html = md.markdown(text, extensions=["extra"])
        self.summary_view.setHtml(html)

    def open_refresh(self):
        dlg = RefreshDialog(self, self.ui_opts)
        if dlg.exec() != QDialog.Accepted:  # type: ignore
            return

        overrides = dlg.overrides()
        self.btn_refresh.setEnabled(False)
        self.lbl_status.setText("running…")
        logging.getLogger("feedsum.qt").info("Refresh overrides=%s", overrides)

        job_id = self._maybe_create_job()
        self._last_job_id = job_id

        self.worker = PipelineWorker(self.cfg, overrides, job_id)
        self.worker.llm_decision_requested.connect(self._on_llm_decision_requested)
        self.worker.status.connect(self.lbl_status.setText)
        self.worker.done.connect(self.on_pipeline_done)
        self.worker.failed.connect(self.on_pipeline_failed)
        self.worker.start()

    def on_pipeline_done(self, payload: object):
        summary_id = None
        job_id = None
        if isinstance(payload, tuple) and len(payload) == 2:
            summary_id, job_id = payload
        else:
            summary_id = payload

        self.lbl_status.setText("done")
        self.btn_refresh.setEnabled(True)
        self._reload_all_config_dependent_ui()
        self._feeds_reload()

        if isinstance(job_id, int) or (
            isinstance(job_id, str) and str(job_id).isdigit()
        ):
            try:
                self._last_job_id = int(job_id)
            except Exception:
                pass

        if summary_id is None:
            QMessageBox.information(
                self, "Klart", "Refresh klar: inga artiklar matchade urvalet."
            )

    def on_pipeline_failed(self, err: str):
        self.lbl_status.setText("error")
        self.btn_refresh.setEnabled(True)
        logging.getLogger("feedsum.qt").error("Pipeline error: %s", err)

        job_id = self._last_job_id
        if job_id is not None and self._looks_like_timeout(err):
            resp = QMessageBox.question(
                self,
                "Timeout",
                "Refresh verkar ha fått timeout.\n\nVill du återuppta från checkpoint?",
                QMessageBox.Yes | QMessageBox.No,  # type: ignore
            )
            if resp == QMessageBox.Yes:  # type: ignore
                self._start_resume(job_id)
                return

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

        self.filters_scroll = QScrollArea()
        self.filters_scroll.setWidgetResizable(True)
        self.filters_container = QWidget()
        self.filters_layout = QHBoxLayout(self.filters_container)
        self.filters_scroll.setWidget(self.filters_container)
        layout.addWidget(self.filters_scroll)

        self.topic_checks_articles: Dict[str, QCheckBox] = {}
        self.source_checks_articles: Dict[str, QCheckBox] = {}
        self.source_topics_articles: Dict[str, List[str]] = {}

        self.btn_apply.clicked.connect(self.reload_articles)
        self.btn_clear.clicked.connect(self.clear_article_filters)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Titel", "Källa", "Publicerad", "Preview"]
        )
        self.table.cellDoubleClicked.connect(self.open_article)
        layout.addWidget(self.table, 1)

        self.tabs.addTab(w, "Artiklar")
        self._rebuild_article_filters_from_ui_opts()

    def _clear_layout(self, layout: QHBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()  # type: ignore
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _rebuild_article_filters_from_ui_opts(self) -> None:
        self._clear_layout(self.filters_layout)
        self.topic_checks_articles = {}
        self.source_checks_articles = {}
        self.source_topics_articles = {}

        gb_topics = QGroupBox("Ämnen")
        vb_t = QVBoxLayout(gb_topics)
        for t in self.ui_opts.topic_options:
            cb = QCheckBox(t)
            self.topic_checks_articles[t] = cb
            vb_t.addWidget(cb)
            cb.stateChanged.connect(
                lambda _=None, topic=t: self._apply_topic_to_sources_articles(topic)
            )

        gb_sources = QGroupBox("Källor")
        vb_s = QVBoxLayout(gb_sources)
        for s in self.ui_opts.source_options:
            name = s["name"]
            self.source_topics_articles[name] = s.get("topics") or []
            cb = QCheckBox(name)
            self.source_checks_articles[name] = cb
            vb_s.addWidget(cb)

        self.filters_layout.addWidget(gb_topics)
        self.filters_layout.addWidget(gb_sources)
        self.filters_layout.addStretch(1)

    def _apply_topic_to_sources_articles(self, topic: str) -> None:
        checked = self.topic_checks_articles[topic].isChecked()
        for src, cb in self.source_checks_articles.items():
            if topic in (self.source_topics_articles.get(src) or []):
                cb.setChecked(checked)

    def clear_article_filters(self):
        for cb in self.topic_checks_articles.values():
            cb.setChecked(False)
        for cb in self.source_checks_articles.values():
            cb.setChecked(False)
        self.from_date.setDate(QDate.currentDate().addDays(-7))
        self.to_date.setDate(QDate.currentDate())
        self.reload_articles()

    def reload_articles(self):
        articles = self.store.list_articles() or []

        sources = [s for s, cb in self.source_checks_articles.items() if cb.isChecked()]
        topics = [t for t, cb in self.topic_checks_articles.items() if cb.isChecked()]

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
            src = a.get("source", "")
            pub = a.get("published", "") or format_ts(published_ts(a))
            prev = (a.get("text", "") or "")[:300].replace("\n", " ")

            it0 = QTableWidgetItem(title)
            it0.setToolTip(str(a.get("url") or ""))
            it0.setData(Qt.UserRole, a)  # type: ignore

            self.table.setItem(r, 0, it0)
            self.table.setItem(r, 1, QTableWidgetItem(src))
            self.table.setItem(r, 2, QTableWidgetItem(pub))
            self.table.setItem(r, 3, QTableWidgetItem(prev))

        self.table.resizeColumnsToContents()

    def open_article(self, row: int, col: int):
        it = self.table.item(row, 0)
        if not it:
            return
        a = it.data(Qt.UserRole)  # type: ignore
        if not isinstance(a, dict):
            return
        dlg = ArticleReaderDialog(self, a)
        dlg.exec()

    # ---- Promptlab tab ----
    def _build_promptlab_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        top = QHBoxLayout()
        top.addWidget(QLabel("Promptlab"))

        self.pl_loaded_summary = QLabel("Laddad summary: (ingen)")
        self.pl_loaded_summary.setTextFormat(Qt.PlainText)  # type: ignore
        top.addSpacing(12)
        top.addWidget(self.pl_loaded_summary, 1)

        self.pl_status = QLabel("idle")
        top.addWidget(self.pl_status)
        layout.addLayout(top)

        main_split = QSplitter(Qt.Horizontal)  # type: ignore

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.addWidget(QLabel("Välj summary"))
        self.pl_summary_list = QListWidget()
        left_l.addWidget(self.pl_summary_list, 1)
        self.btn_pl_load_from_summary = QPushButton("Ladda prompts från summary")
        left_l.addWidget(self.btn_pl_load_from_summary)
        main_split.addWidget(left)

        # --- replace the "right = QWidget(); right_l = QVBoxLayout(right)" block with this ---

        # Scrollable right side (fix: editors can grow beyond screen)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(6, 6, 6, 6)
        right_l.setSpacing(8)

        # IMPORTANT: put the right widget inside the scroll area
        right_scroll.setWidget(right)

        pkg_row = QHBoxLayout()
        self.pl_pkg_combo = QComboBox()
        self.btn_pl_pkg_reload = QPushButton("Ladda lista")
        self.btn_pl_pkg_load = QPushButton("Ladda paket")
        self.pl_pkg_name = QLineEdit()
        self.pl_pkg_name.setPlaceholderText("Namn för att spara paket…")
        self.btn_pl_pkg_save = QPushButton("Spara paket")
        pkg_row.addWidget(QLabel("Promptpaket:"))
        pkg_row.addWidget(self.pl_pkg_combo, 2)
        pkg_row.addWidget(self.btn_pl_pkg_reload)
        pkg_row.addWidget(self.btn_pl_pkg_load)
        pkg_row.addSpacing(10)
        pkg_row.addWidget(self.pl_pkg_name, 2)
        pkg_row.addWidget(self.btn_pl_pkg_save)
        right_l.addLayout(pkg_row)

        self.pl_batch_system = QPlainTextEdit()
        self.pl_batch_user = QPlainTextEdit()
        self.pl_meta_system = QPlainTextEdit()
        self.pl_meta_user = QPlainTextEdit()
        self.pl_super_meta_system = QPlainTextEdit()
        self.pl_super_meta_user = QPlainTextEdit()

        self.pl_batch_system.setPlaceholderText("batch_system…")
        self.pl_batch_user.setPlaceholderText("batch_user_template…")
        self.pl_meta_system.setPlaceholderText("meta_system…")
        self.pl_meta_user.setPlaceholderText("meta_user_template…")
        self.pl_super_meta_system.setPlaceholderText("super_meta_system…")
        self.pl_super_meta_user.setPlaceholderText("super_meta_user_template…")

        # Optional: make them a bit easier to use in a scroll area
        for ed in (
            self.pl_batch_system,
            self.pl_batch_user,
            self.pl_meta_system,
            self.pl_meta_user,
            self.pl_super_meta_system,
            self.pl_super_meta_user,
        ):
            ed.setMinimumHeight(120)

        ed_split = QSplitter(Qt.Vertical)  # type: ignore

        def _wrap_editor(title: str, editor: QPlainTextEdit) -> QWidget:
            ww = QWidget()
            ll = QVBoxLayout(ww)
            ll.setContentsMargins(0, 0, 0, 0)
            ll.setSpacing(4)
            ll.addWidget(QLabel(title))
            ll.addWidget(editor, 1)
            return ww

        ed1 = _wrap_editor("batch_system", self.pl_batch_system)
        ed2 = _wrap_editor("batch_user_template", self.pl_batch_user)
        ed3 = _wrap_editor("meta_system", self.pl_meta_system)
        ed4 = _wrap_editor("meta_user_template", self.pl_meta_user)
        ed5 = _wrap_editor("super_meta_system", self.pl_super_meta_system)
        ed6 = _wrap_editor("super_meta_user_template", self.pl_super_meta_user)

        # FIX 1: actually add the super-meta widgets to the splitter
        ed_split.addWidget(ed1)
        ed_split.addWidget(ed2)
        ed_split.addWidget(ed3)
        ed_split.addWidget(ed4)
        ed_split.addWidget(ed5)
        ed_split.addWidget(ed6)

        # FIX 2: setSizes must match number of widgets (6)
        ed_split.setSizes([160, 220, 160, 220, 160, 220])

        right_l.addWidget(ed_split, 2)

        run_row = QHBoxLayout()
        self.btn_pl_run = QPushButton("Skapa ny summary (ej sparad)")
        run_row.addWidget(self.btn_pl_run)
        run_row.addStretch(1)
        right_l.addLayout(run_row)

        # IMPORTANT: add the scroll area (not the right widget) to the main split
        main_split.addWidget(right_scroll)
        main_split.setSizes([360, 980])
        layout.addWidget(main_split, 1)
        self.tabs.addTab(w, "Promptlab")

        self.pl_summary_list.currentItemChanged.connect(self._pl_on_summary_selected)
        self.btn_pl_load_from_summary.clicked.connect(
            self._pl_load_prompts_from_selected_summary
        )
        self.btn_pl_run.clicked.connect(self._pl_run_replay)
        self.btn_pl_pkg_reload.clicked.connect(self._pl_reload_prompt_packages)
        self.btn_pl_pkg_load.clicked.connect(self._pl_load_selected_package)
        self.btn_pl_pkg_save.clicked.connect(self._pl_save_package)

    def _refresh_promptlab_lists(self) -> None:
        docs = self.store.list_summary_docs() or []
        docs.sort(key=lambda d: int(d.get("created") or 0), reverse=True)

        self.pl_summary_list.blockSignals(True)
        self.pl_summary_list.clear()
        for d in docs:
            sid = str(d.get("id") or "")
            created = int(d.get("created") or 0)
            n = len(d.get("sources") or [])
            it = QListWidgetItem(f"{format_ts(created)} · Artiklar: {n}")
            it.setData(Qt.UserRole, sid)  # type: ignore
            self.pl_summary_list.addItem(it)
        self.pl_summary_list.blockSignals(False)

        if self.pl_summary_list.count():
            self.pl_summary_list.setCurrentRow(0)

        self._pl_reload_prompt_packages()

    def _pl_reload_prompt_packages(self) -> None:
        pkgs = list_prompt_packages(self.cfg, config_path=CONFIG_PATH)
        self.pl_pkg_combo.clear()
        self.pl_pkg_combo.addItems(pkgs)

    def _pl_current_summary_id(self) -> Optional[str]:
        return self.pl_selected_summary_id or (
            str(self.pl_summary_list.currentItem().data(Qt.UserRole))  # type: ignore
            if self.pl_summary_list.currentItem()
            else None
        )

    def _pl_on_summary_selected(self, current: QListWidgetItem, _prev: QListWidgetItem):
        if not current:
            self.pl_selected_summary_id = None
            self.pl_loaded_summary.setText("Laddad summary: (ingen)")
            self.pl_status.setText("idle")
            return

        sid = str(current.data(Qt.UserRole))  # type: ignore
        self.pl_selected_summary_id = sid

        sdoc = self.store.get_summary_doc(sid) or {}
        created = int(sdoc.get("created") or 0)
        created_s = format_ts(created) if created else "(okänd tid)"
        n = len(sdoc.get("sources") or [])
        self.pl_loaded_summary.setText(
            f"Laddad summary: {created_s} · Artiklar: {n} · ID: {sid}"
        )
        self.pl_status.setText("idle")

    def _pl_load_prompts_from_selected_summary(self) -> None:
        sid = self._pl_current_summary_id()
        if not sid:
            QMessageBox.information(self, "Ingen summary", "Välj en summary först.")
            return
        try:
            ps = get_promptset_for_summary(self.store, sid)
        except Exception as e:
            QMessageBox.critical(self, "Fel", str(e))
            return
        self.pl_batch_system.setPlainText(ps.batch_system)
        self.pl_batch_user.setPlainText(ps.batch_user_template)
        self.pl_meta_system.setPlainText(ps.meta_system)
        self.pl_meta_user.setPlainText(ps.meta_user_template)
        self.pl_super_meta_system.setPlainText(ps.super_meta_system)
        self.pl_super_meta_user.setPlainText(ps.super_meta_user_template)
        self.pl_status.setText(f"Prompts laddade från: {sid}")

    def _pl_promptset_from_ui(self) -> PromptSet:
        return PromptSet(
            batch_system=self.pl_batch_system.toPlainText(),
            batch_user_template=self.pl_batch_user.toPlainText(),
            meta_system=self.pl_meta_system.toPlainText(),
            meta_user_template=self.pl_meta_user.toPlainText(),
            super_meta_system=self.pl_super_meta_system.toPlainText(),
            super_meta_user_template=self.pl_super_meta_user.toPlainText(),
        )

    def _pl_load_selected_package(self) -> None:
        name = self.pl_pkg_combo.currentText().strip()
        if not name:
            return
        ps = load_prompt_package(self.cfg, config_path=CONFIG_PATH, package_name=name)
        if not ps:
            QMessageBox.warning(self, "Saknas", f"Kunde inte ladda paket: {name}")
            return
        self.pl_batch_system.setPlainText(ps.batch_system)
        self.pl_batch_user.setPlainText(ps.batch_user_template)
        self.pl_meta_system.setPlainText(ps.meta_system)
        self.pl_meta_user.setPlainText(ps.meta_user_template)
        self.pl_super_meta_system.setPlainText(ps.super_meta_system)
        self.pl_super_meta_user.setPlainText(ps.super_meta_user_template)
        self.pl_status.setText(f"paket laddat: {name}")

    def _pl_save_package(self) -> None:
        name = self.pl_pkg_name.text().strip()
        if not name:
            QMessageBox.information(self, "Namn saknas", "Ange ett namn för paketet.")
            return
        ps = self._pl_promptset_from_ui()
        try:
            path = save_prompt_package(
                self.cfg, config_path=CONFIG_PATH, package_name=name, promptset=ps
            )
        except Exception as e:
            QMessageBox.critical(self, "Fel", str(e))
            return
        self._pl_reload_prompt_packages()
        self.pl_pkg_combo.setCurrentText(name)
        self.pl_status.setText(f"sparat: {name} ({path})")

    def _pl_run_replay(self) -> None:
        sid = self._pl_current_summary_id()
        if not sid:
            QMessageBox.information(self, "Ingen summary", "Välj en summary först.")
            return

        prompts = self._pl_promptset_from_ui()

        self.pl_status.setText("running…")
        self.btn_pl_run.setEnabled(False)

        self.replay_worker = PromptReplayWorker(
            cfg=self.cfg, store=self.store, summary_id=sid, prompts=prompts
        )
        self.replay_worker.status.connect(self.pl_status.setText)
        self.replay_worker.done.connect(self._pl_replay_done)
        self.replay_worker.failed.connect(self._pl_replay_failed)
        self.replay_worker.start()

    def _pl_replay_done(self, result: dict) -> None:
        self.btn_pl_run.setEnabled(True)
        self.pl_status.setText("klart (ej sparad)")

        sid = self._pl_current_summary_id()
        orig_md = ""
        orig_created_s = "(okänd tid)"
        if sid:
            orig_doc = self.store.get_summary_doc(sid)
            if orig_doc:
                orig_md = orig_doc.get("summary", "") or ""
                oc = int(orig_doc.get("created") or 0)
                orig_created_s = format_ts(oc) if oc else orig_created_s

        new_md = (result.get("summary_markdown") or "").strip()
        title = f"Prompt Replay (ej sparad) – från summary {orig_created_s} · ID: {sid}"

        win = ReplayResultWindow(self, original_md=orig_md, new_md=new_md, title=title)
        self._last_replay_window = win
        win.show()

    def _pl_replay_failed(self, err: str) -> None:
        self.btn_pl_run.setEnabled(True)
        self.pl_status.setText("error")
        QMessageBox.critical(self, "Replay misslyckades", err)

    # ---- Feeds tab ----
    def _build_feeds_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        top = QHBoxLayout()
        self.feeds_path_label = QLabel("feeds.yaml: (okänt)")
        self.feeds_status = QLabel("idle")
        top.addWidget(self.feeds_path_label, 1)
        top.addWidget(self.feeds_status)
        layout.addLayout(top)

        btns = QHBoxLayout()
        self.btn_feeds_reload = QPushButton("Ladda om")
        self.btn_feeds_add = QPushButton("Lägg till")
        self.btn_feeds_edit = QPushButton("Redigera")
        self.btn_feeds_delete = QPushButton("Ta bort")
        self.btn_feeds_save = QPushButton("Spara till feeds.yaml")

        btns.addWidget(self.btn_feeds_reload)
        btns.addWidget(self.btn_feeds_add)
        btns.addWidget(self.btn_feeds_edit)
        btns.addWidget(self.btn_feeds_delete)
        btns.addStretch(1)
        btns.addWidget(self.btn_feeds_save)
        layout.addLayout(btns)

        self.feeds_table = QTableWidget(0, 4)
        self.feeds_table.setHorizontalHeaderLabels(
            ["Name", "URL", "Topics", "Category include"]
        )
        self.feeds_table.cellDoubleClicked.connect(
            lambda r, c: self._feeds_edit_selected()
        )
        layout.addWidget(self.feeds_table, 1)

        self.btn_feeds_reload.clicked.connect(self._feeds_reload)
        self.btn_feeds_add.clicked.connect(self._feeds_add)
        self.btn_feeds_edit.clicked.connect(self._feeds_edit_selected)
        self.btn_feeds_delete.clicked.connect(self._feeds_delete_selected)
        self.btn_feeds_save.clicked.connect(self._feeds_save)

        self.tabs.addTab(w, "Feeds")

        self._feeds_items: List[Dict[str, Any]] = []

    def _feeds_reload(self) -> None:
        try:
            self.cfg = load_config(CONFIG_PATH)
            feeds_path = resolve_feeds_path(self.cfg, config_path=CONFIG_PATH)
            self.feeds_path_label.setText(f"feeds.yaml: {feeds_path}")
            self._feeds_items = self._read_feeds_yaml(feeds_path)
            self._feeds_render_table()
            self.feeds_status.setText(f"laddade: {len(self._feeds_items)}")
        except Exception as e:
            self.feeds_status.setText("error")
            QMessageBox.critical(self, "Kunde inte ladda feeds.yaml", str(e))

    def _read_feeds_yaml(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or []
        if not isinstance(data, list):
            raise RuntimeError("feeds.yaml måste vara en YAML-lista.")
        out: List[Dict[str, Any]] = []
        for x in data:
            if isinstance(x, dict):
                out.append(dict(x))
        return out

    def _feeds_render_table(self) -> None:
        items = self._feeds_items
        self.feeds_table.setRowCount(len(items))
        for r, f in enumerate(items):
            name = str(f.get("name") or "")
            url = str(f.get("url") or "")
            topics = ", ".join(_safe_list_str(f.get("topics")))
            cat_inc = ", ".join(_safe_list_str(f.get("category_include")))

            it0 = QTableWidgetItem(name)
            it0.setData(Qt.UserRole, f)  # type: ignore
            self.feeds_table.setItem(r, 0, it0)
            self.feeds_table.setItem(r, 1, QTableWidgetItem(url))
            self.feeds_table.setItem(r, 2, QTableWidgetItem(topics))
            self.feeds_table.setItem(r, 3, QTableWidgetItem(cat_inc))
        self.feeds_table.resizeColumnsToContents()

    def _feeds_selected_row(self) -> Optional[int]:
        row = self.feeds_table.currentRow()
        if row is None or row < 0:
            return None
        if row >= len(self._feeds_items):
            return None
        return row

    def _feeds_add(self) -> None:
        dlg = FeedEditDialog(self, None)
        if dlg.exec() != QDialog.Accepted:  # type: ignore
            return
        f = dlg.value()
        if not f.get("name") or not f.get("url"):
            QMessageBox.information(self, "Saknas", "Name och URL måste anges.")
            return

        if any(
            str(x.get("name") or "").strip().lower() == str(f["name"]).strip().lower()
            for x in self._feeds_items
        ):
            QMessageBox.warning(
                self, "Dublett", "Det finns redan en feed med samma name."
            )
            return

        self._feeds_items.append(f)
        self._feeds_items.sort(key=lambda x: str(x.get("name") or "").lower())
        self._feeds_render_table()
        self.feeds_status.setText("ändringar ej sparade")

    def _feeds_edit_selected(self) -> None:
        row = self._feeds_selected_row()
        if row is None:
            QMessageBox.information(self, "Ingen rad", "Välj en feed att redigera.")
            return
        current = self._feeds_items[row]
        dlg = FeedEditDialog(self, current)
        if dlg.exec() != QDialog.Accepted:  # type: ignore
            return
        updated = dlg.value()
        if not updated.get("name") or not updated.get("url"):
            QMessageBox.information(self, "Saknas", "Name och URL måste anges.")
            return

        new_name = str(updated["name"]).strip().lower()
        for i, x in enumerate(self._feeds_items):
            if i == row:
                continue
            if str(x.get("name") or "").strip().lower() == new_name:
                QMessageBox.warning(
                    self, "Dublett", "Det finns redan en feed med samma name."
                )
                return

        self._feeds_items[row] = updated
        self._feeds_items.sort(key=lambda x: str(x.get("name") or "").lower())
        self._feeds_render_table()
        self.feeds_status.setText("ändringar ej sparade")

    def _feeds_delete_selected(self) -> None:
        row = self._feeds_selected_row()
        if row is None:
            QMessageBox.information(self, "Ingen rad", "Välj en feed att ta bort.")
            return

        name = str(self._feeds_items[row].get("name") or "")
        resp = QMessageBox.question(
            self,
            "Ta bort",
            f"Ta bort '{name}'?",
            QMessageBox.Yes | QMessageBox.No,  # type: ignore
        )
        if resp != QMessageBox.Yes:  # type: ignore
            return

        del self._feeds_items[row]
        self._feeds_render_table()
        self.feeds_status.setText("ändringar ej sparade")

    def _feeds_save(self) -> None:
        try:
            feeds_path = resolve_feeds_path(self.cfg, config_path=CONFIG_PATH)
            feeds_path.parent.mkdir(parents=True, exist_ok=True)
            with open(feeds_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    self._feeds_items, f, sort_keys=False, allow_unicode=True
                )
            self.feeds_status.setText("sparat")

            self._reload_all_config_dependent_ui()
            self._feeds_reload()
        except Exception as e:
            self.feeds_status.setText("error")
            QMessageBox.critical(self, "Kunde inte spara feeds.yaml", str(e))


def main():
    level = _env_log_level(logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
