# qt_gui.py
from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

import markdown as md
from PySide6.QtCore import Qt, QThread, Signal, QDate, QObject
from PySide6.QtGui import QDesktopServices, QTextDocument
from PySide6.QtPrintSupport import QPrinter, QPrintDialog
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import QUrl

from summarizer.main import run_pipeline

from uicommon import (
    load_config,
    get_store,
    format_ts,
    published_ts,
    filter_articles,
    ArticleFilters,
    get_ui_options,
    build_refresh_overrides,
)
from uicommon.bootstrap_ui import resolve_config_path

from summarizer.prompt_replay import (
    PromptSet,
    get_promptset_for_summary,
    rerun_summary_from_existing,  # returns ephemeral dict (NOT persisted)
    list_prompt_packages,
    load_prompt_package,
    save_prompt_package,
)

RUNTIME = resolve_config_path()
CONFIG_PATH = str(RUNTIME.config_path)


# ----------------------------
# Helpers
# ----------------------------
def _env_log_level(default: int = logging.INFO) -> int:
    lvl = (os.environ.get("FEEDSUMMARY_LOG_LEVEL") or "").strip().upper()
    if not lvl:
        return default
    return getattr(logging, lvl, default)


def _fmt_dt_hm(ts: int) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")


# ----------------------------
# Log capture: stdout/stderr + logging -> Qt panel
# ----------------------------
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


# ----------------------------
# Workers
# ----------------------------
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


# ----------------------------
# Refresh dialog (scroll + grouped sources + select all/none)
# ----------------------------
class RefreshDialog(QDialog):
    def __init__(self, parent: QWidget, ui_opts):
        super().__init__(parent)
        self.setWindowTitle("Refresh – inställningar")
        self.ui_opts = ui_opts

        self.topic_checks: Dict[str, QCheckBox] = {}
        self.source_checks: Dict[str, QCheckBox] = {}
        self.source_topics: Dict[str, List[str]] = {}
        self.source_by_primary: Dict[str, List[str]] = {}

        outer = QVBoxLayout(self)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        outer.addWidget(self.scroll, 1)

        content = QWidget()
        self.scroll.setWidget(content)
        root = QVBoxLayout(content)

        form_box = QGroupBox("Grundinställningar")
        form_layout = QFormLayout(form_box)

        self.lb_value = QSpinBox()
        self.lb_value.setMinimum(1)
        self.lb_value.setMaximum(9999)
        self.lb_value.setValue(int(self.ui_opts.default_lookback_value))

        self.lb_unit = QComboBox()
        self.lb_unit.addItems(["h", "d", "w", "m", "y"])
        self.lb_unit.setCurrentText(str(self.ui_opts.default_lookback_unit))

        lb_row = QHBoxLayout()
        lb_row.addWidget(self.lb_value)
        lb_row.addWidget(self.lb_unit)
        lb_wrap = QWidget()
        lb_wrap.setLayout(lb_row)
        form_layout.addRow("Lookback", lb_wrap)

        self.prompt_pkg = QComboBox()
        self.prompt_pkg.addItems(self.ui_opts.prompt_packages)
        if self.ui_opts.default_prompt_package:
            self.prompt_pkg.setCurrentText(self.ui_opts.default_prompt_package)
        form_layout.addRow("Prompt-paket", self.prompt_pkg)

        root.addWidget(form_box)

        if self.ui_opts.topic_options:
            gb_topics = QGroupBox("Ämnen")
            v_topics = QVBoxLayout(gb_topics)

            btns = QHBoxLayout()
            self.btn_topics_all = QPushButton("Markera alla ämnen")
            self.btn_topics_none = QPushButton("Avmarkera alla ämnen")
            btns.addWidget(self.btn_topics_all)
            btns.addWidget(self.btn_topics_none)
            btns.addStretch(1)
            v_topics.addLayout(btns)

            grid = QGridLayout()
            cols = self._choose_columns(
                len(self.ui_opts.topic_options), preferred=3, max_cols=5
            )
            for i, t in enumerate(self.ui_opts.topic_options):
                cb = QCheckBox(t)
                self.topic_checks[t] = cb
                grid.addWidget(cb, i // cols, i % cols)
                cb.stateChanged.connect(
                    lambda _=None, topic=t: self._on_topic_changed(topic)
                )
            v_topics.addLayout(grid)

            self.btn_topics_all.clicked.connect(lambda: self._set_all_topics(True))
            self.btn_topics_none.clicked.connect(lambda: self._set_all_topics(False))

            root.addWidget(gb_topics)

        self._init_sources()

        gb_sources = QGroupBox("Källor (grupperade per ämne)")
        v_sources = QVBoxLayout(gb_sources)

        sbtns = QHBoxLayout()
        self.btn_sources_all = QPushButton("Markera alla källor")
        self.btn_sources_none = QPushButton("Avmarkera alla källor")
        sbtns.addWidget(self.btn_sources_all)
        sbtns.addWidget(self.btn_sources_none)
        sbtns.addStretch(1)
        v_sources.addLayout(sbtns)

        self.btn_sources_all.clicked.connect(lambda: self._set_all_sources(True))
        self.btn_sources_none.clicked.connect(lambda: self._set_all_sources(False))

        group_names = list(self.source_by_primary.keys())
        group_names.sort(key=lambda g: (g == "Okategoriserat", g.lower()))

        for g in group_names:
            names = self.source_by_primary.get(g) or []
            if not names:
                continue
            gb = QGroupBox(f"{g} ({len(names)})")
            grid = QGridLayout(gb)
            cols_s = self._choose_columns(len(names), preferred=2, max_cols=4)
            for i, name in enumerate(sorted(names, key=lambda x: x.lower())):
                grid.addWidget(self.source_checks[name], i // cols_s, i % cols_s)
            v_sources.addWidget(gb)

        root.addWidget(gb_sources)
        root.addStretch(1)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.btn_cancel = QPushButton("Avbryt")
        self.btn_ok = QPushButton("Kör refresh")
        actions.addWidget(self.btn_cancel)
        actions.addWidget(self.btn_ok)
        outer.addLayout(actions)

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self.accept)

        self.resize(900, 720)
        self._cap_to_screen(max_w=1100, max_h=840)

    def _init_sources(self) -> None:
        self.source_checks = {}
        self.source_topics = {}
        self.source_by_primary = {}
        for s in self.ui_opts.source_options:
            name = s["name"]
            topics = list(s.get("topics") or [])
            self.source_topics[name] = topics
            primary = topics[0] if topics else "Okategoriserat"
            self.source_by_primary.setdefault(primary, []).append(name)

            cb = QCheckBox(name)
            cb.setChecked(bool(s.get("default_checked", True)))
            if topics:
                cb.setToolTip(", ".join(topics))
            self.source_checks[name] = cb

    @staticmethod
    def _choose_columns(n: int, *, preferred: int, max_cols: int) -> int:
        if n <= 0:
            return preferred
        if n <= 8:
            return min(2, max_cols)
        if n <= 18:
            return min(preferred, max_cols)
        if n <= 40:
            return min(preferred + 1, max_cols)
        return max_cols

    def _cap_to_screen(self, *, max_w: int, max_h: int) -> None:
        screen = self.screen() or (self.parent().screen() if self.parent() else None)
        if not screen:
            self.setMaximumSize(max_w, max_h)
            return
        geo = screen.availableGeometry()
        w = min(max_w, int(geo.width() * 0.92))
        h = min(max_h, int(geo.height() * 0.92))
        self.setMaximumSize(w, h)

    def _set_all_topics(self, checked: bool) -> None:
        for t, cb in self.topic_checks.items():
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
            self._apply_topic_to_sources(t, checked)

    def _set_all_sources(self, checked: bool) -> None:
        for cb in self.source_checks.values():
            cb.setChecked(checked)

    def _on_topic_changed(self, topic: str) -> None:
        checked = self.topic_checks[topic].isChecked()
        self._apply_topic_to_sources(topic, checked)

    def _apply_topic_to_sources(self, topic: str, checked: bool) -> None:
        for src, cb in self.source_checks.items():
            if topic in (self.source_topics.get(src) or []):
                cb.setChecked(checked)

    def overrides(self) -> Dict[str, Any]:
        selected_sources = [s for s, cb in self.source_checks.items() if cb.isChecked()]
        selected_topics = [t for t, cb in self.topic_checks.items() if cb.isChecked()]
        return build_refresh_overrides(
            lookback_value=self.lb_value.value(),
            lookback_unit=self.lb_unit.currentText(),
            prompt_package=self.prompt_pkg.currentText(),
            selected_sources=selected_sources,
            selected_topics=selected_topics,
        )


# ----------------------------
# Article reader dialog (with Print)
# ----------------------------
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
        printer = QPrinter(QPrinter.HighResolution)
        printer.setDocName("Artikel")

        dlg = QPrintDialog(printer, self)
        dlg.setWindowTitle("Skriv ut artikel")
        if dlg.exec() != QDialog.Accepted:
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


# ----------------------------
# Main window
# ----------------------------
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

        # Splitter: main UI + log panel
        splitter = QSplitter(Qt.Vertical)

        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)

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

        # Connect log streams
        self._log_emitter = QtLogEmitter()
        self._log_emitter.text.connect(self._append_log)

        sys.stdout = QtStream(self._log_emitter)
        sys.stderr = QtStream(self._log_emitter)
        self._install_logging_bridge()

        # Tabs
        self._build_summaries_tab()
        self._build_articles_tab()
        self._build_promptlab_tab()

        # Initial load
        self.reload_summaries()
        self.reload_articles()
        self._refresh_promptlab_lists()

        logging.getLogger("feedsum.qt").info("Bootstrap config_path=%s", CONFIG_PATH)

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

    # -------- printing --------
    def print_current_summary(self) -> None:
        current = self.summary_list.currentItem()
        if not current:
            QMessageBox.information(
                self, "Ingen sammanfattning", "Välj en sammanfattning först."
            )
            return
        sid = current.data(Qt.UserRole)
        sdoc = self.store.get_summary_doc(str(sid))
        if not sdoc:
            QMessageBox.warning(self, "Saknas", "Kunde inte läsa sammanfattningen.")
            return

        md_text = sdoc.get("summary", "") or ""
        html = md.markdown(md_text, extensions=["extra"])

        printer = QPrinter(QPrinter.HighResolution)
        printer.setDocName("Sammanfattning")
        dlg = QPrintDialog(printer, self)
        dlg.setWindowTitle("Skriv ut sammanfattning")
        if dlg.exec() != QDialog.Accepted:
            return

        doc = QTextDocument()
        doc.setHtml(f"<html><body>{html}</body></html>")
        doc.print_(printer)

    # ---- Summaries tab ----
    def _build_summaries_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        top = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh…")
        self.btn_print_summary = QPushButton("Skriv ut")
        self.lbl_status = QLabel("idle")

        top.addWidget(self.btn_refresh)
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
        self.btn_print_summary.clicked.connect(self.print_current_summary)
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
        dlg = RefreshDialog(self, self.ui_opts)
        if dlg.exec() != QDialog.Accepted:
            return

        overrides = dlg.overrides()
        self.btn_refresh.setEnabled(False)
        self.lbl_status.setText("running…")
        logging.getLogger("feedsum.qt").info("Refresh overrides=%s", overrides)

        self.worker = PipelineWorker(self.cfg, overrides)
        self.worker.status.connect(self.lbl_status.setText)
        self.worker.done.connect(self.on_pipeline_done)
        self.worker.failed.connect(self.on_pipeline_failed)
        self.worker.start()

    def on_pipeline_done(self, _sid: object):
        self.lbl_status.setText("done")
        self.btn_refresh.setEnabled(True)

        self.cfg = load_config(CONFIG_PATH)
        self.store = get_store(self.cfg)
        self.ui_opts = get_ui_options(self.cfg, config_path=CONFIG_PATH)

        self.reload_summaries()
        self._rebuild_article_filters_from_ui_opts()
        self.reload_articles()
        self._refresh_promptlab_lists()

    def on_pipeline_failed(self, err: str):
        self.lbl_status.setText("error")
        self.btn_refresh.setEnabled(True)
        logging.getLogger("feedsum.qt").error("Pipeline error: %s", err)
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
            w = item.widget()
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
            it0.setData(Qt.UserRole, a)

            self.table.setItem(r, 0, it0)
            self.table.setItem(r, 1, QTableWidgetItem(src))
            self.table.setItem(r, 2, QTableWidgetItem(pub))
            self.table.setItem(r, 3, QTableWidgetItem(prev))

        self.table.resizeColumnsToContents()

    def open_article(self, row: int, col: int):
        it = self.table.item(row, 0)
        if not it:
            return
        a = it.data(Qt.UserRole)
        if not isinstance(a, dict):
            return
        dlg = ArticleReaderDialog(self, a)
        dlg.exec()

    # ---- Promptlab tab (ephemeral replay, no DB write) ----
    def _build_promptlab_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        top = QHBoxLayout()
        self.pl_status = QLabel("idle")
        top.addWidget(QLabel("Promptlab"))
        top.addStretch(1)
        top.addWidget(self.pl_status)
        layout.addLayout(top)

        main_split = QSplitter(Qt.Horizontal)

        # Left: summary selector
        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.addWidget(QLabel("Välj summary"))
        self.pl_summary_list = QListWidget()
        left_l.addWidget(self.pl_summary_list, 1)

        self.btn_pl_load_from_summary = QPushButton("Ladda prompts från summary")
        left_l.addWidget(self.btn_pl_load_from_summary)

        main_split.addWidget(left)

        # Right: editors + result
        right = QWidget()
        right_l = QVBoxLayout(right)

        # Prompt package controls
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

        # Editors
        self.pl_batch_system = QPlainTextEdit()
        self.pl_batch_user = QPlainTextEdit()
        self.pl_meta_system = QPlainTextEdit()
        self.pl_meta_user = QPlainTextEdit()

        self.pl_batch_system.setPlaceholderText("batch_system…")
        self.pl_batch_user.setPlaceholderText("batch_user_template…")
        self.pl_meta_system.setPlaceholderText("meta_system…")
        self.pl_meta_user.setPlaceholderText("meta_user_template…")

        ed_split = QSplitter(Qt.Vertical)
        ed1 = QWidget()
        ed1_l = QVBoxLayout(ed1)
        ed1_l.addWidget(QLabel("batch_system"))
        ed1_l.addWidget(self.pl_batch_system)
        ed2 = QWidget()
        ed2_l = QVBoxLayout(ed2)
        ed2_l.addWidget(QLabel("batch_user_template"))
        ed2_l.addWidget(self.pl_batch_user)
        ed3 = QWidget()
        ed3_l = QVBoxLayout(ed3)
        ed3_l.addWidget(QLabel("meta_system"))
        ed3_l.addWidget(self.pl_meta_system)
        ed4 = QWidget()
        ed4_l = QVBoxLayout(ed4)
        ed4_l.addWidget(QLabel("meta_user_template"))
        ed4_l.addWidget(self.pl_meta_user)

        ed_split.addWidget(ed1)
        ed_split.addWidget(ed2)
        ed_split.addWidget(ed3)
        ed_split.addWidget(ed4)
        ed_split.setSizes([160, 220, 160, 220])

        right_l.addWidget(ed_split, 2)

        # Run + compare
        run_row = QHBoxLayout()
        self.btn_pl_run = QPushButton("Skapa ny summary (ej sparad)")
        self.btn_pl_compare = QPushButton("Visa jämförelse")
        self.btn_pl_compare.setCheckable(True)
        run_row.addWidget(self.btn_pl_run)
        run_row.addWidget(self.btn_pl_compare)
        run_row.addStretch(1)
        right_l.addLayout(run_row)

        # Result view + optional compare
        self.pl_result_split = QSplitter(Qt.Horizontal)
        self.pl_new_view = QTextBrowser()
        self.pl_original_view = QTextBrowser()
        self.pl_new_view.setHtml("<p>Nytt resultat…</p>")
        self.pl_original_view.setHtml("<p>Original…</p>")
        self.pl_result_split.addWidget(self.pl_new_view)
        self.pl_result_split.addWidget(self.pl_original_view)
        self.pl_result_split.setSizes([700, 500])
        self.pl_original_view.hide()  # start without compare

        right_l.addWidget(self.pl_result_split, 2)

        main_split.addWidget(right)
        main_split.setSizes([320, 980])

        layout.addWidget(main_split, 1)
        self.tabs.addTab(w, "Promptlab")

        # Wire events
        self.pl_summary_list.currentItemChanged.connect(self._pl_on_summary_selected)
        self.btn_pl_load_from_summary.clicked.connect(
            self._pl_load_prompts_from_selected_summary
        )
        self.btn_pl_run.clicked.connect(self._pl_run_replay)
        self.btn_pl_compare.toggled.connect(self._pl_toggle_compare)

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
            it.setData(Qt.UserRole, sid)
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
        it = self.pl_summary_list.currentItem()
        if not it:
            return None
        return str(it.data(Qt.UserRole))

    def _pl_on_summary_selected(self, current: QListWidgetItem, _prev: QListWidgetItem):
        if not current:
            return
        sid = str(current.data(Qt.UserRole))
        sdoc = self.store.get_summary_doc(sid)
        if not sdoc:
            return
        orig_md = sdoc.get("summary", "") or ""
        self.pl_new_view.setHtml(md.markdown(orig_md, extensions=["extra"]))
        self.pl_original_view.setHtml(md.markdown(orig_md, extensions=["extra"]))

    def _pl_load_prompts_from_selected_summary(self) -> None:
        sid = self._pl_current_summary_id()
        if not sid:
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
        self.pl_status.setText("prompts laddade")

    def _pl_promptset_from_ui(self) -> PromptSet:
        return PromptSet(
            batch_system=self.pl_batch_system.toPlainText(),
            batch_user_template=self.pl_batch_user.toPlainText(),
            meta_system=self.pl_meta_system.toPlainText(),
            meta_user_template=self.pl_meta_user.toPlainText(),
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

    def _pl_toggle_compare(self, enabled: bool) -> None:
        if enabled:
            self.pl_original_view.show()
        else:
            self.pl_original_view.hide()

    def _pl_run_replay(self) -> None:
        sid = self._pl_current_summary_id()
        if not sid:
            QMessageBox.information(self, "Ingen summary", "Välj en summary först.")
            return

        # ensure original view is up-to-date for compare
        orig_doc = self.store.get_summary_doc(sid)
        if orig_doc:
            self.pl_original_view.setHtml(
                md.markdown(orig_doc.get("summary", "") or "", extensions=["extra"])
            )

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

        md_text = (result.get("summary_markdown") or "").strip()
        html = md.markdown(md_text, extensions=["extra"])
        self.pl_new_view.setHtml(html)

        # do NOT reload summaries / store (no persistence)

    def _pl_replay_failed(self, err: str) -> None:
        self.btn_pl_run.setEnabled(True)
        self.pl_status.setText("error")
        QMessageBox.critical(self, "Replay misslyckades", err)


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
