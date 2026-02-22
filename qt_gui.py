from __future__ import annotations

import asyncio
import logging
import sys
import threading
from typing import Any, Dict, List, Optional

import markdown as md
from PySide6.QtCore import Qt, QThread, Signal, QDate, QObject
from PySide6.QtGui import QDesktopServices
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
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
    QPlainTextEdit,
)
from PySide6.QtCore import QUrl

from summarizer.helpers import setup_logging
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

RUNTIME = resolve_config_path()
CONFIG_PATH = str(RUNTIME.config_path)
setup_logging()
logger = logging.getLogger(__name__)

# ----------------------------
# Log capture: stdout/stderr -> Qt panel
# ----------------------------
class QtLogEmitter(QObject):
    text = Signal(str)


class QtStream:
    """
    File-like stream that emits text to a Qt signal. Suitable for sys.stdout/sys.stderr redirection.
    """
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


# ----------------------------
# Worker thread: run pipeline
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


# ----------------------------
# Refresh dialog (scroll + grouped sources + select all/none)
# ----------------------------
class RefreshDialog(QDialog):
    """
    Refresh-dialog med:
    - scroll + max storlek (inte utanför skärmen)
    - markera/avmarkera alla (topics + sources)
    - källor grupperade per ämne (primärt ämne = första topic i listan)
    - multi-kolumn grids
    """

    def __init__(self, parent: QWidget, ui_opts):
        super().__init__(parent)
        self.setWindowTitle("Refresh – inställningar")
        self.ui_opts = ui_opts

        # Internal state
        self.topic_checks: Dict[str, QCheckBox] = {}
        self.source_checks: Dict[str, QCheckBox] = {}
        self.source_topics: Dict[str, List[str]] = {}
        self.source_primary_topic: Dict[str, str] = {}
        self.source_by_primary: Dict[str, List[str]] = {}

        outer = QVBoxLayout(self)

        # Scroll area holds all content except bottom actions
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        outer.addWidget(self.scroll, 1)

        content = QWidget()
        self.scroll.setWidget(content)
        root = QVBoxLayout(content)

        # ---- Basic settings
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

        # ---- Topics box (with select/deselect)
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
            cols = self._choose_columns(len(self.ui_opts.topic_options), preferred=3, max_cols=5)

            for i, t in enumerate(self.ui_opts.topic_options):
                cb = QCheckBox(t)
                self.topic_checks[t] = cb
                r = i // cols
                c = i % cols
                grid.addWidget(cb, r, c)
                cb.stateChanged.connect(lambda _=None, topic=t: self._on_topic_changed(topic))

            v_topics.addLayout(grid)
            root.addWidget(gb_topics)

            self.btn_topics_all.clicked.connect(lambda: self._set_all_topics(True))
            self.btn_topics_none.clicked.connect(lambda: self._set_all_topics(False))

        # ---- Build source maps and grouped UI
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

        # group order: keep known topic order, "Okategoriserat" last
        group_names = list(self.source_by_primary.keys())

        def group_sort_key(g: str):
            if g == "Okategoriserat":
                return (999999, g.lower())
            try:
                idx = self.ui_opts.topic_options.index(g)
                return (idx, g.lower())
            except Exception:
                return (500000, g.lower())

        group_names.sort(key=group_sort_key)

        for g in group_names:
            names = self.source_by_primary.get(g) or []
            if not names:
                continue

            gb = QGroupBox(f"{g} ({len(names)})")
            grid = QGridLayout(gb)

            cols_s = self._choose_columns(len(names), preferred=2, max_cols=4)
            for i, name in enumerate(sorted(names, key=lambda x: x.lower())):
                cb = self.source_checks[name]
                r = i // cols_s
                c = i % cols_s
                grid.addWidget(cb, r, c)

            v_sources.addWidget(gb)

        root.addWidget(gb_sources)
        root.addStretch(1)

        # ---- Bottom actions (fixed)
        actions = QHBoxLayout()
        actions.addStretch(1)
        self.btn_cancel = QPushButton("Avbryt")
        self.btn_ok = QPushButton("Kör refresh")
        actions.addWidget(self.btn_cancel)
        actions.addWidget(self.btn_ok)
        outer.addLayout(actions)

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self.accept)

        # ---- Size caps
        self.resize(900, 720)
        self._cap_to_screen(max_w=1100, max_h=840)

    # -------- internals --------
    def _init_sources(self) -> None:
        self.source_checks = {}
        self.source_topics = {}
        self.source_primary_topic = {}
        self.source_by_primary = {}

        for s in self.ui_opts.source_options:
            name = s["name"]
            topics = list(s.get("topics") or [])
            self.source_topics[name] = topics

            primary = topics[0] if topics else "Okategoriserat"
            self.source_primary_topic[name] = primary
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

    # -------- bulk actions --------
    def _set_all_topics(self, checked: bool) -> None:
        # block per-checkbox signals to avoid N cascades, then apply effect manually
        for t, cb in self.topic_checks.items():
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
            self._apply_topic_to_sources(t, checked)

    def _set_all_sources(self, checked: bool) -> None:
        for cb in self.source_checks.values():
            cb.setChecked(checked)

    # -------- topic -> sources mapping --------
    def _on_topic_changed(self, topic: str) -> None:
        checked = self.topic_checks[topic].isChecked()
        self._apply_topic_to_sources(topic, checked)

    def _apply_topic_to_sources(self, topic: str, checked: bool) -> None:
        # Apply to *all* sources that have this topic (not only primary)
        for src, cb in self.source_checks.items():
            if topic in (self.source_topics.get(src) or []):
                cb.setChecked(checked)

    # -------- result --------
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
# Article reader dialog
# ----------------------------
class ArticleReaderDialog(QDialog):
    """
    Shows full stored article text + metadata.
    """
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
        published = _val("published")
        pub_ts = published_ts(self.article)
        fetched_at = self.article.get("fetched_at")
        fetched_h = format_ts(int(fetched_at)) if isinstance(fetched_at, int) else _val("fetched_at")

        pub_h = published
        if not pub_h and pub_ts:
            pub_h = format_ts(pub_ts)

        meta_layout.addRow("Titel", QLabel(title))
        meta_layout.addRow("Källa", QLabel(source))
        meta_layout.addRow("Publicerad", QLabel(pub_h))
        meta_layout.addRow("Hämtad", QLabel(fetched_h))
        meta_layout.addRow("ID", QLabel(_val("id")))
        meta_layout.addRow("URL", QLabel(url))

        if "content_hash" in self.article:
            meta_layout.addRow("content_hash", QLabel(_val("content_hash")))
        if "published_ts" in self.article:
            meta_layout.addRow("published_ts", QLabel(_val("published_ts")))
        if "fetched_at" in self.article:
            meta_layout.addRow("fetched_at", QLabel(_val("fetched_at")))
        if "topics" in self.article:
            meta_layout.addRow("topics", QLabel(_val("topics")))

        outer.addWidget(meta_box)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.btn_open = QPushButton("Öppna i webbläsare")
        self.btn_close = QPushButton("Stäng")
        actions.addWidget(self.btn_open)
        actions.addWidget(self.btn_close)
        outer.addLayout(actions)

        self.btn_close.clicked.connect(self.accept)
        self.btn_open.clicked.connect(self._open_in_browser)

        self.text_view = QTextBrowser()
        self.text_view.setOpenExternalLinks(True)
        outer.addWidget(self.text_view, 1)

        full_text = (self.article.get("text") or "").strip()
        if not full_text:
            full_text = "(Ingen text lagrad för artikeln.)"

        safe = (
            full_text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        self.text_view.setHtml(
            "<pre style='white-space: pre-wrap; font-family: system-ui;'>"
            f"{safe}"
            "</pre>"
        )

    def _open_in_browser(self) -> None:
        url = str(self.article.get("url") or "").strip()
        if not url:
            QMessageBox.information(self, "Ingen URL", "Artikeln saknar URL.")
            return
        QDesktopServices.openUrl(QUrl(url))


# ----------------------------
# Main window
# ----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FeedSummary (Qt)")
        self.resize(1200, 860)

        # Store original stdout/stderr so we can restore on close
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

        self.cfg = load_config(CONFIG_PATH)
        self.store = get_store(self.cfg)
        self.ui_opts = get_ui_options(self.cfg, config_path=CONFIG_PATH)

        # Central splitter (tabs + log panel)
        splitter = QSplitter(Qt.Vertical)

        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)

        # Log panel container
        log_container = QWidget()
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(6, 6, 6, 6)

        log_toolbar = QHBoxLayout()
        log_toolbar.addWidget(QLabel("Logg (stdout/stderr)"))
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

        # Initial splitter sizes: mostly tabs, smaller log
        splitter.setSizes([650, 210])
        splitter.setCollapsible(1, True)

        self.setCentralWidget(splitter)

        self.btn_clear_log.clicked.connect(self.log_view.clear)

        # Redirect stdout/stderr to log panel
        self._log_emitter = QtLogEmitter()
        self._log_emitter.text.connect(self._append_log)
        sys.stdout = QtStream(self._log_emitter)
        sys.stderr = QtStream(self._log_emitter)

        # Build tabs
        self._build_summaries_tab()
        self._build_articles_tab()

        self.reload_summaries()
        self.reload_articles()

        # Emit bootstrap info into log
        print(f"[bootstrap] frozen={RUNTIME.is_frozen} base_dir={RUNTIME.base_dir}")
        print(f"[bootstrap] app_data_dir={RUNTIME.app_data_dir}")
        print(f"[bootstrap] config_path={RUNTIME.config_path}")

    def _append_log(self, text: str) -> None:
        # Normalize: avoid doubling of linebreaks on partial writes
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
        super().closeEvent(event)

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
        dlg = RefreshDialog(self, self.ui_opts)
        if dlg.exec() != QDialog.Accepted:
            return

        overrides = dlg.overrides()
        self.btn_refresh.setEnabled(False)
        self.lbl_status.setText("running…")
        print(f"[ui] refresh overrides={overrides}")

        self.worker = PipelineWorker(self.cfg, overrides)
        self.worker.status.connect(self.lbl_status.setText)
        self.worker.done.connect(self.on_pipeline_done)
        self.worker.failed.connect(self.on_pipeline_failed)
        self.worker.start()

    def on_pipeline_done(self, _sid: object):
        self.lbl_status.setText("done")
        self.btn_refresh.setEnabled(True)
        print("[ui] pipeline done -> reloading config/store/ui options")

        self.cfg = load_config(CONFIG_PATH)
        self.store = get_store(self.cfg)
        self.ui_opts = get_ui_options(self.cfg, config_path=CONFIG_PATH)

        self.reload_summaries()
        self._rebuild_article_filters_from_ui_opts()
        self.reload_articles()

    def on_pipeline_failed(self, err: str):
        self.lbl_status.setText("error")
        self.btn_refresh.setEnabled(True)
        print(f"[ui] pipeline error: {err}")
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

        # Scrollable filter area for topics + sources
        self.filters_scroll = QScrollArea()
        self.filters_scroll.setWidgetResizable(True)
        self.filters_container = QWidget()
        self.filters_layout = QHBoxLayout(self.filters_container)
        self.filters_scroll.setWidget(self.filters_container)
        layout.addWidget(self.filters_scroll)

        # placeholders; built in rebuild method
        self.topic_checks_articles: Dict[str, QCheckBox] = {}
        self.source_checks_articles: Dict[str, QCheckBox] = {}
        self.source_topics_articles: Dict[str, List[str]] = {}
        self.gb_topics: Optional[QGroupBox] = None
        self.gb_sources: Optional[QGroupBox] = None

        self.btn_apply.clicked.connect(self.reload_articles)
        self.btn_clear.clicked.connect(self.clear_article_filters)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Titel", "Källa", "Publicerad", "Preview"])
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

        self.gb_topics = QGroupBox("Ämnen")
        vb_t = QVBoxLayout(self.gb_topics)
        for t in self.ui_opts.topic_options:
            cb = QCheckBox(t)
            self.topic_checks_articles[t] = cb
            vb_t.addWidget(cb)
            cb.stateChanged.connect(lambda _=None, topic=t: self._apply_topic_to_sources_articles(topic))

        self.gb_sources = QGroupBox("Källor")
        vb_s = QVBoxLayout(self.gb_sources)
        for s in self.ui_opts.source_options:
            name = s["name"]
            self.source_topics_articles[name] = s.get("topics") or []
            cb = QCheckBox(name)
            self.source_checks_articles[name] = cb
            vb_s.addWidget(cb)

        self.filters_layout.addWidget(self.gb_topics)
        self.filters_layout.addWidget(self.gb_sources)
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
            it0.setData(Qt.UserRole, a)  # store full dict for reader dialog

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
        a = it.data(Qt.UserRole)
        if not isinstance(a, dict):
            return
        dlg = ArticleReaderDialog(self, a)
        dlg.exec()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()