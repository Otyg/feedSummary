from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional
from datetime import datetime
import markdown as md
import requests
from PySide6.QtCore import Qt, QSettings
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

LOG = logging.getLogger(__name__)


class ApiClient:
    BASEPATH = "/api/v1"
    def __init__(self, base_url: str, timeout: int = 20):
        self.timeout = timeout
        self.s = requests.Session()
        self.set_base(base_url)

    def set_base(self, base_url: str):
        self.base = (base_url or "").strip().rstrip("/")

    def get_json(self, path: str, params: Optional[dict] = None) -> dict:
        if not self.base:
            raise RuntimeError("Base URL is empty")
        url = f"{self.base}{path}"
        r = self.s.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # ---- API endpoints ----
    def ui_options(self) -> dict:
        return self.get_json(f"{self.BASEPATH}/ui_options")

    def list_summaries(self, limit: int = 300) -> List[dict]:
        j = self.get_json(f"{self.BASEPATH}/summaries", params={"limit": limit})
        return j.get("items") or []

    def latest_summary(self) -> Optional[dict]:
        j = self.get_json(f"{self.BASEPATH}/summaries/latest")
        return j.get("item")

    def get_summary(self, sid: str) -> dict:
        j = self.get_json(f"{self.BASEPATH}/summary/{sid}")
        return j["item"]

    def list_articles(self, limit: int = 500) -> List[dict]:
        j = self.get_json(f"{self.BASEPATH}/articles", params={"limit": limit})
        return j.get("items") or []

    def get_article(self, aid: str) -> dict:
        j = self.get_json(f"{self.BASEPATH}/article/{aid}")
        return j["item"]

    def get_prompt(self, name: str) -> dict:
        return self.get_json(f"{self.BASEPATH}/prompt/{name}")


class MainWindow(QMainWindow):
    def __init__(self, initial_base_url: str):
        super().__init__()
        self.setWindowTitle("FeedSummary – Remote (Read-only)")
        self.resize(1250, 860)

        self.settings = QSettings("Otyg", "FeedSummaryQtRemote")

        base = initial_base_url or str(self.settings.value("base_url", "http://localhost:5000"))
        self.api = ApiClient(base)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        self.setCentralWidget(root)

        # Top bar: base URL + connect/reload
        top = QHBoxLayout()
        top.addWidget(QLabel("Base URL:"))
        self.base_edit = QLineEdit(self.api.base)
        self.base_edit.setPlaceholderText("http://server:5000")
        top.addWidget(self.base_edit, 1)

        self.btn_connect = QPushButton("Connect")
        self.btn_reload = QPushButton("Reload")
        top.addWidget(self.btn_connect)
        top.addWidget(self.btn_reload)

        self.lbl_status = QLabel("idle")
        top.addWidget(self.lbl_status)
        root_layout.addLayout(top)

        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs, 1)

        self._build_summaries_tab()
        self._build_articles_tab()
        self._build_prompts_tab()

        self.btn_connect.clicked.connect(self.connect_and_load)
        self.btn_reload.clicked.connect(lambda: self.reload_all(select_latest=False))

        # Auto-connect on start
        self.connect_and_load()

    # -------------------- UI helpers --------------------
    def _set_status(self, s: str):
        self.lbl_status.setText(s)

    def _base_url(self) -> str:
        return self.base_edit.text().strip().rstrip("/")

    def connect_and_load(self):
        base = self._base_url()
        if not base:
            QMessageBox.warning(self, "Base URL", "Skriv in en base-url.")
            return

        self.api.set_base(base)
        self.settings.setValue("base_url", base)

        try:
            self._set_status("connecting…")
            opts = self.api.ui_options()
            self._apply_ui_options(opts)
            self.reload_prompts()  # prompts rely on ui_options too
            self.reload_all(select_latest=True)
            self._set_status("connected")
        except Exception as e:
            self._set_status("error")
            QMessageBox.critical(self, "Connect failed", f"Kunde inte ansluta till {base}\n\n{e}")

    def reload_all(self, select_latest: bool = False):
        self.reload_summaries(select_latest=select_latest)
        self.reload_articles()

    # -------------------- Summaries tab --------------------
    def _build_summaries_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        split = QSplitter()
        self.sum_table = QTableWidget(0, 3)
        self.sum_table.setHorizontalHeaderLabels(["Titel", "Skapad", "Källantal"])
        self.sum_view = QTextBrowser()
        split.addWidget(self.sum_table)
        split.addWidget(self.sum_view)
        split.setStretchFactor(1, 1)
        layout.addWidget(split, 1)

        self.sum_table.cellClicked.connect(self._on_summary_clicked)

        self.tabs.addTab(w, "Summaries")

    def reload_summaries(self, select_latest: bool = False):
        try:
            self._set_status("loading summaries…")
            items = self.api.list_summaries(limit=500)

            self.sum_table.setRowCount(len(items))
            for r, it in enumerate(items):
                created = datetime.fromtimestamp(int(it.get("created") or 0)).strftime("%Y-%m-%d %H:%M")
                title = str(it.get("title") or "").strip()
                sid = str(it.get("id") or "")

                c0 = QTableWidgetItem(str(title) or sid)
                c1 = QTableWidgetItem(str(created))
                c2 = QTableWidgetItem(sid)
                c2.setData(Qt.UserRole, sid)  # type: ignore

                self.sum_table.setItem(r, 0, c0)
                self.sum_table.setItem(r, 1, c1)
                self.sum_table.setItem(r, 2, c2)

            self.sum_table.resizeColumnsToContents()

            if select_latest:
                latest = self.api.latest_summary()
                if latest and latest.get("id"):
                    self._render_summary_by_id(str(latest["id"]))
                    return

            if items and items[0].get("id"):
                self._render_summary_by_id(str(items[0]["id"]))
        except Exception as e:
            QMessageBox.critical(self, "Fel", f"Kunde inte ladda summaries:\n{e}")
        finally:
            self._set_status("idle")

    def _on_summary_clicked(self, row: int, col: int):
        it = self.sum_table.item(row, 2)
        if not it:
            return
        sid = it.data(Qt.UserRole)  # type: ignore
        if sid:
            self._render_summary_by_id(str(sid))

    def _render_summary_by_id(self, sid: str):
        try:
            self._set_status("loading summary…")
            doc = self.api.get_summary(sid)

            title = str(doc.get("title") or "").strip()
            created = datetime.fromtimestamp(int(doc.get("created") or 0)).strftime("%Y-%m-%d %H:%M")
            md_text = str(doc.get("summary") or "")

            body_html = md.markdown(md_text, extensions=["extra"])
            meta_html = f"<div style='color:#666; font-size: 0.9em;'>Created: {created}</div>"

            if title:
                html = f"<h2>{title}</h2>{meta_html}<hr/>{body_html}"
            else:
                html = f"<h2>{sid}</h2>{meta_html}<hr/>{body_html}"

            self.sum_view.setHtml(html)
        except Exception as e:
            self.sum_view.setHtml(f"<p>Kunde inte läsa summary {sid}: {e}</p>")
        finally:
            self._set_status("idle")

    # -------------------- Articles tab --------------------
    def _build_articles_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        # Read-only filters
        filters = QHBoxLayout()
        filters.addWidget(QLabel("Source:"))
        self.cmb_source = QComboBox()
        self.cmb_source.addItem("Alla", "")
        filters.addWidget(self.cmb_source)

        self.btn_apply_filter = QPushButton("Apply")
        filters.addWidget(self.btn_apply_filter)
        filters.addStretch(1)
        layout.addLayout(filters)

        split = QSplitter()
        self.art_table = QTableWidget(0, 4)
        self.art_table.setHorizontalHeaderLabels(["Titel", "Källa", "Publicerad", "ID"])
        self.art_view = QTextBrowser()
        split.addWidget(self.art_table)
        split.addWidget(self.art_view)
        split.setStretchFactor(1, 1)
        layout.addWidget(split, 1)

        self.btn_apply_filter.clicked.connect(self.reload_articles)
        self.art_table.cellClicked.connect(self._on_article_clicked)

        self.tabs.addTab(w, "Articles")

        self._all_articles_cache: List[dict] = []

    def reload_articles(self):
        try:
            self._set_status("loading articles…")
            items = self.api.list_articles(limit=1500)
            self._all_articles_cache = items

            source_filter = self.cmb_source.currentData()  # type: ignore
            if source_filter:
                items = [a for a in items if str(a.get("source") or "") == str(source_filter)]

            self.art_table.setRowCount(len(items))
            for r, it in enumerate(items):
                title = str(it.get("title") or "")
                src = str(it.get("source") or "")
                pub = int(it.get("published_ts") or it.get("fetched_at") or 0)
                aid = str(it.get("id") or "")

                c0 = QTableWidgetItem(title)
                c1 = QTableWidgetItem(src)
                c2 = QTableWidgetItem(str(pub))
                c3 = QTableWidgetItem(aid)
                c3.setData(Qt.UserRole, aid)  # type: ignore

                self.art_table.setItem(r, 0, c0)
                self.art_table.setItem(r, 1, c1)
                self.art_table.setItem(r, 2, c2)
                self.art_table.setItem(r, 3, c3)

            self.art_table.resizeColumnsToContents()

            if items and items[0].get("id"):
                self._render_article_by_id(str(items[0]["id"]))
        except Exception as e:
            QMessageBox.critical(self, "Fel", f"Kunde inte ladda artiklar:\n{e}")
        finally:
            self._set_status("idle")

    def _on_article_clicked(self, row: int, col: int):
        it = self.art_table.item(row, 3)
        if not it:
            return
        aid = it.data(Qt.UserRole)  # type: ignore
        if aid:
            self._render_article_by_id(str(aid))

    def _render_article_by_id(self, aid: str):
        try:
            self._set_status("loading article…")
            a = self.api.get_article(aid)

            title = str(a.get("title") or "")
            source = str(a.get("source") or "")
            url = str(a.get("url") or "")
            text = str(a.get("text") or "")

            html = (
                f"<h2>{title}</h2>"
                f"<p><b>Källa:</b> {source}<br/>"
                f"<b>URL:</b> <a href='{url}'>{url}</a></p>"
                f"<hr/>"
                f"<pre style='white-space: pre-wrap;'>{text}</pre>"
            )
            self.art_view.setHtml(html)
        except Exception as e:
            self.art_view.setHtml(f"<p>Kunde inte läsa artikel {aid}: {e}</p>")
        finally:
            self._set_status("idle")

    # -------------------- Prompts tab --------------------
    def _build_prompts_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        top = QHBoxLayout()
        self.btn_reload_prompts = QPushButton("Ladda om")
        self.lbl_prompts = QLabel("0 paket")
        top.addWidget(self.btn_reload_prompts)
        top.addWidget(self.lbl_prompts, 1)
        layout.addLayout(top)

        split = QSplitter()

        self.prompts_table = QTableWidget(0, 1)
        self.prompts_table.setHorizontalHeaderLabels(["Prompt package"])
        split.addWidget(self.prompts_table)

        self.prompts_view = QTextBrowser()
        self.prompts_view.setHtml(
            "<p>Read-only. Klicka på ett paket för att se dess YAML-innehåll via <code>/api/v1/prompt/&lt;name&gt;</code>.</p>"
        )
        split.addWidget(self.prompts_view)
        split.setStretchFactor(1, 1)

        layout.addWidget(split, 1)

        self.btn_reload_prompts.clicked.connect(self.reload_prompts)
        self.prompts_table.cellClicked.connect(self._on_prompt_clicked)

        self.tabs.addTab(w, "Prompts")

    def reload_prompts(self):
        try:
            self._set_status("loading prompts…")
            opts = self.api.ui_options()
            pkgs = opts.get("prompt_packages") or []
            pkgs = [str(x).strip() for x in pkgs if str(x).strip()]
            pkgs.sort()

            self.prompts_table.setRowCount(len(pkgs))
            for r, name in enumerate(pkgs):
                it = QTableWidgetItem(name)
                it.setData(Qt.UserRole, name)  # type: ignore
                self.prompts_table.setItem(r, 0, it)

            self.prompts_table.resizeColumnsToContents()
            self.lbl_prompts.setText(f"{len(pkgs)} paket")

            feeds_path = str(opts.get("feeds_path") or "")
            prompts_path = str(opts.get("prompts_path") or "")
            self.prompts_view.setHtml(
                "<h3>Prompt packages (read-only)</h3>"
                f"<p><b>Antal:</b> {len(pkgs)}</p>"
                f"<p><b>prompts.yaml:</b> <code>{prompts_path}</code><br/>"
                f"<b>feeds.yaml:</b> <code>{feeds_path}</code></p>"
                "<p>Klicka på ett paket i listan för att visa YAML-innehållet.</p>"
            )

            if pkgs:
                self._render_prompt(pkgs[0])
        except Exception as e:
            QMessageBox.critical(self, "Fel", f"Kunde inte ladda promptpaket:\n{e}")
        finally:
            self._set_status("idle")

    def _on_prompt_clicked(self, row: int, col: int):
        it = self.prompts_table.item(row, 0)
        if not it:
            return
        name = (it.data(Qt.UserRole) or it.text() or "").strip()  # type: ignore
        if name:
            self._render_prompt(name)

    def _render_prompt(self, name: str):
        try:
            self._set_status("loading prompt…")
            j = self.api.get_prompt(name)
            yaml_text = str(j.get("yaml") or "")
            prompts_path = str(j.get("prompts_path") or "")

            html = (
                f"<h3>{name}</h3>"
                f"<div class='text-muted'><b>prompts.yaml:</b> <code>{prompts_path}</code></div>"
                "<hr/>"
                f"<pre style='white-space: pre-wrap;'>{yaml_text}</pre>"
            )
            self.prompts_view.setHtml(html)
        except Exception as e:
            self.prompts_view.setHtml(f"<p>Kunde inte hämta prompt <code>{name}</code>: {e}</p>")
        finally:
            self._set_status("idle")

    # -------------------- UI Options --------------------
    def _apply_ui_options(self, opts: dict):
        sources = opts.get("sources") or []
        self.cmb_source.blockSignals(True)
        self.cmb_source.clear()
        self.cmb_source.addItem("Alla", "")
        for s in sources:
            s = str(s).strip()
            if s:
                self.cmb_source.addItem(s, s)
        self.cmb_source.blockSignals(False)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    ap = argparse.ArgumentParser(description="FeedSummary Qt Remote (read-only)")
    ap.add_argument("--base-url", default="", help="Base URL (optional; can be set in UI), e.g. http://localhost:5000")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    win = MainWindow(args.base_url)
    win.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()