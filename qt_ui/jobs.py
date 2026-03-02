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

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QMessageBox,
    QPlainTextEdit,
)

from uicommon import format_ts
from feedsummary_core.summarizer.helpers import (
    _checkpoint_key,
    _checkpoint_path,
    _meta_ckpt_path,
)


@dataclass(frozen=True)
class JobRow:
    job_id: int
    created_at: int
    started_at: int
    finished_at: int
    status: str
    message: str
    summary_id: str
    has_ckpt: bool
    has_meta: bool
    ckpt_path: str
    meta_path: str
    has_temp: bool
    temp_preview: str


def _job_checkpoint_paths(cfg: Dict[str, Any], job_id: int) -> Tuple[Path, Path]:
    key = _checkpoint_key(int(job_id), [])
    cp = _checkpoint_path(cfg, key)
    mp = _meta_ckpt_path(cfg, key)
    return cp, mp


def _yn(b: bool) -> str:
    return "JA" if b else "NEJ"


def _safe_int(v: Any) -> int:
    try:
        return int(v or 0)
    except Exception:
        return 0


def _clip(s: str, n: int = 220) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


class TempPreviewDialog(QDialog):
    def __init__(self, parent, *, title: str, text: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 600)
        layout = QVBoxLayout(self)

        self.editor = QPlainTextEdit()
        self.editor.setReadOnly(True)
        self.editor.setPlainText(text or "")
        layout.addWidget(self.editor, 1)

        btns = QHBoxLayout()
        btns.addStretch(1)
        b = QPushButton("Stäng")
        b.clicked.connect(self.accept)
        btns.addWidget(b)
        layout.addLayout(btns)


class JobsDialog(QDialog):
    """
    Dialog som listar jobb + checkpoint-status och kan:
      - Återuppta markerat jobb (om checkpoint finns)
      - Visa temp summary (partial)
      - Radera checkpoints (städning)
    """

    def __init__(self, parent, *, cfg: Dict[str, Any], store):
        super().__init__(parent)
        self.setWindowTitle("Avbrutna jobb / Återuppta")
        self.resize(1100, 520)

        self.cfg = cfg
        self.store = store

        root = QVBoxLayout(self)

        info = QLabel(
            "Här kan du se tidigare jobb. Om en checkpoint finns kan du återuppta jobbet.\n"
            "Tips: ‘Temp’ indikerar att det finns ett delresultat sparat under körning."
        )
        info.setWordWrap(True)
        root.addWidget(info)

        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(
            [
                "Jobb-ID",
                "Skapad",
                "Start",
                "Klar",
                "Status",
                "Checkpoint",
                "Meta",
                "Temp",
                "Summary-ID",
                "Meddelande",
            ]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectRows)  # type: ignore
        self.table.setSelectionMode(QTableWidget.SingleSelection)  # type: ignore
        root.addWidget(self.table, 1)

        btn_row = QHBoxLayout()
        self.btn_reload = QPushButton("Ladda om")
        self.btn_preview = QPushButton("Visa temp")
        self.btn_resume = QPushButton("Återuppta")
        self.btn_delete_ckpt = QPushButton("Radera checkpoints…")
        self.btn_open_ckpt_dir = QPushButton("Öppna checkpoint-mapp")
        self.btn_close = QPushButton("Stäng")

        btn_row.addWidget(self.btn_reload)
        btn_row.addWidget(self.btn_preview)
        btn_row.addWidget(self.btn_open_ckpt_dir)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_delete_ckpt)
        btn_row.addWidget(self.btn_resume)
        btn_row.addWidget(self.btn_close)
        root.addLayout(btn_row)

        self.btn_close.clicked.connect(self.reject)
        self.btn_reload.clicked.connect(self.reload)
        self.btn_resume.clicked.connect(self._resume_selected)
        self.btn_preview.clicked.connect(self._preview_temp_selected)
        self.btn_delete_ckpt.clicked.connect(self._delete_ckpt_selected)
        self.btn_open_ckpt_dir.clicked.connect(self._open_ckpt_dir)

        self._rows: List[JobRow] = []
        self.reload()

    def _call_store_list_jobs(self) -> List[Dict[str, Any]]:
        fn = getattr(self.store, "list_jobs", None)
        if not callable(fn):
            raise RuntimeError(
                "Store saknar list_jobs(). Uppdatera persistence-storen."
            )
        return fn(300) or []  # type: ignore

    def reload(self) -> None:
        try:
            jobs = self._call_store_list_jobs()
        except Exception as e:
            QMessageBox.critical(self, "Fel", str(e))
            return

        rows: List[JobRow] = []
        for j in jobs:
            jid = _safe_int(j.get("id"))
            if jid <= 0:
                continue

            created_at = _safe_int(j.get("created_at"))
            started_at = _safe_int(j.get("started_at"))
            finished_at = _safe_int(j.get("finished_at"))
            status = str(j.get("status") or "").strip()
            message = str(j.get("message") or "").strip()
            summary_id = str(j.get("summary_id") or "").strip()

            cp, mp = _job_checkpoint_paths(self.cfg, jid)
            has_ckpt = cp.exists()
            has_meta = mp.exists()

            temp = None
            has_temp = False
            temp_preview = ""
            try:
                temp = self.store.get_temp_summary(jid)
                if (
                    temp
                    and isinstance(temp, dict)
                    and (temp.get("summary") or "").strip()
                ):
                    has_temp = True
                    temp_preview = _clip(str(temp.get("summary") or ""), 220)
            except Exception:
                has_temp = False

            rows.append(
                JobRow(
                    job_id=jid,
                    created_at=created_at,
                    started_at=started_at,
                    finished_at=finished_at,
                    status=status,
                    message=message,
                    summary_id=summary_id,
                    has_ckpt=has_ckpt,
                    has_meta=has_meta,
                    ckpt_path=str(cp),
                    meta_path=str(mp),
                    has_temp=has_temp,
                    temp_preview=temp_preview,
                )
            )

        rows.sort(key=lambda r: r.created_at, reverse=True)
        self._rows = rows

        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            it0 = QTableWidgetItem(str(r.job_id))
            it0.setData(Qt.UserRole, r.job_id)  # type: ignore

            it1 = QTableWidgetItem(format_ts(r.created_at) if r.created_at else "")
            it2 = QTableWidgetItem(format_ts(r.started_at) if r.started_at else "")
            it3 = QTableWidgetItem(format_ts(r.finished_at) if r.finished_at else "")
            it4 = QTableWidgetItem(r.status)
            it5 = QTableWidgetItem(_yn(r.has_ckpt))
            it6 = QTableWidgetItem(_yn(r.has_meta))
            it7 = QTableWidgetItem(_yn(r.has_temp))
            it8 = QTableWidgetItem(r.summary_id)
            it9 = QTableWidgetItem(_clip(r.message, 260))

            # Tooltips
            it5.setToolTip(r.ckpt_path)
            it6.setToolTip(r.meta_path)
            if r.has_temp and r.temp_preview:
                it7.setToolTip(r.temp_preview)

            self.table.setItem(i, 0, it0)
            self.table.setItem(i, 1, it1)
            self.table.setItem(i, 2, it2)
            self.table.setItem(i, 3, it3)
            self.table.setItem(i, 4, it4)
            self.table.setItem(i, 5, it5)
            self.table.setItem(i, 6, it6)
            self.table.setItem(i, 7, it7)
            self.table.setItem(i, 8, it8)
            self.table.setItem(i, 9, it9)

        self.table.resizeColumnsToContents()

    def _selected_job_id(self) -> Optional[int]:
        r = self.table.currentRow()
        if r is None or r < 0:
            return None
        it = self.table.item(r, 0)
        if not it:
            return None
        jid = it.data(Qt.UserRole)  # type: ignore
        try:
            return int(jid)
        except Exception:
            return None

    def _selected_row_obj(self) -> Optional[JobRow]:
        jid = self._selected_job_id()
        if jid is None:
            return None
        for r in self._rows:
            if r.job_id == jid:
                return r
        return None

    def _resume_selected(self) -> None:
        row = self._selected_row_obj()
        if row is None:
            QMessageBox.information(self, "Ingen rad", "Välj ett jobb först.")
            return
        if not row.has_ckpt:
            QMessageBox.warning(
                self,
                "Ingen checkpoint",
                "Det finns ingen checkpoint för det här jobbet, så det går inte att återuppta.",
            )
            return

        self.setProperty("selected_job_id", int(row.job_id))
        self.accept()

    def _preview_temp_selected(self) -> None:
        row = self._selected_row_obj()
        if row is None:
            QMessageBox.information(self, "Ingen rad", "Välj ett jobb först.")
            return

        try:
            temp = self.store.get_temp_summary(row.job_id)
        except Exception as e:
            QMessageBox.warning(self, "Kunde inte läsa temp", str(e))
            return

        if (
            not temp
            or not isinstance(temp, dict)
            or not (temp.get("summary") or "").strip()
        ):
            QMessageBox.information(
                self,
                "Ingen temp summary",
                "Det finns inget delresultat sparat för det här jobbet.",
            )
            return

        text = str(temp.get("summary") or "")
        title = f"Temp summary – jobb {row.job_id}"
        dlg = TempPreviewDialog(self, title=title, text=text)
        dlg.exec()

    def _delete_ckpt_selected(self) -> None:
        row = self._selected_row_obj()
        if row is None:
            QMessageBox.information(self, "Ingen rad", "Välj ett jobb först.")
            return

        cp = Path(row.ckpt_path)
        mp = Path(row.meta_path)
        if not cp.exists() and not mp.exists():
            QMessageBox.information(
                self,
                "Inget att radera",
                "Det finns inga checkpoint-filer för det här jobbet.",
            )
            return

        msg = (
            f"Radera checkpoint-filer för jobb {row.job_id}?\n\n"
            f"- {cp}\n"
            f"- {mp}\n\n"
            "Det går inte att ångra."
        )
        resp = QMessageBox.question(
            self, "Bekräfta", msg, QMessageBox.Yes | QMessageBox.No
        )  # type: ignore
        if resp != QMessageBox.Yes:  # type: ignore
            return

        errs: List[str] = []
        try:
            if cp.exists():
                cp.unlink()
        except Exception as e:
            errs.append(str(e))
        try:
            if mp.exists():
                mp.unlink()
        except Exception as e:
            errs.append(str(e))

        if errs:
            QMessageBox.warning(self, "Delvis misslyckat", "\n".join(errs))
        else:
            QMessageBox.information(self, "Klart", "Checkpoint-filer raderade.")
        self.reload()

    def _open_ckpt_dir(self) -> None:
        try:
            cp, _mp = _job_checkpoint_paths(self.cfg, 1)
            d = cp.parent
            if not d.exists():
                QMessageBox.information(
                    self, "Saknas", f"Checkpoint-mappen finns inte än:\n{d}"
                )
                return

            if os.name == "nt":
                os.startfile(str(d))  # type: ignore[attr-defined]
                return
            if os.uname().sysname.lower() == "darwin":  # type: ignore[attr-defined]
                os.system(f'open "{d}"')
                return
            os.system(f'xdg-open "{d}"')
        except Exception as e:
            QMessageBox.warning(self, "Kunde inte öppna mapp", str(e))
