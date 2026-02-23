from __future__ import annotations

from typing import Any, Dict, List
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from uicommon import build_refresh_overrides


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

        self.scroll = QScrollArea() # type: ignore
        self.scroll.setWidgetResizable(True) # type: ignore
        outer.addWidget(self.scroll, 1) # type: ignore

        content = QWidget()
        self.scroll.setWidget(content) # type: ignore
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
        screen = self.screen() or (self.parent().screen() if self.parent() else None) # type: ignore
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
