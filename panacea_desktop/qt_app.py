from __future__ import annotations

import json
import os
import re
import threading
import tempfile
import time
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets

from .core.db import Database
from .core.engine import MiriamEngine
from .core.prompts import PROMPT_PACKS


# ------------------------------ Styling ------------------------------

QSS = """
* { font-family: Segoe UI, Inter, Arial; }
QMainWindow { background: #0b1220; }
QWidget { color: #d7e3f3; }

/* Make dialogs/message boxes follow the same dark theme */
QDialog, QMessageBox { background: #0b1220; }
QMessageBox QLabel { color: #d7e3f3; }

/* Tabs (Settings) */
QTabWidget::pane {
  border: 1px solid #0f2342;
  background: #07101d;
  border-radius: 12px;
  top: -1px;
}
QTabBar::tab {
  background: #0c1830;
  border: 1px solid #0f2342;
  border-bottom: none;
  padding: 8px 12px;
  border-top-left-radius: 10px;
  border-top-right-radius: 10px;
  margin-right: 6px;
}
QTabBar::tab:selected {
  background: rgba(95,241,182,0.12);
  border-color: rgba(95,241,182,0.22);
}

/* Common form controls */
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox {
  background: #0c1830;
  border: 1px solid #0f2342;
  border-radius: 10px;
  padding: 8px 10px;
}
QComboBox::drop-down { border: none; }

QPushButton {
  background: #0c1830;
  border: 1px solid #0f2342;
  border-radius: 10px;
  padding: 8px 12px;
}
QPushButton:hover { background: rgba(255,255,255,0.06); }
QPushButton:disabled { color: rgba(215,227,243,0.35); }
QComboBox QAbstractItemView { background: #0c1830; border: 1px solid #0f2342; selection-background-color: rgba(95,241,182,0.12); }

/* Menus ("...") */
QMenu {
  background: #0f1723;
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 12px;
  padding: 6px;
}
QMenu::item {
  background: transparent;
  color: #d7e3f3;
  padding: 8px 12px;
  border-radius: 10px;
}
QMenu::item:selected { background: rgba(95,241,182,0.14); }
QMenu::item:disabled { color: rgba(215,227,243,0.55); }
QMenu::separator {
  height: 1px;
  background: rgba(255,255,255,0.08);
  margin: 6px 8px;
}

QDialogButtonBox QPushButton {
  background: #0c1830;
  border: 1px solid #0f2342;
  border-radius: 10px;
  padding: 8px 14px;
}
QDialogButtonBox QPushButton:hover { background: rgba(255,255,255,0.05); }
QDialogButtonBox QPushButton:pressed { background: rgba(255,255,255,0.08); }

/* Sidebar */
#Sidebar { background: #07101d; border-right: 1px solid #0f1b2e; }
#NewChatBtn {
  background: #5ff1b6;
  color: #072018;
  border: none;
  border-radius: 10px;
  padding: 12px 14px;
  font-weight: 600;
}
#NewChatBtn:hover { background: #74f6c4; }
#Search {
  background: #0c1830;
  border: 1px solid #0f2342;
  border-radius: 10px;
  padding: 10px 12px;
}
#ChatList { background: transparent; border: none; }
#ChatList::item { padding: 10px 12px; border-radius: 10px; }
#ChatList::item:selected { background: rgba(95,241,182,0.12); }
#ChatList::item:hover { background: rgba(255,255,255,0.04); }

/* Tables (Settings: Memory list, etc.) */
QTableWidget, QTreeWidget, QListWidget {
  background: #07101d;
  border: 1px solid #0f2342;
  border-radius: 12px;
  alternate-background-color: rgba(255,255,255,0.02);
  gridline-color: rgba(255,255,255,0.08);
}
QTableWidget::viewport, QTreeWidget::viewport, QListWidget::viewport {
  background: #07101d;
}
QTableWidget::item, QTreeWidget::item, QListWidget::item {
  padding: 6px 8px;
  color: #d7e3f3;
}
QTableWidget::item:selected, QTreeWidget::item:selected, QListWidget::item:selected {
  background: rgba(95,241,182,0.12);
}
QHeaderView::section {
  background: #0c1830;
  color: #d7e3f3;
  padding: 6px 10px;
  border: none;
  border-right: 1px solid rgba(255,255,255,0.08);
}
QTableCornerButton::section { background: #0c1830; border: none; }

/* Scrollbars (dark) */
QScrollBar:vertical {
  background: rgba(255,255,255,0.02);
  width: 12px;
  margin: 0px;
  border-radius: 6px;
}
QScrollBar::handle:vertical {
  background: rgba(215,227,243,0.22);
  min-height: 24px;
  border-radius: 6px;
}
QScrollBar::handle:vertical:hover { background: rgba(215,227,243,0.30); }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar:horizontal {
  background: rgba(255,255,255,0.02);
  height: 12px;
  margin: 0px;
  border-radius: 6px;
}
QScrollBar::handle:horizontal {
  background: rgba(215,227,243,0.22);
  min-width: 24px;
  border-radius: 6px;
}
QScrollBar::handle:horizontal:hover { background: rgba(215,227,243,0.30); }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }

/* Top bar */
#TopBar { background: transparent; }
#TopSpacer { background: transparent; }


/* Mode dropdown */
QComboBox#ModeDrop {
  min-height: 28px;
  padding: 4px 10px;
  border-radius: 10px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.06);
}
QComboBox#ModeDrop:hover { background: rgba(255,255,255,0.06); }
QComboBox#ModeDrop::drop-down { border: none; width: 22px; }
QComboBox#ModeDrop::down-arrow { image: none; }
QComboBox QAbstractItemView {
  background: #0f1723;
  border: 1px solid rgba(255,255,255,0.10);
  selection-background-color: rgba(95,241,182,0.18);
  outline: 0px;
}
/* Switch */
QCheckBox#Switch { spacing: 8px; }
QCheckBox#Switch::indicator {
  width: 42px; height: 24px;
  border-radius: 12px;
  background: #1a2a45;
}
QCheckBox#Switch::indicator:checked { background: #5ff1b6; }
QCheckBox#Switch::indicator::unchecked { background: #1a2a45; }
QCheckBox#Switch::indicator:unchecked { border: 1px solid #243a5f; }

/* Chat area */
#ChatScroll { background: transparent; border: none; }
#ChatViewport { background: transparent; }

/* Input */
#InputBox {
  background: #0c1830;
  border: 1px solid #0f2342;
  border-radius: 14px;
}
#MessageEdit {
  background: transparent;
  border: none;
  padding: 10px 12px;
  font-size: 13px;
}
QToolButton#IconBtn {
  background: transparent;
  border: none;
  padding: 8px;
  border-radius: 10px;
}
QToolButton#IconBtn:hover { background: rgba(255,255,255,0.06); }
#SendBtn {
  background: #5ff1b6;
  color: #072018;
  border: none;
  padding: 10px 14px;
  border-radius: 12px;
  font-weight: 700;
}
#SendBtn:hover { background: #74f6c4; }

/* Bubble */
QFrame#BubbleUser {
  background: rgba(95,241,182,0.12);
  border: 1px solid rgba(95,241,182,0.22);
  border-radius: 14px;
}
QFrame#BubbleAssistant {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 14px;
}
QLabel#BubbleText { padding: 10px 12px; }
"""


# ------------------------------ Widgets ------------------------------


def _md_to_html(md: str) -> str:
    """Render Markdown to HTML using Qt's built-in Markdown engine."""
    md = md or ""
    doc = QtGui.QTextDocument()
    doc.setDefaultStyleSheet(
        "body{color:#d7e3f3;font-family:'Segoe UI',Inter,Arial;"
        "font-size:13px;line-height:1.35;}"
        "a{color:#5ff1b6;text-decoration:none;}"
        "pre{background:#0c1830;border:1px solid #0f2342;"
        "border-radius:10px;padding:8px;white-space:pre-wrap;}"
        "code{font-family:Consolas,Menlo,Monaco,monospace;}"
        "ul,ol{margin-left:18px;}"
    )
    # Qt 6 supports Markdown directly.
    doc.setMarkdown(md)
    return doc.toHtml()


class Bubble(QtWidgets.QFrame):
    def __init__(self, text: str, *, is_user: bool):
        super().__init__()
        self.setObjectName("BubbleUser" if is_user else "BubbleAssistant")

        self._is_user = is_user

        self._label = QtWidgets.QLabel()
        self._label.setObjectName("BubbleText")
        self._label.setWordWrap(True)
        self._label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._label.setOpenExternalLinks(True)
        self._label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard
        )

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._label)

        self.set_text(text)

    def set_text(self, text: str) -> None:
        # Render markdown to HTML for nicer display (lists/bold/headers).
        html = _md_to_html(text)
        self._label.setText(html)

    def text(self) -> str:
        # Return raw html; for raw markdown keep it in caller if needed.
        return self._label.text()


class FlowLayout(QtWidgets.QLayout):
    """Simple flow layout for message bubbles."""

    def __init__(self, parent=None, margin=0, spacing=10):
        super().__init__(parent)
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)
        self._items: list[QtWidgets.QLayoutItem] = []

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return QtCore.Qt.Orientations(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self.doLayout(QtCore.QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self.doLayout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def doLayout(self, rect: QtCore.QRect, testOnly: bool):
        x = rect.x()
        y = rect.y()
        lineHeight = 0
        spacing = self.spacing()

        for item in self._items:
            w = item.sizeHint().width()
            h = item.sizeHint().height()
            nextX = x + w + spacing
            if nextX - spacing > rect.right() and lineHeight > 0:
                x = rect.x()
                y = y + lineHeight + spacing
                nextX = x + w + spacing
                lineHeight = 0
            if not testOnly:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), item.sizeHint()))
            x = nextX
            lineHeight = max(lineHeight, h)

        return y + lineHeight - rect.y()


# ------------------------------ Dialogs ------------------------------


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget, db: Database, initial_tab: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.db = db
        self._initial_tab = initial_tab

        tabs = QtWidgets.QTabWidget()

        def _optional_import_ok(mods: list[str]) -> bool:
            try:
                for m in mods:
                    importlib.import_module(m)
                return True
            except Exception:
                return False

        # Providers tab
        prov = QtWidgets.QWidget()
        prov_l = QtWidgets.QFormLayout(prov)
        self.provider_mode = QtWidgets.QComboBox()
        self.provider_mode.addItems(["auto", "novita", "openrouter", "custom"])
        self.provider_mode.setCurrentText(db.get_setting("provider_mode", "auto") or "auto")

        self.novita_base = QtWidgets.QLineEdit(db.get_setting("novita_base_url", "") or "")
        self.novita_key = QtWidgets.QLineEdit(db.get_setting("novita_api_key", "") or "")
        self.novita_key.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.openrouter_base = QtWidgets.QLineEdit(db.get_setting("openrouter_base_url", "") or "")
        self.openrouter_key = QtWidgets.QLineEdit(db.get_setting("openrouter_api_key", "") or "")
        self.openrouter_key.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.custom_base = QtWidgets.QLineEdit(db.get_setting("custom_base_url", "") or "")
        self.custom_key = QtWidgets.QLineEdit(db.get_setting("custom_api_key", "") or "")
        self.custom_key.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)

        prov_l.addRow("Provider", self.provider_mode)
        prov_l.addRow("Novita base_url", self.novita_base)
        prov_l.addRow("Novita api_key", self.novita_key)
        prov_l.addRow("OpenRouter base_url", self.openrouter_base)
        prov_l.addRow("OpenRouter api_key", self.openrouter_key)
        prov_l.addRow("Custom base_url", self.custom_base)
        prov_l.addRow("Custom api_key", self.custom_key)
        tabs.addTab(prov, "Providers")

                # Models tab
        models = QtWidgets.QWidget()
        models_l = QtWidgets.QFormLayout(models)

        # Novita models
        hdr1 = QtWidgets.QLabel("Novita models")
        hdr1.setStyleSheet("font-weight: 700; padding-top: 6px;")
        models_l.addRow(hdr1, QtWidgets.QLabel(""))

        self.novita_model_light = QtWidgets.QLineEdit(
            db.get_setting("novita_model_light", "qwen/qwen2.5-vl-72b-instruct") or "qwen/qwen2.5-vl-72b-instruct"
        )
        self.novita_model_medium = QtWidgets.QLineEdit(
            db.get_setting("novita_model_medium", "baidu/ernie-4.5-vl-424b-a47b") or "baidu/ernie-4.5-vl-424b-a47b"
        )
        self.novita_model_heavy = QtWidgets.QLineEdit(
            db.get_setting("novita_model_heavy", "zai-org/glm-4.5v") or "zai-org/glm-4.5v"
        )
        self.novita_model_arbiter = QtWidgets.QLineEdit(
            db.get_setting("novita_model_arbiter", "openai/gpt-oss-120b") or "openai/gpt-oss-120b"
        )
        models_l.addRow("Novita light", self.novita_model_light)
        models_l.addRow("Novita medium", self.novita_model_medium)
        models_l.addRow("Novita heavy", self.novita_model_heavy)
        models_l.addRow("Novita arbiter", self.novita_model_arbiter)

        # OpenRouter models
        hdr2 = QtWidgets.QLabel("OpenRouter models")
        hdr2.setStyleSheet("font-weight: 700; padding-top: 10px;")
        models_l.addRow(hdr2, QtWidgets.QLabel(""))

        self.openrouter_model_light = QtWidgets.QLineEdit(
            db.get_setting("openrouter_model_light", "qwen/qwen3-vl-32b-instruct") or "qwen/qwen3-vl-32b-instruct"
        )
        self.openrouter_model_medium = QtWidgets.QLineEdit(
            db.get_setting("openrouter_model_medium", "baidu/ernie-4.5-vl-424b-a47b") or "baidu/ernie-4.5-vl-424b-a47b"
        )
        self.openrouter_model_heavy = QtWidgets.QLineEdit(
            db.get_setting("openrouter_model_heavy", "z-ai/glm-4.6v") or "z-ai/glm-4.6v"
        )
        self.openrouter_model_arbiter = QtWidgets.QLineEdit(
            db.get_setting("openrouter_model_arbiter", "openai/gpt-oss-120b") or "openai/gpt-oss-120b"
        )
        models_l.addRow("OpenRouter light", self.openrouter_model_light)
        models_l.addRow("OpenRouter medium", self.openrouter_model_medium)
        models_l.addRow("OpenRouter heavy", self.openrouter_model_heavy)
        models_l.addRow("OpenRouter arbiter", self.openrouter_model_arbiter)

        # Custom provider models (generic)
        hdr3 = QtWidgets.QLabel("Custom provider models")
        hdr3.setStyleSheet("font-weight: 700; padding-top: 10px;")
        models_l.addRow(hdr3, QtWidgets.QLabel(""))

        self.model_light = QtWidgets.QLineEdit(db.get_setting("model_light", "qwen/qwen3-vl-32b-instruct") or "qwen/qwen3-vl-32b-instruct")
        self.model_medium = QtWidgets.QLineEdit(db.get_setting("model_medium", "baidu/ernie-4.5-vl-424b-a47b") or "baidu/ernie-4.5-vl-424b-a47b")
        self.model_heavy = QtWidgets.QLineEdit(db.get_setting("model_heavy", "z-ai/glm-4.6v") or "z-ai/glm-4.6v")
        self.model_arbiter = QtWidgets.QLineEdit(db.get_setting("model_arbiter", "openai/gpt-oss-120b") or "openai/gpt-oss-120b")
        models_l.addRow("Custom light", self.model_light)
        models_l.addRow("Custom medium", self.model_medium)
        models_l.addRow("Custom heavy", self.model_heavy)
        models_l.addRow("Custom arbiter", self.model_arbiter)

        # Generation params
        hdr4 = QtWidgets.QLabel("Generation")
        hdr4.setStyleSheet("font-weight: 700; padding-top: 10px;")
        models_l.addRow(hdr4, QtWidgets.QLabel(""))

        self.temperature = QtWidgets.QLineEdit(db.get_setting("temperature", "0.2") or "0.2")
        self.max_tokens = QtWidgets.QLineEdit(db.get_setting("max_tokens", "1200") or "1200")
        self.history_max = QtWidgets.QLineEdit(db.get_setting("history_max_messages", "30") or "30")
        models_l.addRow("Temperature", self.temperature)
        models_l.addRow("Max tokens", self.max_tokens)
        models_l.addRow("History max msgs", self.history_max)

        tabs.addTab(models, "Models")

        # Behavior tab
        beh = QtWidgets.QWidget()
        beh_l = QtWidgets.QFormLayout(beh)
        self.prompt_pack = QtWidgets.QComboBox()
        self.prompt_pack.addItems(list(PROMPT_PACKS.keys()))
        self.prompt_pack.setCurrentText(db.get_setting("prompt_pack", "gp") or "gp")

        self.reasoning_mode = QtWidgets.QComboBox()
        self.reasoning_mode.addItems(["light", "medium", "heavy", "consensus"])
        self.reasoning_mode.setCurrentText(db.get_setting("reasoning_mode", "medium") or "medium")

        beh_l.addRow("Pack", self.prompt_pack)
        beh_l.addRow("Mode", self.reasoning_mode)
        tabs.addTab(beh, "Behavior")

        # Prompts tab (override per pack)
        prm = QtWidgets.QWidget()
        prm_l = QtWidgets.QVBoxLayout(prm)
        prm_l.setContentsMargins(0, 0, 0, 0)
        prm_form = QtWidgets.QFormLayout()
        self.prompts_pack = QtWidgets.QComboBox()
        self.prompts_pack.addItems(list(PROMPT_PACKS.keys()))
        self.prompts_pack.setCurrentText(self.prompt_pack.currentText())
        self.system_override = QtWidgets.QPlainTextEdit()
        self.system_override.setPlaceholderText("Override system prompt for selected pack...")
        self.arbiter_override = QtWidgets.QPlainTextEdit()
        self.arbiter_override.setPlaceholderText("Override arbiter prompt for selected pack...")
        prm_form.addRow("Pack", self.prompts_pack)
        prm_form.addRow("System", self.system_override)
        prm_form.addRow("Arbiter", self.arbiter_override)
        prm_l.addLayout(prm_form)
        prm_btns = QtWidgets.QHBoxLayout()
        self.btn_prm_defaults = QtWidgets.QPushButton("Load defaults")
        self.btn_prm_clear = QtWidgets.QPushButton("Clear override")
        prm_btns.addWidget(self.btn_prm_defaults)
        prm_btns.addWidget(self.btn_prm_clear)
        prm_btns.addStretch(1)
        prm_l.addLayout(prm_btns)
        tabs.addTab(prm, "Prompts")

        self._prompt_edits: dict[str, tuple[str, str]] = {}

        def _load_pack_prompts(pack: str):
            pack = pack if pack in PROMPT_PACKS else "gp"
            ov = self.db.get_prompt_override(pack) or {}
            sys_txt = (ov.get("system_prompt") or "").strip() or PROMPT_PACKS[pack]["system"]
            arb_txt = (ov.get("arbiter_prompt") or "").strip() or PROMPT_PACKS[pack]["arbiter"]
            self.system_override.setPlainText(sys_txt)
            self.arbiter_override.setPlainText(arb_txt)

        def _stash_current_prompts():
            p = self.prompts_pack.currentText()
            if not p:
                return
            self._prompt_edits[p] = (self.system_override.toPlainText(), self.arbiter_override.toPlainText())

        def _on_prompts_pack_changed(new_pack: str):
            _stash_current_prompts()
            # Load from cache first
            if new_pack in self._prompt_edits:
                sys_txt, arb_txt = self._prompt_edits[new_pack]
                self.system_override.setPlainText(sys_txt)
                self.arbiter_override.setPlainText(arb_txt)
            else:
                _load_pack_prompts(new_pack)

        self.prompts_pack.currentTextChanged.connect(_on_prompts_pack_changed)

        def _defaults_clicked():
            p = self.prompts_pack.currentText() or "gp"
            self.system_override.setPlainText(PROMPT_PACKS[p]["system"])
            self.arbiter_override.setPlainText(PROMPT_PACKS[p]["arbiter"])
            _stash_current_prompts()

        def _clear_clicked():
            p = self.prompts_pack.currentText() or "gp"
            self.db.delete_prompt_override(p)
            self.system_override.setPlainText(PROMPT_PACKS[p]["system"])
            self.arbiter_override.setPlainText(PROMPT_PACKS[p]["arbiter"])
            self._prompt_edits.pop(p, None)

        self.btn_prm_defaults.clicked.connect(_defaults_clicked)
        self.btn_prm_clear.clicked.connect(_clear_clicked)

        # init prompts editor content
        _load_pack_prompts(self.prompts_pack.currentText())

        # Memory tab
        mem = QtWidgets.QWidget()
        mem_l = QtWidgets.QVBoxLayout(mem)
        mem_l.setContentsMargins(0, 0, 0, 0)
        mem_top = QtWidgets.QHBoxLayout()
        self.auto_memory = QtWidgets.QCheckBox("Auto-extract cross-dialog memories")
        self.auto_memory.setChecked((db.get_setting("auto_memory_enabled", "0") or "0") == "1")
        mem_top.addWidget(self.auto_memory)
        mem_top.addStretch(1)
        mem_top.addWidget(QtWidgets.QLabel("Max items"))
        self.mem_max = QtWidgets.QLineEdit(db.get_setting("global_memory_max_items", "12") or "12")
        self.mem_max.setFixedWidth(80)
        mem_top.addWidget(self.mem_max)
        mem_l.addLayout(mem_top)

        # Quick manual memory input (paste text and add)
        quick = QtWidgets.QGroupBox("Add memory")
        quick_l = QtWidgets.QVBoxLayout(quick)
        self.mem_quick_text = QtWidgets.QPlainTextEdit()
        self.mem_quick_text.setTabChangesFocus(True)
        self.mem_quick_text.setPlaceholderText(
            "Paste an important memory here (preferences, constraints, ongoing projects)."
        )
        self.mem_quick_text.setMinimumHeight(110)
        quick_l.addWidget(self.mem_quick_text)

        quick_row = QtWidgets.QHBoxLayout()
        self.mem_quick_tags = QtWidgets.QLineEdit()
        self.mem_quick_tags.setPlaceholderText("tags (optional)")
        self.mem_quick_pin = QtWidgets.QCheckBox("Pin")
        self.mem_quick_enable = QtWidgets.QCheckBox("Enabled")
        self.mem_quick_enable.setChecked(True)
        self.mem_quick_add = QtWidgets.QPushButton("Add")
        self.mem_quick_add.setObjectName("PrimaryBtn")
        quick_row.addWidget(self.mem_quick_tags, 1)
        quick_row.addWidget(self.mem_quick_pin)
        quick_row.addWidget(self.mem_quick_enable)
        quick_row.addWidget(self.mem_quick_add)
        quick_l.addLayout(quick_row)

        mem_l.addWidget(quick)


        self.mem_table = QtWidgets.QTableWidget(0, 5)
        self.mem_table.setHorizontalHeaderLabels(["Enabled", "Pinned", "Memory", "Tags", "Source"])
        self.mem_table.horizontalHeader().setStretchLastSection(True)
        self.mem_table.verticalHeader().setVisible(False)
        self.mem_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.mem_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        mem_l.addWidget(self.mem_table, 1)

        mem_btns = QtWidgets.QHBoxLayout()
        self.btn_mem_add = QtWidgets.QPushButton("Add")
        self.btn_mem_edit = QtWidgets.QPushButton("Edit")
        self.btn_mem_del = QtWidgets.QPushButton("Delete")
        mem_btns.addWidget(self.btn_mem_add)
        mem_btns.addWidget(self.btn_mem_edit)
        mem_btns.addWidget(self.btn_mem_del)
        mem_btns.addStretch(1)
        mem_l.addLayout(mem_btns)
        tabs.addTab(mem, "Memory")

        self._mem_rows: list[dict] = []

        def _refresh_mem_table():
            self._mem_rows = self.db.list_global_memories(include_disabled=True)
            self.mem_table.setRowCount(0)
            for r in self._mem_rows:
                row = self.mem_table.rowCount()
                self.mem_table.insertRow(row)
                self.mem_table.setItem(row, 0, QtWidgets.QTableWidgetItem("✓" if int(r.get("enabled") or 0) == 1 else ""))
                self.mem_table.setItem(row, 1, QtWidgets.QTableWidgetItem("📌" if int(r.get("pinned") or 0) == 1 else ""))
                self.mem_table.setItem(row, 2, QtWidgets.QTableWidgetItem((r.get("content") or "")[:500]))
                self.mem_table.setItem(row, 3, QtWidgets.QTableWidgetItem((r.get("tags") or "")[:200]))
                self.mem_table.setItem(row, 4, QtWidgets.QTableWidgetItem((r.get("source") or "")[:50]))
            self.mem_table.resizeColumnsToContents()

        def _selected_mem_row() -> Optional[dict]:
            idxs = self.mem_table.selectionModel().selectedRows()
            if not idxs:
                return None
            row = int(idxs[0].row())
            if row < 0 or row >= len(self._mem_rows):
                return None
            return self._mem_rows[row]

        class MemoryEditDialog(QtWidgets.QDialog):
            def __init__(self, parent: QtWidgets.QWidget, existing: Optional[dict] = None):
                super().__init__(parent)
                self.setWindowTitle("Memory")
                self.existing = existing or {}
                l = QtWidgets.QVBoxLayout(self)
                form = QtWidgets.QFormLayout()
                self.content = QtWidgets.QPlainTextEdit()
                self.content.setPlaceholderText("Paste a memory here…")
                self.content.setPlainText(self.existing.get("content") or "")
                self.content.setMinimumHeight(180)
                self.tags = QtWidgets.QLineEdit(self.existing.get("tags") or "")
                self.tags.setPlaceholderText("tags (optional), comma-separated")
                self.enabled = QtWidgets.QCheckBox("Enabled")
                self.enabled.setChecked(int(self.existing.get("enabled") or 1) == 1)
                self.pinned = QtWidgets.QCheckBox("Pinned")
                self.pinned.setChecked(int(self.existing.get("pinned") or 0) == 1)
                form.addRow("Content", self.content)
                form.addRow("Tags", self.tags)
                form.addRow("", self.enabled)
                form.addRow("", self.pinned)
                l.addLayout(form)
                bb = QtWidgets.QDialogButtonBox(
                    QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
                )
                bb.accepted.connect(self.accept)
                bb.rejected.connect(self.reject)
                l.addWidget(bb)
        
            def values(self) -> dict:
                return {
                    "content": self.content.toPlainText().strip(),
                    "tags": self.tags.text().strip(),
                    "enabled": self.enabled.isChecked(),
                    "pinned": self.pinned.isChecked(),
                }

        # Apply initial tab selection if provided
        if getattr(self, "_initial_tab", None):
            _want = str(self._initial_tab).strip().lower()
            for _i in range(tabs.count()):
                if tabs.tabText(_i).strip().lower() == _want:
                    tabs.setCurrentIndex(_i)
                    break

                bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
                bb.accepted.connect(self.accept)
                bb.rejected.connect(self.reject)
                l.addWidget(bb)

            def values(self) -> dict:
                return {
                    "content": self.content.toPlainText().strip(),
                    "tags": self.tags.text().strip(),
                    "enabled": self.enabled.isChecked(),
                    "pinned": self.pinned.isChecked(),
                }

        def _mem_quick_add():
            txt = (self.mem_quick_text.toPlainText() or "").strip()
            if not txt:
                return
            tags = (self.mem_quick_tags.text() or "").strip()
            pinned = self.mem_quick_pin.isChecked()
            enabled = self.mem_quick_enable.isChecked()
            self.db.add_global_memory(txt, tags=tags, pinned=pinned, enabled=enabled, source="manual")
            self.mem_quick_text.clear()
            _refresh_mem_table()
        def _mem_add():
            d = MemoryEditDialog(self)
            if d.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return
            v = d.values()
            if not v["content"]:
                return
            self.db.add_global_memory(v["content"], tags=v["tags"], pinned=v["pinned"], enabled=v["enabled"], source="manual")
            _refresh_mem_table()

        def _mem_edit():
            r = _selected_mem_row()
            if not r:
                return
            d = MemoryEditDialog(self, existing=r)
            if d.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return
            v = d.values()
            if not v["content"]:
                return
            self.db.update_global_memory(int(r["id"]), content=v["content"], tags=v["tags"], pinned=v["pinned"], enabled=v["enabled"])
            _refresh_mem_table()

        def _mem_del():
            r = _selected_mem_row()
            if not r:
                return
            if QtWidgets.QMessageBox.question(self, "Confirm", f"Delete memory #{r['id']}?") != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            self.db.delete_global_memory(int(r["id"]))
            _refresh_mem_table()

        self.btn_mem_add.clicked.connect(_mem_add)
        self.btn_mem_edit.clicked.connect(_mem_edit)
        self.btn_mem_del.clicked.connect(_mem_del)
        self.mem_quick_add.clicked.connect(_mem_quick_add)
        _refresh_mem_table()


        # RAG tab
        rag = QtWidgets.QWidget()
        rag_l = QtWidgets.QFormLayout(rag)
        self.rag_mode = QtWidgets.QComboBox()
        self.rag_mode.addItems(["lexical", "bm25", "faiss"])
        self.rag_mode.setCurrentText(db.get_setting("rag_retrieval_mode", "bm25") or "bm25")
        self.rag_topk = QtWidgets.QLineEdit(db.get_setting("rag_top_k", "6") or "6")
        self.rag_chunk = QtWidgets.QLineEdit(db.get_setting("rag_chunk_chars", "1200") or "1200")
        self.rag_overlap = QtWidgets.QLineEdit(db.get_setting("rag_chunk_overlap_chars", "200") or "200")
        self.rag_emb = QtWidgets.QLineEdit(db.get_setting("rag_embedding_model", "sentence-transformers/all-MiniLM-L6-v2") or "sentence-transformers/all-MiniLM-L6-v2")

        rebuild = QtWidgets.QPushButton("Rebuild FAISS index")
        rebuild.clicked.connect(self._rebuild_index)

        rag_l.addRow("Retrieval mode", self.rag_mode)
        rag_l.addRow("Top-k", self.rag_topk)
        rag_l.addRow("Chunk chars", self.rag_chunk)
        rag_l.addRow("Overlap chars", self.rag_overlap)
        rag_l.addRow("Embedding model", self.rag_emb)
        rag_l.addRow("", rebuild)
        tabs.addTab(rag, "RAG")

        # Voice tab
        voice = QtWidgets.QWidget()
        voice_l = QtWidgets.QFormLayout(voice)

        self.voice_input_enabled = QtWidgets.QCheckBox("Enable voice input (mic)")
        self.voice_input_enabled.setChecked((db.get_setting("voice_input_enabled", "0") or "0") == "1")

        self.voice_auto_send = QtWidgets.QCheckBox("Auto-send after transcription")
        self.voice_auto_send.setChecked((db.get_setting("voice_auto_send", "0") or "0") == "1")

        self.voice_stt_model = QtWidgets.QLineEdit(db.get_setting("voice_stt_model", "whisper-1") or "whisper-1")
        self.voice_stt_language = QtWidgets.QLineEdit(db.get_setting("voice_stt_language", "") or "")
        self.voice_stt_prompt = QtWidgets.QLineEdit(db.get_setting("voice_stt_prompt", "") or "")

        self.tts_enabled = QtWidgets.QCheckBox("Speak Miriam answers (TTS)")
        self.tts_enabled.setChecked((db.get_setting("tts_enabled", "0") or "0") == "1")
        self.tts_rate = QtWidgets.QLineEdit(db.get_setting("tts_rate", "175") or "175")

        # On clean macOS installs (no Homebrew), native audio libs are often absent.
        # Keep the app/build usable by disabling voice toggles when deps are missing.
        if not _optional_import_ok(["sounddevice", "soundfile"]):
            self.voice_input_enabled.setChecked(False)
            self.voice_input_enabled.setEnabled(False)
            self.voice_input_enabled.setToolTip("Voice input requires optional audio dependencies (sounddevice/soundfile).")

        if not _optional_import_ok(["pyttsx3"]):
            self.tts_enabled.setChecked(False)
            self.tts_enabled.setEnabled(False)
            self.tts_enabled.setToolTip("TTS requires optional dependency pyttsx3.")

        voice_l.addRow("", self.voice_input_enabled)
        voice_l.addRow("", self.voice_auto_send)
        voice_l.addRow("STT model", self.voice_stt_model)
        voice_l.addRow("STT language (optional)", self.voice_stt_language)
        voice_l.addRow("STT prompt (optional)", self.voice_stt_prompt)
        voice_l.addRow("", self.tts_enabled)
        voice_l.addRow("TTS rate", self.tts_rate)
        tabs.addTab(voice, "Voice")

        # Buttons
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save | QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(tabs)
        lay.addWidget(buttons)
        self.resize(540, 420)

    def _rebuild_index(self):
        self.db.set_setting("rag_index_dirty", "1")
        QtWidgets.QMessageBox.information(self, "RAG", "Index marked for rebuild. It will rebuild on next retrieval.")

    def _save(self):
        # Provider
        self.db.set_setting("provider_mode", self.provider_mode.currentText())
        self.db.set_setting("novita_base_url", self.novita_base.text().strip())
        self.db.set_setting("novita_api_key", self.novita_key.text().strip())
        self.db.set_setting("openrouter_base_url", self.openrouter_base.text().strip())
        self.db.set_setting("openrouter_api_key", self.openrouter_key.text().strip())
        self.db.set_setting("custom_base_url", self.custom_base.text().strip())
        self.db.set_setting("custom_api_key", self.custom_key.text().strip())

        # Models
        # Novita
        self.db.set_setting("novita_model_light", self.novita_model_light.text().strip())
        self.db.set_setting("novita_model_medium", self.novita_model_medium.text().strip())
        self.db.set_setting("novita_model_heavy", self.novita_model_heavy.text().strip())
        self.db.set_setting("novita_model_arbiter", self.novita_model_arbiter.text().strip())

        # OpenRouter
        self.db.set_setting("openrouter_model_light", self.openrouter_model_light.text().strip())
        self.db.set_setting("openrouter_model_medium", self.openrouter_model_medium.text().strip())
        self.db.set_setting("openrouter_model_heavy", self.openrouter_model_heavy.text().strip())
        self.db.set_setting("openrouter_model_arbiter", self.openrouter_model_arbiter.text().strip())

        # Custom/generic
        self.db.set_setting("model_light", self.model_light.text().strip())
        self.db.set_setting("model_medium", self.model_medium.text().strip())
        self.db.set_setting("model_heavy", self.model_heavy.text().strip())
        self.db.set_setting("model_arbiter", self.model_arbiter.text().strip())

        # Generation params
        self.db.set_setting("temperature", self.temperature.text().strip())
        self.db.set_setting("max_tokens", self.max_tokens.text().strip())
        self.db.set_setting("history_max_messages", self.history_max.text().strip())

        # Behavior
        self.db.set_setting("prompt_pack", self.prompt_pack.currentText())
        self.db.set_setting("reasoning_mode", self.reasoning_mode.currentText())

        # Prompt overrides (optional)
        try:
            pcur = self.prompts_pack.currentText() or self.prompt_pack.currentText() or "gp"
            self._prompt_edits[pcur] = (self.system_override.toPlainText(), self.arbiter_override.toPlainText())
            for p, (sys_txt, arb_txt) in self._prompt_edits.items():
                if p not in PROMPT_PACKS:
                    continue
                sys_txt = (sys_txt or "").strip()
                arb_txt = (arb_txt or "").strip()
                if sys_txt == PROMPT_PACKS[p]["system"] and arb_txt == PROMPT_PACKS[p]["arbiter"]:
                    self.db.delete_prompt_override(p)
                else:
                    self.db.set_prompt_override(p, sys_txt, arb_txt)
        except Exception:
            pass

        # Global memory settings
        self.db.set_setting("auto_memory_enabled", "1" if self.auto_memory.isChecked() else "0")
        self.db.set_setting("global_memory_max_items", self.mem_max.text().strip())

        # RAG
        self.db.set_setting("rag_retrieval_mode", self.rag_mode.currentText())
        self.db.set_setting("rag_top_k", self.rag_topk.text().strip())
        self.db.set_setting("rag_chunk_chars", self.rag_chunk.text().strip())
        self.db.set_setting("rag_chunk_overlap_chars", self.rag_overlap.text().strip())
        self.db.set_setting("rag_embedding_model", self.rag_emb.text().strip())

        # Voice
        self.db.set_setting("voice_input_enabled", "1" if self.voice_input_enabled.isChecked() else "0")
        self.db.set_setting("voice_auto_send", "1" if self.voice_auto_send.isChecked() else "0")
        self.db.set_setting("voice_stt_model", self.voice_stt_model.text().strip())
        self.db.set_setting("voice_stt_language", self.voice_stt_language.text().strip())
        self.db.set_setting("voice_stt_prompt", self.voice_stt_prompt.text().strip())
        self.db.set_setting("tts_enabled", "1" if self.tts_enabled.isChecked() else "0")
        self.db.set_setting("tts_rate", self.tts_rate.text().strip())

        self.accept()


class HistoDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget, db: Database, dialog_id: int):
        super().__init__(parent)
        self.setWindowTitle("Histo metadata")
        self.db = db
        self.dialog_id = dialog_id
        st = db.histo_get(dialog_id)

        form = QtWidgets.QFormLayout()
        self.stain = QtWidgets.QLineEdit(st.get("stain", ""))
        self.mag = QtWidgets.QLineEdit(st.get("magnification", ""))
        self.quality = QtWidgets.QLineEdit(st.get("quality", ""))
        self.note = QtWidgets.QLineEdit(st.get("note", ""))
        form.addRow("Stain", self.stain)
        form.addRow("Magnification", self.mag)
        form.addRow("Quality", self.quality)
        form.addRow("Note", self.note)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save | QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(buttons)
        self.resize(520, 240)

    def _save(self):
        self.db.histo_set(
            self.dialog_id,
            stain=self.stain.text().strip(),
            magnification=self.mag.text().strip(),
            quality=self.quality.text().strip(),
            note=self.note.text().strip(),
        )
        self.accept()


class RadiologyIngestWorker(QtCore.QObject):
    done = QtCore.Signal(int, int)  # study_id, n_previews
    progress = QtCore.Signal(int, str, str)  # pct, message, preview_path
    error = QtCore.Signal(str)

    def __init__(
        self,
        db: Database,
        dialog_id: int,
        source_path: str,
        series_uid: str,
        modality: str,
        body_region: str,
        contrast: str,
        clinical_question: str,
    ):
        super().__init__()
        self.db = db
        self.dialog_id = int(dialog_id)
        self.source_path = source_path
        self.series_uid = (series_uid or "").strip()
        self.modality = modality
        self.body_region = body_region
        self.contrast = contrast
        self.clinical_question = clinical_question

    @QtCore.Slot()
    def run(self):
        try:
            from .core.radiology import ingest_imaging_study

            def _cb(pct: int, msg: str, preview_path: str = "") -> None:
                # Runs in worker thread; Qt will queue the signal to UI thread.
                self.progress.emit(int(pct), str(msg or ""), str(preview_path or ""))

            res = ingest_imaging_study(
                self.db,
                dialog_id=self.dialog_id,
                source_path=self.source_path,
                series_uid=self.series_uid,
                modality=self.modality,
                body_region=self.body_region,
                contrast=self.contrast,
                clinical_question=self.clinical_question,
                progress_cb=_cb,
            )
            self.done.emit(int(res.study_id), int(res.n_previews))
        except Exception as e:
            self.error.emit(str(e))


class RadiologySeriesScanWorker(QtCore.QObject):
    """Scan a DICOM folder/zip and return available SeriesInstanceUIDs."""

    done = QtCore.Signal(list)  # list of dicts
    progress = QtCore.Signal(int, str)
    error = QtCore.Signal(str)

    def __init__(self, source_path: str):
        super().__init__()
        self.source_path = source_path

    @QtCore.Slot()
    def run(self):
        try:
            from .core.radiology import scan_dicom_series

            def _cb(pct: int, msg: str, _preview: str = "") -> None:
                self.progress.emit(int(pct), str(msg or ""))

            series = scan_dicom_series(self.source_path, progress_cb=_cb)
            self.done.emit(series)
        except Exception as e:
            self.error.emit(str(e))


class RadiologyDialog(QtWidgets.QDialog):
    """Import a radiology study (CT/MRI) and generate preview slices."""

    def __init__(self, parent: QtWidgets.QWidget, db: Database, dialog_id: int):
        super().__init__(parent)
        self.setWindowTitle("Радиология: импорт исследования")
        self.db = db
        self.dialog_id = int(dialog_id)
        self._source_path = ""

        form = QtWidgets.QFormLayout()

        self.source = QtWidgets.QLineEdit("")
        self.source.setReadOnly(True)
        btn_dir = QtWidgets.QPushButton("Выбрать папку DICOM…")
        btn_file = QtWidgets.QPushButton("Выбрать ZIP/NIfTI…")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.source, 1)
        row.addWidget(btn_file)
        row.addWidget(btn_dir)

        hint = QtWidgets.QLabel(
            "Совет: для DICOM выбери папку, которая содержит исследование (сессии/серии). "
            "Приложение сканирует подпапки рекурсивно и генерирует preview-срезы. "
            "Для переносимости папки внутри приложения сохраняются как ZIP."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: rgba(215,227,243,0.65);")

        # Series selector (only for DICOM folder/zip). Filled after scanning headers.
        self.series = QtWidgets.QComboBox()
        self.series.addItem("Автовыбор (лучшая серия)", "")
        self.series.setEnabled(False)
        self.btn_scan_series = QtWidgets.QPushButton("Сканировать серии")
        self.btn_scan_series.setEnabled(False)
        self.series_status = QtWidgets.QLabel("Серия: (не просканировано)")
        self.series_status.setStyleSheet("color: rgba(215,227,243,0.65); font-size: 11px;")
        series_row = QtWidgets.QHBoxLayout()
        series_row.addWidget(self.series, 1)
        series_row.addWidget(self.btn_scan_series)

        self.modality = QtWidgets.QComboBox()
        self.modality.addItems(["", "CT", "MR", "XR", "US", "PET", "NM"])
        self.body_region = QtWidgets.QLineEdit("")
        self.contrast = QtWidgets.QComboBox()
        self.contrast.addItems(["", "без контраста", "с контрастом", "неизвестно"])
        self.question = QtWidgets.QLineEdit("")

        self.chk_switch_pack = QtWidgets.QCheckBox("Переключить режим ассистента на RAD")
        self.chk_switch_pack.setChecked(True)

        self.chk_auto_report = QtWidgets.QCheckBox("Сразу сформировать подробный отчёт (LLM)")
        self.chk_auto_report.setChecked(True)

        form.addRow("Источник", row)
        form.addRow("", hint)
        form.addRow("Серия DICOM", series_row)
        form.addRow("", self.series_status)
        form.addRow("Модальность", self.modality)
        form.addRow("Область", self.body_region)
        form.addRow("Контраст", self.contrast)
        form.addRow("Клинический вопрос", self.question)
        form.addRow("", self.chk_switch_pack)
        form.addRow("", self.chk_auto_report)

        # Progress + previews
        self.progress_label = QtWidgets.QLabel("")
        self.progress_label.setStyleSheet("color: rgba(215,227,243,0.75);")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        self._thumb_grid = QtWidgets.QGridLayout()
        self._thumb_grid.setHorizontalSpacing(10)
        self._thumb_grid.setVerticalSpacing(10)
        self._thumb_grid.setContentsMargins(0, 0, 0, 0)

        thumb_wrap = QtWidgets.QWidget()
        thumb_wrap.setLayout(self._thumb_grid)
        self.thumb_scroll = QtWidgets.QScrollArea()
        self.thumb_scroll.setWidgetResizable(True)
        self.thumb_scroll.setWidget(thumb_wrap)
        self.thumb_scroll.setMinimumHeight(180)
        self.thumb_scroll.setStyleSheet("QScrollArea{border:1px solid rgba(255,255,255,0.06);border-radius:12px;background:rgba(255,255,255,0.02);}")

        self._thumb_widgets: list[QtWidgets.QWidget] = []

        self.btn_import = QtWidgets.QPushButton("Импорт")
        self.btn_close = QtWidgets.QPushButton("Закрыть")
        self.btn_import.clicked.connect(self._import)
        self.btn_close.clicked.connect(self.reject)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_import)
        btn_row.addWidget(self.btn_close)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.progress_label)
        lay.addWidget(self.progress_bar)
        lay.addWidget(self.thumb_scroll)
        lay.addLayout(btn_row)
        self.resize(900, 520)

        btn_file.clicked.connect(self._pick_file)
        btn_dir.clicked.connect(self._pick_dir)
        self.btn_scan_series.clicked.connect(self._start_scan_series)

        self._thread: QtCore.QThread | None = None
        self._worker: RadiologyIngestWorker | None = None

        self._scan_thread: QtCore.QThread | None = None
        self._scan_worker: RadiologySeriesScanWorker | None = None

    def _pick_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Выберите ZIP с DICOM или файл NIfTI",
            "",
            "DICOM ZIP (*.zip);;NIfTI (*.nii *.nii.gz);;All files (*.*)",
        )
        if not path:
            return
        self._source_path = path
        self.source.setText(path)

        low = path.lower()
        is_zip = low.endswith(".zip")
        is_nifti = low.endswith(".nii") or low.endswith(".nii.gz")
        self.series.setEnabled(is_zip)
        self.btn_scan_series.setEnabled(is_zip)
        if is_zip:
            self._start_scan_series()
        else:
            # NIfTI or other: series selection not applicable
            self.series.clear()
            self.series.addItem("Автовыбор (лучшая серия)", "")
            self.series_status.setText("Серия: (не применимо)")

    def _pick_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку DICOM", "")
        if not path:
            return
        self._source_path = path
        self.source.setText(path)
        self.series.setEnabled(True)
        self.btn_scan_series.setEnabled(True)
        self._start_scan_series()

    def _start_scan_series(self):
        """Scan available DICOM series and fill the series selector.

        Runs in a background QThread to keep UI responsive.
        """
        if not self._source_path:
            return
        low = self._source_path.lower()
        if not (Path(self._source_path).is_dir() or low.endswith(".zip")):
            return

        # Cancel any previous scan thread.
        try:
            if self._scan_thread and self._scan_thread.isRunning():
                self._scan_thread.quit()
        except Exception:
            pass

        self.series_status.setText("Сканирую серии DICOM…")
        self.btn_scan_series.setEnabled(False)
        self.series.setEnabled(False)
        self.series.clear()
        self.series.addItem("Автовыбор (лучшая серия)", "")

        self._scan_thread = QtCore.QThread(self)
        self._scan_worker = RadiologySeriesScanWorker(self._source_path)
        self._scan_worker.moveToThread(self._scan_thread)
        self._scan_thread.started.connect(self._scan_worker.run)
        self._scan_worker.progress.connect(self._on_scan_progress)
        self._scan_worker.done.connect(self._on_scan_done)
        self._scan_worker.error.connect(self._on_scan_error)
        self._scan_worker.done.connect(self._scan_thread.quit)
        self._scan_worker.error.connect(self._scan_thread.quit)
        self._scan_thread.finished.connect(self._cleanup_scan_thread)
        self._scan_thread.start()

    def _cleanup_scan_thread(self):
        try:
            if self._scan_thread:
                self._scan_thread.deleteLater()
        except Exception:
            pass
        self._scan_thread = None
        self._scan_worker = None

    def _on_scan_progress(self, pct: int, msg: str):
        if msg:
            self.series_status.setText(str(msg))

    def _on_scan_done(self, series_list: list):
        # series_list: list of dicts
        self.series.clear()
        self.series.addItem("Автовыбор (лучшая серия)", "")

        if not series_list:
            self.series_status.setText("Серия: не найдено DICOM-серий")
            self.series.setEnabled(False)
            self.btn_scan_series.setEnabled(True)
            return

        # Populate selector
        for s in series_list:
            uid = str(s.get("series_uid") or "")
            desc = str(s.get("series_desc") or "").strip()
            mod = str(s.get("modality") or "").strip().upper()
            n = int(s.get("n_slices") or 0)
            mat = s.get("matrix") or ""
            title = f"{mod or 'DICOM'}  {mat}  slices={n}"
            if desc:
                title += f"  | {desc}"
            # show shortened UID to keep it readable
            if uid:
                title += f"  ({uid[-8:]})"
            self.series.addItem(title, uid)

        self.series_status.setText(f"Серия: найдено {len(series_list)}")
        self.series.setEnabled(True)
        self.btn_scan_series.setEnabled(True)

    def _on_scan_error(self, err: str):
        self.series_status.setText("Серия: ошибка сканирования")
        self.series.setEnabled(False)
        self.btn_scan_series.setEnabled(True)
        QtWidgets.QMessageBox.warning(self, "Радиология", f"Не удалось просканировать серии:\n\n{err}")

    def _import(self):
        if not self._source_path:
            QtWidgets.QMessageBox.information(self, "Радиология", "Выберите папку DICOM, ZIP с DICOM или файл NIfTI.")
            return

        self.btn_import.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Импортирую…")
        self._clear_thumbnails()

        self._thread = QtCore.QThread(self)
        self._worker = RadiologyIngestWorker(
            self.db,
            self.dialog_id,
            self._source_path,
            (self.series.currentData() or "").strip(),
            (self.modality.currentText() or "").strip(),
            self.body_region.text().strip(),
            (self.contrast.currentText() or "").strip(),
            self.question.text().strip(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.done.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    def _cleanup_thread(self):
        try:
            if self._thread:
                self._thread.deleteLater()
        except Exception:
            pass
        self._thread = None
        self._worker = None

    def _clear_thumbnails(self):
        for w in self._thumb_widgets:
            try:
                w.deleteLater()
            except Exception:
                pass
        self._thumb_widgets.clear()
        while self._thumb_grid.count():
            it = self._thumb_grid.takeAt(0)
            if it and it.widget():
                try:
                    it.widget().deleteLater()
                except Exception:
                    pass

    def _add_thumbnail(self, img_path: str, label: str = ""):
        p = (img_path or "").strip()
        if not p:
            return
        try:
            pix = QtGui.QPixmap(p)
        except Exception:
            return
        if pix.isNull():
            return

        card = QtWidgets.QFrame()
        card.setStyleSheet(
            "QFrame{border:1px solid rgba(255,255,255,0.07);border-radius:12px;background:rgba(255,255,255,0.02);}"
        )
        v = QtWidgets.QVBoxLayout(card)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        img = QtWidgets.QLabel()
        img.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        img.setPixmap(pix.scaled(180, 180, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation))
        cap = QtWidgets.QLabel(label or "")
        cap.setWordWrap(True)
        cap.setStyleSheet("color: rgba(215,227,243,0.75); font-size: 11px;")
        v.addWidget(img)
        if label:
            v.addWidget(cap)

        idx = len(self._thumb_widgets)
        r = idx // 4
        c = idx % 4
        self._thumb_grid.addWidget(card, r, c)
        self._thumb_widgets.append(card)

    def _on_progress(self, pct: int, msg: str, preview_path: str):
        try:
            self.progress_bar.setValue(int(max(0, min(100, pct))))
        except Exception:
            pass
        if msg:
            self.progress_label.setText(str(msg))
        if preview_path:
            # Add thumbnail for newly created preview
            try:
                self._add_thumbnail(preview_path, "")
            except Exception:
                pass

    def _on_done(self, study_id: int, n_previews: int):
        self.progress_bar.setValue(100)
        self.progress_label.setText(f"Импортировано исследование №{study_id}. Preview: {n_previews}.")
        self.btn_import.setEnabled(True)
        if self.chk_switch_pack.isChecked():
            try:
                self.db.set_setting("prompt_pack", "radio")
            except Exception:
                pass

        # Optionally trigger an immediate detailed report.
        if self.chk_auto_report.isChecked():
            try:
                parent = self.parent()
                if parent is not None and hasattr(parent, "send_programmatic"):
                    q = (self.question.text() or "").strip()
                    user_prompt = (
                        "🩻 Авто-отчёт по импортированному исследованию.\n"
                        "Сформируй максимально подробное ориентировочное описание по предпросмотрам и метаданным.\n"
                        "1) Краткое резюме.\n"
                        "2) Подробное описание видимых находок/структур.\n"
                        "3) Возможные ROI (зоны интереса) с привязкой к анатомии: где и почему подозрительно.\n"
                        "4) Дифдиагноз ТОП-3 с аргументами.\n"
                        "5) Что уточнить (серии/параметры/клиника) и как подтвердить/исключить.\n"
                        "Обязательно: не давай полный отказ и не пиши, что 'только врач может'."
                    )
                    if q:
                        user_prompt += f"\n\nКлинический вопрос: {q}"
                    parent.send_programmatic(user_prompt)
            except Exception:
                pass

            # Close the dialog to return focus to chat.
            self.accept()
            return

        QtWidgets.QMessageBox.information(
            self,
            "Радиология",
            f"Исследование импортировано. Сгенерировано preview: {n_previews}.\n\nТеперь можно спрашивать, например: «Что видно на КТ/МРТ?»",
        )

    def _on_error(self, err: str):
        self.progress_label.setText("Импорт не удался")
        self.btn_import.setEnabled(True)
        QtWidgets.QMessageBox.warning(self, "Радиология", f"Не удалось импортировать исследование:\n\n{err}")


class RadiologyStudiesDialog(QtWidgets.QDialog):
    """Manage radiology studies for a chat: switch active, open viewer, delete."""

    def __init__(self, parent: QtWidgets.QWidget, db: Database, dialog_id: int):
        super().__init__(parent)
        self.setWindowTitle("Radiology studies")
        self.db = db
        self.dialog_id = int(dialog_id)

        self.list = QtWidgets.QListWidget()

        btn_set_active = QtWidgets.QPushButton("Set active")
        btn_view = QtWidgets.QPushButton("Open viewer")
        btn_delete = QtWidgets.QPushButton("Delete")
        btn_close = QtWidgets.QPushButton("Close")

        btn_set_active.clicked.connect(self._set_active)
        btn_view.clicked.connect(self._open_viewer)
        btn_delete.clicked.connect(self._delete)
        btn_close.clicked.connect(self.reject)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(btn_set_active)
        row.addWidget(btn_view)
        row.addWidget(btn_delete)
        row.addStretch(1)
        row.addWidget(btn_close)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.list)
        lay.addLayout(row)

        self.resize(860, 420)
        self._refresh()

    def _refresh(self):
        studies = self.db.imaging_list_studies(self.dialog_id)
        active = self.db.imaging_get_active(self.dialog_id)
        active_id = int(active.get("id")) if active else None

        self._items = studies
        self.list.clear()
        for s in studies:
            sid = int(s.get("id"))
            tag = "ACTIVE" if active_id == sid else "     "
            mod = (s.get("modality") or "").upper()
            desc = (s.get("meta") or {}).get("series_desc") or (s.get("meta") or {}).get("study_desc") or ""
            q = (s.get("clinical_question") or "").strip()
            created = str(s.get("created_at") or "")
            line = f"[{tag}]  #{sid}  {mod or 'DICOM'}  {created}"
            if desc:
                line += f"  | {desc}"
            if q:
                line += f"  | Q: {q}"
            self.list.addItem(line)

    def _selected(self) -> dict[str, Any] | None:
        idx = self.list.currentRow()
        if idx < 0 or idx >= len(self._items):
            return None
        return self._items[idx]

    def _set_active(self):
        s = self._selected()
        if not s:
            return
        self.db.imaging_set_active(self.dialog_id, int(s["id"]))
        self._refresh()

    def _open_viewer(self):
        s = self._selected()
        if not s:
            return
        dlg = RadiologyViewerDialog(self, self.db, int(s["id"]))
        dlg.exec()

    def _delete(self):
        s = self._selected()
        if not s:
            return
        sid = int(s["id"])
        if QtWidgets.QMessageBox.question(
            self,
            "Delete study",
            f"Delete study #{sid}? Previews will be removed from DB (files remain in store).",
        ) != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.db.imaging_delete_study(sid)
        self._refresh()


class RadiologyViewerDialog(QtWidgets.QDialog):
    """Simple viewer: slice scrolling, CT WL/WW, and draft MPR."""

    def __init__(self, parent: QtWidgets.QWidget, db: Database, study_id: int):
        super().__init__(parent)
        self.setWindowTitle(f"Radiology viewer · Study #{study_id}")
        self.db = db
        self.study_id = int(study_id)

        self._volume = None
        self._meta: dict[str, Any] = {}

        self.plane = QtWidgets.QComboBox()
        self.plane.addItems(["Axial", "Coronal", "Sagittal"])

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._render)

        self.lbl = QtWidgets.QLabel()
        self.lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl.setMinimumSize(640, 480)

        # CT window controls
        self.wl = QtWidgets.QSpinBox()
        self.wl.setRange(-2000, 3000)
        self.ww = QtWidgets.QSpinBox()
        self.ww.setRange(1, 5000)
        self.wl.setValue(40)
        self.ww.setValue(400)
        self.wl.valueChanged.connect(self._render)
        self.ww.valueChanged.connect(self._render)

        btn_medi = QtWidgets.QPushButton("CT Mediastinum")
        btn_lung = QtWidgets.QPushButton("CT Lung")
        btn_bone = QtWidgets.QPushButton("CT Bone")
        btn_medi.clicked.connect(lambda: self._set_window(40, 400))
        btn_lung.clicked.connect(lambda: self._set_window(-600, 1500))
        btn_bone.clicked.connect(lambda: self._set_window(300, 1500))

        self.info = QtWidgets.QLabel("Loading…")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color: rgba(215,227,243,0.7);")

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Plane"))
        top.addWidget(self.plane)
        top.addSpacing(12)
        top.addWidget(QtWidgets.QLabel("WL"))
        top.addWidget(self.wl)
        top.addWidget(QtWidgets.QLabel("WW"))
        top.addWidget(self.ww)
        top.addWidget(btn_medi)
        top.addWidget(btn_lung)
        top.addWidget(btn_bone)
        top.addStretch(1)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.slider)
        lay.addWidget(self.lbl, 1)
        lay.addWidget(self.info)

        self.plane.currentIndexChanged.connect(self._update_slider)

        self.resize(1100, 780)
        self._load()

    def _set_window(self, wl: int, ww: int):
        self.wl.setValue(int(wl))
        self.ww.setValue(int(ww))

    def _load(self):
        try:
            from .core.radiology import load_volume_for_viewer

            vol, meta = load_volume_for_viewer(self.db, self.study_id)
            self._volume = vol
            self._meta = meta or {}
        except Exception as e:
            self.info.setText("Failed to load: " + str(e))
            return

        modality = (self._meta.get("Modality") or "").upper()
        if modality != "CT":
            # WL/WW only for CT
            self.wl.setEnabled(False)
            self.ww.setEnabled(False)

        self._update_slider()

        # info
        shape = getattr(self._volume, "shape", None)
        uid = self._meta.get("series_uid") or ""
        self.info.setText(
            f"Draft viewer. Modality: {modality or 'unknown'} · shape: {shape} · series: {str(uid)[-8:]}\n"
            "Note: MPR is a rough re-slice without full DICOM orientation handling."
        )

    def _update_slider(self):
        if self._volume is None:
            return
        vol = self._volume
        plane = self.plane.currentText()
        if plane == "Axial":
            n = vol.shape[0]
        elif plane == "Coronal":
            n = vol.shape[1]
        else:
            n = vol.shape[2]
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, int(n) - 1))
        self.slider.setValue(int(n // 2) if n > 0 else 0)
        self.slider.blockSignals(False)
        self._render()

    def _slice2d(self, plane: str, idx: int) -> np.ndarray:
        vol = self._volume
        assert vol is not None
        if plane == "Axial":
            img = vol[int(idx), :, :]
            # display-friendly orientation
            return np.rot90(img)
        if plane == "Coronal":
            img = vol[:, int(idx), :]
            return np.rot90(img)
        img = vol[:, :, int(idx)]
        return np.rot90(img)

    def _render(self):
        if self._volume is None:
            return
        plane = self.plane.currentText()
        idx = int(self.slider.value())
        img = self._slice2d(plane, idx)

        from .core.radiology import _to_uint8_percentile, _to_uint8_windowed

        modality = (self._meta.get("Modality") or "").upper()
        if modality == "CT":
            img8 = _to_uint8_windowed(img, wl=float(self.wl.value()), ww=float(self.ww.value()))
        else:
            img8 = _to_uint8_percentile(img)

        h, w = img8.shape
        qimg = QtGui.QImage(img8.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.lbl.setPixmap(
            pix.scaled(self.lbl.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
        )


class RagDocsDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget, db: Database):
        super().__init__(parent)
        self.setWindowTitle("RAG docs")
        self.db = db

        self.list = QtWidgets.QListWidget()

        add_btn = QtWidgets.QPushButton("Add")
        toggle_btn = QtWidgets.QPushButton("Toggle")
        del_btn = QtWidgets.QPushButton("Delete")
        close_btn = QtWidgets.QPushButton("Close")

        add_btn.clicked.connect(self._add)
        toggle_btn.clicked.connect(self._toggle)
        del_btn.clicked.connect(self._delete)
        close_btn.clicked.connect(self.reject)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(add_btn)
        row.addWidget(toggle_btn)
        row.addWidget(del_btn)
        row.addStretch(1)
        row.addWidget(close_btn)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.list)
        lay.addLayout(row)
        self.resize(760, 360)
        self._refresh()

    def _refresh(self):
        self.docs = self.db.rag_list_docs()
        self.list.clear()
        for d in self.docs:
            flag = "ON" if d.get("enabled") else "OFF"
            self.list.addItem(f"[{flag}] #{d['id']}  {d['title']}  ({d['path']})")

    def _selected(self):
        idx = self.list.currentRow()
        if idx < 0 or idx >= len(self.docs):
            return None
        return self.docs[idx]

    def _add(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Add document",
            "",
            "Docs (*.pdf *.docx *.txt *.rtf *.md);;All files (*.*)",
        )
        if not paths:
            return
        # Mark dirty; actual ingestion is done by engine helper in main window
        self.parent().add_rag_docs(paths)  # type: ignore
        self._refresh()

    def _toggle(self):
        d = self._selected()
        if not d:
            return
        self.db.rag_set_doc_enabled(int(d["id"]), not bool(d.get("enabled")))
        self._refresh()

    def _delete(self):
        d = self._selected()
        if not d:
            return
        if QtWidgets.QMessageBox.question(self, "Confirm", f"Delete doc #{d['id']}?") != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.db.rag_delete_doc(int(d["id"]))
        self._refresh()


# ------------------------------ Worker ------------------------------


class StreamWorker(QtCore.QObject):
    delta = QtCore.Signal(str)
    done = QtCore.Signal(str, str, str)  # final_text, provider, model
    error = QtCore.Signal(str)

    def __init__(self, engine: MiriamEngine, dialog_id: int, user_text: str, image_paths: list[str]):
        super().__init__()
        self.engine = engine
        self.dialog_id = dialog_id
        self.user_text = user_text
        self.image_paths = image_paths

    @QtCore.Slot()
    def run(self):
        try:
            # Guard against boilerplate refusals (common for radiology image prompts on some providers).
            # We buffer the first part before streaming it into the UI; if it looks like a refusal, we retry once
            # with a stronger non-refusal instruction and then emit the useful answer.
            provider, model, it, _cands = self.engine.generate_stream(
                self.dialog_id, self.user_text, image_paths=self.image_paths
            )

            prebuf: list[str] = []
            buf: list[str] = []
            streaming_started = False

            def _looks_like_refusal(txt: str) -> bool:
                try:
                    from panacea_desktop.core.engine import looks_like_refusal

                    return looks_like_refusal(txt)
                except Exception:
                    t = (txt or "").lower()
                    return ("к сожалению" in t and "не могу" in t) or ("только квалифицирован" in t)

            for piece in it:
                if not piece:
                    continue
                buf.append(piece)

                if not streaming_started:
                    prebuf.append(piece)
                    head = "".join(prebuf)

                    # Wait until we have enough signal to classify the start of the answer.
                    # For refusals we want to catch them early (they're often short).
                    head_low = head.lower()
                    early_refusal_hint = ("к сожалению" in head_low and "не могу" in head_low) or (
                        "только квалифицирован" in head_low and ("специалист" in head_low or "врач" in head_low)
                    )
                    if (len(head) < 200 and "\n" not in head) and not early_refusal_hint:
                        continue

                    if _looks_like_refusal(head):
                        # Retry once with anti-refusal prompt (non-stream). Do not show the refusal to the user.
                        alt = self.engine.generate(
                            self.dialog_id,
                            self.user_text,
                            image_paths=self.image_paths,
                            anti_refusal=True,
                        )
                        final2 = (alt.text or "").strip()
                        if final2:
                            self.delta.emit(final2)
                            self.done.emit(final2, alt.provider_used, alt.model_used)
                            return
                        # If retry failed, fall back to original head.
                        self.delta.emit(head)
                        streaming_started = True
                        continue

                    # Not a refusal: start streaming normally (emit buffered head once).
                    self.delta.emit(head)
                    streaming_started = True
                    continue

                # Normal streaming after the initial guard.
                self.delta.emit(piece)

            final = "".join(buf) if streaming_started else "".join(prebuf)
            self.done.emit(final, provider, model)
        except Exception as e:
            self.error.emit(str(e))


class VoiceSignals(QtCore.QObject):
    transcript = QtCore.Signal(str)
    error = QtCore.Signal(str)


# ------------------------------ Main Window ------------------------------


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Miriam")
        self.db = Database()
        self.engine = MiriamEngine(self.db)
        self._pending_images: list[str] = []
        self._pending_docs: list[str] = []
        self._pending_imaging: list[str] = []  # DICOM ZIP / folder marker / NIfTI

        # Voice
        self._voice_signals = VoiceSignals()
        self._voice_signals.transcript.connect(self._on_voice_transcript)
        self._voice_signals.error.connect(self._on_voice_error)
        self._voice_recording = False
        self._voice_stop = threading.Event()
        self._voice_thread: threading.Thread | None = None
        self._voice_fs = 16000
        self._tts_lock = threading.Lock()
        self._tts_engine = None

        root = QtWidgets.QWidget()
        root_l = QtWidgets.QHBoxLayout(root)
        root_l.setContentsMargins(0, 0, 0, 0)
        root_l.setSpacing(0)

        # Sidebar
        self.sidebar = QtWidgets.QWidget(objectName="Sidebar")
        sb_l = QtWidgets.QVBoxLayout(self.sidebar)
        sb_l.setContentsMargins(18, 18, 18, 18)
        sb_l.setSpacing(12)

        self.btn_new = QtWidgets.QPushButton("+  New chat", objectName="NewChatBtn")
        self.btn_new.clicked.connect(self.new_chat)
        sb_l.addWidget(self.btn_new)

        self.search = QtWidgets.QLineEdit(objectName="Search")
        self.search.setPlaceholderText("Search")
        self.search.textChanged.connect(self._filter_chats)
        sb_l.addWidget(self.search)

        self.chat_list = QtWidgets.QListWidget(objectName="ChatList")
        self.chat_list.itemSelectionChanged.connect(self._on_chat_selected)
        sb_l.addWidget(self.chat_list, 1)

        root_l.addWidget(self.sidebar, 0)

        # Main panel
        main = QtWidgets.QWidget()
        main_l = QtWidgets.QVBoxLayout(main)
        main_l.setContentsMargins(24, 16, 24, 16)
        main_l.setSpacing(10)

        # Top bar
        top = QtWidgets.QWidget(objectName="TopBar")
        top_l = QtWidgets.QHBoxLayout(top)
        top_l.setContentsMargins(0, 0, 0, 0)
        top_l.setSpacing(14)

        top_l.addStretch(1)

        self.rag_switch = QtWidgets.QCheckBox("RAG", objectName="Switch")
        self.rag_switch.setChecked((self.db.get_setting("rag_enabled", "1") or "1") == "1")
        self.rag_switch.stateChanged.connect(self._toggle_rag)
        top_l.addWidget(self.rag_switch)

        self.mem_switch = QtWidgets.QCheckBox("Memory", objectName="Switch")
        self.mem_switch.setChecked((self.db.get_setting("memory_enabled", "1") or "1") == "1")
        self.mem_switch.stateChanged.connect(self._toggle_memory)
        top_l.addWidget(self.mem_switch)

        self.btn_mem_edit = QtWidgets.QToolButton(objectName="IconBtn")
        self.btn_mem_edit.setText("🧠")
        self.btn_mem_edit.setToolTip("Edit memories")
        self.btn_mem_edit.clicked.connect(lambda: self.open_settings("Memory"))
        top_l.addWidget(self.btn_mem_edit)
        # Mode dropdown (prompt pack)
        self.mode_combo = QtWidgets.QComboBox(objectName="ModeDrop")
        self.mode_combo.setToolTip("Режим ассистента")
        self.mode_combo.setFixedWidth(130)
        _pack = (self.db.get_setting("prompt_pack", "daughter") or "daughter").strip()
        # Prompt-pack selector.
        # Keep labels short to fit the top bar; "RAD" = Radiology (CT/MRI).
        _items = [("Daughter","daughter"), ("GP","gp"), ("Derm","derm"), ("Histo","histo"), ("RAD","radio")]
        for _label, _key in _items:
            if _key in PROMPT_PACKS:
                self.mode_combo.addItem(_label, _key)
        # select current
        for _i in range(self.mode_combo.count()):
            if self.mode_combo.itemData(_i) == _pack:
                self.mode_combo.setCurrentIndex(_i)
                break
        self.mode_combo.currentIndexChanged.connect(self._on_pack_changed)
        top_l.addWidget(self.mode_combo)

        # Radiology quick action (import DICOM study from a folder / ZIP / NIfTI)
        self.btn_radio = QtWidgets.QToolButton(objectName="IconBtn")
        self.btn_radio.setText("🩻")
        self.btn_radio.setToolTip("Радиология: импорт исследования (DICOM папка / ZIP / NIfTI)")
        self.btn_radio.clicked.connect(self.open_radiology)
        top_l.addWidget(self.btn_radio)
        # Shortcut: open radiology import (Ctrl+R)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), self, activated=self.open_radiology)


        self.btn_settings = QtWidgets.QToolButton(objectName="IconBtn")
        self.btn_settings.setText("⚙")
        self.btn_settings.setToolTip("Settings")
        self.btn_settings.clicked.connect(self.open_settings)
        top_l.addWidget(self.btn_settings)
        # Shortcut: open settings (Ctrl+,)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+,"), self, activated=lambda: self.open_settings())


        main_l.addWidget(top)

        # Chat scroll
        self.chat_scroll = QtWidgets.QScrollArea(objectName="ChatScroll")
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        self.chat_view = QtWidgets.QWidget(objectName="ChatViewport")
        self.chat_v = QtWidgets.QVBoxLayout(self.chat_view)
        self.chat_v.setContentsMargins(0, 0, 0, 0)
        self.chat_v.setSpacing(10)
        self.chat_v.addStretch(1)
        self.chat_scroll.setWidget(self.chat_view)
        main_l.addWidget(self.chat_scroll, 1)

        # Input row
        self.input_box = QtWidgets.QFrame(objectName="InputBox")
        ib_l = QtWidgets.QHBoxLayout(self.input_box)
        ib_l.setContentsMargins(8, 8, 8, 8)
        ib_l.setSpacing(6)

        self.message = QtWidgets.QTextEdit(objectName="MessageEdit")
        self.message.setPlaceholderText("Message Miriam...")
        self.message.setFixedHeight(56)
        ib_l.addWidget(self.message, 1)

        self.btn_attach = QtWidgets.QToolButton(objectName="IconBtn")
        self.btn_attach.setText("🖼")
        self.btn_attach.setToolTip("Attach files")
        self.btn_attach.clicked.connect(self.attach_files)
        ib_l.addWidget(self.btn_attach)

        self.btn_docs = QtWidgets.QToolButton(objectName="IconBtn")
        self.btn_docs.setText("📎")
        self.btn_docs.setToolTip("Add document to RAG")
        self.btn_docs.clicked.connect(self.add_doc)
        ib_l.addWidget(self.btn_docs)

        # Mic (voice input)
        self.btn_mic = QtWidgets.QToolButton(objectName="IconBtn")
        self.btn_mic.setText("🎤")
        self.btn_mic.setToolTip("Voice input")
        self.btn_mic.clicked.connect(self.toggle_voice_recording)
        ib_l.addWidget(self.btn_mic)

        if not self._voice_deps_available():
            # Keep builds friendly to clean macOS installs (no Homebrew).
            self.btn_mic.setEnabled(False)
            self.btn_mic.setToolTip("Voice input unavailable (optional audio dependencies missing).")

        # More menu
        self.btn_menu = QtWidgets.QToolButton(objectName="IconBtn")
        self.btn_menu.setText("⋯")
        self.btn_menu.setToolTip("More")
        self.btn_menu.clicked.connect(self.open_more_menu)
        ib_l.addWidget(self.btn_menu)

        self.btn_send = QtWidgets.QPushButton("➤", objectName="SendBtn")
        self.btn_send.clicked.connect(self.send)
        ib_l.addWidget(self.btn_send)

        main_l.addWidget(self.input_box)

        root_l.addWidget(main, 1)
        self.setCentralWidget(root)

        self._dialogs_cache: list[dict] = []
        self._active_dialog_id: Optional[int] = None
        self._stream_bubble: Optional[Bubble] = None
        self._stream_text: list[str] = []

        self.refresh_dialogs(select_active=True)
        self._load_active_chat()

    # -------- dialogs list --------
    def refresh_dialogs(self, *, select_active: bool = False):
        self._dialogs_cache = self.db.list_dialogs()
        self._filter_chats(self.search.text())
        if select_active:
            aid = self.db.get_active_dialog()
            if aid:
                self._select_dialog_in_list(aid)

    def _filter_chats(self, text: str):
        q = (text or "").lower().strip()
        self.chat_list.clear()
        for d in self._dialogs_cache:
            title = d.get("title", "")
            if q and q not in title.lower():
                continue
            item = QtWidgets.QListWidgetItem(title)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, int(d["id"]))
            self.chat_list.addItem(item)

    def _select_dialog_in_list(self, dialog_id: int):
        for i in range(self.chat_list.count()):
            it = self.chat_list.item(i)
            if int(it.data(QtCore.Qt.ItemDataRole.UserRole)) == int(dialog_id):
                self.chat_list.setCurrentItem(it)
                return

    def _on_chat_selected(self):
        it = self.chat_list.currentItem()
        if not it:
            return
        dialog_id = int(it.data(QtCore.Qt.ItemDataRole.UserRole))
        self.db.set_active_dialog(dialog_id)
        self._active_dialog_id = dialog_id
        self._load_active_chat()

    def new_chat(self):
        # Ensure the new dialog is visible in the list even if the user has a search filter.
        if self.search.text().strip():
            self.search.blockSignals(True)
            self.search.clear()
            self.search.blockSignals(False)
        did = self.db.create_dialog('New chat')
        self.db.set_active_dialog(did)
        self._active_dialog_id = did
        self.refresh_dialogs(select_active=True)
        self._load_active_chat()

    def _load_active_chat(self):
        did = self.db.get_active_dialog()
        if not did:
            did = self.db.create_dialog("New chat")
            self.db.set_active_dialog(did)
        self._active_dialog_id = did
        self._render_chat()

    def _clear_chat_view(self):
        # remove all widgets except stretch
        while self.chat_v.count() > 1:
            item = self.chat_v.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _render_chat(self):
        self._clear_chat_view()
        did = self._active_dialog_id
        if not did:
            return
        msgs = self.db.get_messages(did)
        for m in msgs:
            self._add_message_bubble(m.get("role") == "user", m.get("content") or "")
        self._scroll_to_bottom()

    # -------- bubbles --------
    def _add_message_bubble(self, is_user: bool, text: str) -> Bubble:
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        if is_user:
            row.addStretch(1)
        bubble = Bubble(text, is_user=is_user)
        bubble.setMaximumWidth(780)
        row.addWidget(bubble, 0)
        if not is_user:
            row.addStretch(1)
        container = QtWidgets.QWidget()
        container.setLayout(row)
        self.chat_v.insertWidget(self.chat_v.count() - 1, container)
        return bubble

    def _scroll_to_bottom(self):
        sb = self.chat_scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    # -------- top toggles --------
    def _toggle_rag(self, state: int):
        self.db.set_setting("rag_enabled", "1" if state else "0")

    def _toggle_memory(self, state: int):
        self.db.set_setting("memory_enabled", "1" if state else "0")

    # -------- actions --------
    def _on_pack_changed(self):
        try:
            key = self.mode_combo.currentData()
        except Exception:
            key = None
        key = (key or "gp").strip()
        if key not in PROMPT_PACKS:
            key = "gp"
        self.db.set_setting("prompt_pack", key)
        # No hard reset: just informs that next message will use new pack.
        # Optionally, update placeholder/title subtly.
        self._set_status(f"Mode: {key}")

    def open_settings(self, initial_tab: str | None = None):
        """Open the Settings dialog (robust in packaged .exe runs).

        In some packaging scenarios, exceptions can be swallowed (no console).
        We always surface errors in a message box so the user can configure API keys.
        """
        try:
            dlg = SettingsDialog(self, self.db, initial_tab=initial_tab)
            # Ensure it comes to front
            dlg.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
            dlg.setModal(True)
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
            if dlg.exec():
                # refresh switch state (in case changed elsewhere)
                self.rag_switch.setChecked((self.db.get_setting("rag_enabled", "1") or "1") == "1")
                self.mem_switch.setChecked((self.db.get_setting("memory_enabled", "1") or "1") == "1")
        except Exception:
            import traceback
            QtWidgets.QMessageBox.critical(self, "Settings error", traceback.format_exc())

    def _ensure_provider_configured(self) -> bool:
        """Ensure at least one provider is configured.

        In packaged .exe runs, environment variables are often not set. If the
        user hasn't opened Settings yet, API keys will be empty, which would
        otherwise surface as a runtime error. We proactively open Settings.
        """
        mode = (self.db.get_setting("provider_mode", "auto") or "auto").strip().lower()

        def nonempty(k: str) -> bool:
            return bool((self.db.get_setting(k, "") or "").strip())

        # Check the selected mode
        if mode == "novita":
            ok = nonempty("novita_api_key") and nonempty("novita_base_url")
        elif mode == "openrouter":
            ok = nonempty("openrouter_api_key") and nonempty("openrouter_base_url")
        elif mode == "custom":
            ok = nonempty("custom_api_key") and nonempty("custom_base_url")
        else:
            ok = (nonempty("novita_api_key") and nonempty("novita_base_url")) or (
                nonempty("openrouter_api_key") and nonempty("openrouter_base_url")
            ) or (nonempty("custom_api_key") and nonempty("custom_base_url"))

        if ok:
            return True

        QtWidgets.QMessageBox.information(
            self,
            "Settings required",
            "No API keys configured. Please set NOVITA / OpenRouter (or Custom) credentials in Settings.",
        )
        self.open_settings()

        # Re-check after dialog
        return self._ensure_provider_configured_post()

    def _ensure_provider_configured_post(self) -> bool:
        mode = (self.db.get_setting("provider_mode", "auto") or "auto").strip().lower()

        def nonempty(k: str) -> bool:
            return bool((self.db.get_setting(k, "") or "").strip())

        if mode == "novita":
            return nonempty("novita_api_key") and nonempty("novita_base_url")
        if mode == "openrouter":
            return nonempty("openrouter_api_key") and nonempty("openrouter_base_url")
        if mode == "custom":
            return nonempty("custom_api_key") and nonempty("custom_base_url")
        return (nonempty("novita_api_key") and nonempty("novita_base_url")) or (
            nonempty("openrouter_api_key") and nonempty("openrouter_base_url")
        ) or (nonempty("custom_api_key") and nonempty("custom_base_url"))

    def open_histo(self):
        if not self._active_dialog_id:
            return
        dlg = HistoDialog(self, self.db, self._active_dialog_id)
        dlg.exec()

    def open_radiology(self):
        if not self._active_dialog_id:
            QtWidgets.QMessageBox.information(
                self,
                "Radiology",
                "Please create/select a chat first, then import a radiology study for that chat.",
            )
            return
        dlg = RadiologyDialog(self, self.db, self._active_dialog_id)
        dlg.exec()

    def open_radiology_studies(self):
        if not self._active_dialog_id:
            QtWidgets.QMessageBox.information(
                self,
                "Radiology",
                "Please create/select a chat first, then import/view radiology studies for that chat.",
            )
            return
        dlg = RadiologyStudiesDialog(self, self.db, self._active_dialog_id)
        dlg.exec()

    def open_rag_docs(self):
        dlg = RagDocsDialog(self, self.db)
        dlg.exec()

    def add_doc(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Add document(s) to RAG",
            "",
            "Documents (*.pdf *.docx *.txt *.md *.rtf);;All files (*.*)",
        )
        if not paths:
            return
        self.add_rag_docs(paths)
        self.db.set_setting("rag_index_dirty", "1")

    def open_more_menu(self):
        menu = QtWidgets.QMenu(self)
        act_export = menu.addAction("Export chat")
        act_docs = menu.addAction("RAG documents")
        act_histo = menu.addAction("Histo metadata")
        act_radio = menu.addAction("Радиология: импорт КТ/МРТ")
        act_radio_mgr = menu.addAction("Радиология: исследования / viewer")

        chosen = menu.exec(self.btn_menu.mapToGlobal(QtCore.QPoint(0, self.btn_menu.height())))
        if chosen == act_export:
            self.export_chat()
        elif chosen == act_docs:
            self.open_rag_docs()
        elif chosen == act_histo:
            self.open_histo()
        elif chosen == act_radio:
            self.open_radiology()
        elif chosen == act_radio_mgr:
            self.open_radiology_studies()

    # ------------------------------ Voice ------------------------------
    def _voice_deps_available(self) -> bool:
        try:
            importlib.import_module("sounddevice")
            importlib.import_module("soundfile")
            return True
        except Exception:
            return False

    def toggle_voice_recording(self):
        """Start/stop microphone recording.

        Records audio locally and then uses the configured provider's
        /audio/transcriptions endpoint to transcribe it.
        """
        if not self._voice_deps_available():
            QtWidgets.QMessageBox.information(
                self,
                "Voice input",
                "Voice input is not available in this build (missing optional audio dependencies).\n\n"
                "You can still use the app normally. To enable mic recording, install sounddevice + soundfile and the required system audio libraries.",
            )
            return

        # Respect the setting (can be toggled in Settings → Voice).
        if (self.db.get_setting("voice_input_enabled", "0") or "0") != "1":
            QtWidgets.QMessageBox.information(
                self,
                "Voice input",
                "Voice input is disabled. Enable it in Settings → Voice.",
            )
            return

        if not self._voice_recording:
            self._voice_recording = True
            self._voice_stop.clear()
            self.btn_mic.setText("⏺")
            self.btn_mic.setToolTip("Stop recording")

            self._voice_thread = threading.Thread(target=self._voice_record_worker, daemon=True)
            self._voice_thread.start()
        else:
            self._voice_recording = False
            self._voice_stop.set()
            self.btn_mic.setText("🎤")
            self.btn_mic.setToolTip("Voice input")

    def _voice_record_worker(self):
        try:
            import numpy as np
            import sounddevice as sd
            import soundfile as sf

            frames: list[np.ndarray] = []

            def callback(indata, _frames, _time, status):
                if status:
                    # Just collect; we surface errors at the end.
                    pass
                frames.append(indata.copy())
                if self._voice_stop.is_set():
                    raise sd.CallbackStop()

            with sd.InputStream(
                samplerate=self._voice_fs,
                channels=1,
                dtype="int16",
                callback=callback,
            ):
                while not self._voice_stop.is_set():
                    sd.sleep(50)

            if not frames:
                return

            audio = np.concatenate(frames, axis=0)
            tmpdir = Path(tempfile.gettempdir())
            out = tmpdir / f"miriam_recording_{int(time.time())}.wav"
            sf.write(str(out), audio, self._voice_fs, subtype="PCM_16")

            # Transcribe in a separate thread to keep the audio callback clean.
            threading.Thread(target=self._voice_transcribe_worker, args=(str(out),), daemon=True).start()
        except Exception as e:
            self._voice_signals.error.emit(str(e))

    def _voice_transcribe_worker(self, wav_path: str):
        try:
            text, _prov = self.engine.transcribe_audio(wav_path)
            text = (text or "").strip()
            if text:
                self._voice_signals.transcript.emit(text)
        except Exception as e:
            self._voice_signals.error.emit(str(e))

    def _on_voice_transcript(self, text: str):
        # Fill the input box (user can edit). Optionally auto-send.
        self.message.setPlainText(text)
        self.message.setFocus()
        if (self.db.get_setting("voice_auto_send", "0") or "0") == "1":
            self.send()

    def _on_voice_error(self, err: str):
        QtWidgets.QMessageBox.warning(
            self,
            "Voice input",
            f"Voice input failed: {err}\n\nTip: make sure your microphone works and your provider supports /audio/transcriptions.",
        )

    def _strip_markdown_for_tts(self, text: str) -> str:
        # Light markdown cleanup so TTS sounds better.
        t = (text or "")
        # remove code blocks
        t = re.sub(r"```.*?```", "", t, flags=re.DOTALL)
        # inline code
        t = re.sub(r"`([^`]+)`", r"\\1", t)
        # headings/bullets markers
        t = re.sub(r"^#{1,6}\s+", "", t, flags=re.MULTILINE)
        t = re.sub(r"^[\-\*\+]\s+", "", t, flags=re.MULTILINE)
        # links [text](url) -> text
        t = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\\1", t)
        return t.strip()

    def _tts_speak_worker(self, text: str):
        try:
            import pyttsx3

            clean = self._strip_markdown_for_tts(text)
            if not clean:
                return
            rate = int((self.db.get_setting("tts_rate", "175") or "175").strip() or 175)

            with self._tts_lock:
                if self._tts_engine is None:
                    self._tts_engine = pyttsx3.init()
                eng = self._tts_engine
                try:
                    eng.setProperty("rate", rate)
                except Exception:
                    pass
                eng.say(clean)
                eng.runAndWait()
        except Exception:
            # TTS is best-effort; don't crash the app.
            return

    def add_rag_docs(self, paths: list[str]):
        # Ingest documents now (synchronously) to keep UI simple and robust.
        from .core.rag import add_document_to_rag

        ok = 0
        failed: list[str] = []
        for p in paths:
            try:
                title = Path(p).name
                add_document_to_rag(self.db, file_path=p, title=title)
                ok += 1
            except Exception as e:
                failed.append(f"{p}: {e}")

        if ok:
            QtWidgets.QMessageBox.information(self, "RAG", f"Added {ok} document(s) to RAG.")
        if failed:
            QtWidgets.QMessageBox.warning(self, "RAG", "Some documents failed:\n\n" + "\n".join(failed))

    def attach_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Attach files",
            "",
            "Supported (*.png *.jpg *.jpeg *.webp *.pdf *.docx *.txt *.md *.rtf *.zip *.nii *.nii.gz);;Images (*.png *.jpg *.jpeg *.webp);;Docs (*.pdf *.docx *.txt *.md *.rtf);;Radiology (*.zip *.nii *.nii.gz);;All files (*.*)",
        )
        if not paths:
            return

        img_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        doc_ext = {".pdf", ".docx", ".txt", ".md", ".rtf"}
        radio_ext = {".zip", ".nii"}
        for fp in paths:
            ext = Path(fp).suffix.lower()
            # Handle .nii.gz
            if fp.lower().endswith(".nii.gz"):
                self._pending_imaging.append(fp)
            elif ext in img_ext:
                self._pending_images.append(fp)
            elif ext in doc_ext:
                self._pending_docs.append(fp)
            elif ext in radio_ext:
                self._pending_imaging.append(fp)
            else:
                # treat unknown as doc
                self._pending_docs.append(fp)

        tip = "Attach files"
        if self._pending_images:
            tip += f" | images: {len(self._pending_images)}"
        if self._pending_docs:
            tip += f" | docs: {len(self._pending_docs)}"
        if self._pending_imaging:
            tip += f" | radiology: {len(self._pending_imaging)}"
        self.btn_attach.setToolTip(tip)

    def attach_image(self):
        # Backward-compatible alias
        self.attach_files()

    def export_chat(self):
        did = self._active_dialog_id
        if not did:
            return
        title = (self.chat_list.currentItem().text() if self.chat_list.currentItem() else "chat")
        default_name = f"{title}.md".replace("/", "-").replace("\\", "-")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export chat",
            default_name,
            "Markdown (*.md);;Text (*.txt);;JSON (*.json)",
        )
        if not path:
            return
        msgs = self.db.get_messages(did)
        suffix = Path(path).suffix.lower()
        try:
            if suffix == ".json":
                Path(path).write_text(json.dumps(msgs, ensure_ascii=False, indent=2), encoding="utf-8")
            elif suffix == ".txt":
                lines = []
                for m in msgs:
                    who = "You" if m.get("role") == "user" else "Miriam"
                    lines.append(f"{who}:\n{m.get('content','')}\n")
                Path(path).write_text("\n".join(lines), encoding="utf-8")
            else:
                lines = [f"# {title}\n"]
                for m in msgs:
                    who = "You" if m.get("role") == "user" else "Miriam"
                    lines.append(f"**{who}**\n\n{m.get('content','')}\n")
                Path(path).write_text("\n".join(lines), encoding="utf-8")
            QtWidgets.QMessageBox.information(self, "Export", "Exported successfully.")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export", f"Failed to export: {e}")

    def send(self):
        did = self._active_dialog_id
        if not did:
            return
        text = self.message.toPlainText().strip()
        self._pending_user_text = text
        if not text:
            return

        # Ensure provider credentials are set before we persist the turn.
        if not self._ensure_provider_configured():
            return

        # Store and render user message
        images = list(self._pending_images)
        docs = list(getattr(self, "_pending_docs", []))
        radio_files = list(getattr(self, "_pending_imaging", []))
        self._pending_images.clear()
        self._pending_docs.clear()
        self._pending_imaging.clear()
        self.btn_attach.setToolTip("Attach files")

        attachments = {}
        if images:
            attachments["images"] = images
        if docs:
            attachments["docs"] = docs
        if radio_files:
            attachments["radiology"] = radio_files

        self.db.add_message(did, "user", text, attachments=attachments)
        self._add_message_bubble(True, text)

        # If the user attached documents, ingest them into dialog-scoped RAG now.
        if docs:
            try:
                from pathlib import Path as _Path
                from .core.rag import add_document_to_rag

                failed = []
                for fp in docs:
                    try:
                        add_document_to_rag(self.db, fp, title=_Path(fp).name, scope="dialog", dialog_id=did)
                    except Exception as e:
                        failed.append(f"{fp}: {e}")
                if failed:
                    # Non-fatal: the message still goes through.
                    QtWidgets.QMessageBox.warning(self, "RAG", "Some documents could not be indexed\n\n" + "\n".join(failed))
            except Exception:
                pass

        # If the user attached radiology files, import the first one as an active study (best-effort).
        # For large studies prefer: More → Radiology study.
        if radio_files:
            try:
                from .core.radiology import ingest_imaging_study

                # Import only the first file; additional files are ignored in this quick path.
                ingest_imaging_study(
                    self.db,
                    dialog_id=did,
                    source_path=radio_files[0],
                    modality="",
                    body_region="",
                    contrast="",
                    clinical_question="",
                )
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Radiology",
                    "Failed to import radiology file.\n\n" + str(e) + "\n\nTry: More → Radiology study.",
                )

        # Prepare assistant bubble
        self._stream_text = []
        self._stream_bubble = self._add_message_bubble(False, "")
        self._scroll_to_bottom()

        self.btn_send.setEnabled(False)
        self.message.setEnabled(False)

        # Worker thread (Qt)
        self._thread = QtCore.QThread(self)
        self._worker = StreamWorker(self.engine, did, text, images)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.delta.connect(self._on_stream_delta)
        self._worker.done.connect(self._on_stream_done)
        self._worker.error.connect(self._on_stream_error)
        self._worker.done.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

        self.message.clear()

    def send_programmatic(self, text: str, image_paths: list[str] | None = None):
        """Send a message without touching the input box.

        Used for post-import auto-reports (e.g., radiology).
        """
        did = self._active_dialog_id
        if not did:
            return
        text = (text or "").strip()
        if not text:
            return

        # Ensure provider credentials are set before we persist the turn.
        if not self._ensure_provider_configured():
            return

        images = list(image_paths or [])

        self.db.add_message(did, "user", text, attachments={})
        self._add_message_bubble(True, text)

        self._stream_text = []
        self._stream_bubble = self._add_message_bubble(False, "")
        self._scroll_to_bottom()

        self.btn_send.setEnabled(False)
        self.message.setEnabled(False)

        self._thread = QtCore.QThread(self)
        self._worker = StreamWorker(self.engine, did, text, images)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.delta.connect(self._on_stream_delta)
        self._worker.done.connect(self._on_stream_done)
        self._worker.error.connect(self._on_stream_error)
        self._worker.done.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    def _on_stream_delta(self, chunk: str):
        if not self._stream_bubble:
            return
        self._stream_text.append(chunk)
        self._stream_bubble.set_text("".join(self._stream_text))
        self._scroll_to_bottom()

    def _on_stream_done(self, final_text: str, provider: str, model: str):
        did = self._active_dialog_id
        if did:
            self.db.add_message(did, "assistant", final_text, attachments={"provider": provider, "model": model})
        self.btn_send.setEnabled(True)
        self.message.setEnabled(True)
        self.message.setFocus()

        if (self.db.get_setting("tts_enabled", "0") or "0") == "1":
            threading.Thread(target=self._tts_speak_worker, args=(final_text,), daemon=True).start()

    def _on_stream_error(self, err: str):
        self.btn_send.setEnabled(True)
        self.message.setEnabled(True)
        # If this looks like a missing-credentials error, guide user to Settings.
        low = (err or "").lower()
        if "api keys" in low or "api_key is empty" in low or "no api keys" in low:
            QtWidgets.QMessageBox.warning(self, "Error", err)
            self.open_settings()
            return
        QtWidgets.QMessageBox.warning(self, "Error", err)

    def _cleanup_thread(self):
        try:
            self._worker.deleteLater()
            self._thread.deleteLater()
        except Exception:
            pass


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyleSheet(QSS)
    w = MainWindow()
    w.resize(1200, 780)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
