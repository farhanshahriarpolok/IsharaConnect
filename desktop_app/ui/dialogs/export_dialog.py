"""Export Transcript Dialog for IsharaConnect Desktop Client.

Provides a sleek modal for exporting conversation logs into PDF, TXT, or JSON format.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from desktop_app.ui.theme import ThemeColors, ThemeStyles
from desktop_app.utils.transcript_exporter import (
    export_to_json,
    export_to_pdf,
    export_to_txt,
)

logger = logging.getLogger(__name__)


class ExportTranscriptDialog(QDialog):
    """Modern modal dialog allowing users to export communication session transcripts."""

    def __init__(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        room_id: str = "room_public_01",
        mode: str = "Signer",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.messages = messages or []
        self.room_id = room_id
        self.mode = mode

        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("Export Conversation Transcript")
        self.setMinimumWidth(540)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: #0F172A;
                color: #F8FAFC;
                border: 1px solid #334155;
                border-radius: 12px;
            }}
            QLabel {{
                color: #F8FAFC;
                font-family: 'Segoe UI', Arial;
            }}
            QLineEdit {{
                background-color: #1E293B;
                color: #F8FAFC;
                border: 1px solid #475569;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 13px;
            }}
            QLineEdit:focus {{
                border: 1px solid #06B6D4;
            }}
            QRadioButton {{
                color: #E2E8F0;
                font-size: 13px;
                font-weight: 500;
                spacing: 8px;
            }}
            QRadioButton::indicator {{
                width: 16px;
                height: 16px;
            }}
            QRadioButton::indicator:checked {{
                background-color: #06B6D4;
                border: 2px solid #0891B2;
                border-radius: 8px;
            }}
            QRadioButton::indicator:unchecked {{
                background-color: #1E293B;
                border: 2px solid #64748B;
                border-radius: 8px;
            }}
            QCheckBox {{
                color: #CBD5E1;
                font-size: 12px;
                spacing: 6px;
            }}
            QCheckBox::indicator:checked {{
                background-color: #10B981;
                border: 1px solid #059669;
                border-radius: 3px;
            }}
            QPushButton {{
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 13px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        # Header Title & Description
        header_layout = QVBoxLayout()
        title = QLabel("💾 Export Conversation Transcript")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #06B6D4;")
        
        subtitle = QLabel(
            f"Save the live two-way dialogue log ({len(self.messages)} message(s) recorded) "
            "into a permanent document."
        )
        subtitle.setStyleSheet("color: #94A3B8; font-size: 12px;")
        subtitle.setWordWrap(True)
        
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        layout.addLayout(header_layout)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #334155; max-height: 1px;")
        layout.addWidget(line)

        # 1. Format Selection
        format_label = QLabel("Choose Export Format:")
        format_label.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        layout.addWidget(format_label)

        self.format_group = QButtonGroup(self)
        self.radio_pdf = QRadioButton("📄 PDF Document (.pdf) — Styled with Timestamps & Participant Badges")
        self.radio_txt = QRadioButton("📝 Formatted Text (.txt) — Plaintext Log suitable for any reader")
        self.radio_json = QRadioButton("📦 Structured JSON (.json) — Machine-readable Raw Log & Metadata")
        
        self.radio_pdf.setChecked(True)
        self.format_group.addButton(self.radio_pdf, 1)
        self.format_group.addButton(self.radio_txt, 2)
        self.format_group.addButton(self.radio_json, 3)

        self.radio_pdf.toggled.connect(self._on_format_changed)
        self.radio_txt.toggled.connect(self._on_format_changed)
        self.radio_json.toggled.connect(self._on_format_changed)

        format_layout = QVBoxLayout()
        format_layout.setSpacing(10)
        format_layout.addWidget(self.radio_pdf)
        format_layout.addWidget(self.radio_txt)
        format_layout.addWidget(self.radio_json)
        layout.addLayout(format_layout)

        # 2. File Destination
        dest_label = QLabel("Destination File Path:")
        dest_label.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        layout.addWidget(dest_label)

        path_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self._update_default_path()
        self.browse_btn = QPushButton("📁 Browse...")
        self.browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: #F8FAFC;
                border: 1px solid #475569;
            }
            QPushButton:hover {
                background-color: #475569;
            }
        """)
        self.browse_btn.clicked.connect(self._browse_destination)
        path_layout.addWidget(self.path_input, stretch=1)
        path_layout.addWidget(self.browse_btn)
        layout.addLayout(path_layout)

        # 3. Options
        options_layout = QHBoxLayout()
        self.cb_timestamps = QCheckBox("Include Exact Millisecond Timestamps")
        self.cb_timestamps.setChecked(True)
        self.cb_metadata = QCheckBox("Include Room & System Metadata Header")
        self.cb_metadata.setChecked(True)
        options_layout.addWidget(self.cb_timestamps)
        options_layout.addWidget(self.cb_metadata)
        layout.addLayout(options_layout)

        # 4. Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: #94A3B8;
                border: 1px solid #475569;
            }
            QPushButton:hover {
                background-color: #475569;
                color: #F8FAFC;
            }
        """)
        self.cancel_btn.clicked.connect(self.reject)

        self.export_btn = QPushButton("🚀 Export Now")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #10B981;
                color: #064E3B;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #34D399;
            }
        """)
        self.export_btn.clicked.connect(self._do_export)

        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.export_btn)
        layout.addLayout(button_layout)

    def _get_extension(self) -> str:
        if self.radio_pdf.isChecked():
            return "pdf"
        elif self.radio_txt.isChecked():
            return "txt"
        else:
            return "json"

    def _update_default_path(self):
        ext = self._get_extension()
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_dir = Path.home() / "Documents"
        if not default_dir.exists():
            default_dir = Path.cwd()
        default_file = default_dir / f"IsharaConnect_Transcript_{date_str}.{ext}"
        self.path_input.setText(str(default_file))

    def _on_format_changed(self):
        curr_text = self.path_input.text().strip()
        new_ext = self._get_extension()
        if curr_text:
            p = Path(curr_text)
            new_path = p.with_suffix(f".{new_ext}")
            self.path_input.setText(str(new_path))
        else:
            self._update_default_path()

    def _browse_destination(self):
        ext = self._get_extension()
        filter_str = f"{ext.upper()} Files (*.{ext});;All Files (*.*)"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Transcript Destination",
            self.path_input.text().strip(),
            filter_str,
        )
        if file_path:
            self.path_input.setText(file_path)

    def _do_export(self):
        target_path = self.path_input.text().strip()
        if not target_path:
            QMessageBox.warning(self, "Invalid Path", "Please provide a valid destination file path.")
            return

        metadata = {
            "room_id": self.room_id,
            "mode": self.mode,
            "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "include_timestamps": self.cb_timestamps.isChecked(),
            "include_metadata": self.cb_metadata.isChecked(),
        }

        ext = self._get_extension()
        try:
            if ext == "pdf":
                out = export_to_pdf(self.messages, target_path, metadata)
            elif ext == "txt":
                out = export_to_txt(self.messages, target_path, metadata)
            else:
                out = export_to_json(self.messages, target_path, metadata)

            QMessageBox.information(
                self,
                "Export Complete",
                f"Transcript successfully exported to:\n{out}\n\nTotal messages: {len(self.messages)}",
            )
            self.accept()
        except Exception as e:
            logger.exception("Failed to export transcript: %s", e)
            QMessageBox.critical(
                self,
                "Export Error",
                f"An error occurred while exporting the transcript:\n{e}",
            )
