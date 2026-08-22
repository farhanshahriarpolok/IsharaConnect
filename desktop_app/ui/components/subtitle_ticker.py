"""Live Subtitle Ticker HUD Component for Real-Time Continuous BdSL Translation.

Features:
- Glassmorphic dark HUD container (#0F172A with border #334155 and 12px rounded corners)
- Top Tier: Active gloss chips in glowing amber capsule badges (#F59E0B)
- Bottom Tier: Grammatically inflected continuous Bengali translation (#10B981 emerald) with confidence score pill
- Smooth updates, audio re-voice trigger, and buffer flush
"""

import logging
from typing import List, Optional
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core_engine.nlp.gloss_to_sentence import GlossToSentenceTranslator
from desktop_app.controllers.audio_player import audio_controller
from desktop_app.ui.theme import ThemeColors

logger = logging.getLogger(__name__)


class SubtitleTickerWidget(QFrame):
    """Floating/Docked Glassmorphic HUD for live gesture glosses and continuous natural Bengali translation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gloss_buffer: List[str] = []
        self.current_sentence_bn = ""
        self.current_sentence_en = ""
        self.current_confidence = 0.0
        self.translator = GlossToSentenceTranslator()

        self.setObjectName("SubtitleTickerHUD")
        self._init_ui()

    def _init_ui(self):
        self.setStyleSheet("""
            QFrame#SubtitleTickerHUD {
                background-color: rgba(15, 23, 42, 0.88);
                border: 1px solid rgba(51, 65, 85, 0.8);
                border-radius: 12px;
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 12, 16, 14)
        main_layout.setSpacing(8)

        # 1. Top Tier: Active Gloss Strip (Horizontal Scroll of Amber Chips)
        gloss_header_layout = QHBoxLayout()
        gloss_tag_lbl = QLabel("GESTURE STREAM:")
        gloss_tag_lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        gloss_tag_lbl.setStyleSheet("color: #94A3B8; letter-spacing: 1px;")
        gloss_header_layout.addWidget(gloss_tag_lbl)

        self.chips_layout = QHBoxLayout()
        self.chips_layout.setContentsMargins(0, 0, 0, 0)
        self.chips_layout.setSpacing(6)
        self.chips_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        chips_container = QWidget()
        chips_container.setStyleSheet("background: transparent;")
        chips_container.setLayout(self.chips_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(36)
        scroll.setStyleSheet("border: none; background: transparent;")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(chips_container)

        gloss_header_layout.addWidget(scroll, stretch=1)
        main_layout.addLayout(gloss_header_layout)

        # 2. Bottom Tier: Inflected Bengali Sentence & Confidence Badge
        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(12)

        self.sentence_label = QLabel("অপেক্ষা করছি...")
        self.sentence_label.setFont(QFont("SolaimanLipi", 16, QFont.Weight.Bold))
        self.sentence_label.setStyleSheet("color: #10B981;")
        self.sentence_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        bottom_layout.addWidget(self.sentence_label, stretch=1)

        # Confidence Score Pill Badge
        self.confidence_pill = QLabel("95%")
        self.confidence_pill.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.confidence_pill.setStyleSheet("""
            background-color: rgba(6, 182, 212, 0.15);
            color: #06B6D4;
            border: 1px solid rgba(6, 182, 212, 0.4);
            border-radius: 10px;
            padding: 3px 8px;
        """)
        self.confidence_pill.setVisible(False)
        bottom_layout.addWidget(self.confidence_pill)

        # Re-voice Action Button
        self.revoice_btn = QPushButton("🔊 Re-Voice")
        self.revoice_btn.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        self.revoice_btn.setStyleSheet("""
            QPushButton {
                background-color: #1E293B;
                color: #F8FAFC;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 4px 10px;
            }
            QPushButton:hover {
                background-color: #334155;
                color: #06B6D4;
            }
        """)
        self.revoice_btn.clicked.connect(self._on_revoice)
        bottom_layout.addWidget(self.revoice_btn)

        main_layout.addLayout(bottom_layout)

    def update_active_glosses(self, gloss_list: List[str]):
        """Renders active gloss chips in glowing amber pill badges."""
        self.gloss_buffer = list(gloss_list)

        # Clear existing chips
        while self.chips_layout.count():
            item = self.chips_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        for g in self.gloss_buffer:
            chip = QLabel(f"[{g}]")
            chip.setFont(QFont("SolaimanLipi", 11, QFont.Weight.Bold))
            chip.setStyleSheet("""
                background-color: rgba(245, 158, 11, 0.15);
                color: #F59E0B;
                border: 1px solid rgba(245, 158, 11, 0.4);
                border-radius: 10px;
                padding: 2px 8px;
            """)
            self.chips_layout.addWidget(chip)

    def update_translation(self, translated_text: str, confidence: float, is_final: bool = False):
        """Updates the translated sentence display and confidence rating badge."""
        clean_text = translated_text.strip() if translated_text else ""
        if clean_text:
            self.current_sentence_bn = clean_text
            self.current_confidence = confidence
            self.sentence_label.setText(clean_text)

            conf_pct = int(confidence * 100) if confidence <= 1.0 else int(confidence)
            self.confidence_pill.setText(f"{conf_pct}%")
            self.confidence_pill.setVisible(True)

            if is_final:
                self.sentence_label.setStyleSheet("color: #10B981; font-weight: bold;")
            else:
                self.sentence_label.setStyleSheet("color: #34D399;")
        else:
            self.sentence_label.setText("অপেক্ষা করছি...")
            self.confidence_pill.setVisible(False)

    @pyqtSlot(dict)
    def update_ticker(self, sign_data: dict):
        """Backward compatibility adapter for single-frame inference dictionary."""
        if not sign_data or sign_data.get("sign_id") == -1:
            return

        label_bn = sign_data.get("label_bn", "").strip()
        confidence = float(sign_data.get("confidence", 0.9))
        if not label_bn:
            return

        stream_res = self.translator.process_stream(label_bn, confidence)
        self.update_active_glosses(stream_res.get("raw_glosses", []))
        self.update_translation(
            stream_res.get("translated_text", ""),
            stream_res.get("confidence", confidence),
            stream_res.get("is_final", False)
        )

    def _on_revoice(self):
        """Plays synthesized Bengali vocalization for current text."""
        if self.current_sentence_bn and self.current_sentence_bn != "অপেক্ষা করছি...":
            clean_text = self.current_sentence_bn.rstrip("।!?")
            audio_controller.speak_bengali(clean_text)

    def clear_ticker(self):
        """Clears active chips, sentence text, and resets translation state."""
        self.gloss_buffer.clear()
        self.current_sentence_bn = ""
        self.current_confidence = 0.0
        self.translator.reset()

        while self.chips_layout.count():
            item = self.chips_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        self.sentence_label.setText("অপেক্ষা করছি...")
        self.confidence_pill.setVisible(False)

    def clear_buffer(self):
        """Backward compatibility alias for clear_ticker."""
        self.clear_ticker()
