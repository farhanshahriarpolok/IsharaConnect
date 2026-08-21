"""Live Sentence Ticker HUD Component."""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont
from desktop_app.ui.theme import ThemeColors
from core_engine.audio.audio_player import player_instance
from core_engine.nlp.advanced_grammar_engine import AdvancedBdSLGrammarEngine
import logging

logger = logging.getLogger(__name__)


class SentenceTickerWidget(QWidget):
    """HUD for building sentences from incoming glosses."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.gloss_buffer = []
        self.finalized_sentence_bn = ""
        self.finalized_sentence_en = ""
        self.grammar_engine = AdvancedBdSLGrammarEngine()
        
        self.setObjectName("GlassCard")
        self._init_ui()
        
    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Top Row: Gloss Chips
        self.chips_layout = QHBoxLayout()
        self.chips_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background: transparent;")
        scroll.setMaximumHeight(50)
        
        chips_container = QWidget()
        chips_container.setStyleSheet("background: transparent;")
        chips_container.setLayout(self.chips_layout)
        scroll.setWidget(chips_container)
        
        main_layout.addWidget(scroll)
        
        # Center: Main Bengali Sentence
        self.sentence_label = QLabel("অপেক্ষা করছি...")
        self.sentence_label.setFont(QFont("SolaimanLipi", 24, QFont.Weight.Bold))
        self.sentence_label.setStyleSheet(f"color: {ThemeColors.CYAN_ACCENT};")
        self.sentence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.sentence_label)
        
        # Bottom Row: English + Action
        bottom_layout = QHBoxLayout()
        self.subtitle_label = QLabel("Waiting for signs...")
        self.subtitle_label.setFont(QFont("Inter", 14))
        self.subtitle_label.setStyleSheet(f"color: {ThemeColors.TEXT_SECONDARY};")
        
        self.revoice_btn = QPushButton("🔊 Re-voice")
        self.revoice_btn.clicked.connect(self._on_revoice)
        
        bottom_layout.addWidget(self.subtitle_label)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.revoice_btn)
        
        main_layout.addLayout(bottom_layout)
        
    @pyqtSlot(dict)
    def update_ticker(self, sign_data: dict):
        """Called when a new sign is stabilized."""
        if not sign_data or sign_data.get("sign_id") == -1:
            return
            
        bn = sign_data.get("label_bn", "").strip()
        en = sign_data.get("label_en", "").strip()
        
        if not bn:
            return
            
        self.gloss_buffer.append((bn, en))
        
        # Add chip UI
        chip = QLabel(f"[ {bn} ]")
        chip.setStyleSheet(f"""
            background-color: {ThemeColors.SURFACE_DARK};
            color: {ThemeColors.TEXT_PRIMARY};
            border-radius: 12px;
            padding: 4px 10px;
            font-size: 12px;
        """)
        self.chips_layout.addWidget(chip)
        
        # Advanced Continuous Grammar NLP Generation
        gloss_list = [g[0] for g in self.gloss_buffer]
        nlp_result = self.grammar_engine.generate_natural_sentence(gloss_list)
        
        self.finalized_sentence_bn = nlp_result["bengali"]
        self.finalized_sentence_en = nlp_result["english"]
        
        self.sentence_label.setText(self.finalized_sentence_bn)
        self.subtitle_label.setText(self.finalized_sentence_en)
        
    def _on_revoice(self):
        if self.finalized_sentence_bn:
            # Play text using the audio engine
            clean_text = self.finalized_sentence_bn.rstrip("।!?")
            player_instance.play_text(clean_text, lang="bn")
            
    def clear_buffer(self):
        self.gloss_buffer = []
        # Remove chips
        while self.chips_layout.count():
            item = self.chips_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
                
        self.sentence_label.setText("অপেক্ষা করছি...")
        self.subtitle_label.setText("Waiting for signs...")
