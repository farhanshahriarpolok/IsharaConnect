import logging
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QBrush, QPen
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFrame, QProgressBar, QListWidget, QListWidgetItem, QSplitter
)

from desktop_app.controllers.practice_controller import PracticeSessionManager
from core_engine.audio.audio_player import player_instance

logger = logging.getLogger(__name__)

# Modern Palette
BG_COLOR = "#1E1E2E"
PANEL_COLOR = "#2A2B3D"
TEXT_COLOR = "#CDD6F4"
ACCENT_BLUE = "#89B4FA"
ACCENT_GREEN = "#A6E3A1"
ACCENT_RED = "#F38BA8"
BORDER_RADIUS = "8px"

STYLESHEET = f"""
    QWidget {{ background-color: {BG_COLOR}; color: {TEXT_COLOR}; }}
    QFrame {{ background-color: {PANEL_COLOR}; border-radius: {BORDER_RADIUS}; }}
    QPushButton {{ 
        background-color: {ACCENT_BLUE}; 
        color: #11111B; 
        font-weight: bold; 
        border-radius: {BORDER_RADIUS}; 
        padding: 8px 16px; 
    }}
    QPushButton:hover {{ background-color: #74C7EC; }}
    QPushButton:disabled {{ background-color: #45475A; color: #6C7086; }}
    QListWidget {{ 
        background-color: #181825; 
        border: 1px solid #313244; 
        border-radius: 4px; 
        outline: 0;
    }}
    QListWidget::item {{ padding: 10px; }}
    QListWidget::item:selected {{ background-color: {ACCENT_BLUE}; color: #11111B; font-weight: bold; }}
    QProgressBar {{ 
        border: 1px solid #313244; 
        border-radius: 4px; 
        text-align: center; 
        color: white; 
    }}
    QProgressBar::chunk {{ background-color: {ACCENT_GREEN}; width: 10px; }}
"""

class LearningHubWidget(QWidget):
    # Signals to request camera feed route changes
    request_camera_start = pyqtSignal()
    request_camera_stop = pyqtSignal()

    def __init__(self, labels_path: str = "dataset/labels.json"):
        super().__init__()
        self.session_manager = PracticeSessionManager(labels_path=labels_path)
        self.active_sign_id = -1
        
        self.setStyleSheet(STYLESHEET)
        self._init_ui()
        self._load_categories()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- Left Sidebar / Lesson Navigator ---
        sidebar_frame = QFrame()
        sidebar_layout = QVBoxLayout(sidebar_frame)
        
        title = QLabel("📚 লেসন (Lessons)")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        sidebar_layout.addWidget(title)
        
        self.category_list = QListWidget()
        self.category_list.currentItemChanged.connect(self._on_sign_selected)
        sidebar_layout.addWidget(self.category_list)
        
        splitter.addWidget(sidebar_frame)
        
        # --- Center / Practice Arena ---
        arena_frame = QFrame()
        arena_layout = QVBoxLayout(arena_frame)
        
        # Top Banner
        self.target_banner = QLabel("🎯 প্র্যাকটিস করুন: নির্বাচন করুন")
        self.target_banner.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        self.target_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.target_banner.setStyleSheet(f"color: {ACCENT_BLUE}; padding: 10px;")
        arena_layout.addWidget(self.target_banner)
        
        # Center Video View
        self.camera_label = QLabel("Camera Offline")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet(f"background-color: #000000; border-radius: {BORDER_RADIUS};")
        self.camera_label.setMinimumSize(640, 480)
        arena_layout.addWidget(self.camera_label, stretch=1)
        
        # Bottom Feedback HUD
        hud_layout = QVBoxLayout()
        
        self.feedback_badge = QLabel("অপেক্ষা করছি... (Waiting...)")
        self.feedback_badge.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self.feedback_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hud_layout.addWidget(self.feedback_badge)
        
        self.accuracy_bar = QProgressBar()
        self.accuracy_bar.setRange(0, 100)
        hud_layout.addWidget(self.accuracy_bar)
        
        # Action Buttons
        btn_layout = QHBoxLayout()
        self.skip_btn = QPushButton("Skip (এড়িয়ে যান)")
        self.skip_btn.clicked.connect(self._next_sign)
        
        self.next_btn = QPushButton("Next Sign (পরবর্তী)")
        self.next_btn.clicked.connect(self._next_sign)
        self.next_btn.setEnabled(False)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.skip_btn)
        btn_layout.addWidget(self.next_btn)
        btn_layout.addStretch()
        
        hud_layout.addLayout(btn_layout)
        arena_layout.addLayout(hud_layout)
        
        splitter.addWidget(arena_frame)
        
        # Set splitter sizes (25% sidebar, 75% arena)
        splitter.setSizes([250, 750])
        main_layout.addWidget(splitter)

    def _load_categories(self):
        """Populate the sidebar with grouped signs based on tier."""
        tier_names = {
            "vowel": "স্বরবর্ণ (Vowels)",
            "consonant": "ব্যঞ্জনবর্ণ (Consonants)",
            "digit": "সংখ্যা (Digits)",
            "word": "প্রাথমিক শব্দ (Words)",
            "external": "অতিরিক্ত (Extra)"
        }
        
        # Group signs by tier
        grouped = {}
        for sign in self.session_manager.signs_map.values():
            t = sign.get("tier", "external")
            if t not in grouped:
                grouped[t] = []
            grouped[t].append(sign)
            
        for tier_key in ["vowel", "consonant", "digit", "word", "external"]:
            if tier_key in grouped:
                # Add Header Item (Non-selectable)
                header = QListWidgetItem(f"--- {tier_names[tier_key]} ---")
                header.setFlags(Qt.ItemFlag.NoItemFlags)
                header.setForeground(QColor(ACCENT_GREEN))
                header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
                self.category_list.addItem(header)
                
                # Add Signs
                for sign in sorted(grouped[tier_key], key=lambda x: x["id"]):
                    item = QListWidgetItem(f"{sign['label_bn']} ({sign['label_en']})")
                    item.setData(Qt.ItemDataRole.UserRole, sign["id"])
                    self.category_list.addItem(item)

    def _on_sign_selected(self, current: QListWidgetItem, previous: QListWidgetItem):
        if not current:
            return
            
        sign_id = current.data(Qt.ItemDataRole.UserRole)
        if sign_id is None:
            return
            
        self.active_sign_id = sign_id
        self.session_manager.set_target_sign(sign_id)
        
        sign = self.session_manager.target_sign
        self.target_banner.setText(f"🎯 প্র্যাকটিস করুন: {sign['label_bn']} ({sign['label_en']})")
        self.feedback_badge.setText("প্রস্তুত (Ready)")
        self.accuracy_bar.setValue(0)
        self.next_btn.setEnabled(False)
        self.request_camera_start.emit()

    def _next_sign(self):
        """Move to the next sign in the list."""
        current_row = self.category_list.currentRow()
        for i in range(current_row + 1, self.category_list.count()):
            item = self.category_list.item(i)
            if item.flags() & Qt.ItemFlag.ItemIsSelectable:
                self.category_list.setCurrentRow(i)
                return
        # Reached end
        self.target_banner.setText("✅ সব লেসন শেষ! (All Lessons Complete!)")
        self.active_sign_id = -1
        self.request_camera_stop.emit()

    @pyqtSlot(QImage)
    def update_camera_feed(self, image: QImage):
        """Receives camera frames from main window."""
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(self.camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.camera_label.setPixmap(scaled)

    @pyqtSlot(dict)
    def process_prediction(self, prediction: dict):
        """Receives predictions to evaluate against the target sign."""
        if self.active_sign_id == -1:
            return
            
        # Ignore predictions if already passed
        if self.next_btn.isEnabled():
            return
            
        feedback = self.session_manager.evaluate_frame(prediction)
        
        self.accuracy_bar.setValue(int(feedback["match_score"]))
        self.feedback_badge.setText(feedback["feedback_text"])
        
        if feedback["is_passed"]:
            self.feedback_badge.setStyleSheet(f"color: {ACCENT_GREEN};")
            self.next_btn.setEnabled(True)
            self.accuracy_bar.setValue(100)
            player_instance.play_chime("success")
        else:
            self.feedback_badge.setStyleSheet(f"color: {TEXT_COLOR};")
