import logging
import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QBrush, QPen
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFrame, QProgressBar, QTreeWidget, QTreeWidgetItem, QSplitter,
    QStackedWidget, QTextEdit
)

from core_engine.vision.spatial_hand_engine import SpatialHandEngine

logger = logging.getLogger(__name__)

# Modern Palette
BG_COLOR = "#1E1E2E"
TEXT_COLOR = "#CDD6F4"
ACCENT_COLOR = "#89B4FA"
SUCCESS_COLOR = "#A6E3A1"
ERROR_COLOR = "#F38BA8"
SURFACE_COLOR = "#313244"
PANEL_COLOR = "#181825"

class AcademyDashboard(QWidget):
    """Gamified BdSL Interpreter Academy."""
    
    request_back = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.spatial_engine = SpatialHandEngine()
        self._init_ui()
        
    def _init_ui(self):
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {BG_COLOR};
                color: {TEXT_COLOR};
                font-family: 'Segoe UI', Arial, sans-serif;
            }}
            QTreeWidget {{
                background-color: {PANEL_COLOR};
                border: 1px solid {SURFACE_COLOR};
                border-radius: 8px;
                padding: 5px;
            }}
            QTreeWidget::item:selected {{
                background-color: {ACCENT_COLOR};
                color: #11111B;
                border-radius: 4px;
            }}
            QPushButton {{
                background-color: {SURFACE_COLOR};
                color: {TEXT_COLOR};
                border: 1px solid {ACCENT_COLOR};
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {ACCENT_COLOR};
                color: #11111B;
            }}
            QPushButton#ActionBtn {{
                background-color: {ACCENT_COLOR};
                color: #11111B;
            }}
            QPushButton#ActionBtn:hover {{
                background-color: #74c7ec;
            }}
            QLabel#Header {{
                font-size: 24px;
                font-weight: bold;
                color: {ACCENT_COLOR};
            }}
            QLabel#SubHeader {{
                font-size: 18px;
                font-weight: bold;
                color: {SUCCESS_COLOR};
            }}
            QTextEdit {{
                background-color: {PANEL_COLOR};
                border: 1px solid {SURFACE_COLOR};
                border-radius: 8px;
                color: {TEXT_COLOR};
                font-size: 14px;
            }}
        """)
        
        main_layout = QVBoxLayout(self)
        
        # --- Top Header ---
        header_layout = QHBoxLayout()
        back_btn = QPushButton("← Back to Main")
        back_btn.clicked.connect(self.request_back.emit)
        
        title_lbl = QLabel("BdSL Interpreter Academy")
        title_lbl.setObjectName("Header")
        
        score_lbl = QLabel("XP: 1450 ⭐")
        score_lbl.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        score_lbl.setStyleSheet(f"color: {SUCCESS_COLOR};")
        
        header_layout.addWidget(back_btn)
        header_layout.addStretch()
        header_layout.addWidget(title_lbl)
        header_layout.addStretch()
        header_layout.addWidget(score_lbl)
        main_layout.addLayout(header_layout)
        
        # --- Splitter ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left Panel (Curriculum)
        curriculum_panel = QWidget()
        curriculum_layout = QVBoxLayout(curriculum_panel)
        curriculum_layout.setContentsMargins(0, 0, 0, 0)
        
        c_header = QLabel("Curriculum")
        c_header.setObjectName("SubHeader")
        curriculum_layout.addWidget(c_header)
        
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self._build_curriculum()
        curriculum_layout.addWidget(self.tree)
        
        # Right Panel (Content)
        content_panel = QWidget()
        content_layout = QVBoxLayout(content_panel)
        content_layout.setContentsMargins(10, 0, 0, 0)
        
        self.content_stack = QStackedWidget()
        
        # Page 1: Anatomy & Info
        self.anatomy_page = QWidget()
        anatomy_layout = QVBoxLayout(self.anatomy_page)
        
        self.sign_title = QLabel("Select a lesson to begin")
        self.sign_title.setObjectName("SubHeader")
        anatomy_layout.addWidget(self.sign_title, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.anatomy_viewer = QLabel("[Sign Anatomy / Vector Illustration]")
        self.anatomy_viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.anatomy_viewer.setStyleSheet(f"background-color: {PANEL_COLOR}; border-radius: 10px;")
        self.anatomy_viewer.setMinimumHeight(250)
        anatomy_layout.addWidget(self.anatomy_viewer)
        
        start_practice_btn = QPushButton("Start Live Practice Quiz")
        start_practice_btn.setObjectName("ActionBtn")
        start_practice_btn.clicked.connect(self._start_practice)
        anatomy_layout.addWidget(start_practice_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Page 2: Live Practice Arena
        self.practice_page = QWidget()
        practice_layout = QVBoxLayout(self.practice_page)
        
        arena_layout = QHBoxLayout()
        self.camera_feed = QLabel("Initializing Camera...")
        self.camera_feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_feed.setStyleSheet(f"background-color: {PANEL_COLOR}; border: 2px solid {ACCENT_COLOR}; border-radius: 10px;")
        self.camera_feed.setFixedSize(400, 300)
        
        self.target_overlay = QLabel("[Target Overlay]")
        self.target_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.target_overlay.setStyleSheet(f"background-color: {PANEL_COLOR}; border-radius: 10px;")
        self.target_overlay.setFixedSize(400, 300)
        
        arena_layout.addWidget(self.camera_feed)
        arena_layout.addWidget(self.target_overlay)
        practice_layout.addLayout(arena_layout)
        
        self.posture_hud = QTextEdit()
        self.posture_hud.setReadOnly(True)
        self.posture_hud.setMaximumHeight(80)
        self.posture_hud.setHtml(f"<b style='color:{ACCENT_COLOR};'>AI Posture Tutor:</b> Ready.")
        practice_layout.addWidget(self.posture_hud)
        
        self.match_progress = QProgressBar()
        self.match_progress.setRange(0, 100)
        self.match_progress.setValue(0)
        self.match_progress.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid {SURFACE_COLOR};
                border-radius: 5px;
                text-align: center;
                color: white;
                font-weight: bold;
            }}
            QProgressBar::chunk {{
                background-color: {SUCCESS_COLOR};
                width: 20px;
            }}
        """)
        practice_layout.addWidget(self.match_progress)
        
        stop_btn = QPushButton("Stop Practice")
        stop_btn.clicked.connect(self._stop_practice)
        practice_layout.addWidget(stop_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.content_stack.addWidget(self.anatomy_page)
        self.content_stack.addWidget(self.practice_page)
        content_layout.addWidget(self.content_stack)
        
        splitter.addWidget(curriculum_panel)
        splitter.addWidget(content_panel)
        splitter.setSizes([300, 700])
        
        main_layout.addWidget(splitter)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_practice_frame)
        self.cap = None
        
    def _build_curriculum(self):
        l1 = QTreeWidgetItem(self.tree, ["Level 1: Alphabets & Digits"])
        QTreeWidgetItem(l1, ["স্বরবর্ণ (Vowels)"])
        QTreeWidgetItem(l1, ["ব্যঞ্জনবর্ণ (Consonants)"])
        QTreeWidgetItem(l1, ["সংখ্যা (Numbers)"])
        
        l2 = QTreeWidgetItem(self.tree, ["Level 2: Daily Words"])
        QTreeWidgetItem(l2, ["Greetings"])
        QTreeWidgetItem(l2, ["Family"])
        QTreeWidgetItem(l2, ["Emotions"])
        
        l3 = QTreeWidgetItem(self.tree, ["Level 3: Sentence Grammar"])
        QTreeWidgetItem(l3, ["Subject-Object-Verb (SOV) Structure"])
        QTreeWidgetItem(l3, ["Questions & Negations"])
        
        l4 = QTreeWidgetItem(self.tree, ["Level 4: Interpreter Simulation"])
        QTreeWidgetItem(l4, ["News Broadcasting"])
        QTreeWidgetItem(l4, ["Public Speech Translation"])
        
        self.tree.expandAll()
        
    def _start_practice(self):
        self.content_stack.setCurrentWidget(self.practice_page)
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30) # ~33fps
        self.posture_hud.setHtml(f"<b style='color:{ACCENT_COLOR};'>AI Posture Tutor:</b> ডান হাতের তর্জনী দিয়ে বাম হাতের তালু স্পর্শ করুন। (Touch the left palm with the right index finger).")
        
    def _stop_practice(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.content_stack.setCurrentWidget(self.anatomy_page)
        self.match_progress.setValue(0)
        
    def _update_practice_frame(self):
        if not self.cap:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            return
            
        frame = cv2.flip(frame, 1)
        
        # Extract features
        features = self.spatial_engine.extract_spatial_features(frame)
        
        # Simple simulated grading based on hand presence
        score = 0
        if features["has_left"] and features["has_right"]:
            score = 85
        elif features["has_right"] or features["has_left"]:
            score = 45
            
        self.match_progress.setValue(score)
        if score > 80:
            self.posture_hud.setHtml(f"<b style='color:{SUCCESS_COLOR};'>Excellent! Hold it.</b>")
        else:
            self.posture_hud.setHtml(f"<b style='color:{ACCENT_COLOR};'>AI Posture Tutor:</b> Adjust your hand orientation.")
        
        # Draw basic landmarks for visual feedback
        for i in range(42):
            if np.any(features["raw_landmarks"][i]):
                x, y = int(features["raw_landmarks"][i][0] * frame.shape[1]), int(features["raw_landmarks"][i][1] * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(400, 300, Qt.AspectRatioMode.KeepAspectRatio)
        self.camera_feed.setPixmap(pixmap)
        
    def closeEvent(self, event):
        self._stop_practice()
        super().closeEvent(event)
