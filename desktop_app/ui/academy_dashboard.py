import logging
import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFrame, QProgressBar, QTreeWidget, QTreeWidgetItem, QSplitter,
    QStackedWidget, QTextEdit, QMessageBox
)
from PyQt6.QtGui import QDesktopServices, QPixmap, QImage, QFont
from PyQt6.QtCore import QUrl, pyqtSignal, pyqtSlot, QTimer, Qt
from desktop_app.controllers.certificate_generator import CertificateGenerator
from desktop_app.ui.theme import ThemeStyles
from desktop_app.ui.components.circular_gauge import CircularAccuracyGauge
from desktop_app.ui.components.motion_trajectory_viewer import MotionTrajectoryViewer

from core_engine.vision.spatial_hand_engine import SpatialHandEngine
from desktop_app.ui.components.ghost_overlay import GhostOverlayPainter
import random

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
        self.ghost_painter = GhostOverlayPainter()
        
        # Interactive Quiz State
        self.quiz_active = False
        self.exam_active = False
        self.countdown_ticks = 0
        self.xp_score = 0
        self.streak = 0
        self.exam_signs = []
        self.current_exam_index = 0
        self.exam_correct = 0
        
        # Simulated Reference Data (for demonstration of ghost overlay tracking)
        # In production this would be loaded from a dataset
        self.mock_ref_landmarks = []
        for i in range(42):
            self.mock_ref_landmarks.append((100 + i*5, 100 + (i%5)*5))
            
        self._init_ui()
        
    def _init_ui(self):
        self.setStyleSheet(ThemeStyles.get_global_stylesheet())
        
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
        
        self.exam_btn = QPushButton("🎓 Take Certification Exam")
        self.exam_btn.setStyleSheet(f"background-color: {SUCCESS_COLOR}; color: #11111B;")
        self.exam_btn.clicked.connect(self._start_exam)
        curriculum_layout.addWidget(self.exam_btn)
        
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
        
        # Wrap camera in a widget to overlay the trajectory viewer
        cam_container = QWidget()
        cam_container.setFixedSize(400, 300)
        
        self.camera_feed = QLabel(cam_container)
        self.camera_feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_feed.setText("Initializing Camera...")
        self.camera_feed.setFixedSize(400, 300)
        self.camera_feed.setObjectName("GlassCard")
        
        self.trajectory_viewer = MotionTrajectoryViewer(cam_container)
        self.trajectory_viewer.setFixedSize(400, 300)
        
        self.target_overlay = QLabel("[Target Overlay]")
        self.target_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.target_overlay.setObjectName("GlassCard")
        self.target_overlay.setFixedSize(400, 300)
        
        arena_layout.addWidget(cam_container)
        arena_layout.addWidget(self.target_overlay)
        practice_layout.addLayout(arena_layout)
        
        self.posture_hud = QTextEdit()
        self.posture_hud.setReadOnly(True)
        self.posture_hud.setMaximumHeight(80)
        self.posture_hud.setHtml(f"<b style='color:#06B6D4;'>AI Posture Tutor:</b> Ready.")
        practice_layout.addWidget(self.posture_hud)
        
        # Score HUD
        score_layout = QHBoxLayout()
        self.xp_label = QLabel("XP: 0")
        self.xp_label.setStyleSheet("color: #06B6D4; font-weight: bold;")
        self.streak_label = QLabel("Streak: 0 🔥")
        self.streak_label.setStyleSheet("color: #F43F5E; font-weight: bold;")
        score_layout.addWidget(self.xp_label)
        score_layout.addWidget(self.streak_label)
        practice_layout.addLayout(score_layout)
        
        # Circular Accuracy Gauge
        gauge_layout = QHBoxLayout()
        self.match_gauge = CircularAccuracyGauge(radius=40, thickness=10)
        gauge_layout.addWidget(self.match_gauge, alignment=Qt.AlignmentFlag.AlignCenter)
        practice_layout.addLayout(gauge_layout)
        
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
        self.timer.start(33) # ~30fps
        
        # Start Quiz
        self.quiz_active = True
        self.exam_active = False
        self.countdown_ticks = 150 # 5 seconds at 30fps
        
        self.posture_hud.setHtml(f"<b style='color:{ACCENT_COLOR};'>Quiz Starting!</b> Get ready to sign...")
        self.target_overlay.setText("Quiz Active")
        
    def _start_exam(self):
        self.content_stack.setCurrentWidget(self.practice_page)
        self.cap = cv2.VideoCapture(0)
        self.timer.start(33)
        
        all_signs = ["স্বরবর্ণ (Vowels)", "ব্যঞ্জনবর্ণ (Consonants)", "সংখ্যা (Numbers)", "Greetings", "Family", "Emotions", "Subject-Object-Verb (SOV) Structure", "Questions & Negations", "News Broadcasting", "Public Speech Translation"]
        self.exam_signs = random.choices(all_signs, k=10)
        self.current_exam_index = 0
        self.exam_correct = 0
        
        self.quiz_active = False
        self.exam_active = True
        self.countdown_ticks = 150
        
        self.posture_hud.setHtml(f"<b style='color:{ACCENT_COLOR};'>Exam Started!</b> Sign 1/10: {self.exam_signs[0]}")
        self.target_overlay.setText(f"Exam: {self.exam_signs[0]}")
        
    def _end_exam(self):
        self._stop_practice()
        score_percent = (self.exam_correct / 10.0) * 100
        
        msg = QMessageBox(self)
        msg.setWindowTitle("Certification Exam Complete")
        msg.setStyleSheet(f"background-color: {PANEL_COLOR}; color: {TEXT_COLOR};")
        
        if score_percent >= 70:
            msg.setText(f"Congratulations! You passed with a score of {score_percent}%.\nWould you like to download your verifiable certificate?")
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            ret = msg.exec()
            if ret == QMessageBox.StandardButton.Yes:
                cg = CertificateGenerator()
                filepath = cg.generate("Jane Doe", "Tier 4 Master", score_percent, 45)
                QDesktopServices.openUrl(QUrl.fromLocalFile(filepath))
        else:
            msg.setText(f"You scored {score_percent}%. Keep practicing and try again!")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
        
    def _stop_practice(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.content_stack.setCurrentWidget(self.anatomy_page)
        if hasattr(self, 'match_gauge'):
            self.match_gauge.set_value(0)
        
        
    def _update_practice_frame(self):
        if not self.cap:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            return
            
        frame = cv2.flip(frame, 1)
        
        # Extract features
        features = self.spatial_engine.extract_spatial_features(frame)
        
        # Process Live Landmarks
        live_landmarks = []
        left_wrist = None
        right_wrist = None
        
        if features["has_left"] or features["has_right"]:
            for i in range(42):
                if np.any(features["raw_landmarks"][i]):
                    x = float(features["raw_landmarks"][i][0] * frame.shape[1])
                    y = float(features["raw_landmarks"][i][1] * frame.shape[0])
                    live_landmarks.append((x, y))
                    
                    if i == 0:  # left wrist
                        left_wrist = (x, y)
                    elif i == 21: # right wrist
                        right_wrist = (x, y)
                        
        self.trajectory_viewer.update_trajectory(left_wrist, right_wrist)
        
        # Ghost Overlay rendering
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Add overlay
        final_img = self.ghost_painter.draw_overlay(qt_img, self.mock_ref_landmarks, live_landmarks)
        pixmap = QPixmap.fromImage(final_img).scaled(400, 300, Qt.AspectRatioMode.KeepAspectRatio)
        self.camera_feed.setPixmap(pixmap)
        
        # Quiz Logic
        if self.quiz_active or self.exam_active:
            if self.countdown_ticks > 0:
                self.countdown_ticks -= 1
                secs = self.countdown_ticks // 30
                if self.exam_active:
                    self.posture_hud.setHtml(f"<b style='color:{ACCENT_COLOR};'>Exam [{self.current_exam_index+1}/10]:</b> Show sign in {secs}s...")
                else:
                    self.posture_hud.setHtml(f"<b style='color:{ACCENT_COLOR};'>Quiz:</b> Show sign in {secs}s...")
                
                score = self.ghost_painter.calculate_alignment_score(self.mock_ref_landmarks, live_landmarks)
                self.match_gauge.set_value(score)
                
                if score > 80.0:
                    self.posture_hud.setHtml(f"<b style='color:#10B981;'>Excellent! Perfect Match.</b>")
                    self.xp_score += 50
                    self.streak += 1
                    if self.exam_active:
                        self.exam_correct += 1
                        self.current_exam_index += 1
                        if self.current_exam_index >= 10:
                            self._end_exam()
                            return
                        else:
                            self.countdown_ticks = 150
                            self.target_overlay.setText(f"Exam: {self.exam_signs[self.current_exam_index]}")
                    else:
                        self.quiz_active = False # End quiz early on success
            else:
                self.posture_hud.setHtml(f"<b style='color:#F43F5E;'>Time's Up!</b>")
                self.streak = 0
                if self.exam_active:
                    self.current_exam_index += 1
                    if self.current_exam_index >= 10:
                        self._end_exam()
                        return
                    else:
                        self.countdown_ticks = 150
                        self.target_overlay.setText(f"Exam: {self.exam_signs[self.current_exam_index]}")
                else:
                    self.quiz_active = False
                
            self.xp_label.setText(f"XP: {self.xp_score}")
            self.streak_label.setText(f"Streak: {self.streak} 🔥")
            
        else:
            self.match_gauge.set_value(0)
        
    def closeEvent(self, event):
        self._stop_practice()
        super().closeEvent(event)
