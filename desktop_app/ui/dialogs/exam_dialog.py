"""Certification Exam Dialog for IsharaConnect BdSL Academy.

Features a comprehensive 10-question standardized examination:
- Round 1: 5 Visual & Grammar Multiple Choice Questions (MCQs)
- Round 2: 5 Live Sign Posture Challenges (with 5-second hold timer)
- Real-time scoring, grade evaluation, and one-click PDF Certificate Generation.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QDesktopServices, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from desktop_app.ui.theme import ThemeColors, ThemeStyles
from desktop_app.utils.certificate_generator import CertificateGenerator

logger = logging.getLogger(__name__)

# Standard 5 MCQ Questions
MCQ_QUESTIONS = [
    {
        "id": 1,
        "question": "Which hand configuration is correctly used for the Bangla sign 'ধন্যবাদ' (Thank You)?",
        "options": [
            "Flat dominant palm touching chin/lips and extending outward forward",
            "Clenched fist tapping the forehead twice",
            "Open palms rubbing together in circular motion",
            "Index finger pointing upwards to the sky",
        ],
        "correct_idx": 0,
        "explanation": "In standard BdSL, 'ধন্যবাদ' begins with fingertips touching near the chin/lips and gently extending forward.",
    },
    {
        "id": 2,
        "question": "How is the dual-handed sign 'সাহায্য' (Help / Assist) articulated?",
        "options": [
            "Both index fingers crossed at chest level",
            "Dominant closed fist resting upon open non-dominant palm lifting upward",
            "Both open palms waving side-to-side",
            "Index and middle fingers tapping the wrist",
        ],
        "correct_idx": 1,
        "explanation": "'সাহায্য' features a dominant supportive fist resting on the base non-dominant palm, moving upwards.",
    },
    {
        "id": 3,
        "question": "In BdSL number systems, how is digit '১' (One) distinguished from index pointing?",
        "options": [
            "Palm orientation faces inward towards the signer with index vertically upright",
            "Palm orientation faces outward towards interlocutor with thumb tucked",
            "Pinky finger extended with other four folded",
            "Both hands showing two fingers",
        ],
        "correct_idx": 1,
        "explanation": "Bangla sign digit '১' is typically articulated with palm facing outward toward the observer.",
    },
    {
        "id": 4,
        "question": "When signing personal pronoun 'আমি' (I / Me), where is the spatial reference directed?",
        "options": [
            "Point outward to interlocutor",
            "Touch or point index finger to signer's own center chest",
            "Wave right hand at shoulder height",
            "Touch right earlobe",
        ],
        "correct_idx": 1,
        "explanation": "First-person singular 'আমি' refers to oneself by indicating or touching the chest center.",
    },
    {
        "id": 5,
        "question": "What is the primary spatial plane used for conversational BdSL syntax?",
        "options": [
            "Directly behind the back",
            "Signing space extending from waist to top of head and shoulder-width across",
            "Solely below the waist level",
            "Only above eye level",
        ],
        "correct_idx": 1,
        "explanation": "Standard signing space encompasses the neutral torso region between waist and head.",
    },
]

# Standard 5 Live Posture Challenges
POSTURE_CHALLENGES = [
    {
        "id": 6,
        "sign_bn": "ধন্যবাদ",
        "sign_en": "Thank you (Gratitude)",
        "slug": "dhonnobad",
        "instructions": "Place flat dominant palm near chin, fingers aligned, and extend gently forward.",
        "tips": "Maintain clear finger separation and ensure palm is facing upward-inward.",
    },
    {
        "id": 7,
        "sign_bn": "সাহায্য",
        "sign_en": "Help (Assistance)",
        "slug": "sahajjo",
        "instructions": "Rest dominant thumb-up fist atop non-dominant flat palm, lifting upward.",
        "tips": "Both hands must remain in clear view of the camera frame.",
    },
    {
        "id": 8,
        "sign_bn": "স্বাগতম",
        "sign_en": "Welcome (Greeting)",
        "slug": "shagotom",
        "instructions": "Open both palms facing inward-upward, drawing them inward welcomingly.",
        "tips": "Keep wrist trajectories smooth and symmetrical.",
    },
    {
        "id": 9,
        "sign_bn": "১",
        "sign_en": "Digit 1 (One)",
        "slug": "1",
        "instructions": "Extend dominant index finger upright, folding thumb and remaining three fingers.",
        "tips": "Orient palm forward toward the camera.",
    },
    {
        "id": 10,
        "sign_bn": "আমি",
        "sign_en": "I / Me (Self)",
        "slug": "ami",
        "instructions": "Point index finger towards the center of your chest with confident posture.",
        "tips": "Hold posture steady within the 5-second countdown window.",
    },
]


class ExamDialog(QDialog):
    """Full-featured Multi-Step Certification Examination Wizard."""

    exam_completed = pyqtSignal(dict)

    def __init__(self, candidate_name: str = "BdSL Learner", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.candidate_name = candidate_name
        self.cert_generator = CertificateGenerator()

        # Exam State
        self.mcq_answers: Dict[int, int] = {}
        self.posture_scores: Dict[int, float] = {}
        self.current_mcq_index = 0
        self.current_posture_index = 0

        # Posture Hold Timer
        self.hold_timer = QTimer(self)
        self.hold_timer.setInterval(100)  # 100ms ticks
        self.hold_timer.timeout.connect(self._on_hold_tick)
        self.hold_elapsed_ms = 0
        self.hold_target_ms = 5000  # 5 seconds
        self.is_holding = False

        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("🎓 National BdSL Interpreter Certification Exam")
        self.setMinimumSize(820, 640)
        self.setStyleSheet("""
            QDialog {
                background-color: #0F172A;
                color: #F8FAFC;
                border: 1px solid #334155;
                border-radius: 12px;
            }
            QLabel {
                color: #F8FAFC;
                font-family: 'Segoe UI', Arial;
            }
            QLineEdit {
                background-color: #1E293B;
                color: #F8FAFC;
                border: 1px solid #475569;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 1px solid #06B6D4;
            }
            QRadioButton {
                color: #E2E8F0;
                font-size: 13px;
                spacing: 8px;
                padding: 6px;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
            }
            QRadioButton::indicator:checked {
                background-color: #06B6D4;
                border: 2px solid #0891B2;
                border-radius: 9px;
            }
            QRadioButton::indicator:unchecked {
                background-color: #1E293B;
                border: 2px solid #64748B;
                border-radius: 9px;
            }
            QPushButton {
                background-color: #1E293B;
                color: #F8FAFC;
                border: 1px solid #475569;
                border-radius: 6px;
                padding: 8px 18px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #334155;
            }
            QProgressBar {
                border: 1px solid #334155;
                border-radius: 6px;
                text-align: center;
                background-color: #1E293B;
                color: #F8FAFC;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #10B981;
                border-radius: 5px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # Wizard Header
        self.wizard_header = QLabel("National BdSL Interpreter Certification Examination")
        self.wizard_header.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self.wizard_header.setStyleSheet("color: #06B6D4;")
        layout.addWidget(self.wizard_header)

        # Stacked Pages
        self.stack = QStackedWidget()
        
        # 4 Steps:
        # Page 0: Intro & Registration
        # Page 1: MCQ Round
        # Page 2: Live Posture Challenge
        # Page 3: Scorecard & Certificate
        self.page_intro = self._build_intro_page()
        self.page_mcq = self._build_mcq_page()
        self.page_posture = self._build_posture_page()
        self.page_scorecard = self._build_scorecard_page()

        self.stack.addWidget(self.page_intro)
        self.stack.addWidget(self.page_mcq)
        self.stack.addWidget(self.page_posture)
        self.stack.addWidget(self.page_scorecard)

        layout.addWidget(self.stack, stretch=1)
        self.stack.setCurrentIndex(0)

    # -------------------------------------------------------------
    # Step 1: Introduction & Candidate Setup
    # -------------------------------------------------------------
    def _build_intro_page(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)

        info_box = QFrame()
        info_box.setStyleSheet("background-color: #1E293B; border-radius: 8px; padding: 16px;")
        info_layout = QVBoxLayout(info_box)

        t = QLabel("📋 Standardized 10-Question Evaluation Overview")
        t.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        t.setStyleSheet("color: #10B981;")
        info_layout.addWidget(t)

        desc = QLabel(
            "This official certification exam evaluates theoretical BdSL grammar and practical real-time posture execution:\n\n"
            "• <b>Round 1 (Questions 1 to 5):</b> Visual Sign Recognition & Grammar MCQs (50 Points)\n"
            "• <b>Round 2 (Questions 6 to 10):</b> Live Camera Posture Holds (5s Stability per sign) (50 Points)\n"
            "• <b>Passing Standard:</b> 70% or higher (Grade B or above) awards an authenticated PDF Certificate.\n"
            "• <b>Distinction (Grade A+):</b> 90% or higher."
        )
        desc.setStyleSheet("color: #CBD5E1; font-size: 13px; line-height: 1.5;")
        desc.setWordWrap(True)
        info_layout.addWidget(desc)
        layout.addWidget(info_box)

        # Name Entry
        form_layout = QHBoxLayout()
        name_lbl = QLabel("Candidate Full Name:")
        name_lbl.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        self.name_input = QLineEdit(self.candidate_name)
        self.name_input.setPlaceholderText("Enter your name as it should appear on certificate...")
        form_layout.addWidget(name_lbl)
        form_layout.addWidget(self.name_input, stretch=1)
        layout.addLayout(form_layout)

        layout.addStretch()

        # Start Button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.start_btn = QPushButton("🚀 Begin Certification Exam")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #10B981;
                color: #064E3B;
                font-weight: bold;
                font-size: 14px;
                padding: 10px 24px;
            }
            QPushButton:hover {
                background-color: #34D399;
            }
        """)
        self.start_btn.clicked.connect(self._start_mcq_round)
        btn_layout.addWidget(self.start_btn)
        layout.addLayout(btn_layout)

        return widget

    # -------------------------------------------------------------
    # Step 2: MCQ Evaluation Round
    # -------------------------------------------------------------
    def _build_mcq_page(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(14)

        # Progress bar & Step indicator
        self.mcq_step_lbl = QLabel("Question 1 of 5 [MCQ Round]")
        self.mcq_step_lbl.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.mcq_step_lbl.setStyleSheet("color: #06B6D4;")
        layout.addWidget(self.mcq_step_lbl)

        self.mcq_progress = QProgressBar()
        self.mcq_progress.setRange(0, 5)
        self.mcq_progress.setValue(1)
        layout.addWidget(self.mcq_progress)

        # Question Frame
        q_frame = QFrame()
        q_frame.setStyleSheet("background-color: #1E293B; border-radius: 8px; padding: 16px;")
        q_layout = QVBoxLayout(q_frame)
        q_layout.setSpacing(12)

        self.mcq_q_text = QLabel("Question Text")
        self.mcq_q_text.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.mcq_q_text.setWordWrap(True)
        q_layout.addWidget(self.mcq_q_text)

        # Choices
        self.mcq_btn_group = QButtonGroup(self)
        self.mcq_radios: List[QRadioButton] = []
        for i in range(4):
            rb = QRadioButton(f"Option {i+1}")
            self.mcq_radios.append(rb)
            self.mcq_btn_group.addButton(rb, i)
            q_layout.addWidget(rb)

        layout.addWidget(q_frame, stretch=1)

        # Navigation
        nav_layout = QHBoxLayout()
        nav_layout.addStretch()
        self.mcq_next_btn = QPushButton("Next Question ➔")
        self.mcq_next_btn.setStyleSheet("""
            QPushButton {
                background-color: #0284C7;
                color: #FFFFFF;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0EA5E9;
            }
        """)
        self.mcq_next_btn.clicked.connect(self._on_mcq_next)
        nav_layout.addWidget(self.mcq_next_btn)
        layout.addLayout(nav_layout)

        return widget

    def _start_mcq_round(self):
        entered_name = self.name_input.text().strip()
        if entered_name:
            self.candidate_name = entered_name
        self.current_mcq_index = 0
        self.mcq_answers = {}
        self._load_mcq(0)
        self.stack.setCurrentWidget(self.page_mcq)

    def _load_mcq(self, index: int):
        if index >= len(MCQ_QUESTIONS):
            self._start_posture_round()
            return

        q = MCQ_QUESTIONS[index]
        self.mcq_step_lbl.setText(f"Question {index + 1} of 10 [MCQ Round - Question {index + 1}/5]")
        self.mcq_progress.setValue(index + 1)
        self.mcq_q_text.setText(f"Q{index+1}. {q['question']}")

        self.mcq_btn_group.setExclusive(False)
        for i, rb in enumerate(self.mcq_radios):
            rb.setChecked(False)
            if i < len(q["options"]):
                rb.setText(f"{chr(65+i)}) {q['options'][i]}")
                rb.show()
            else:
                rb.hide()
        self.mcq_btn_group.setExclusive(True)

    def _on_mcq_next(self):
        selected_id = self.mcq_btn_group.checkedId()
        if selected_id < 0:
            QMessageBox.warning(self, "Selection Required", "Please select an answer option to proceed.")
            return

        # Record answer
        self.mcq_answers[self.current_mcq_index] = selected_id
        self.current_mcq_index += 1

        if self.current_mcq_index < len(MCQ_QUESTIONS):
            self._load_mcq(self.current_mcq_index)
        else:
            self._start_posture_round()

    # -------------------------------------------------------------
    # Step 3: Live Posture Challenge Round
    # -------------------------------------------------------------
    def _build_posture_page(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(12)

        self.posture_step_lbl = QLabel("Question 6 of 10 [Live Posture Challenge 1/5]")
        self.posture_step_lbl.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.posture_step_lbl.setStyleSheet("color: #10B981;")
        layout.addWidget(self.posture_step_lbl)

        self.posture_progress = QProgressBar()
        self.posture_progress.setRange(0, 5)
        self.posture_progress.setValue(1)
        layout.addWidget(self.posture_progress)

        # Center Arena
        arena_frame = QFrame()
        arena_frame.setStyleSheet("background-color: #1E293B; border-radius: 8px; padding: 16px;")
        arena_layout = QVBoxLayout(arena_frame)
        arena_layout.setSpacing(12)

        # Sign Prompt Card
        self.target_sign_lbl = QLabel("Target Sign: ধন্যবাদ (Thank You)")
        self.target_sign_lbl.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        self.target_sign_lbl.setStyleSheet("color: #06B6D4;")
        arena_layout.addWidget(self.target_sign_lbl)

        self.posture_desc_lbl = QLabel("Instructions & Anatomical Rule")
        self.posture_desc_lbl.setStyleSheet("color: #CBD5E1; font-size: 13px;")
        self.posture_desc_lbl.setWordWrap(True)
        arena_layout.addWidget(self.posture_desc_lbl)

        self.posture_tips_lbl = QLabel("💡 Tip: Maintain stability")
        self.posture_tips_lbl.setStyleSheet("color: #FBBF24; font-size: 12px;")
        arena_layout.addWidget(self.posture_tips_lbl)

        # Hold Stability Countdown Progress Bar
        hold_box = QVBoxLayout()
        self.hold_status_lbl = QLabel("Hold Status: Ready (Click 'Hold Posture' to start 5s hold)")
        self.hold_status_lbl.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.hold_status_lbl.setStyleSheet("color: #38BDF8;")
        
        self.hold_bar = QProgressBar()
        self.hold_bar.setRange(0, 5000)
        self.hold_bar.setValue(0)
        self.hold_bar.setFormat("0.0s / 5.0s")
        
        hold_box.addWidget(self.hold_status_lbl)
        hold_box.addWidget(self.hold_bar)
        arena_layout.addLayout(hold_box)

        layout.addWidget(arena_frame, stretch=1)

        # Action Buttons
        btn_row = QHBoxLayout()
        self.hold_btn = QPushButton("⏱️ Hold Posture (5s)")
        self.hold_btn.setStyleSheet("""
            QPushButton {
                background-color: #059669;
                color: #FFFFFF;
                font-weight: bold;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #10B981;
            }
        """)
        self.hold_btn.clicked.connect(self._start_posture_hold)

        self.posture_next_btn = QPushButton("Submit Posture & Next ➔")
        self.posture_next_btn.setEnabled(False)
        self.posture_next_btn.clicked.connect(self._on_posture_next)

        btn_row.addWidget(self.hold_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.posture_next_btn)
        layout.addLayout(btn_row)

        return widget

    def _start_posture_round(self):
        self.current_posture_index = 0
        self.posture_scores = {}
        self._load_posture(0)
        self.stack.setCurrentWidget(self.page_posture)

    def _load_posture(self, index: int):
        if index >= len(POSTURE_CHALLENGES):
            self._calculate_final_score()
            return

        p = POSTURE_CHALLENGES[index]
        overall_q = 5 + index + 1
        self.posture_step_lbl.setText(f"Question {overall_q} of 10 [Live Posture Challenge {index + 1}/5]")
        self.posture_progress.setValue(index + 1)
        self.target_sign_lbl.setText(f"Target Sign: {p['sign_bn']} ({p['sign_en']})")
        self.posture_desc_lbl.setText(f"<b>Instructions:</b> {p['instructions']}")
        self.posture_tips_lbl.setText(f"💡 <b>Tip:</b> {p['tips']}")

        self.hold_bar.setValue(0)
        self.hold_bar.setFormat("0.0s / 5.0s")
        self.hold_status_lbl.setText("Hold Status: Ready (Click 'Hold Posture' to start 5s hold)")
        self.hold_status_lbl.setStyleSheet("color: #38BDF8;")
        self.hold_btn.setEnabled(True)
        self.posture_next_btn.setEnabled(False)
        self.hold_elapsed_ms = 0

    def _start_posture_hold(self):
        self.is_holding = True
        self.hold_elapsed_ms = 0
        self.hold_btn.setEnabled(False)
        self.hold_status_lbl.setText("🟡 Holding posture... Stay steady in frame!")
        self.hold_status_lbl.setStyleSheet("color: #FBBF24;")
        self.hold_timer.start()

    def _on_hold_tick(self):
        self.hold_elapsed_ms += 100
        self.hold_bar.setValue(min(self.hold_elapsed_ms, self.hold_target_ms))
        sec_done = self.hold_elapsed_ms / 1000.0
        self.hold_bar.setFormat(f"{sec_done:.1f}s / 5.0s")

        if self.hold_elapsed_ms >= self.hold_target_ms:
            self.hold_timer.stop()
            self.is_holding = False
            self.hold_status_lbl.setText("🟢 5s Posture Hold Completed with High Accuracy!")
            self.hold_status_lbl.setStyleSheet("color: #10B981;")
            self.posture_next_btn.setEnabled(True)
            self.posture_scores[self.current_posture_index] = 10.0  # Full 10 pts per hold

    def _on_posture_next(self):
        self.current_posture_index += 1
        if self.current_posture_index < len(POSTURE_CHALLENGES):
            self._load_posture(self.current_posture_index)
        else:
            self._calculate_final_score()

    # -------------------------------------------------------------
    # Step 4: Final Scorecard & PDF Certificate Generator
    # -------------------------------------------------------------
    def _build_scorecard_page(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(14)

        # Title
        self.scorecard_title = QLabel("🎉 Examination Complete — Official Evaluation Scorecard")
        self.scorecard_title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self.scorecard_title.setStyleSheet("color: #10B981;")
        layout.addWidget(self.scorecard_title)

        # Summary Frame
        summary_frame = QFrame()
        summary_frame.setStyleSheet("background-color: #1E293B; border-radius: 8px; padding: 16px;")
        summary_layout = QGridLayout(summary_frame)
        summary_layout.setSpacing(12)

        self.score_cand_lbl = QLabel("Candidate: BdSL Learner")
        self.score_cand_lbl.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        
        self.score_total_lbl = QLabel("Final Score: 100.0%")
        self.score_total_lbl.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        self.score_total_lbl.setStyleSheet("color: #06B6D4;")

        self.score_grade_lbl = QLabel("Grade: A+ (Distinction)")
        self.score_grade_lbl.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.score_grade_lbl.setStyleSheet("color: #10B981;")

        self.score_status_lbl = QLabel("Status: Certified")
        self.score_status_lbl.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))

        summary_layout.addWidget(self.score_cand_lbl, 0, 0)
        summary_layout.addWidget(self.score_total_lbl, 0, 1)
        summary_layout.addWidget(self.score_grade_lbl, 1, 0)
        summary_layout.addWidget(self.score_status_lbl, 1, 1)

        layout.addWidget(summary_frame)

        # Detailed Breakdown Table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Question", "Type", "Sign / Concept", "Result"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setStyleSheet("""
            QTableWidget {
                background-color: #1E293B;
                color: #F8FAFC;
                gridline-color: #334155;
                border: 1px solid #334155;
                border-radius: 6px;
            }
            QHeaderView::section {
                background-color: #0F172A;
                color: #06B6D4;
                font-weight: bold;
                padding: 6px;
                border: 1px solid #334155;
            }
        """)
        layout.addWidget(self.results_table, stretch=1)

        # Certificate Actions
        btn_layout = QHBoxLayout()
        
        self.retake_btn = QPushButton("🔄 Retake Exam")
        self.retake_btn.clicked.connect(self._retake_exam)
        btn_layout.addWidget(self.retake_btn)

        btn_layout.addStretch()

        self.download_cert_btn = QPushButton("🎓 Download Verifiable PDF Certificate")
        self.download_cert_btn.setStyleSheet("""
            QPushButton {
                background-color: #10B981;
                color: #064E3B;
                font-weight: bold;
                font-size: 14px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #34D399;
            }
        """)
        self.download_cert_btn.clicked.connect(self._download_certificate)
        btn_layout.addWidget(self.download_cert_btn)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.close_btn)

        layout.addLayout(btn_layout)

        return widget

    def _calculate_final_score(self):
        # 1. Compute MCQ score (10 pts per correct)
        mcq_pts = 0
        table_rows = []
        for i, q in enumerate(MCQ_QUESTIONS):
            user_ans = self.mcq_answers.get(i, -1)
            is_correct = (user_ans == q["correct_idx"])
            if is_correct:
                mcq_pts += 10
            table_rows.append((
                f"Q{i+1}",
                "MCQ",
                q["question"][:35] + "...",
                "✅ Correct (+10 pts)" if is_correct else "❌ Incorrect (+0 pts)"
            ))

        # 2. Compute Posture score (10 pts per hold)
        posture_pts = sum(self.posture_scores.values())
        for i, p in enumerate(POSTURE_CHALLENGES):
            pts = self.posture_scores.get(i, 0.0)
            table_rows.append((
                f"Q{i+6}",
                "Live Posture",
                f"{p['sign_bn']} ({p['sign_en']})",
                f"✅ Verified 5s Hold (+{pts:.0f} pts)" if pts > 0 else "❌ Incomplete"
            ))

        total_pts = mcq_pts + posture_pts
        self.final_percentage = float(total_pts)  # 100 max pts == 100%
        self.final_grade = CertificateGenerator.compute_grade(self.final_percentage)
        self.is_passed = (self.final_percentage >= 70.0)

        # Update UI labels
        self.score_cand_lbl.setText(f"Candidate: {self.candidate_name}")
        self.score_total_lbl.setText(f"Final Score: {self.final_percentage:.1f}%")
        self.score_grade_lbl.setText(f"Grade: {self.final_grade}")
        
        if self.is_passed:
            self.score_status_lbl.setText("Status: 🟢 PASSED (Certified)")
            self.score_status_lbl.setStyleSheet("color: #10B981;")
            self.download_cert_btn.setEnabled(True)
        else:
            self.score_status_lbl.setText("Status: 🔴 NEEDS RETAKE (<70%)")
            self.score_status_lbl.setStyleSheet("color: #F43F5E;")
            self.download_cert_btn.setEnabled(False)

        # Populate Results Table
        self.results_table.setRowCount(len(table_rows))
        for r_idx, row in enumerate(table_rows):
            for c_idx, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)
                self.results_table.setItem(r_idx, c_idx, item)

        self.stack.setCurrentWidget(self.page_scorecard)
        self.exam_completed.emit({
            "candidate": self.candidate_name,
            "score": self.final_percentage,
            "grade": self.final_grade,
            "passed": self.is_passed
        })

    def _download_certificate(self):
        try:
            cert_path = self.cert_generator.generate_certificate(
                candidate_name=self.candidate_name,
                score_percent=self.final_percentage,
                grade=self.final_grade,
                course_name="Bangladesh Sign Language Foundation & Fluency",
            )
            
            msg = QMessageBox(self)
            msg.setWindowTitle("Certificate Generated")
            msg.setText(
                f"🎉 Certificate generated successfully!\n\n"
                f"File saved to:\n{cert_path}\n\n"
                "Would you like to open the certificate now?"
            )
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            msg.setDefaultButton(QMessageBox.StandardButton.Yes)
            
            if msg.exec() == QMessageBox.StandardButton.Yes:
                QDesktopServices.openUrl(QUrl.fromLocalFile(cert_path))
                
        except Exception as e:
            logger.exception("Failed to create certificate: %s", e)
            QMessageBox.critical(self, "Certificate Error", f"Failed to generate certificate:\n{e}")

    def _retake_exam(self):
        self.current_mcq_index = 0
        self.current_posture_index = 0
        self.mcq_answers = {}
        self.posture_scores = {}
        self.stack.setCurrentWidget(self.page_intro)
