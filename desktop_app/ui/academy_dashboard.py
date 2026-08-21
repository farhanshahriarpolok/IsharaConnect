"""Gamified BdSL Interpreter Academy with 3-Column Split Layout:
- Left Panel: Lessons List (Curriculum Tree)
- Center Panel: Live Camera Practice Arena
- Right Panel: Target Sign Reference Guide (SVG Card, Audio, & Anatomy)
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QDesktopServices, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMessageBox, QProgressBar,
    QPushButton, QScrollArea, QSplitter, QStackedWidget, QTextEdit,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget
)

try:
    from PyQt6.QtSvgWidgets import QSvgWidget
    SVG_WIDGET_AVAILABLE = True
except ImportError:
    SVG_WIDGET_AVAILABLE = False

from core_engine.audio.audio_player import player_instance
from core_engine.vision.dtw_matcher import DTWMotionMatcher
from core_engine.vision.geometric_rule_engine import BdSLGeometricRuleEngine
from core_engine.vision.spatial_hand_engine import SpatialHandEngine
from desktop_app.controllers.audio_player import audio_controller
from desktop_app.controllers.certificate_generator import CertificateGenerator
from desktop_app.ui.components.circular_gauge import CircularAccuracyGauge
from desktop_app.ui.components.ghost_overlay import GhostOverlayPainter
from desktop_app.ui.components.motion_trajectory_viewer import MotionTrajectoryViewer
from desktop_app.ui.components.sign_card_viewer import SignCardViewer
from desktop_app.ui.theme import ThemeStyles

logger = logging.getLogger(__name__)

# Cyberpunk Modern Palette
BG_COLOR = "#1E1E2E"
TEXT_COLOR = "#CDD6F4"
ACCENT_COLOR = "#89B4FA"
SUCCESS_COLOR = "#A6E3A1"
ERROR_COLOR = "#F38BA8"
SURFACE_COLOR = "#313244"
PANEL_COLOR = "#181825"
CYAN_ACCENT = "#06B6D4"


class AcademyDashboard(QWidget):
    """Gamified BdSL Interpreter Academy featuring a 3-Column Split View."""

    request_back = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.spatial_engine = SpatialHandEngine()
        self.dtw_matcher = DTWMotionMatcher()
        self.rule_engine = BdSLGeometricRuleEngine()
        self.ghost_painter = GhostOverlayPainter()

        # Curriculum metadata database
        self.curriculum_data = self._load_curriculum_database()
        self.curriculum_items_order: List[str] = []

        # Temporal Frame Buffer for DTW Dynamic Sign Evaluation
        self.temporal_frame_buffer = []
        self.is_dynamic_sign = False
        self.current_sign_slug = "dhonnobad"
        self.current_sign_bn = "ধন্যবাদ"
        self.current_sign_en = "Thank you"

        # Interactive Quiz & Practice State
        self.practice_running = False
        self.quiz_active = False
        self.exam_active = False
        self.countdown_ticks = 0
        self.xp_score = 0
        self.streak = 0
        self.exam_signs = []
        self.current_exam_index = 0
        self.exam_correct = 0

        # Simulated Reference Data for ghost overlay
        self.mock_ref_landmarks = [(100 + i * 5, 100 + (i % 5) * 5) for i in range(42)]

        self._init_ui()

    def _load_curriculum_database(self) -> Dict[str, Dict[str, Any]]:
        """Loads and indexes curriculum metadata from JSON files."""
        database: Dict[str, Dict[str, Any]] = {}

        # 1. Load curriculum_data.json
        curr_file = Path("dataset/curriculum_data.json")
        if curr_file.exists():
            try:
                with open(curr_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for tier in data.get("tiers", []):
                    for module in tier.get("modules", []):
                        for lesson in module.get("lessons", []):
                            slug = lesson.get("sign_id", "").replace("vowel_", "").replace("cons_", "").replace("num_", "")
                            sym = lesson.get("symbol", "")
                            meta = {
                                "slug": slug or sym,
                                "label_bn": sym,
                                "label_en": lesson.get("name_en", ""),
                                "phonetic": lesson.get("phonetic", ""),
                                "category": tier.get("tier_name", "Foundational"),
                                "handedness": lesson.get("handedness", "single"),
                                "mnemonic": lesson.get("mnemonic", ""),
                                "touch_rule": lesson.get("touch_rule", ""),
                                "exercise_prompt": lesson.get("exercise_prompt", "")
                            }
                            database[sym] = meta
                            database[meta["label_en"]] = meta
                            if slug:
                                database[slug] = meta
            except Exception as e:
                logger.warning("Failed to load curriculum_data.json: %s", e)

        # 2. Enrich with labels.json
        labels_file = Path("dataset/labels.json")
        if labels_file.exists():
            try:
                with open(labels_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for s in data.get("signs", []):
                    slug = s.get("slug", "")
                    bn = s.get("label_bn", "")
                    en = s.get("label_en", "")
                    if slug and (slug not in database):
                        database[slug] = {
                            "slug": slug,
                            "label_bn": bn,
                            "label_en": en,
                            "phonetic": slug.replace("_", " ").title(),
                            "category": s.get("category", "General"),
                            "handedness": s.get("handedness", "single"),
                            "mnemonic": s.get("description", ""),
                            "touch_rule": "Follow standard BdSL motion trajectory.",
                            "exercise_prompt": f"Demonstrate '{bn}' ({en})."
                        }
                    database[bn] = database.get(slug, {})
            except Exception as e:
                logger.warning("Failed to load labels.json: %s", e)

        return database

    def _init_ui(self):
        self.setStyleSheet(ThemeStyles.get_global_stylesheet())

        main_layout = QVBoxLayout(self)

        # --- Top Header ---
        header_layout = QHBoxLayout()
        back_btn = QPushButton("← Back to Main")
        back_btn.clicked.connect(self.request_back.emit)

        title_lbl = QLabel("BdSL Interpreter Academy")
        title_lbl.setObjectName("Header")

        self.xp_header_lbl = QLabel("XP: 0 ⭐")
        self.xp_header_lbl.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.xp_header_lbl.setStyleSheet(f"color: {SUCCESS_COLOR};")

        header_layout.addWidget(back_btn)
        header_layout.addStretch()
        header_layout.addWidget(title_lbl)
        header_layout.addStretch()
        header_layout.addWidget(self.xp_header_lbl)
        main_layout.addLayout(header_layout)

        # --- 3-COLUMN SPLITTER ---
        # Panel 1: Left Curriculum (20%) | Panel 2: Center Camera Arena (45%) | Panel 3: Right Reference Guide (35%)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # =========================================================================
        # 1. LEFT PANEL (20% width): Lessons List (Curriculum Tree)
        # =========================================================================
        left_panel = QFrame()
        left_panel.setObjectName("GlassCard")
        left_panel.setStyleSheet(f"background-color: {PANEL_COLOR}; border-radius: 12px; padding: 8px;")
        left_panel.setMinimumWidth(220)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_header = QLabel("📚 পাঠ্যতালিকা (Curriculum)")
        left_header.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        left_header.setStyleSheet(f"color: {CYAN_ACCENT}; padding: 4px;")
        left_layout.addWidget(left_header)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self._build_curriculum()
        self.tree.itemClicked.connect(self._on_curriculum_selected)
        left_layout.addWidget(self.tree)

        self.exam_btn = QPushButton("🎓 Take Certification Exam")
        self.exam_btn.setStyleSheet(f"background-color: {SUCCESS_COLOR}; color: #11111B; font-weight: bold; padding: 8px;")
        self.exam_btn.clicked.connect(self._start_exam)
        left_layout.addWidget(self.exam_btn)

        splitter.addWidget(left_panel)

        # =========================================================================
        # 2. CENTER PANEL (45% width): Live Camera Practice Arena
        # =========================================================================
        center_panel = QFrame()
        center_panel.setObjectName("GlassCard")
        center_panel.setStyleSheet(f"background-color: {PANEL_COLOR}; border-radius: 12px; padding: 10px;")
        center_panel.setMinimumWidth(400)
        center_layout = QVBoxLayout(center_panel)

        # Center Arena Header
        self.arena_title = QLabel("🎯 প্র্যাকটিস করুন: ধন্যবাদ (Thank you)")
        self.arena_title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.arena_title.setStyleSheet(f"color: {CYAN_ACCENT};")
        self.arena_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_layout.addWidget(self.arena_title)

        # Camera Container with MotionTrajectoryViewer overlay
        cam_container = QWidget()
        cam_container.setFixedSize(400, 270)

        self.camera_feed = QLabel(cam_container)
        self.camera_feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_feed.setText("Webcam Idle. Click 'Start Practice' to begin.")
        self.camera_feed.setFixedSize(400, 270)
        self.camera_feed.setStyleSheet(f"background-color: {BG_COLOR}; border-radius: 8px;")

        self.trajectory_viewer = MotionTrajectoryViewer(cam_container)
        self.trajectory_viewer.setFixedSize(400, 270)

        center_layout.addWidget(cam_container, alignment=Qt.AlignmentFlag.AlignCenter)

        # Accuracy Progress Bar (0-100%)
        progress_layout = QHBoxLayout()
        prog_label = QLabel("Match Accuracy:")
        prog_label.setStyleSheet("color: #94A3B8; font-size: 11px;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {SURFACE_COLOR};
                border-radius: 6px;
                text-align: center;
                color: #FFFFFF;
                font-weight: bold;
                height: 14px;
            }}
            QProgressBar::chunk {{
                background-color: {CYAN_ACCENT};
                border-radius: 6px;
            }}
        """)
        progress_layout.addWidget(prog_label)
        progress_layout.addWidget(self.progress_bar)
        center_layout.addLayout(progress_layout)

        # AI Posture Tutor / Checklist HUD
        self.posture_hud = QTextEdit()
        self.posture_hud.setReadOnly(True)
        self.posture_hud.setMaximumHeight(55)
        self.posture_hud.setHtml(f"<b style='color:{CYAN_ACCENT};'>AI Posture Tutor:</b> Ready.")
        center_layout.addWidget(self.posture_hud)

        # Live Posture Coach & Real-time Error Notification Banner
        self.posture_coach_banner = QLabel("💡 পরামর্শ: ক্যামেরার সামনে আপনার হাত প্রদর্শন করুন...")
        self.posture_coach_banner.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.posture_coach_banner.setStyleSheet("""
            background: rgba(30, 41, 59, 0.8);
            color: #94A3B8;
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 8px;
            padding: 8px 12px;
        """)
        self.posture_coach_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.posture_coach_banner.setWordWrap(True)
        center_layout.addWidget(self.posture_coach_banner)

        # Telemetry Row (XP, Streak, Circular Gauge)
        telemetry_row = QHBoxLayout()
        self.xp_label = QLabel("XP: 0")
        self.xp_label.setStyleSheet(f"color: {CYAN_ACCENT}; font-weight: bold;")
        self.streak_label = QLabel("Streak: 0 🔥")
        self.streak_label.setStyleSheet("color: #F43F5E; font-weight: bold;")
        self.match_gauge = CircularAccuracyGauge(radius=24, thickness=6)

        telemetry_row.addWidget(self.xp_label)
        telemetry_row.addStretch()
        telemetry_row.addWidget(self.match_gauge)
        telemetry_row.addStretch()
        telemetry_row.addWidget(self.streak_label)
        center_layout.addLayout(telemetry_row)

        # Arena Action Controls (Start/Stop, Next Sign, Restart)
        action_row = QHBoxLayout()
        self.start_practice_btn = QPushButton("▶ Start Practice")
        self.start_practice_btn.setObjectName("ActionBtn")
        self.start_practice_btn.setStyleSheet(f"background-color: {CYAN_ACCENT}; color: #11111B; font-weight: bold; padding: 8px 14px;")
        self.start_practice_btn.clicked.connect(self._toggle_practice)

        self.next_sign_btn = QPushButton("⏭ Next Sign")
        self.next_sign_btn.setStyleSheet(f"background-color: {SURFACE_COLOR}; color: {TEXT_COLOR}; font-weight: bold; padding: 8px 12px;")
        self.next_sign_btn.clicked.connect(self._on_next_sign)

        self.restart_sign_btn = QPushButton("🔄 Restart")
        self.restart_sign_btn.setStyleSheet(f"background-color: {SURFACE_COLOR}; color: {TEXT_COLOR}; font-weight: bold; padding: 8px 12px;")
        self.restart_sign_btn.clicked.connect(self._on_restart_sign)

        action_row.addWidget(self.start_practice_btn)
        action_row.addWidget(self.next_sign_btn)
        action_row.addWidget(self.restart_sign_btn)
        center_layout.addLayout(action_row)

        splitter.addWidget(center_panel)

        # =========================================================================
        # 3. RIGHT PANEL (35% width): Target Sign Reference Guide (SVG Card & Steps)
        # =========================================================================
        right_scroll = QScrollArea()
        right_scroll.setObjectName("RightScrollArea")
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollBar:vertical { width: 6px; background: #11111B; border-radius: 3px; }"
            "QScrollBar::handle:vertical { background: #313244; border-radius: 3px; }"
        )

        right_panel = QFrame()
        right_panel.setObjectName("GlassCard")
        right_panel.setStyleSheet(f"background-color: {PANEL_COLOR}; border-radius: 12px; padding: 12px; border: 1px solid rgba(6, 182, 212, 0.25);")
        right_panel.setMinimumWidth(300)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(10, 10, 10, 10)

        # Row 1: Header
        right_header = QLabel("📖 ইশারা নির্দেশিকা (Sign Guide)")
        right_header.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        right_header.setStyleSheet(f"color: {CYAN_ACCENT}; padding: 2px;")
        right_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(right_header)

        # Row 2: Bengali character (34px) + English subtitle (16px)
        self.ref_sign_bn = QLabel("ধন্যবাদ")
        self.ref_sign_bn.setFont(QFont("SolaimanLipi", 34, QFont.Weight.Black))
        self.ref_sign_bn.setStyleSheet("color: #F8FAFC; font-weight: 900; padding: 0px;")
        self.ref_sign_bn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.ref_sign_bn)

        self.ref_sign_en = QLabel("Thank you")
        self.ref_sign_en.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        self.ref_sign_en.setStyleSheet("color: #38BDF8; font-weight: bold;")
        self.ref_sign_en.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.ref_sign_en)

        # Row 3: Horizontal Layout for Pronunciation Badge + Interactive Listen Button
        pronounce_row = QHBoxLayout()
        pronounce_row.setSpacing(8)

        self.ref_phonetic = QLabel("🔊 উচ্চারণ: [dhon-no-baad]")
        self.ref_phonetic.setStyleSheet("color: #10B981; background-color: rgba(16, 185, 129, 0.12); border-radius: 6px; padding: 6px 10px; font-weight: 600; font-size: 13px;")
        self.ref_phonetic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pronounce_row.addWidget(self.ref_phonetic, stretch=3)

        self.listen_btn = QPushButton("🔊 শুনুন")
        self.listen_btn.setStyleSheet(f"background-color: {SURFACE_COLOR}; color: {CYAN_ACCENT}; font-weight: bold; font-size: 12px; padding: 6px 10px; border-radius: 6px;")
        self.listen_btn.clicked.connect(self._on_listen_pronunciation)
        pronounce_row.addWidget(self.listen_btn, stretch=2)

        right_layout.addLayout(pronounce_row)

        # Row 4: Category & Handedness Badge
        self.ref_badge = QLabel("Type: Dynamic | Single Hand")
        self.ref_badge.setStyleSheet(f"background-color: {SURFACE_COLOR}; color: #CBD5E1; border-radius: 6px; padding: 4px 8px; font-size: 11.5px;")
        self.ref_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.ref_badge)

        # Middle: High-Fidelity Dedicated SVG Visual Card Viewer
        self.sign_card_viewer = SignCardViewer("dhonnobad", "ধন্যবাদ", "Thank you")
        self.sign_card_viewer.setFixedSize(280, 220)
        self.svg_widget = self.sign_card_viewer  # compatibility alias
        right_layout.addWidget(self.sign_card_viewer, alignment=Qt.AlignmentFlag.AlignCenter)

        # Bottom: Anatomical Step-by-Step Instruction Guide
        self.ref_instructions = QTextEdit()
        self.ref_instructions.setReadOnly(True)
        self.ref_instructions.setMinimumHeight(150)
        self.ref_instructions.setStyleSheet("background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px; padding: 8px; font-size: 13px; color: #E2E8F0;")
        right_layout.addWidget(self.ref_instructions)

        right_scroll.setWidget(right_panel)
        splitter.addWidget(right_scroll)

        # Explicit Splitter Stretch Factors: Left 20%, Center 45%, Right 35%
        splitter.setSizes([220, 480, 360])
        splitter.setStretchFactor(0, 20)
        splitter.setStretchFactor(1, 45)
        splitter.setStretchFactor(2, 35)
        main_layout.addWidget(splitter)

        # Practice / Quiz countdown timer (pure state countdown, no camera capture)
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_practice_tick)
        self.cap = None

        # Load initial sign
        self._update_reference_card("dhonnobad")

    def _build_curriculum(self):
        """Populates the curriculum tree widget."""
        self.curriculum_items_order.clear()
        curriculum_file = "dataset/curriculum_data.json"
        if os.path.exists(curriculum_file):
            try:
                with open(curriculum_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for tier in data.get("tiers", []):
                    t_item = QTreeWidgetItem(self.tree, [tier.get("tier_name", "Tier")])
                    if "modules" in tier:
                        for mod in tier["modules"]:
                            m_item = QTreeWidgetItem(t_item, [mod.get("module_title", "Module")])
                            for lesson in mod.get("lessons", []):
                                sym = lesson.get("symbol", "")
                                name = lesson.get("name_en", "")
                                item_text = f"{sym} - {name}"
                                QTreeWidgetItem(m_item, [item_text])
                                self.curriculum_items_order.append(item_text)
                self.tree.expandAll()
                return
            except Exception as e:
                logger.debug("Tree parse fallback: %s", e)

        # Fallback items
        l1 = QTreeWidgetItem(self.tree, ["Level 1: Alphabets & Digits"])
        for sign in ["অ - Vowel A", "আ - Vowel Aa", "ই - Vowel I", "ক - Consonant Ka", "১ - One", "২ - Two"]:
            QTreeWidgetItem(l1, [sign])
            self.curriculum_items_order.append(sign)

        l2 = QTreeWidgetItem(self.tree, ["Level 2: Daily Words"])
        for sign in ["ধন্যবাদ - Thank you", "সাহায্য - Help", "স্বাগতম - Welcome"]:
            QTreeWidgetItem(l2, [sign])
            self.curriculum_items_order.append(sign)

        self.tree.expandAll()

    def _update_reference_card(self, sign_key: str):
        """Updates the Center Arena title and Right Sub-Panel with visual illustration and instructions."""
        meta = self.curriculum_data.get(sign_key, {})
        if not meta and " - " in sign_key:
            parts = [p.strip() for p in sign_key.split(" - ")]
            for part in parts:
                if part in self.curriculum_data:
                    meta = self.curriculum_data[part]
                    break

        if not meta:
            # Try fuzzy match
            for k, v in self.curriculum_data.items():
                if k and (k == sign_key or k in sign_key or sign_key in k):
                    meta = v
                    break

        raw_bn = sign_key.split(" - ")[0].strip() if " - " in sign_key else sign_key
        raw_en = sign_key.split(" - ")[1].strip() if " - " in sign_key else sign_key

        slug = meta.get("slug") or "dhonnobad"
        bn = meta.get("label_bn") or raw_bn
        en = meta.get("label_en") or raw_en
        phonetic = meta.get("phonetic") or slug.replace("_", " ").title()
        cat = meta.get("category", "General")
        
        handedness_raw = str(meta.get("handedness", "single")).lower()
        is_dual = handedness_raw in ["dual", "both", "2"]
        handedness_badge = "👐 উভয় হাত (Both Hands)" if is_dual else "✋ ডান হাত (Right Hand)"
        handedness_color = "#10B981" if is_dual else "#38BDF8"
        
        mnemonic = meta.get("mnemonic") or meta.get("description") or "ক্যামেরার সামনে হাতের অঙ্গুলি ও তালু নির্দেশিত চিত্রমোতাবেক প্রস্তুত রাখুন।"
        touch_rule = meta.get("touch_rule") or ("কোনো স্পর্শ নেই (উন্মুক্ত হাতের একক ভঙ্গি)" if not is_dual else "উভয় হাতের নির্দেশিত সংযোগ বিন্দু স্পর্শ করুন")

        self.current_sign_slug = slug
        self.current_sign_bn = bn
        self.current_sign_en = en

        # Update Center Panel Title
        self.arena_title.setText(f"🎯 প্র্যাকটিস করুন: {bn} ({en})")

        # Update Right Panel Labels
        self.ref_sign_bn.setText(bn)
        self.ref_sign_en.setText(en)
        self.ref_phonetic.setText(f"🔊 উচ্চারণ: [{phonetic}]")
        self.ref_badge.setText(f"ক্যাটাগরি: {cat} | {handedness_badge}")
        self.ref_badge.setStyleSheet(f"background-color: {SURFACE_COLOR}; color: {handedness_color}; border-radius: 6px; padding: 4px 8px; font-size: 11.5px; font-weight: bold;")

        pos_desc = "ক্যামেরা ফ্রেমের ঠিক মাঝে হাত ও বাহু স্থির রাখুন।" if not is_dual else "উভয় হাত সমান উচ্চতায় বুকের সামনে আনুন।"
        fingers_desc = mnemonic
        action_desc = "স্থিতিশীল ভঙ্গিতে ২ সেকেন্ড ধরে রাখুন।" if not self.is_dynamic_sign else "ধীর ও মসৃণ গতিতে নির্দেশিত পথ অনুসরণ করুন।"

        instructions_html = f"""
        <div style="font-family: 'Segoe UI', Arial; font-size: 13px; line-height: 1.5; color: #E2E8F0;">
          <b style="color: #38BDF8; font-size: 14px;">🎯 ধাপ ও কৌশল (Steps & Technique):</b>
          <div style="margin-top: 6px; padding: 8px; background: rgba(15, 23, 42, 0.6); border-radius: 6px; font-size: 12.5px; line-height: 1.6;">
            <b style="color: #38BDF8;">১. অবস্থান (Position):</b> {pos_desc}<br>
            <b style="color: #38BDF8;">২. আঙুল (Fingers):</b> {fingers_desc}<br>
            <b style="color: #38BDF8;">৩. কৌশল (Action):</b> {action_desc}
          </div>
          <div style="margin-top: 8px; padding: 8px 10px; background: rgba(16, 185, 129, 0.12); border-left: 3px solid #10B981; border-radius: 6px; color: #34D399; font-weight: 600; font-size: 12.5px;">
            ✨ <b>স্পর্শের নিয়ম:</b> {touch_rule}
          </div>
        </div>
        """
        self.ref_instructions.setHtml(instructions_html)

        # Reset coach banner
        self.update_posture_coach(f"ক্যামেরার সামনে '{bn}' এর ভঙ্গি প্রদর্শন করুন...", state="idle")

        # Load Visual Card via bulletproof SignCardViewer
        self.sign_card_viewer.load_sign(slug, bn, en)

    def update_posture_coach(self, advice_text: str, state: str = "idle"):
        """Updates the Live Posture Coach notification banner."""
        if state == "perfect":
            style = "background: rgba(16, 185, 129, 0.22); color: #34D399; border: 1.5px solid #10B981; border-radius: 8px; padding: 8px 12px; font-weight: bold;"
            icon = "🟢"
        elif state == "warning":
            style = "background: rgba(244, 63, 94, 0.18); color: #FDA4AF; border: 1.5px solid #F43F5E; border-radius: 8px; padding: 8px 12px; font-weight: bold;"
            icon = "🔴"
        elif state == "info":
            style = "background: rgba(6, 182, 212, 0.18); color: #38BDF8; border: 1.5px solid #06B6D4; border-radius: 8px; padding: 8px 12px; font-weight: bold;"
            icon = "💡"
        else:  # idle
            style = "background: rgba(30, 41, 59, 0.8); color: #94A3B8; border: 1px solid rgba(148, 163, 184, 0.3); border-radius: 8px; padding: 8px 12px;"
            icon = "⚪"

        self.posture_coach_banner.setStyleSheet(style)
        self.posture_coach_banner.setText(f"{icon} {advice_text}")

    def _on_curriculum_selected(self, item: QTreeWidgetItem, column: int):
        """Triggered when user clicks a lesson in the curriculum tree."""
        text = item.text(0)
        self._update_reference_card(text)

        dynamic_keywords = [
            "greetings", "sov", "structure", "questions", "news",
            "speech", "dhonnobad", "kemon", "sahajjo", "dynamic", "স্বাগতম"
        ]
        self.is_dynamic_sign = any(k in text.lower() for k in dynamic_keywords)

    def _on_lesson_selected(self, lesson_text: str):
        """Programmatic selection of a lesson."""
        self._update_reference_card(lesson_text)

    def _on_listen_pronunciation(self):
        """Plays the spoken pronunciation of the active sign."""
        if self.current_sign_bn:
            audio_controller.speak_bengali(self.current_sign_bn)
        elif self.current_sign_en:
            audio_controller.speak_bengali(self.current_sign_en)

    def _on_next_sign(self):
        """Advances to the next curriculum sign."""
        if not self.curriculum_items_order:
            return

        current_idx = 0
        for idx, item_text in enumerate(self.curriculum_items_order):
            if self.current_sign_bn in item_text or self.current_sign_slug in item_text:
                current_idx = idx
                break

        next_idx = (current_idx + 1) % len(self.curriculum_items_order)
        next_item = self.curriculum_items_order[next_idx]
        self._update_reference_card(next_item)
        self._on_restart_sign()

    def _on_restart_sign(self):
        """Clears buffers and resets progress for the current sign."""
        self.temporal_frame_buffer.clear()
        self.progress_bar.setValue(0)
        self.countdown_ticks = 150
        mode_hint = "Dynamic Gesture" if self.is_dynamic_sign else "Static Posture"
        self.posture_hud.setHtml(f"<b style='color:{ACCENT_COLOR};'>Practice ({mode_hint}):</b> Sign '{self.current_sign_bn}'...")

    def _toggle_practice(self):
        """Toggles webcam capture and live practice mode."""
        if not self.practice_running:
            self._start_practice()
        else:
            self._stop_practice()

    def _start_practice(self):
        self.practice_running = True
        self.start_practice_btn.setText("⏹ Stop Practice")
        self.start_practice_btn.setStyleSheet(f"background-color: {ERROR_COLOR}; color: #11111B; font-weight: bold; padding: 8px 14px;")
        self.temporal_frame_buffer.clear()
        self.quiz_active = True
        self.exam_active = False
        self.countdown_ticks = 150  # 5s
        mode_hint = "Dynamic Gesture" if self.is_dynamic_sign else "Static Posture"
        self.posture_hud.setHtml(f"<b style='color:{ACCENT_COLOR};'>Quiz Starting ({mode_hint})!</b> Sign '{self.current_sign_bn}'...")
        self.timer.start(100)

    def _on_practice_tick(self):
        """Countdown and exam progression tick."""
        if self.countdown_ticks > 0:
            self.countdown_ticks -= 1
            if self.countdown_ticks == 0 and self.exam_active:
                self.current_exam_index += 1
                if self.current_exam_index < len(self.exam_signs):
                    sign_curr = self.exam_signs[self.current_exam_index]
                    self._update_reference_card(sign_curr)
                    self.countdown_ticks = 150
                    self.posture_hud.setHtml(f"<b style='color:{ACCENT_COLOR};'>Exam:</b> Sign {self.current_exam_index + 1}/10: {sign_curr}")
                else:
                    self._end_exam()

    @pyqtSlot(object)
    def update_camera_feed(self, image):
        """Updates the practice arena camera video feed.
        
        Supports QImage, QPixmap, and NumPy ndarray.
        """
        if image is None:
            return

        pixmap = None
        if isinstance(image, QPixmap):
            pixmap = image
        elif isinstance(image, QImage):
            pixmap = QPixmap.fromImage(image)
        elif isinstance(image, np.ndarray):
            try:
                h, w, ch = image.shape
                bytes_per_line = ch * w
                if ch == 3:
                    qt_img = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
                else:
                    qt_img = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
                pixmap = QPixmap.fromImage(qt_img)
            except Exception as e:
                logger.debug(f"Failed converting ndarray to pixmap: {e}")
                return

        if pixmap and not pixmap.isNull():
            target_size = self.camera_feed.size()
            if target_size.width() <= 10 or target_size.height() <= 10:
                target_size = self.camera_feed.sizeHint()
            scaled = pixmap.scaled(target_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.camera_feed.setPixmap(scaled)

    def update_frame(self, image):
        """Alias for update_camera_feed."""
        self.update_camera_feed(image)

    def set_frame(self, image):
        """Alias for update_camera_feed."""
        self.update_camera_feed(image)

    @pyqtSlot(dict)
    def process_prediction(self, data: dict):
        """Processes real-time prediction data forwarded from MainWindow camera worker."""
        if not data:
            return

        label_bn = data.get("label_bn", "")
        conf = float(data.get("confidence", 0.0)) * 100.0
        left_lm = data.get("left_landmarks")
        right_lm = data.get("right_landmarks")

        # Evaluate target posture directly if landmarks are available
        if left_lm is not None or right_lm is not None:
            score, is_match, checklist, advice = self.rule_engine.evaluate_target_posture(
                left_lm, right_lm, self.current_sign_slug
            )
            pct = int(score * 100.0)
            self.progress_bar.setValue(pct)
            self.match_gauge.set_value(float(pct))
            
            if checklist:
                items_html = " ".join([
                    f"<span style='color: {'#10B981' if c.get('matched') else '#F43F5E'};'>{'✓' if c.get('matched') else '✗'} {c.get('item_bn')}</span>"
                    for c in checklist
                ])
                self.posture_hud.setHtml(f"<b>চেকলিস্ট:</b> {items_html}")

            if is_match:
                self.update_posture_coach(advice, state="perfect")
                if self.practice_running:
                    self.xp_score += 5
                    self.streak += 1
                    self.xp_label.setText(f"XP: {self.xp_score}")
                    self.streak_label.setText(f"Streak: {self.streak} 🔥")
                    self.xp_header_lbl.setText(f"XP: {self.xp_score} ⭐")
            else:
                self.update_posture_coach(advice, state="warning")
            return

        # Fallback if only prediction classification dict is forwarded
        if label_bn and (label_bn == self.current_sign_bn or self.current_sign_slug in str(data.get("label_en", "")).lower()):
            self.progress_bar.setValue(int(min(100.0, max(0.0, conf))))
            self.match_gauge.set_value(conf)
            if conf >= 70.0:
                self.update_posture_coach("ভঙ্গি নিখুঁত! হাত এই অবস্থায় ধরে রাখুন...", state="perfect")
                self.posture_hud.setHtml(f"<b style='color:#10B981;'>Detected: {label_bn} ({conf:.1f}%)</b>")
                if self.practice_running:
                    self.xp_score += 5
                    self.streak += 1
                    self.xp_label.setText(f"XP: {self.xp_score}")
                    self.streak_label.setText(f"Streak: {self.streak} 🔥")
                    self.xp_header_lbl.setText(f"XP: {self.xp_score} ⭐")
            else:
                self.update_posture_coach(f"পরামর্শ: লক্ষ্য ইশারা '{self.current_sign_bn}' এর সাথে মেলাতে হাত প্রস্তুত করুন...", state="info")
        else:
            self.update_posture_coach("পরামর্শ: নির্দেশিত চিত্রের সাথে হাত ও আঙুল মেলান...", state="warning")

    def _stop_practice(self):
        self.practice_running = False
        self.start_practice_btn.setText("▶ Start Practice")
        self.start_practice_btn.setStyleSheet(f"background-color: {CYAN_ACCENT}; color: #11111B; font-weight: bold; padding: 8px 14px;")
        self.timer.stop()
        if self.cap and hasattr(self.cap, "release"):
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.camera_feed.setText("Webcam Idle. Click 'Start Practice' to begin.")
        self.quiz_active = False
        self.exam_active = False
        self.progress_bar.setValue(0)

    def _start_exam(self):
        self._start_practice()
        all_signs = list(self.curriculum_data.keys())
        if len(all_signs) > 10:
            self.exam_signs = random.sample(all_signs, 10)
        else:
            self.exam_signs = ["ধন্যবাদ", "সাহায্য", "স্বাগতম", "১", "২", "৩", "অ", "আ", "ক", "আমি"]
        self.current_exam_index = 0
        self.exam_correct = 0

        self.quiz_active = False
        self.exam_active = True
        self.countdown_ticks = 150
        sign_curr = self.exam_signs[0]
        self._update_reference_card(sign_curr)
        self.posture_hud.setHtml(f"<b style='color:{ACCENT_COLOR};'>Exam Started!</b> Sign 1/10: {sign_curr}")

    def _end_exam(self):
        self._stop_practice()
        score_percent = (self.exam_correct / 10.0) * 100

        msg = QMessageBox(self)
        msg.setWindowTitle("Certification Exam Complete")
        msg.setStyleSheet(f"background-color: {PANEL_COLOR}; color: {TEXT_COLOR};")

        if score_percent >= 70:
            msg.setText(f"Congratulations! You passed with a score of {score_percent}%.\nWould you like to download your verifiable certificate?")
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if msg.exec() == QMessageBox.StandardButton.Yes:
                cert_path = CertificateGenerator.generate_certificate("BdSL Learner", "National Interpreter Certification", int(score_percent))
                QDesktopServices.openUrl(QUrl.fromLocalFile(cert_path))
        else:
            msg.setText(f"Exam Completed. Your score: {score_percent}%.\nPractice more and retake to earn your certificate!")
            msg.exec()

    def _update_practice_frame(self):
        """Processes a frame if self.cap is opened (e.g. for mock unit tests)."""
        if not self.cap or not hasattr(self.cap, "isOpened") or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return

        frame = cv2.flip(frame, 1)

        # Extract features defensively
        try:
            features = self.spatial_engine.extract_spatial_features(frame)
        except Exception as e:
            logger.debug(f"Spatial feature extraction error: {e}")
            features = None

        if not features or not isinstance(features, dict):
            return

        raw_lm = features.get("raw_landmarks")
        has_left = features.get("has_left", False)
        has_right = features.get("has_right", False)

        spatial_vec = None
        for k in ("spatial_vector", "spatial_features", "landmarks_151d", "features"):
            val = features.get(k)
            if val is not None:
                spatial_vec = val
                break

        if spatial_vec is None and "normalized_landmarks" in features and "touch_matrix" in features:
            norm_lm = features["normalized_landmarks"].flatten()
            touch = np.nan_to_num(features["touch_matrix"], nan=0.0, posinf=1.0, neginf=0.0).flatten()
            spatial_vec = np.concatenate([norm_lm, touch]).astype(np.float32)

        # Append to rolling temporal buffer for DTW
        if spatial_vec is not None:
            self.temporal_frame_buffer.append(spatial_vec)
            if len(self.temporal_frame_buffer) > 30:
                self.temporal_frame_buffer.pop(0)

        # Landmarks for ghost alignment
        live_landmarks = []
        left_wrist, right_wrist = None, None
        if raw_lm is not None:
            h, w, _ = frame.shape
            for i, lm in enumerate(raw_lm):
                if lm is not None and len(lm) >= 2 and (lm[0] != 0 or lm[1] != 0):
                    x = int(lm[0] * w)
                    y = int(lm[1] * h)
                    live_landmarks.append((x, y))
                    if i == 0:
                        left_wrist = (x, y)
                    elif i == 21:
                        right_wrist = (x, y)

        self.trajectory_viewer.update_trajectory(left_wrist, right_wrist)

        # Ghost Overlay rendering
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        final_img = self.ghost_painter.draw_overlay(qt_img, self.mock_ref_landmarks, live_landmarks)
        pixmap = QPixmap.fromImage(final_img).scaled(400, 270, Qt.AspectRatioMode.KeepAspectRatio)
        self.camera_feed.setPixmap(pixmap)

        # Geometric Checklist Evaluation
        left_lm = raw_lm if has_left else None
        right_lm = raw_lm if has_right else None
        _, _, finger_status = self.rule_engine.evaluate_rules(left_lm, right_lm)
        checklist = finger_status.get("checklist", [])

        # Score calculation: Dynamic DTW or Static Ghost Alignment
        if self.is_dynamic_sign and len(self.temporal_frame_buffer) >= 10:
            user_seq = np.array(self.temporal_frame_buffer, dtype=np.float32)
            dtw_eval = self.dtw_matcher.evaluate_gesture_accuracy(user_seq, self.current_sign_slug)
            current_score = float(dtw_eval["score"])
        else:
            current_score = float(self.ghost_painter.calculate_alignment_score(self.mock_ref_landmarks, live_landmarks))

        # Update Accuracy Progress Bar & Circular Gauge
        self.progress_bar.setValue(int(min(100.0, max(0.0, current_score))))
        self.match_gauge.set_value(current_score if (has_left or has_right) else 0)

        # Checklist HTML formatting
        checklist_html = ""
        if checklist:
            items = []
            for itm in checklist:
                col = "#10B981" if itm["matched"] else "#94A3B8"
                sym = "✓" if itm["matched"] else "○"
                items.append(f"<span style='color:{col};'>[{sym}] {itm['item_bn']}</span>")
            checklist_html = "<br>" + " &nbsp;|&nbsp; ".join(items)

        # Quiz / Exam Logic
        if self.quiz_active or self.exam_active:
            if self.countdown_ticks > 0:
                self.countdown_ticks -= 1
                secs = self.countdown_ticks // 30
                sign_type = "DTW Motion" if self.is_dynamic_sign else "Static Pose"
                if self.exam_active:
                    self.posture_hud.setHtml(f"<b style='color:{ACCENT_COLOR};'>Exam [{self.current_exam_index + 1}/10 - {sign_type}]:</b> Sign in {secs}s...{checklist_html}")
                else:
                    self.posture_hud.setHtml(f"<b style='color:{ACCENT_COLOR};'>Quiz ({sign_type}):</b> Sign in {secs}s...{checklist_html}")

                if current_score > 75.0:
                    audio_controller.play_chime("notify")
                    self.posture_hud.setHtml(f"<b style='color:#10B981;'>Excellent! Match: {current_score:.1f}%</b>{checklist_html}")
                    self.xp_score += 50
                    self.streak += 1
                    self.xp_header_lbl.setText(f"XP: {self.xp_score} ⭐")

                    if self.exam_active:
                        self.exam_correct += 1
                        self.current_exam_index += 1
                        if self.current_exam_index >= 10:
                            self._end_exam()
                            return
                        else:
                            self.countdown_ticks = 150
                            next_sign = self.exam_signs[self.current_exam_index]
                            self._update_reference_card(next_sign)
                    else:
                        self.quiz_active = False
            else:
                self.posture_hud.setHtml(f"<b style='color:#F43F5E;'>Time's Up!</b>{checklist_html}")
                self.streak = 0
                if self.exam_active:
                    self.current_exam_index += 1
                    if self.current_exam_index >= 10:
                        self._end_exam()
                        return
                    else:
                        self.countdown_ticks = 150
                        next_sign = self.exam_signs[self.current_exam_index]
                        self._update_reference_card(next_sign)
                else:
                    self.quiz_active = False

            self.xp_label.setText(f"XP: {self.xp_score}")
            self.streak_label.setText(f"Streak: {self.streak} 🔥")
        else:
            if checklist_html:
                self.posture_hud.setHtml(f"<b style='color:{CYAN_ACCENT};'>AI Posture Checklist:</b>{checklist_html}")

    def closeEvent(self, event):
        self._stop_practice()
        super().closeEvent(event)
