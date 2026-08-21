"""Unit tests for the Certification Exam Dialog and Examination Wizard."""

import os
from pathlib import Path
import pytest
from PyQt6.QtWidgets import QApplication

from desktop_app.ui.dialogs.exam_dialog import (
    ExamDialog,
    MCQ_QUESTIONS,
    POSTURE_CHALLENGES,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_exam_dialog_flow(qapp, tmp_path):
    """Test full multi-step examination workflow from intro to scorecard."""
    dialog = ExamDialog(candidate_name="Test Student")
    dialog.cert_generator.output_dir = Path(tmp_path)
    
    # 1. Check Intro Page
    assert dialog.stack.currentIndex() == 0
    assert dialog.name_input.text() == "Test Student"
    
    # 2. Start MCQ Round
    dialog._start_mcq_round()
    assert dialog.stack.currentIndex() == 1
    assert dialog.current_mcq_index == 0
    
    # Answer 5 MCQs (Answer all correctly)
    for i in range(len(MCQ_QUESTIONS)):
        correct = MCQ_QUESTIONS[i]["correct_idx"]
        dialog.mcq_radios[correct].setChecked(True)
        dialog._on_mcq_next()
        
    # 3. Posture Round should have started automatically after MCQ 5
    assert dialog.stack.currentIndex() == 2
    assert dialog.current_posture_index == 0
    
    # Complete 5 posture challenges
    for i in range(len(POSTURE_CHALLENGES)):
        dialog.posture_scores[i] = 10.0
        dialog.posture_next_btn.setEnabled(True)
        dialog._on_posture_next()
        
    # 4. Scorecard Page
    assert dialog.stack.currentIndex() == 3
    assert dialog.final_percentage == 100.0
    assert "A+ (Distinction)" in dialog.final_grade
    assert dialog.is_passed is True
    assert dialog.download_cert_btn.isEnabled() is True
    
    # 5. Generate Certificate from dialog
    cert_path = dialog.cert_generator.generate_certificate(
        candidate_name=dialog.candidate_name,
        score_percent=dialog.final_percentage,
        grade=dialog.final_grade,
    )
    assert os.path.exists(cert_path)
    assert cert_path.endswith(".pdf")
    
    dialog.close()


def test_exam_dialog_retake_flow(qapp):
    """Test failed exam state with retake capability."""
    dialog = ExamDialog(candidate_name="Needs Practice Student")
    
    # Set mock answers with 0 score
    dialog.mcq_answers = {0: 3, 1: 3, 2: 3, 3: 3, 4: 3}  # Wrong answers
    dialog.posture_scores = {}
    
    dialog._calculate_final_score()
    assert dialog.final_percentage == 0.0
    assert dialog.is_passed is False
    assert "NEEDS RETAKE" in dialog.score_status_lbl.text()
    assert dialog.download_cert_btn.isEnabled() is False
    
    # Click Retake Exam
    dialog._retake_exam()
    assert dialog.stack.currentIndex() == 0
    
    dialog.close()
