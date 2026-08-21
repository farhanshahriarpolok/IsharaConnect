import pytest
import time
from desktop_app.controllers.practice_controller import PracticeSessionManager

@pytest.fixture
def manager(tmp_path):
    import json
    labels_path = tmp_path / "test_labels.json"
    data = {
        "signs": [
            {"id": 0, "slug": "hello", "label_bn": "হ্যালো", "label_en": "Hello"},
            {"id": 1, "slug": "thanks", "label_bn": "ধন্যবাদ", "label_en": "Thanks"}
        ]
    }
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
        
    m = PracticeSessionManager(labels_path=str(labels_path))
    m.hold_duration = 0.5  # lower for testing
    return m

def test_initialization(manager):
    assert len(manager.signs_map) == 2
    assert manager.target_sign is None

def test_set_target_sign(manager):
    manager.set_target_sign(0)
    assert manager.target_sign["id"] == 0
    assert manager.target_sign["slug"] == "hello"

def test_evaluate_frame_wrong_sign(manager):
    manager.set_target_sign(0)
    pred = {"sign_id": 1, "confidence": 0.9, "label_bn": "ধন্যবাদ", "is_stable": True}
    res = manager.evaluate_frame(pred)
    assert res["is_passed"] is False
    assert res["match_score"] <= 10.0
    assert "ধন্যবাদ" in res["feedback_text"]

def test_evaluate_frame_correct_sign_low_confidence(manager):
    manager.set_target_sign(0)
    pred = {"sign_id": 0, "confidence": 0.5, "is_stable": True}
    res = manager.evaluate_frame(pred)
    assert res["is_passed"] is False
    assert res["match_score"] == 50.0
    assert "কাছাকাছি" in res["feedback_text"]

def test_evaluate_frame_hold_duration(manager):
    manager.set_target_sign(0)
    pred = {"sign_id": 0, "confidence": 0.9, "is_stable": True}
    
    # First frame (starts hold)
    res = manager.evaluate_frame(pred)
    assert res["is_passed"] is False
    assert manager._is_holding is True
    
    # Immediate second frame (not enough time passed)
    res = manager.evaluate_frame(pred)
    assert res["is_passed"] is False
    assert "ধরে রাখুন" in res["feedback_text"]
    
    # Wait for hold duration
    time.sleep(0.6)
    res = manager.evaluate_frame(pred)
    assert res["is_passed"] is True
    assert "দারুণ" in res["feedback_text"]
    
def test_evaluate_frame_hold_interrupted(manager):
    manager.set_target_sign(0)
    
    # Start hold
    manager.evaluate_frame({"sign_id": 0, "confidence": 0.9})
    assert manager._is_holding is True
    
    # Interrupted by bad prediction
    manager.evaluate_frame({"sign_id": 1, "confidence": 0.8})
    assert manager._is_holding is False
    
    # Start hold again
    manager.evaluate_frame({"sign_id": 0, "confidence": 0.9})
    assert manager._is_holding is True
