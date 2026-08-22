"""Unit Test Suite for Face-First Facial Keypoint Matrix & Fingertip-Centric Precision Contact (Sprint 34).

Tests:
1. Face-first keypoint cluster resolution (Forehead, Nose, Upper Lip, Chin, Cheek).
2. Fingertip-centric (not wrist) active articulator selection for all facial signs.
3. Precise Fingertip-to-Face distance metric for Chin (dhonnobad) yielding >= 92% at chin and < 15% at forehead.
4. Corrective Bengali directional hints mentioning 'আঙুলের ডগা'.
5. Multi-feature face contacts (Nose for 'kho', Cheek for 'ma', Forehead for 'salam').
"""

import numpy as np
import pytest

from core_engine.vision.spatial_normalizer import SpatialNormalizer
from core_engine.vision.sign_correction_advisor import SignCorrectionAdvisor


def _create_synthetic_facemesh() -> np.ndarray:
    """Creates a 468 FaceMesh array with realistic keypoint clusters."""
    face = np.zeros((468, 3), dtype=np.float32)
    # Forehead cluster: 10, 151, 9 at y=0.20
    face[10] = [0.50, 0.20, 0.0]
    face[151] = [0.50, 0.21, 0.0]
    face[9] = [0.50, 0.22, 0.0]
    # Nose cluster: 1, 4, 195 at y=0.30
    face[1] = [0.50, 0.30, 0.0]
    face[4] = [0.50, 0.31, 0.0]
    face[195] = [0.50, 0.29, 0.0]
    # Upper Lip cluster: 0, 13, 267 at y=0.36
    face[0] = [0.50, 0.36, 0.0]
    face[13] = [0.50, 0.36, 0.0]
    face[267] = [0.52, 0.36, 0.0]
    # Chin cluster: 152, 175, 199 at y=0.45
    face[152] = [0.50, 0.45, 0.0]
    face[175] = [0.50, 0.45, 0.0]
    face[199] = [0.50, 0.46, 0.0]
    # Cheeks: Right 234, 93 | Left 454, 323 at y=0.34
    face[234] = [0.62, 0.34, 0.0]
    face[93] = [0.60, 0.34, 0.0]
    face[454] = [0.38, 0.34, 0.0]
    face[323] = [0.40, 0.34, 0.0]
    return face


def _create_hand_with_fingertip_at(tip_x: float, tip_y: float) -> np.ndarray:
    """Creates a 21-landmark hand where Index Tip (8) and Middle Tip (12) are at (tip_x, tip_y)."""
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[0] = [tip_x, tip_y + 0.16, 0.0]  # Wrist is 16cm below tips
    for i in range(1, 21):
        hand[i] = [tip_x, tip_y + 0.05, 0.0]
    hand[8] = [tip_x, tip_y, 0.0]          # Index Tip
    hand[12] = [tip_x, tip_y, 0.0]         # Middle Tip
    return hand


def test_face_first_keypoint_clusters():
    """Verify get_anatomical_anchor_3d computes mean coordinates for keypoint clusters."""
    normalizer = SpatialNormalizer()
    facemesh = _create_synthetic_facemesh()

    forehead = normalizer.get_anatomical_anchor_3d("FOREHEAD", face_landmarks=facemesh)
    nose = normalizer.get_anatomical_anchor_3d("NOSE", face_landmarks=facemesh)
    chin = normalizer.get_anatomical_anchor_3d("CHIN", face_landmarks=facemesh)
    cheek = normalizer.get_anatomical_anchor_3d("CHEEK_RIGHT", face_landmarks=facemesh)

    np.testing.assert_allclose(forehead[:2], [0.50, 0.21], atol=0.01)
    np.testing.assert_allclose(nose[:2], [0.50, 0.30], atol=0.01)
    np.testing.assert_allclose(chin[:2], [0.50, 0.453], atol=0.01)
    np.testing.assert_allclose(cheek[:2], [0.61, 0.34], atol=0.01)


def test_fingertip_centric_dhonnobad_chin_vs_forehead():
    """Verify dhonnobad achieves >=92% when fingertips touch Chin and <15% at Forehead."""
    normalizer = SpatialNormalizer()
    facemesh = _create_synthetic_facemesh()

    # Fingertips at Chin (y=0.45)
    hand_chin = _create_hand_with_fingertip_at(0.50, 0.45)
    score_chin, dist_cm_chin, _, _ = normalizer.calculate_anchor_alignment(
        hand_landmarks=hand_chin,
        target_anchor_name="CHIN",
        face_landmarks=facemesh
    )

    # Fingertips at Forehead (y=0.20)
    hand_forehead = _create_hand_with_fingertip_at(0.50, 0.20)
    score_forehead, dist_cm_forehead, hint_forehead, _ = normalizer.calculate_anchor_alignment(
        hand_landmarks=hand_forehead,
        target_anchor_name="CHIN",
        face_landmarks=facemesh
    )

    assert score_chin >= 0.92
    assert score_forehead < 0.15
    assert "আঙুলের ডগা" in hint_forehead
    assert "নামিয়ে" in hint_forehead or "চিবুক" in hint_forehead


def test_directional_fingertip_hints_salam_forehead():
    """Verify salam (Forehead target) generates hint when fingertips are placed at Chin."""
    normalizer = SpatialNormalizer()
    facemesh = _create_synthetic_facemesh()

    # Fingertips at Chin (y=0.45) when target is Forehead
    hand_chin = _create_hand_with_fingertip_at(0.50, 0.45)
    score, _, hint, _ = normalizer.calculate_anchor_alignment(
        hand_landmarks=hand_chin,
        target_anchor_name="FOREHEAD",
        face_landmarks=facemesh
    )

    assert score < 0.15
    assert hint is not None
    assert "আঙুলের ডগা" in hint
    assert "চিবুকে" in hint or "উপরে" in hint


def test_multi_feature_contacts_nose_cheek_forehead():
    """Verify distinct anchor scoring for Nose ('kho'), Cheek ('ma'), and Forehead ('salam')."""
    normalizer = SpatialNormalizer()
    facemesh = _create_synthetic_facemesh()

    # Nose touch (y=0.30)
    hand_nose = _create_hand_with_fingertip_at(0.50, 0.30)
    score_nose, _, _, _ = normalizer.calculate_anchor_alignment(hand_nose, "NOSE", facemesh)
    assert score_nose >= 0.92

    # Cheek touch (x=0.61, y=0.34)
    hand_cheek = _create_hand_with_fingertip_at(0.61, 0.34)
    score_cheek, _, _, _ = normalizer.calculate_anchor_alignment(hand_cheek, "CHEEK_RIGHT", facemesh)
    assert score_cheek >= 0.92

    # Forehead touch (y=0.21)
    hand_forehead = _create_hand_with_fingertip_at(0.50, 0.21)
    score_forehead, _, _, _ = normalizer.calculate_anchor_alignment(hand_forehead, "FOREHEAD", facemesh)
    assert score_forehead >= 0.92


def test_wrist_strictly_prohibited_for_facial_signs():
    """Verify facial signs never use wrist (0) as the articulator in AUTO mode."""
    normalizer = SpatialNormalizer()
    hand = _create_hand_with_fingertip_at(0.50, 0.45)  # Wrist is at y=0.61, tips at y=0.45
    chin_3d = np.array([0.50, 0.45, 0.0], dtype=np.float32)

    art = normalizer.resolve_active_articulator(
        hand, articulator_type="AUTO", anchor_3d=chin_3d, target_anchor_name="CHIN"
    )

    # Articulator must be near fingertip y=0.45, NOT wrist y=0.61
    assert art[1] < 0.50
    assert abs(art[1] - 0.45) < 0.02
