"""Unit Test Suite for Universal 3D Biometric Face Anchor Resolver (Sprint 33).

Tests:
1. Universal 3D landmark binding for MediaPipe FaceMesh & Pose anchor points.
2. Biometrically normalized distance for Chin vs Forehead (e.g. 'dhonnobad' target).
3. Directional text guidance vectors ("হাত উপরে তুলুন" vs "হাত নিচে নামান").
4. Active articulator resolver (Fingertips vs Wrist vs Palm Center).
5. Target Anchor Aura and directional vector arrow rendering in CameraHUDOverlay.
"""

import numpy as np
import pytest

from core_engine.vision.spatial_normalizer import SpatialNormalizer
from core_engine.vision.sign_correction_advisor import SignCorrectionAdvisor
from desktop_app.ui.components.camera_hud_overlay import CameraHUDOverlay


def _create_synthetic_facemesh() -> np.ndarray:
    """Creates synthetic 468 FaceMesh array with realistic proportions."""
    face = np.zeros((468, 3), dtype=np.float32)
    # Forehead (10) at y=0.20, Chin (152) at y=0.45 (Height = 0.25)
    face[10] = [0.50, 0.20, 0.0]
    face[151] = [0.50, 0.22, 0.0]
    face[152] = [0.50, 0.45, 0.0]
    face[175] = [0.50, 0.44, 0.0]
    # Upper Lip (0, 13) at y=0.38
    face[0] = [0.50, 0.38, 0.0]
    face[13] = [0.50, 0.38, 0.0]
    # Cheek Right (234) at x=0.62, y=0.34
    face[234] = [0.62, 0.34, 0.0]
    # Cheek Left (454) at x=0.38, y=0.34
    face[454] = [0.38, 0.34, 0.0]
    return face


def _create_hand_at_location(x: float, y: float) -> np.ndarray:
    """Creates a 21-landmark hand centered at given (x, y)."""
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[0] = [x, y + 0.10, 0.0]  # Wrist
    for i in range(1, 21):
        hand[i] = [x, y, 0.0]     # Fingertips at (x, y)
    return hand


def test_universal_3d_anchor_binding():
    """Verify get_anatomical_anchor_3d correctly extracts coordinates from FaceMesh."""
    normalizer = SpatialNormalizer()
    facemesh = _create_synthetic_facemesh()

    chin_3d = normalizer.get_anatomical_anchor_3d("CHIN", face_landmarks=facemesh)
    forehead_3d = normalizer.get_anatomical_anchor_3d("FOREHEAD", face_landmarks=facemesh)
    cheek_3d = normalizer.get_anatomical_anchor_3d("CHEEK_RIGHT", face_landmarks=facemesh)

    np.testing.assert_allclose(chin_3d[:2], [0.50, 0.445], atol=0.01)
    np.testing.assert_allclose(forehead_3d[:2], [0.50, 0.21], atol=0.01)
    np.testing.assert_allclose(cheek_3d[:2], [0.62, 0.34], atol=0.01)


def test_biometric_distance_chin_vs_forehead():
    """Verify 'dhonnobad' (Chin target) scores >= 90% at chin and < 20% at forehead."""
    normalizer = SpatialNormalizer()
    facemesh = _create_synthetic_facemesh()

    # Hand at Chin (y=0.45)
    hand_chin = _create_hand_at_location(0.50, 0.45)
    score_chin, dist_cm_chin, _, _ = normalizer.calculate_anchor_alignment(
        hand_landmarks=hand_chin,
        target_anchor_name="CHIN",
        face_landmarks=facemesh,
        articulator_type="FINGERTIPS"
    )

    # Hand at Forehead (y=0.20)
    hand_forehead = _create_hand_at_location(0.50, 0.20)
    score_forehead, dist_cm_forehead, _, _ = normalizer.calculate_anchor_alignment(
        hand_landmarks=hand_forehead,
        target_anchor_name="CHIN",
        face_landmarks=facemesh,
        articulator_type="FINGERTIPS"
    )

    assert score_chin >= 0.90
    assert score_forehead < 0.20
    assert dist_cm_chin < dist_cm_forehead


def test_directional_guidance_text():
    """Verify exact directional Bengali hints for high vs low hand position."""
    normalizer = SpatialNormalizer()
    facemesh = _create_synthetic_facemesh()

    # Hand too low (Chest y=0.70) when target is Chin (y=0.45)
    hand_low = _create_hand_at_location(0.50, 0.70)
    _, _, hint_low, _ = normalizer.calculate_anchor_alignment(
        hand_landmarks=hand_low,
        target_anchor_name="CHIN",
        face_landmarks=facemesh
    )
    assert hint_low is not None
    assert "হাত নিচে রয়েছে" in hint_low
    assert "উপরে" in hint_low

    # Hand too high (Forehead y=0.15) when target is Chin (y=0.45)
    hand_high = _create_hand_at_location(0.50, 0.15)
    _, _, hint_high, _ = normalizer.calculate_anchor_alignment(
        hand_landmarks=hand_high,
        target_anchor_name="CHIN",
        face_landmarks=facemesh
    )
    assert hint_high is not None
    assert "উপরে" in hint_high
    assert "নিচে নামিয়ে" in hint_high


def test_active_articulator_modes():
    """Verify resolve_active_articulator differentiates between wrist and fingertips."""
    normalizer = SpatialNormalizer()
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[0] = [0.50, 0.60, 0.0]  # Wrist
    hand[8] = [0.50, 0.35, 0.0]  # Index Tip
    hand[12] = [0.50, 0.35, 0.0] # Middle Tip

    art_wrist = normalizer.resolve_active_articulator(hand, articulator_type="WRIST")
    art_tips = normalizer.resolve_active_articulator(hand, articulator_type="FINGERTIPS")

    np.testing.assert_allclose(art_wrist[:2], [0.50, 0.60], atol=1e-5)
    np.testing.assert_allclose(art_tips[:2], [0.50, 0.35], atol=1e-5)


def test_camera_hud_target_anchor_aura():
    """Verify CameraHUDOverlay draws on-face aura and directional vector arrow."""
    hud = CameraHUDOverlay()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    facemesh = _create_synthetic_facemesh()
    hand_away = _create_hand_at_location(0.50, 0.75)

    annotated = hud.draw_hud(
        frame=frame,
        right_landmarks=hand_away,
        target_anchor="CHIN",
        face_landmarks=facemesh,
        ghost_target_slug="dhonnobad"
    )

    assert annotated.shape == (480, 640, 3)
    assert np.count_nonzero(annotated) > 0
