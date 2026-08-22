"""Unit tests for BdSL Video v3 Hyper-Kinematic Schema Extractor."""

import numpy as np
import pytest
from dataset.tools.bdsl_video_v3_extractor import BdSLVideoV3Extractor


def test_fit_cubic_bezier():
    """Test least-squares optimization for 3D Bézier curve fitting."""
    extractor = BdSLVideoV3Extractor()

    # Generate synthetic parabolic 3D trajectory
    t = np.linspace(0, 1, 30)
    x = 0.5 + 0.1 * t
    y = 0.4 + 0.2 * t - 0.1 * (t ** 2)
    z = -0.1 + 0.05 * t
    pts = np.stack([x, y, z], axis=1)

    p0, p1, p2, p3 = extractor._fit_cubic_bezier(pts)

    assert p0.shape == (3,)
    assert p1.shape == (3,)
    assert p2.shape == (3,)
    assert p3.shape == (3,)

    # Endpoints match trajectory boundary
    assert np.allclose(p0, pts[0], atol=1e-3)
    assert np.allclose(p3, pts[-1], atol=1e-3)


def test_detect_temporal_phases():
    """Test velocity profiling and temporal phase duration segmentation."""
    extractor = BdSLVideoV3Extractor()

    # Synthetic trajectory of 30 frames at 30 fps (1.0 second = 1000ms)
    traj = np.zeros((30, 3))
    traj[:, 1] = np.linspace(0, 0.3, 30)  # linear movement

    phases = extractor._detect_temporal_phases(traj, fps=30.0)

    assert phases["total_ms"] == 1000
    assert phases["prep_ms"] == 200
    assert phases["stroke_ms"] == 500
    assert phases["hold_ms"] == 150
    assert phases["retract_ms"] == 150
    assert phases["peak_vel"] > 0


def test_classify_handshape():
    """Test classifying handshape codes based on landmark geometry."""
    extractor = BdSLVideoV3Extractor()

    # Synthetic open hand
    open_hand = np.zeros((21, 3))
    open_hand[0] = [0.5, 0.5, 0.0]    # wrist
    open_hand[4] = [0.45, 0.35, 0.0]  # thumb
    open_hand[8] = [0.50, 0.30, 0.0]  # index tip (distance = 0.20 > 0.15)
    open_hand[20] = [0.55, 0.30, 0.0] # pinky tip (distance = 0.20 > 0.15)

    code = extractor._classify_handshape([open_hand])
    assert code == "HS_FLAT_BENT_THUMB"

    # Synthetic index pointing hand
    index_pointing = np.zeros((21, 3))
    index_pointing[0] = [0.5, 0.5, 0.0]
    index_pointing[8] = [0.50, 0.30, 0.0]  # index extended (> 0.15)
    index_pointing[20] = [0.52, 0.48, 0.0] # pinky curled (< 0.15)

    code_pointing = extractor._classify_handshape([index_pointing])
    assert code_pointing == "HS_INDEX_EXTENDED"

    # Synthetic fist
    fist = np.zeros((21, 3))
    fist[0] = [0.5, 0.5, 0.0]
    fist[8] = [0.51, 0.48, 0.0]   # index curled (< 0.15)
    fist[20] = [0.52, 0.48, 0.0]  # pinky curled (< 0.15)

    code_fist = extractor._classify_handshape([fist])
    assert code_fist == "HS_FIST"
