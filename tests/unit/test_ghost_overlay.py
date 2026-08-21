import pytest
from desktop_app.ui.components.ghost_overlay import GhostOverlayPainter

def test_ghost_overlay_score():
    painter = GhostOverlayPainter()
    
    # Create mock perfect match
    ref = [(float(i), float(i)) for i in range(42)]
    live = [(float(i), float(i)) for i in range(42)]
    
    score = painter.calculate_alignment_score(ref, live)
    assert score == 100.0
    
    # Create complete mismatch
    live_bad = [(float(i+100), float(i+100)) for i in range(42)]
    score_bad = painter.calculate_alignment_score(ref, live_bad)
    assert score_bad == 0.0

def test_ghost_overlay_empty():
    painter = GhostOverlayPainter()
    assert painter.calculate_alignment_score([], []) == 0.0
    assert painter.calculate_alignment_score([(0,0)], []) == 0.0
