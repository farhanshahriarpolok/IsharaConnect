"""Unit tests for the Certificate Generator."""

import os
import pytest
from desktop_app.controllers.certificate_generator import CertificateGenerator
import qrcode

def test_certificate_generation(tmp_path):
    """Test generating a certificate PDF."""
    cg = CertificateGenerator(output_dir=str(tmp_path))
    
    candidate_name = "Test User"
    tier_rank = "Tier 4 Master"
    score_percent = 95.0
    speed_wpm = 50
    
    filepath = cg.generate(candidate_name, tier_rank, score_percent, speed_wpm)
    
    assert os.path.exists(filepath)
    assert filepath.endswith(".pdf")
    
    # Check that QR code image was cleaned up
    qr_files = [f for f in os.listdir(tmp_path) if f.startswith("qr_")]
    assert len(qr_files) == 0
