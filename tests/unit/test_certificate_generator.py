"""Unit tests for the Certificate Generator."""

import os
from pathlib import Path
import pytest
from desktop_app.utils.certificate_generator import CertificateGenerator
from desktop_app.controllers.certificate_generator import CertificateGenerator as ControllerCertificateGenerator


def test_certificate_generation_with_tmp_path(tmp_path):
    """Test generating a certificate PDF using CertificateGenerator instance."""
    cg = CertificateGenerator(output_dir=str(tmp_path))
    
    candidate_name = "Farhan Shahriar Polok"
    score_percent = 96.5
    
    filepath = cg.generate_certificate(
        candidate_name=candidate_name,
        score_percent=score_percent,
        course_name="Bangladesh Sign Language Foundation & Fluency"
    )
    
    assert os.path.exists(filepath)
    assert filepath.endswith(".pdf")
    assert os.path.getsize(filepath) > 1000
    
    # Check that temporary QR code image was cleaned up
    qr_files = [f for f in os.listdir(tmp_path) if f.startswith("qr_")]
    assert len(qr_files) == 0


def test_certificate_generator_backward_compatibility(tmp_path):
    """Test backward compatibility with controller alias and .generate() method."""
    cg = ControllerCertificateGenerator(output_dir=str(tmp_path))
    filepath = cg.generate("Candidate Two", "Tier 3 Advanced", 88.0, speed_wpm=45)
    
    assert os.path.exists(filepath)
    assert filepath.endswith(".pdf")


def test_certificate_generator_static_method(tmp_path):
    """Test class method generate_static."""
    filepath = CertificateGenerator.generate_static(
        candidate_name="Anika Tabassum",
        score_percent=92.0,
        grade="A+ (Excellent)",
        output_dir=str(tmp_path)
    )
    assert os.path.exists(filepath)
    assert Path(filepath).is_file()


def test_certificate_compute_grade():
    """Test letter grade computations across tiers."""
    assert "A+ (Distinction)" in CertificateGenerator.compute_grade(98.0)
    assert "A+ (Excellent)" in CertificateGenerator.compute_grade(92.0)
    assert "A (Very Good)" in CertificateGenerator.compute_grade(85.0)
    assert "B (Proficient)" in CertificateGenerator.compute_grade(75.0)
    assert "C (Pass)" in CertificateGenerator.compute_grade(62.0)
    assert "Needs Retake" in CertificateGenerator.compute_grade(55.0)
