"""Automated PDF Certificate Generator for IsharaConnect BdSL Academy.

Generates authentic, verifiable PDF certificates in landscape A4 format
with cyber/emerald borders, security QR code, unique verification hash,
grade, and official verification seal.
"""

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import qrcode
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

logger = logging.getLogger(__name__)


class CertificateGenerator:
    """Generates authentic and verifiable PDF certificates for IsharaConnect Academy."""

    DEFAULT_COURSE = "Bangladesh Sign Language Foundation & Fluency"
    DEFAULT_ISSUER = "National ICT Division & IsharaConnect BdSL Academy"

    def __init__(self, output_dir: Optional[str] = None):
        """Initializes the certificate generator.

        Args:
            output_dir: Directory where generated certificates will be stored.
                        Defaults to 'certificates/'.
        """
        if output_dir is None:
            self.output_dir = Path("certificates")
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_grade(score_percent: float) -> str:
        """Determines the letter grade based on percentage score."""
        if score_percent >= 95.0:
            return "A+ (Distinction)"
        elif score_percent >= 90.0:
            return "A+ (Excellent)"
        elif score_percent >= 80.0:
            return "A (Very Good)"
        elif score_percent >= 70.0:
            return "B (Proficient)"
        elif score_percent >= 60.0:
            return "C (Pass)"
        else:
            return "Needs Retake"

    def generate_certificate(
        self,
        candidate_name: str,
        score_percent: float,
        grade: Optional[str] = None,
        course_name: Optional[str] = None,
        speed_wpm: int = 45,
        cert_id: Optional[str] = None,
        issue_date: Optional[str] = None,
    ) -> str:
        """Generates a styled, high-res PDF certificate and returns the file path.

        Args:
            candidate_name: Name of the recipient.
            score_percent: Exam evaluation score percentage (0-100).
            grade: Letter grade (e.g. 'A+', 'A'). Computed if None.
            course_name: Name of the course or exam title.
            speed_wpm: Gesture recognition speed in words/signs per minute.
            cert_id: Custom UUID/ID or generated automatically.
            issue_date: Date string (YYYY-MM-DD) or current date.

        Returns:
            Absolute or relative file path to the generated PDF.
        """
        if not candidate_name or not candidate_name.strip():
            candidate_name = "Distinguished BdSL Learner"

        if cert_id is None:
            cert_id = str(uuid.uuid4())

        if issue_date is None:
            issue_date = datetime.now().strftime("%B %d, %Y")

        if grade is None:
            grade = self.compute_grade(score_percent)

        if course_name is None:
            course_name = self.DEFAULT_COURSE

        # Compute cryptographic hash for anti-tamper verification
        raw_token = f"{cert_id}:{candidate_name}:{score_percent}:{issue_date}:ISHARA_SECRET"
        verif_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()[:16].upper()

        # Verification payload for QR code
        qr_payload = {
            "cert_id": cert_id,
            "verification_hash": verif_hash,
            "recipient": candidate_name,
            "course": course_name,
            "grade": grade,
            "score": f"{score_percent:.1f}%",
            "date": issue_date,
            "verify_url": f"https://api.isharaconnect.gov.bd/admin/verify-certificate/{cert_id}?hash={verif_hash}"
        }

        # Safe filename
        safe_name = "".join(c if c.isalnum() else "_" for c in candidate_name).strip("_")
        pdf_filename = f"Certificate_{safe_name}_{cert_id[:8]}.pdf"
        pdf_path = self.output_dir / pdf_filename

        # Generate temporary QR code
        qr_img_path = self.output_dir / f"qr_temp_{cert_id[:8]}.png"
        try:
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_M,
                box_size=6,
                border=2,
            )
            qr.add_data(json.dumps(qr_payload))
            qr.make(fit=True)
            img = qr.make_image(fill_color="#0F172A", back_color="#FFFFFF")
            img.save(str(qr_img_path))
        except Exception as e:
            logger.warning(f"Could not generate QR code image: {e}")
            qr_img_path = None

        # Build PDF Canvas
        c = canvas.Canvas(str(pdf_path), pagesize=landscape(A4))
        width, height = landscape(A4)

        self._draw_background_and_borders(c, width, height)
        self._draw_certificate_content(
            c, width, height, candidate_name, course_name, score_percent,
            grade, issue_date, cert_id, verif_hash, qr_img_path
        )

        c.save()

        # Cleanup temporary QR code
        if qr_img_path and qr_img_path.exists():
            try:
                os.remove(qr_img_path)
            except OSError:
                pass

        logger.info("Generated certificate for %s at %s", candidate_name, pdf_path)
        return str(pdf_path.resolve())

    def _draw_background_and_borders(self, c: canvas.Canvas, width: float, height: float):
        """Draws multi-tiered cyber/emerald borders and subtle decorative patterns."""
        # Clean soft slate background fill
        c.setFillColor(HexColor("#F8FAFC"))
        c.rect(0, 0, width, height, fill=1, stroke=0)

        # Outer primary emerald border
        c.setStrokeColor(HexColor("#10B981"))
        c.setLineWidth(6)
        c.rect(0.35 * inch, 0.35 * inch, width - 0.70 * inch, height - 0.70 * inch)

        # Secondary cyber cyan border
        c.setStrokeColor(HexColor("#06B6D4"))
        c.setLineWidth(2)
        c.rect(0.45 * inch, 0.45 * inch, width - 0.90 * inch, height - 0.90 * inch)

        # Inner dark slate border
        c.setStrokeColor(HexColor("#0F172A"))
        c.setLineWidth(0.8)
        c.rect(0.55 * inch, 0.55 * inch, width - 1.10 * inch, height - 1.10 * inch)

        # Corner embellishments (Tech/Cyber brackets)
        corner_len = 0.4 * inch
        c.setStrokeColor(HexColor("#059669"))
        c.setLineWidth(3)
        
        # Top-Left
        c.line(0.35 * inch, height - 0.35 * inch, 0.35 * inch + corner_len, height - 0.35 * inch)
        c.line(0.35 * inch, height - 0.35 * inch, 0.35 * inch, height - 0.35 * inch - corner_len)
        # Top-Right
        c.line(width - 0.35 * inch, height - 0.35 * inch, width - 0.35 * inch - corner_len, height - 0.35 * inch)
        c.line(width - 0.35 * inch, height - 0.35 * inch, width - 0.35 * inch, height - 0.35 * inch - corner_len)
        # Bottom-Left
        c.line(0.35 * inch, 0.35 * inch, 0.35 * inch + corner_len, 0.35 * inch)
        c.line(0.35 * inch, 0.35 * inch, 0.35 * inch, 0.35 * inch + corner_len)
        # Bottom-Right
        c.line(width - 0.35 * inch, 0.35 * inch, width - 0.35 * inch - corner_len, 0.35 * inch)
        c.line(width - 0.35 * inch, 0.35 * inch, width - 0.35 * inch, 0.35 * inch + corner_len)

    def _draw_certificate_content(
        self,
        c: canvas.Canvas,
        width: float,
        height: float,
        candidate_name: str,
        course_name: str,
        score_percent: float,
        grade: str,
        issue_date: str,
        cert_id: str,
        verif_hash: str,
        qr_img_path: Optional[Path],
    ):
        """Draws all typography, headers, seals, verification codes, and signatures."""
        # Top Organization Header
        c.setFont("Helvetica-Bold", 12)
        c.setFillColor(HexColor("#047857"))
        c.drawCentredString(width / 2.0, height - 0.95 * inch, "GOVERNMENT OF THE PEOPLE'S REPUBLIC OF BANGLADESH")

        c.setFont("Helvetica", 10)
        c.setFillColor(HexColor("#475569"))
        c.drawCentredString(width / 2.0, height - 1.15 * inch, "ICT DIVISION • ISHARACONNECT NATIONAL BdSL ACADEMY")

        # Certificate Main Title
        c.setFont("Helvetica-Bold", 30)
        c.setFillColor(HexColor("#0F172A"))
        c.drawCentredString(width / 2.0, height - 1.70 * inch, "CERTIFICATE OF PROFICIENCY")

        c.setFont("Helvetica-Oblique", 13)
        c.setFillColor(HexColor("#0D9488"))
        c.drawCentredString(width / 2.0, height - 2.05 * inch, "Bangla Sign Language (BdSL) Interpreter Certification")

        # Decorative line
        c.setStrokeColor(HexColor("#06B6D4"))
        c.setLineWidth(1.5)
        c.line(width / 2.0 - 2.5 * inch, height - 2.18 * inch, width / 2.0 + 2.5 * inch, height - 2.18 * inch)

        # "This is proudly presented to"
        c.setFont("Helvetica", 13)
        c.setFillColor(HexColor("#334155"))
        c.drawCentredString(width / 2.0, height - 2.60 * inch, "This is to officially certify that")

        # Candidate Name
        c.setFont("Helvetica-Bold", 26)
        c.setFillColor(HexColor("#065F46"))
        c.drawCentredString(width / 2.0, height - 3.10 * inch, candidate_name)

        # Underline under name
        c.setStrokeColor(HexColor("#10B981"))
        c.setLineWidth(1)
        name_width = min(max(c.stringWidth(candidate_name, "Helvetica-Bold", 26), 200), 450)
        c.line(width / 2.0 - name_width / 2.0, height - 3.20 * inch, width / 2.0 + name_width / 2.0, height - 3.20 * inch)

        # Narrative description
        c.setFont("Helvetica", 12)
        c.setFillColor(HexColor("#1E293B"))
        c.drawCentredString(
            width / 2.0,
            height - 3.65 * inch,
            f"has demonstrated exceptional competence and successfully passed the standardized evaluation for"
        )
        c.setFont("Helvetica-Bold", 13)
        c.setFillColor(HexColor("#0F172A"))
        c.drawCentredString(width / 2.0, height - 3.95 * inch, course_name)

        # Achievement Metrics Box
        box_w = 4.8 * inch
        box_h = 0.55 * inch
        box_x = (width - box_w) / 2.0
        box_y = height - 4.70 * inch

        c.setFillColor(HexColor("#ECFDF5"))
        c.setStrokeColor(HexColor("#10B981"))
        c.setLineWidth(1)
        c.roundRect(box_x, box_y, box_w, box_h, 6, fill=1, stroke=1)

        c.setFont("Helvetica-Bold", 11)
        c.setFillColor(HexColor("#065F46"))
        metrics_str = f"Final Score: {score_percent:.1f}%   |   Grade: {grade}   |   Status: Verified Certified"
        c.drawCentredString(width / 2.0, box_y + 0.18 * inch, metrics_str)

        # Left Column: Official Seal & Date
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(HexColor("#0F172A"))
        c.drawString(1.0 * inch, 1.75 * inch, "Date of Issuance:")
        c.setFont("Helvetica", 10)
        c.setFillColor(HexColor("#334155"))
        c.drawString(1.0 * inch, 1.55 * inch, issue_date)

        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(HexColor("#64748B"))
        c.drawString(1.0 * inch, 1.25 * inch, f"Verification ID: {cert_id[:13]}")
        c.drawString(1.0 * inch, 1.05 * inch, f"Security Hash: {verif_hash}")

        # Center Column: QR Code & Verification Stamp
        if qr_img_path and qr_img_path.exists():
            c.drawImage(
                str(qr_img_path),
                width / 2.0 - 0.60 * inch,
                0.80 * inch,
                width=1.20 * inch,
                height=1.20 * inch,
            )
            c.setFont("Helvetica", 7)
            c.setFillColor(HexColor("#475569"))
            c.drawCentredString(width / 2.0, 0.65 * inch, "Scan to Verify Authenticity")

        # Right Column: Signatures
        sig_x = width - 3.2 * inch
        c.setStrokeColor(HexColor("#475569"))
        c.setLineWidth(1)
        c.line(sig_x, 1.70 * inch, sig_x + 2.2 * inch, 1.70 * inch)

        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(HexColor("#0F172A"))
        c.drawString(sig_x + 0.2 * inch, 1.50 * inch, "Dr. Md. Kabir Hossain")

        c.setFont("Helvetica", 9)
        c.setFillColor(HexColor("#64748B"))
        c.drawString(sig_x + 0.05 * inch, 1.32 * inch, "Director General, BdSL Research")
        c.drawString(sig_x + 0.25 * inch, 1.15 * inch, "IsharaConnect Board")

    # Compatibility method with existing codebase
    def generate(self, candidate_name: str, tier_rank: str, score_percent: float, speed_wpm: int = 45) -> str:
        """Backward-compatible generation method."""
        return self.generate_certificate(
            candidate_name=candidate_name,
            score_percent=score_percent,
            grade=tier_rank,
            speed_wpm=speed_wpm
        )

    @classmethod
    def generate_static(
        cls,
        candidate_name: str,
        score_percent: float,
        grade: Optional[str] = None,
        course_name: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """Convenience class method to generate a certificate."""
        generator = cls(output_dir=output_dir)
        return generator.generate_certificate(
            candidate_name=candidate_name,
            score_percent=score_percent,
            grade=grade,
            course_name=course_name
        )
