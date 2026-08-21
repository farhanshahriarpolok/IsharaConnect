"""Certificate Generator for BdSL Academy."""

import os
import json
import uuid
from datetime import datetime
import qrcode
from reportlab.lib.pagesizes import landscape, A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor

class CertificateGenerator:
    """Generates verifiable PDF certificates for Academy graduates."""
    
    def __init__(self, output_dir: str = "output/certificates"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate(self, candidate_name: str, tier_rank: str, score_percent: float, speed_wpm: int) -> str:
        """Generate a PDF certificate and return its filepath."""
        cert_id = str(uuid.uuid4())
        issue_date = datetime.now().strftime("%Y-%m-%d")
        
        # QR Code Data
        qr_data = {
            "cert_id": cert_id,
            "name": candidate_name,
            "tier": tier_rank,
            "score": score_percent,
            "date": issue_date,
            "verify_url": f"https://api.isharaconnect.gov.bd/admin/verify-certificate/{cert_id}"
        }
        
        qr = qrcode.make(json.dumps(qr_data))
        qr_path = os.path.join(self.output_dir, f"qr_{cert_id}.png")
        qr.save(qr_path)
        
        filename = f"Certificate_{candidate_name.replace(' ', '_')}_{cert_id[:8]}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        c = canvas.Canvas(filepath, pagesize=landscape(A4))
        width, height = landscape(A4)
        
        # Border
        c.setStrokeColor(HexColor("#89B4FA"))
        c.setLineWidth(5)
        c.rect(0.5 * inch, 0.5 * inch, width - 1 * inch, height - 1 * inch)
        
        # Inner border
        c.setStrokeColor(HexColor("#F38BA8"))
        c.setLineWidth(1)
        c.rect(0.6 * inch, 0.6 * inch, width - 1.2 * inch, height - 1.2 * inch)
        
        # Title
        c.setFont("Helvetica-Bold", 36)
        c.setFillColor(HexColor("#11111B"))
        c.drawCentredString(width / 2.0, height - 2 * inch, "Certificate of BdSL Proficiency")
        
        c.setFont("Helvetica", 18)
        c.drawCentredString(width / 2.0, height - 2.5 * inch, "National ICT Division")
        
        c.setFont("Helvetica", 14)
        c.drawCentredString(width / 2.0, height - 3.5 * inch, "This is to certify that")
        
        c.setFont("Helvetica-Bold", 28)
        c.setFillColor(HexColor("#313244"))
        c.drawCentredString(width / 2.0, height - 4.2 * inch, candidate_name)
        
        c.setFont("Helvetica", 14)
        c.setFillColor(HexColor("#11111B"))
        c.drawCentredString(width / 2.0, height - 4.8 * inch, "has successfully completed the BdSL Interpreter Academy Training")
        c.drawCentredString(width / 2.0, height - 5.2 * inch, f"Rank: {tier_rank} | Accuracy: {score_percent}% | Speed: {speed_wpm} WPM")
        
        # Date & Signature
        c.setFont("Helvetica", 12)
        c.drawString(1.5 * inch, 1.5 * inch, f"Date: {issue_date}")
        c.line(1.5 * inch, 1.4 * inch, 3.5 * inch, 1.4 * inch)
        
        c.drawString(width - 3.5 * inch, 1.5 * inch, "Authorized Signature")
        c.line(width - 3.5 * inch, 1.4 * inch, width - 1.5 * inch, 1.4 * inch)
        
        # Draw QR Code
        c.drawImage(qr_path, width / 2.0 - 0.75 * inch, 1.0 * inch, width=1.5 * inch, height=1.5 * inch)
        
        c.setFont("Helvetica", 8)
        c.drawCentredString(width / 2.0, 0.8 * inch, f"ID: {cert_id}")
        
        c.save()
        
        # Cleanup QR image
        if os.path.exists(qr_path):
            os.remove(qr_path)
            
        return filepath
