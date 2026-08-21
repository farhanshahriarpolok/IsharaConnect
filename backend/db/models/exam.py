from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey
from backend.db.models.base import Base

class ExamRecord(Base):
    __tablename__ = "exam_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    score_percentage = Column(Float, nullable=False)
    grade = Column(String(50), nullable=False)
    certificate_hash = Column(String(255), unique=True, index=True, nullable=True)
    verification_qr = Column(String(255), nullable=True)
    issued_at = Column(DateTime, default=datetime.utcnow, nullable=False)
