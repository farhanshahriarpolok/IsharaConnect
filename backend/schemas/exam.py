from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ExamRecordCreate(BaseModel):
    score_percentage: float
    grade: str
    certificate_hash: Optional[str] = None
    verification_qr: Optional[str] = None

class ExamRecordResponse(ExamRecordCreate):
    id: int
    user_id: str
    issued_at: datetime

    class Config:
        from_attributes = True
