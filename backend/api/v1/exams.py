from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from backend.db.session import get_async_db
from backend.db.models.user import User
from backend.db.models.exam import ExamRecord
from backend.schemas.exam import ExamRecordCreate, ExamRecordResponse
from backend.core.security import get_current_active_user
import uuid

router = APIRouter()

@router.post("/save", response_model=ExamRecordResponse)
async def save_exam(
    exam_in: ExamRecordCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    certificate_hash = exam_in.certificate_hash or str(uuid.uuid4())
    
    exam = ExamRecord(
        user_id=current_user.id,
        score_percentage=exam_in.score_percentage,
        grade=exam_in.grade,
        certificate_hash=certificate_hash,
        verification_qr=exam_in.verification_qr
    )
    db.add(exam)
    await db.commit()
    await db.refresh(exam)
    return exam

@router.get("/history", response_model=List[ExamRecordResponse])
async def get_exam_history(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    stmt = select(ExamRecord).where(ExamRecord.user_id == current_user.id).order_by(ExamRecord.issued_at.desc())
    result = await db.execute(stmt)
    return result.scalars().all()
