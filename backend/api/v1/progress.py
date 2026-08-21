from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from backend.db.session import get_async_db
from backend.db.models.user import User
from backend.db.models.progress import LearningProgress
from backend.schemas.progress import ProgressSyncRequest, ProgressResponse
from backend.core.security import get_current_active_user
from datetime import datetime

router = APIRouter()

@router.post("/sync", response_model=ProgressResponse)
async def sync_progress(
    progress_in: ProgressSyncRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    stmt = select(LearningProgress).where(
        LearningProgress.user_id == current_user.id,
        LearningProgress.lesson_slug == progress_in.lesson_slug
    )
    result = await db.execute(stmt)
    progress = result.scalars().first()

    if progress:
        progress.accuracy_score = max(progress.accuracy_score, progress_in.accuracy_score)
        progress.practice_count += progress_in.practice_count
        progress.last_practiced_at = progress_in.last_practiced_at or datetime.utcnow()
    else:
        progress = LearningProgress(
            user_id=current_user.id,
            lesson_slug=progress_in.lesson_slug,
            tier_id=progress_in.tier_id,
            accuracy_score=progress_in.accuracy_score,
            practice_count=progress_in.practice_count,
            last_practiced_at=progress_in.last_practiced_at or datetime.utcnow()
        )
        db.add(progress)

    await db.commit()
    await db.refresh(progress)
    return progress

@router.get("/sync", response_model=List[ProgressResponse])
async def get_progress(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    stmt = select(LearningProgress).where(LearningProgress.user_id == current_user.id)
    result = await db.execute(stmt)
    return result.scalars().all()
