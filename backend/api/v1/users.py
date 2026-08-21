from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from backend.db.session import get_async_db
from backend.db.models.user import User
from backend.schemas.user import UserResponse, UserBase
from backend.core.security import get_current_active_user

router = APIRouter()

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@router.put("/me", response_model=UserResponse)
async def update_user_me(
    user_in: UserBase,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    current_user.full_name = user_in.full_name
    # Don't allow changing email or role directly here without checks in a real app,
    # but for simplicity we'll just allow full_name update.
    await db.commit()
    await db.refresh(current_user)
    return current_user
