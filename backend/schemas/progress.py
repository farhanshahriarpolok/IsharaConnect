from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ProgressSyncRequest(BaseModel):
    lesson_slug: str
    tier_id: int
    accuracy_score: float
    practice_count: int
    last_practiced_at: Optional[datetime] = None

class ProgressResponse(ProgressSyncRequest):
    id: int
    user_id: str
    last_practiced_at: datetime

    class Config:
        from_attributes = True
