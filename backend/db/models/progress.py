from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, UniqueConstraint
from backend.db.models.base import Base

class LearningProgress(Base):
    __tablename__ = "learning_progress"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    lesson_slug = Column(String(255), nullable=False)
    tier_id = Column(Integer, nullable=False)
    accuracy_score = Column(Float, default=0.0)
    practice_count = Column(Integer, default=0)
    last_practiced_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "lesson_slug", name="uq_user_lesson"),
    )
