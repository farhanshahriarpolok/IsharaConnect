from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, Float, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="USER")
    consent_status = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)

class Sign(Base):
    __tablename__ = "signs"
    id = Column(Integer, primary_key=True, index=True)
    slug = Column(String, unique=True, index=True, nullable=False)
    label_bn = Column(String, nullable=False)
    label_en = Column(String, nullable=False)
    tier = Column(Integer, default=1)
    handedness = Column(String, default="auto")
    motion_type = Column(String, default="static")

class DatasetSample(Base):
    __tablename__ = "dataset_samples"
    id = Column(Integer, primary_key=True, index=True)
    sign_id = Column(Integer, ForeignKey("signs.id"))
    signer_id = Column(Integer, ForeignKey("users.id"))
    landmarks_file = Column(String, nullable=False)
    quality_score = Column(Float, default=0.0)
    status = Column(String, default="PENDING")
    
    sign = relationship("Sign")
    signer = relationship("User")

class ModelVersion(Base):
    __tablename__ = "model_versions"
    version_tag = Column(String, primary_key=True, index=True)
    f1_score = Column(Float)
    latency_p95_ms = Column(Float)
    onnx_path = Column(String, nullable=False)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class LearningProgress(Base):
    __tablename__ = "learning_progress"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    sign_id = Column(Integer, ForeignKey("signs.id"))
    best_score = Column(Float, default=0.0)
    is_completed = Column(Boolean, default=False)
