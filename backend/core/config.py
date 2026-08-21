from typing import List, Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SECRET_KEY: str = "super-secret-key-for-isharaconnect-dev"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_SECRET_KEY: str = "super-secret-jwt-key"
    JWT_ALGORITHM: str = "HS256"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]
    DATABASE_URL: str = "sqlite+aiosqlite:///./isharaconnect.db"
    REDIS_URL: Optional[str] = None
    
    class Config:
        env_file = ".env"

settings = Settings()
