from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from backend.core.config import settings

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
    # specific arguments for sqlite to allow multi-threading (in aiosqlite it uses a single background thread)
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

async_session_maker = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)

async def get_async_db():
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()
