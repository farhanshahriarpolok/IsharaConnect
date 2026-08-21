import asyncio
from backend.db.session import engine
from backend.db.models import Base

async def init_db():
    async with engine.begin() as conn:
        # Create all tables if they don't exist
        await conn.run_sync(Base.metadata.create_all)

if __name__ == "__main__":
    asyncio.run(init_db())
    print("Database initialized successfully")
