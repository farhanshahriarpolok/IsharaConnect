from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.core.config import settings
from backend.database.models import Base, User, Sign
from backend.auth.security import get_password_hash

engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    
    # Seed SuperAdmin
    if not db.query(User).filter(User.email == "admin@isharaconnect.local").first():
        admin = User(
            email="admin@isharaconnect.local",
            hashed_password=get_password_hash("admin123"),
            role="SUPER_ADMIN",
            is_active=True
        )
        db.add(admin)
        
    # Seed default sign if not exists
    if not db.query(Sign).filter(Sign.slug == "ami").first():
        sign = Sign(
            slug="ami",
            label_bn="আমি",
            label_en="I",
            tier=1
        )
        db.add(sign)
        
    db.commit()
    db.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
