import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.database.models import Base, User, Sign

# Use in-memory SQLite for tests
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def db_session():
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)

def test_create_user(db_session):
    user = User(email="test@local", hashed_password="pwd", role="USER")
    db_session.add(user)
    db_session.commit()
    
    saved_user = db_session.query(User).filter_by(email="test@local").first()
    assert saved_user is not None
    assert saved_user.role == "USER"
    assert saved_user.is_active is True

def test_create_sign(db_session):
    sign = Sign(slug="hello", label_bn="হ্যালো", label_en="Hello", tier=1)
    db_session.add(sign)
    db_session.commit()
    
    saved_sign = db_session.query(Sign).filter_by(slug="hello").first()
    assert saved_sign is not None
    assert saved_sign.label_en == "Hello"
