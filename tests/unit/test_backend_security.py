import pytest
from backend.core.security import verify_password, get_password_hash, create_access_token

def test_password_hashing():
    password = "supersecretpassword123"
    hashed = get_password_hash(password)
    assert verify_password(password, hashed) is True
    assert verify_password("wrongpassword", hashed) is False

def test_create_access_token():
    token = create_access_token(subject="user_123")
    assert isinstance(token, str)
    assert len(token) > 20
