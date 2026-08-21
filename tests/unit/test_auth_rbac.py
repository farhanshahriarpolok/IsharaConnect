import pytest
from backend.auth.security import get_password_hash, verify_password, create_access_token, decode_access_token

def test_password_hashing():
    password = "secret_password123"
    hashed = get_password_hash(password)
    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("wrong_password", hashed) is False

def test_jwt_token():
    data = {"sub": "admin@isharaconnect.local", "role": "SUPER_ADMIN"}
    token = create_access_token(data)
    decoded = decode_access_token(token)
    
    assert decoded is not None
    assert decoded["sub"] == "admin@isharaconnect.local"
    assert decoded["role"] == "SUPER_ADMIN"
    assert "exp" in decoded

def test_jwt_invalid_token():
    assert decode_access_token("invalid.token.here") is None
