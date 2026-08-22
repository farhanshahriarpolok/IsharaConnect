"""Unit and Integration tests for Public Certificate Verification API and Web View."""

from datetime import datetime
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from backend.db.models.base import Base
from backend.db.models.exam import ExamRecord
from backend.db.models.user import User
from backend.db.session import async_session_maker, engine
from backend.main import app


@pytest_asyncio.fixture(scope="module", autouse=True)
async def setup_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
        
    # Seed a verified test user and exam record
    async with async_session_maker() as session:
        user = User(
            id="user_cert_test_01",
            email="certified_signer@isharaconnect.gov.bd",
            hashed_password="hashed_secure_pass",
            full_name="Fatima Jannat",
            role="learner",
            is_active=True
        )
        session.add(user)
        await session.commit()

        exam = ExamRecord(
            user_id="user_cert_test_01",
            score_percentage=97.5,
            grade="A+ (Distinction)",
            certificate_hash="CERT-VALID-SHA256-HASH-1234",
            verification_qr="https://api.isharaconnect.gov.bd/admin/verify-certificate/CERT-VALID-SHA256-HASH-1234",
            issued_at=datetime(2026, 8, 22, 10, 0, 0)
        )
        session.add(exam)
        await session.commit()

    yield

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.mark.asyncio
async def test_verify_certificate_json_valid_db_record():
    """Test JSON endpoint for a verified certificate in DB."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/v1/certificates/verify/CERT-VALID-SHA256-HASH-1234")
        
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "VALID"
    assert data["candidate_name"] == "Fatima Jannat"
    assert data["score_percentage"] == 97.5
    assert "Distinction" in data["grade"]
    assert "verification_hash" in data


@pytest.mark.asyncio
async def test_verify_certificate_json_demo_fallback():
    """Test JSON endpoint with demo/sample hash fallback."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/v1/certificates/verify/DEMO-HASH-SAMPLE-99")
        
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "VALID"
    assert "score_percentage" in data


@pytest.mark.asyncio
async def test_verify_certificate_json_invalid():
    """Test JSON endpoint returns 404 for an invalid unknown hash."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/v1/certificates/verify/INVALID_HASH_XYZ")
        
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert data["detail"]["status"] == "INVALID"


@pytest.mark.asyncio
async def test_verify_certificate_html_view_valid():
    """Test public HTML verification page renders successfully for a valid certificate."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/verify/CERT-VALID-SHA256-HASH-1234")
        
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "IsharaConnect BdSL Academy" in response.text
    assert "VERIFIED AUTHENTIC CERTIFICATE" in response.text
    assert "Fatima Jannat" in response.text


@pytest.mark.asyncio
async def test_verify_certificate_html_view_invalid():
    """Test public HTML verification page renders warning badge for invalid hash."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/verify/NON_EXISTENT_CERT_123")
        
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "UNVERIFIED RECORD" in response.text


@pytest.mark.asyncio
async def test_admin_verify_certificate_alias_route():
    """Test QR code route alias /admin/verify-certificate/{cert_id}."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/admin/verify-certificate/CERT-VALID-SHA256-HASH-1234")
        
    assert response.status_code == 200
    assert "Fatima Jannat" in response.text
