"""Unit tests for NLP Translation API endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import app


@pytest.mark.asyncio
async def test_nlp_translate_api_endpoint():
    """Test POST /api/v1/nlp/translate with an array of glosses."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/api/v1/nlp/translate", json={
            "glosses": ["আমি", "ভাত", "খাওয়া"]
        })

    assert response.status_code == 200
    data = response.json()
    assert data["raw_glosses"] == ["আমি", "ভাত", "খাওয়া"]
    assert data["translated_text"] == "আমি ভাত খাচ্ছি।"
    assert data["confidence"] >= 0.9
    assert data["is_final"] is True


@pytest.mark.asyncio
async def test_nlp_debounce_stream_and_reset_endpoints():
    """Test POST /api/v1/nlp/debounce-stream and /reset endpoints."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Reset first
        reset_res = await ac.post("/api/v1/nlp/reset")
        assert reset_res.status_code == 200

        # Send 3 stream tokens
        for _ in range(3):
            stream_res = await ac.post("/api/v1/nlp/debounce-stream", json={
                "token": "ধন্যবাদ",
                "confidence": 0.95,
                "timestamp": 12345.67
            })
            assert stream_res.status_code == 200

        data = stream_res.json()
        assert "ধন্যবাদ" in data["translated_text"]
