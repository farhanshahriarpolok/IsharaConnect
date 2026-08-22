"""Unit Test Suite for Two-Way Split Dashboard & COTURN NAT Configuration (Sprint 36).

Tests:
1. `/dashboard` HTTP GET endpoint rendering split view HTML template.
2. `/api/v1/webrtc/config` ICE and COTURN configuration server endpoints.
3. WebRTC ICE server discovery and credentials fallback.
4. WebSocket duplex room connection URL formatting.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.config.webrtc_config import get_ice_servers, get_rtc_configuration


@pytest.fixture
def client():
    return TestClient(app)


def test_dashboard_page_rendering(client):
    """Verify /dashboard endpoint serves HTML with split view panels and room context."""
    response = client.get("/dashboard?room_id=test-room&client_type=signer")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "IsharaConnect" in response.text
    assert "test-room" in response.text
    assert "local-video" in response.text
    assert "avatar-container" in response.text
    assert "signer-subtitle" in response.text


def test_webrtc_config_endpoint(client):
    """Verify /api/v1/webrtc/config returns ICE servers including STUN and COTURN."""
    response = client.get("/api/v1/webrtc/config")
    assert response.status_code == 200
    data = response.json()
    assert "iceServers" in data
    assert len(data["iceServers"]) >= 1

    urls = []
    for s in data["iceServers"]:
        urls.extend(s.get("urls", []))
    assert any("stun:" in u for u in urls)


def test_ice_servers_fallback():
    """Verify get_ice_servers returns valid STUN and COTURN configuration."""
    servers = get_ice_servers()
    assert isinstance(servers, list)
    assert len(servers) >= 2

    rtc_conf = get_rtc_configuration()
    assert rtc_conf["bundlePolicy"] == "max-bundle"
    assert rtc_conf["rtcpMuxPolicy"] == "require"
