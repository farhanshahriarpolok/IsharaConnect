"""Unit tests for WebRTC Connection Manager & Signaling Hub."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from backend.websockets.webrtc_signaling import WebRTCConnectionManager, webrtc_hub
from fastapi.testclient import TestClient
from backend.main import app


class TestWebRTCConnectionManager:
    @pytest.mark.asyncio
    async def test_connect_disconnect_and_broadcast(self):
        manager = WebRTCConnectionManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        room_id = "test_room_101"

        # 1. Connect ws1 and ws2
        await manager.connect(room_id, ws1)
        await manager.connect(room_id, ws2)
        assert len(manager.rooms[room_id]) == 2
        ws1.accept.assert_awaited_once()
        ws2.accept.assert_awaited_once()

        # 2. Broadcast from ws1 -> only ws2 receives
        offer_msg = {"type": "offer", "sdp": "v=0..."}
        await manager.broadcast_to_room(room_id, offer_msg, sender=ws1)
        ws2.send_text.assert_awaited_once()
        ws1.send_text.assert_not_awaited()

        # 3. Disconnect
        manager.disconnect(room_id, ws1)
        assert len(manager.rooms[room_id]) == 1
        manager.disconnect(room_id, ws2)
        assert room_id not in manager.rooms

    def test_dashboard_endpoint(self):
        client = TestClient(app)
        response = client.get("/dashboard")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
