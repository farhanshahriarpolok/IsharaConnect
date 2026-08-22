"""Unit tests for Distributed Redis Pub/Sub Room Adapter and Resilient Room Manager."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import app
from backend.websockets.base_adapter import ClientType, InMemoryRoomAdapter
from backend.websockets.redis_adapter import RedisRoomAdapter
from backend.websockets.room_manager import ConnectionManager, get_room_adapter


class MockWebSocket:
    """Mock WebSocket for unit testing room adapters."""
    def __init__(self):
        self.accepted = False
        self.sent_messages = []
        self.closed = False

    async def accept(self):
        self.accepted = True

    async def send_json(self, data: dict):
        self.sent_messages.append(data)

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_in_memory_room_adapter_workflow():
    """Test standard lifecycle on the in-memory fallback adapter."""
    adapter = InMemoryRoomAdapter()
    ws1 = MockWebSocket()
    ws2 = MockWebSocket()

    room_id = "test_room_memory_01"
    await adapter.connect(ws1, room_id, ClientType.SIGNER)
    await adapter.connect(ws2, room_id, ClientType.SPEAKER)

    assert ws1.accepted and ws2.accepted
    assert len(adapter.active_connections[room_id]) == 2

    # Broadcast message from ws1 -> ws2 should receive it
    test_msg = {"type": "SIGN_TRANSLATION", "data": {"label_bn": "ধন্যবাদ"}}
    await adapter.broadcast_message(room_id, test_msg, sender=ws1)

    assert len(ws2.sent_messages) == 1
    assert ws2.sent_messages[0]["data"]["label_bn"] == "ধন্যবাদ"
    assert len(ws1.sent_messages) == 0

    # Active rooms inspection
    active_rooms = await adapter.get_active_rooms()
    assert len(active_rooms) == 1
    assert active_rooms[0]["room_id"] == room_id
    assert active_rooms[0]["signers_count"] == 1
    assert active_rooms[0]["speakers_count"] == 1

    # Disconnect
    await adapter.disconnect(ws1, room_id)
    await adapter.disconnect(ws2, room_id)
    assert room_id not in adapter.active_connections
    await adapter.close()


@pytest.mark.asyncio
async def test_get_room_adapter_fallback_on_invalid_url():
    """Test transparent fallback to InMemoryRoomAdapter when Redis is unconfigured or unreachable."""
    # 1. No URL provided
    adapter1 = await get_room_adapter(redis_url=None)
    assert isinstance(adapter1, InMemoryRoomAdapter)

    # 2. Invalid / unreachable Redis URL
    adapter2 = await get_room_adapter(redis_url="redis://127.0.0.1:59999/0")
    assert isinstance(adapter2, InMemoryRoomAdapter)


@pytest.mark.asyncio
async def test_redis_room_adapter_pubsub_lifecycle():
    """Test RedisRoomAdapter subscription, publish, listener message routing, and unsubscription."""
    mock_redis = AsyncMock()
    mock_pubsub = AsyncMock()

    # Mock pubsub.listen() generator
    async def mock_listen():
        # Yield one message then stop
        yield {
            "type": "message",
            "data": json.dumps({
                "type": "SPEECH_TEXT",
                "data": {"text": "শুভ সকাল"},
                "_node_id": "remote_node_xyz",
                "_sender_socket_id": 99999,
                "_timestamp": 123456.78
            })
        }

    mock_pubsub.listen = mock_listen
    mock_redis.pubsub = MagicMock(return_value=mock_pubsub)
    mock_redis.ping = AsyncMock(return_value=True)
    mock_redis.keys = AsyncMock(return_value=["ishara:room_meta:redis_room_01"])
    mock_redis.get = AsyncMock(return_value=json.dumps({
        "room_id": "redis_room_01",
        "signers_count": 1,
        "speakers_count": 0,
        "total_participants": 1,
        "created_at": "2026-08-22T10:00:00",
        "last_active": "2026-08-22T10:05:00",
        "node_id": "test_node"
    }))
    mock_redis.set = AsyncMock()
    mock_redis.publish = AsyncMock()
    mock_redis.expire = AsyncMock()
    mock_redis.close = AsyncMock()

    with patch("backend.websockets.redis_adapter.aioredis.from_url", return_value=mock_redis):
        adapter = RedisRoomAdapter("redis://localhost:6379/0")
        ws = MockWebSocket()
        room_id = "redis_room_01"

        # 1. Connect first client -> subscribes to Redis channel
        await adapter.connect(ws, room_id, ClientType.SIGNER)
        mock_pubsub.subscribe.assert_called_once_with(f"channel:ishara:rooms:{room_id}")
        mock_redis.set.assert_called()

        # 2. Broadcast message -> publishes to Redis
        msg = {"type": "SIGN_TRANSLATION", "data": {"label_bn": "স্বাগতম"}}
        await adapter.broadcast_message(room_id, msg, sender=ws)
        mock_redis.publish.assert_called()

        # Let the listener loop process the mock pubsub message
        await asyncio.sleep(0.05)
        assert len(ws.sent_messages) == 1
        assert ws.sent_messages[0]["data"]["text"] == "শুভ সকাল"

        # 3. Active rooms stats
        rooms = await adapter.get_active_rooms()
        assert len(rooms) == 1
        assert rooms[0]["room_id"] == "redis_room_01"

        # 4. Disconnect -> unsubs from Redis
        await adapter.disconnect(ws, room_id)
        mock_pubsub.unsubscribe.assert_called_once_with(f"channel:ishara:rooms:{room_id}")
        await adapter.close()


@pytest.mark.asyncio
async def test_active_rooms_api_endpoint():
    """Test GET /api/v1/rooms/active endpoint returns cluster room statistics."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/v1/rooms/active")

    assert response.status_code == 200
    data = response.json()
    assert "active_rooms_count" in data
    assert "adapter_type" in data
    assert "rooms" in data
    assert isinstance(data["rooms"], list)
