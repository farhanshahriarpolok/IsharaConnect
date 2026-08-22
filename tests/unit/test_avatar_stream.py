"""Unit tests for WebSocket Avatar Streaming Router."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from backend.main import app

client = TestClient(app)


def test_avatar_stream_valid_gloss():
    """Test avatar stream with valid v3 gloss sign."""
    with client.websocket_connect("/ws/avatar_stream") as ws:
        # Send gloss request with zero sleep in test to stream fast
        with patch("asyncio.sleep", return_value=None):
            ws.send_json({
                "glosses": ["ধন্যবাদ"]
            })

            # 1. Start stream message
            msg_start = ws.receive_json()
            assert msg_start["status"] == "start_stream"
            assert msg_start["total_frames"] > 0
            assert msg_start["fps"] == 60

            # 2. Receive first frame
            msg_frame = ws.receive_json()
            assert msg_frame["status"] == "frame"
            assert "data" in msg_frame
            assert "right_wrist" in msg_frame["data"]
            assert len(msg_frame["data"]["right_hand"]) == 21

            # 3. Read through remaining frames to end_stream
            for _ in range(msg_start["total_frames"] - 1):
                f = ws.receive_json()
                assert f["status"] == "frame"

            msg_end = ws.receive_json()
            assert msg_end["status"] == "end_stream"


def test_avatar_stream_invalid_gloss():
    """Test avatar stream returns error status when sign is not found."""
    with client.websocket_connect("/ws/avatar_stream") as ws:
        ws.send_json({
            "glosses": ["invalid_unknown_sign_12345"]
        })

        msg_err = ws.receive_json()
        assert msg_err["status"] == "error"
        assert "No valid v3 signs found" in msg_err["message"]
