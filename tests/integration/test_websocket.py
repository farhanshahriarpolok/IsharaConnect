"""Integration tests for FastAPI WebSocket Backend."""

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

def test_health_check():
    """Test health check REST API."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_websocket_room_duplex():
    """Test duplex WebSocket communication between SIGNER and SPEAKER."""
    room_id = "test_room_1"
    
    # Connect Signer
    with client.websocket_connect(f"/ws/room/{room_id}/signer") as ws_signer:
        # Connect Speaker
        with client.websocket_connect(f"/ws/room/{room_id}/speaker") as ws_speaker:
            
            # Signer sends translation event
            ws_signer.send_json({
                "type": "SIGN_TRANSLATION",
                "data": {
                    "label_bn": "হ্যালো",
                    "confidence": 0.95
                }
            })
            
            # Speaker should receive the broadcasted event
            speaker_recv = ws_speaker.receive_json()
            assert speaker_recv["type"] == "SIGN_TRANSLATION"
            assert speaker_recv["data"]["label_bn"] == "হ্যালো"
            # Audio synthesis adds a base64 payload
            assert "audio_payload_base64" in speaker_recv["data"]
            
            # Speaker sends speech-to-text event
            ws_speaker.send_json({
                "type": "SPEECH_TEXT",
                "data": {
                    "transcript": "Hello there",
                    "is_final": True
                }
            })
            
            # Signer should receive the text subtitle
            signer_recv = ws_signer.receive_json()
            assert signer_recv["type"] == "SPEECH_TEXT"
            assert signer_recv["data"]["transcript"] == "Hello there"
