"""Integration tests for FastAPI WebSocket Backend."""

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

def test_health_check():
    """Test health check REST APIs."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

    # Fast root health check for launcher
    root_health = client.get("/health")
    assert root_health.status_code == 200
    assert root_health.json()["status"] == "healthy"
    assert "IsharaConnect" in root_health.json()["service"]

def test_websocket_room_duplex():
    """Test duplex WebSocket communication between SIGNER and SPEAKER."""
    from unittest.mock import patch
    from backend.websockets.room_manager import manager

    room_id = "test_room_1"
    
    with patch.object(manager.tts_engine, "synthesize_to_bytes", return_value=b"RIFFdummywaveaudio"):
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
                assert "হ্যালো" in speaker_recv["data"]["label_bn"]
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
