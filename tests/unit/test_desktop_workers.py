"""Unit tests for Desktop Client workers."""

import json
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import QThread

from desktop_app.controllers.camera_worker import CameraWorker
from desktop_app.controllers.network_worker import NetworkWorker


def test_network_worker_initialization():
    """Test NetworkWorker instantiates correctly without immediately connecting."""
    worker = NetworkWorker("ws://localhost:8000", "test_room", "signer")
    assert worker.server_url == "ws://localhost:8000"
    assert worker.room_id == "test_room"
    assert worker.client_type == "signer"
    assert worker._is_running is True


def test_network_worker_send_queuing():
    """Test that send_sign_event safely enqueues the payload."""
    worker = NetworkWorker("ws://localhost:8000", "test_room", "signer")
    
    sign_payload = {"label_bn": "হ্যালো", "confidence": 0.9}
    
    # We call the thread-safe method
    worker.send_sign_event(sign_payload)
    
    # Check the queue in the worker's asyncio loop
    assert worker._send_queue.qsize() == 1
    
    queued_item = worker._send_queue.get_nowait()
    assert queued_item["type"] == "SIGN_TRANSLATION"
    assert queued_item["data"] == sign_payload


@patch("desktop_app.controllers.camera_worker.cv2.VideoCapture")
def test_camera_worker_initialization(mock_video_capture):
    """Test CameraWorker initialization logic."""
    # Mocking capture so we don't open real webcam
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False # Force dummy loop fallback
    mock_video_capture.return_value = mock_cap
    
    worker = CameraWorker()
    assert worker.camera_id == 0
    assert worker._is_running is True
    
    # We won't call start() as it runs QThread which blocks or requires event loop setup in pytest
    # But we can verify attributes
    assert worker.detector is None
