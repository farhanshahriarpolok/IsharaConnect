"""Unit tests for NetworkWorker resilient WebSocket duplex and offline fallback."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from PyQt6.QtCore import QCoreApplication

from desktop_app.controllers.network_worker import NetworkWorker


@pytest.fixture(scope="module")
def qapp():
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    yield app


def test_network_worker_default_instantiation(qapp):
    """Test NetworkWorker can be instantiated with default parameters."""
    worker = NetworkWorker()
    assert worker.server_url == "ws://127.0.0.1:8000"
    assert worker.room_id == "default_room"
    assert worker.client_type == "signer"
    assert worker.max_retries == 3
    assert worker.retry_count == 0
    assert worker.is_offline is False


def test_network_worker_queue_bounding(qapp):
    """Test that send queue safely caps size without unbounded memory accumulation."""
    worker = NetworkWorker()
    for i in range(100):
        worker.send_sign_event({"index": i, "label_bn": "টেস্ট"})

    assert worker._send_queue.qsize() <= 50


def test_network_worker_speech_event_queuing(qapp):
    """Test that speech event is properly formatted and enqueued."""
    worker = NetworkWorker()
    worker.send_speech_event("আমি ভালো আছি")

    assert worker._send_queue.qsize() == 1
    item = worker._send_queue.get_nowait()
    assert item["type"] == "SPEECH_TEXT"
    assert item["data"]["transcript"] == "আমি ভালো আছি"
    assert item["data"]["is_final"] is True


@pytest.mark.asyncio
async def test_network_worker_offline_fallback(qapp):
    """Test that failed connection attempts transition to Offline / Standalone Mode."""
    worker = NetworkWorker(server_url="ws://127.0.0.1:59999", room_id="test_offline")
    worker.max_retries = 3

    emitted_statuses = []
    emitted_network_states = []

    worker.connection_status.connect(lambda ok, msg: emitted_statuses.append((ok, msg)))
    worker.network_status_changed.connect(lambda msg: emitted_network_states.append(msg))

    real_sleep = asyncio.sleep

    async def fast_sleep(d):
        await real_sleep(0.001)

    with patch("websockets.connect", side_effect=ConnectionRefusedError("WinError 1225")), \
         patch("desktop_app.controllers.network_worker.asyncio.sleep", side_effect=fast_sleep):

        task = asyncio.create_task(worker._websocket_loop())

        for _ in range(30):
            await real_sleep(0.005)
            if worker.is_offline:
                break

        worker._is_running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert worker.is_offline is True
    assert any("Offline / Standalone Mode" in s for s in emitted_network_states)
    assert any("Offline Mode" in s[1] for s in emitted_statuses)



def test_network_worker_trigger_reconnect(qapp):
    """Test trigger_reconnect resets retry count and offline state."""
    worker = NetworkWorker()
    worker.is_offline = True
    worker.retry_count = 3

    worker.trigger_reconnect()
    assert worker.is_offline is False
    assert worker.retry_count == 0
