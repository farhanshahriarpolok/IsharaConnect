import pytest
from PyQt6.QtWidgets import QApplication
from desktop_app.controllers.network_worker import NetworkWorker
import sys

# Basic application instance for tests
@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app

def test_network_worker_resilience(qapp):
    """Verify that calling send_* methods when loop is None or stopped doesn't crash."""
    worker = NetworkWorker(server_url="ws://localhost:8000", room_id="test", client_type="signer")
    
    # 1. Before loop is started (is_running() is False)
    # Shouldn't raise any unhandled exceptions
    try:
        worker.send_sign_event({"label": "test"})
        worker.send_speech_event("test transcript")
    except Exception as e:
        pytest.fail(f"send_* methods raised exception before loop start: {e}")
        
    # 2. Simulate loop being None
    worker._loop = None
    try:
        worker.send_sign_event({"label": "test"})
        worker.send_speech_event("test transcript")
    except Exception as e:
        pytest.fail(f"send_* methods raised exception when loop is None: {e}")
