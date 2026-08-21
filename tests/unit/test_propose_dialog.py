import pytest
import json
from unittest.mock import patch, MagicMock

from desktop_app.utils.url_helpers import get_http_url, get_ws_url
from desktop_app.ui.propose_sign_dialog import ProposeSignDialog
from PyQt6.QtWidgets import QApplication
import sys

@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app

def test_url_helpers():
    # Test HTTP conversions
    assert get_http_url("ws://localhost:8000") == "http://localhost:8000"
    assert get_http_url("wss://api.example.com", "/test") == "https://api.example.com/test"
    assert get_http_url("http://localhost:8000", "test") == "http://localhost:8000/test"
    
    # Test WS conversions
    assert get_ws_url("http://localhost:8000") == "ws://localhost:8000"
    assert get_ws_url("https://api.example.com", "/ws") == "wss://api.example.com/ws"
    assert get_ws_url("ws://localhost:8000", "ws") == "ws://localhost:8000/ws"

@patch('urllib.request.urlopen')
def test_propose_dialog_http_submit(mock_urlopen, qapp):
    # Mock successful response
    mock_response = MagicMock()
    mock_response.status = 200
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    # Initialize dialog with a websocket url to test the conversion
    dialog = ProposeSignDialog(server_url="ws://127.0.0.1:8000")
    
    # Bypass camera/recording setup for the unit test
    dialog.bn_input.setText("test_bn")
    dialog.en_input.setText("test_en")
    dialog.user_input.setText("test_user")
    dialog.recorded_samples = [[[0.0]*3]*42] * 5
    
    # Mock accept and QMessageBox
    with patch.object(dialog, 'accept') as mock_accept, \
         patch('desktop_app.ui.propose_sign_dialog.QMessageBox.information') as mock_info:
        
        # Manually invoke the submit slot
        dialog._submit_proposal()
        
        # Verify the correct Request was created and sent
        assert mock_urlopen.call_count == 1
        req = mock_urlopen.call_args[0][0]
        
        # Check HTTP scheme
        assert req.full_url == "http://127.0.0.1:8000/api/v1/signs/propose"
        
        # Check headers and method
        assert req.headers['Content-type'] == 'application/json'
        
        # Check payload
        payload = json.loads(req.data.decode('utf-8'))
        assert payload["user_id"] == "test_user"
        assert payload["bangla"] == "test_bn"
        assert payload["english"] == "test_en"
        assert payload["category"] == "proposed"
        
        mock_accept.assert_called_once()
