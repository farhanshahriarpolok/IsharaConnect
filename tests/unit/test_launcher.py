"""Unit tests for Unified Launcher helper utilities."""

from unittest.mock import MagicMock, patch
import pytest
from launch import is_port_in_use, check_server_health, free_port, stream_logs


def test_is_port_in_use_closed():
    """Verify is_port_in_use returns False for an unassigned ephemeral port."""
    assert is_port_in_use(port=59999) is False


def test_check_server_health_success():
    """Verify check_server_health succeeds when endpoint returns 200."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        result = check_server_health(["http://127.0.0.1:8000/health"], retries=2, delay=0.01)
        assert result is True


def test_check_server_health_process_crash():
    """Verify check_server_health immediately fails if subprocess exits."""
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 1  # Process crashed with exit code 1

    result = check_server_health(["http://127.0.0.1:8000/health"], process=mock_proc, retries=5, delay=0.01)
    assert result is False


def test_free_port_no_op_when_not_in_use():
    """free_port should be a no-op if port is already free."""
    with patch("launch.is_port_in_use", return_value=False), \
         patch("subprocess.run") as mock_run:
        free_port(8000)
        mock_run.assert_not_called()
