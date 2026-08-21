"""Unit tests for the Transcript Exporter and Export Dialog."""

import json
import os
from pathlib import Path
import pytest
from PyQt6.QtWidgets import QApplication

from desktop_app.utils.transcript_exporter import (
    export_to_txt,
    export_to_json,
    export_to_pdf,
)
from desktop_app.ui.dialogs.export_dialog import ExportTranscriptDialog


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def sample_messages():
    return [
        {
            "timestamp": "2026-08-22 10:00:00",
            "sender": "Signer",
            "text": "ধন্যবাদ (Thank you)"
        },
        {
            "timestamp": "2026-08-22 10:00:05",
            "sender": "Speaker",
            "text": "You are welcome! How are you today?"
        },
        {
            "timestamp": "2026-08-22 10:00:15",
            "sender": "You (Signer)",
            "text": "আমি ভালো আছি (I am fine)"
        }
    ]


def test_export_to_txt(tmp_path, sample_messages):
    """Test exporting conversation to plain text."""
    out_file = tmp_path / "chat_transcript.txt"
    metadata = {"room_id": "test_room_101", "mode": "Signer"}
    
    result = export_to_txt(sample_messages, str(out_file), metadata=metadata)
    assert os.path.exists(result)
    
    content = out_file.read_text(encoding="utf-8")
    assert "ISHARACONNECT" in content
    assert "test_room_101" in content
    assert "ধন্যবাদ" in content
    assert "You are welcome" in content


def test_export_to_json(tmp_path, sample_messages):
    """Test exporting conversation to JSON."""
    out_file = tmp_path / "chat_transcript.json"
    metadata = {"room_id": "test_room_101", "mode": "Signer"}
    
    result = export_to_json(sample_messages, str(out_file), metadata=metadata)
    assert os.path.exists(result)
    
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["session"]["room_id"] == "test_room_101"
    assert len(data["messages"]) == 3
    assert data["messages"][0]["text"] == "ধন্যবাদ (Thank you)"


def test_export_to_pdf(tmp_path, sample_messages):
    """Test exporting conversation to styled PDF."""
    out_file = tmp_path / "chat_transcript.pdf"
    metadata = {"room_id": "test_room_101", "mode": "Signer"}
    
    result = export_to_pdf(sample_messages, str(out_file), metadata=metadata)
    assert os.path.exists(result)
    assert os.path.getsize(result) > 1000


def test_export_dialog_instantiation(qapp, sample_messages, tmp_path):
    """Test ExportTranscriptDialog UI functionality."""
    dialog = ExportTranscriptDialog(
        messages=sample_messages,
        room_id="room_alpha",
        mode="Signer"
    )
    assert dialog.windowTitle() == "Export Conversation Transcript"
    assert dialog.radio_pdf.isChecked()
    
    # Test changing format switches file extension
    dialog.radio_txt.setChecked(True)
    assert dialog._get_extension() == "txt"
    assert dialog.path_input.text().endswith(".txt")
    
    dialog.radio_json.setChecked(True)
    assert dialog._get_extension() == "json"
    assert dialog.path_input.text().endswith(".json")
    
    dialog.close()
