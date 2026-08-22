"""Unit Test Suite for Real-Time CSLR ONNX Pipeline & WebRTC Track Processor (Sprint 36).

Tests:
1. `CSLRSlidingWindowBuffer` circular buffer appending and stride triggering.
2. `CSLROnnxEngine` CTC inference, confidence, and Bengali syntax translation.
3. `SignLanguageTrackProcessor` frame ingestion, inference invocation, and callback emission.
"""

import asyncio
import numpy as np
import pytest
from av import VideoFrame

from core_engine.inference.cslr_onnx_engine import CSLROnnxEngine, CSLRSlidingWindowBuffer
from backend.webrtc.track_processor import SignLanguageTrackProcessor


@pytest.mark.asyncio
async def test_cslr_sliding_window_buffer():
    """Verify circular buffer appends frames and triggers ready signal on stride."""
    buf = CSLRSlidingWindowBuffer(window_size=32, stride=8)
    assert len(buf.buffer) == 0

    triggered = []
    for i in range(40):
        dummy_frame = np.ones((75, 3), dtype=np.float32) * i
        is_ready = buf.append(dummy_frame)
        if is_ready:
            triggered.append(i)

    # With window_size=32 and stride=8: triggers at frame 31 (32nd frame) and 39 (40th frame)
    assert len(triggered) == 2
    window = buf.get_window()
    assert window.shape == (32, 75, 3)


@pytest.mark.asyncio
async def test_cslr_onnx_engine_prediction_and_translation():
    """Verify CSLROnnxEngine processes 32-frame window and translates gloss to fluent Bengali."""
    engine = CSLROnnxEngine(window_size=32, stride=8)

    # Active motion window
    active_window = np.ones((32, 75, 3), dtype=np.float32) * 0.5
    gloss, conf, latency_ms = await engine.predict_cslr_ctc(active_window)

    assert isinstance(gloss, str)
    assert conf >= 0.0
    assert latency_ms < 50.0  # Sub-50ms latency target

    translated = await engine.translate_gloss_to_text("আমি স্কুল যাওয়া")
    assert "স্কুলে যাচ্ছি" in translated


class MockMediaStreamTrack:
    kind = "video"

    def __init__(self):
        self.frame_idx = 0

    async def recv(self):
        self.frame_idx += 1
        arr = np.zeros((480, 640, 3), dtype=np.uint8)
        arr[200:280, 280:360] = 200  # Central brightness
        frame = VideoFrame.from_ndarray(arr, format="bgr24")
        return frame


class MockDataChannel:
    def __init__(self):
        self.readyState = "open"
        self.messages = []

    def send(self, msg):
        self.messages.append(msg)


@pytest.mark.asyncio
async def test_track_processor_integration():
    """Verify SignLanguageTrackProcessor ingests video frames and delivers predictions."""
    mock_track = MockMediaStreamTrack()
    mock_dc = MockDataChannel()
    engine = CSLROnnxEngine(window_size=32, stride=8)

    received_callbacks = []

    async def on_pred(payload):
        received_callbacks.append(payload)

    processor = SignLanguageTrackProcessor(
        track=mock_track,
        data_channel=mock_dc,
        engine=engine,
        on_prediction=on_pred,
        window_size=32,
        stride=8
    )

    # Feed 33 frames
    for _ in range(33):
        out_frame = await processor.recv()
        assert out_frame is not None

    # Trigger inference manually to verify pipeline execution
    payload = await processor.run_inference_if_ready()
    assert payload is not None
    assert "gloss" in payload
    assert "text" in payload
    assert len(mock_dc.messages) >= 1
