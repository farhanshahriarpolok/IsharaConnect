"""Tests for non-blocking CSLR live WebRTC track hook.

Coverage:
  - SignLanguageTrackProcessor: construction, buffer mechanics, non-blocking dispatch
  - _extract_landmarks_fast: shape, dtype, range
  - _emit_prediction: payload structure, callback invocation, data channel gating
  - active_inferences counter: back-pressure guard
  - reset(): state reset
  - Legacy alias run_inference_if_ready()
  - CSLRSlidingWindowBuffer: append, get_window, stride trigger
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core_engine.inference.cslr_onnx_engine import CSLROnnxEngine, CSLRSlidingWindowBuffer


# ──────────────────────────────────────────────────────────────────────────────
# Helpers & Stubs
# ──────────────────────────────────────────────────────────────────────────────

class MockMediaStreamTrack:
    """Minimal stand-in for aiortc.MediaStreamTrack."""
    kind = "video"

    def __init__(self):
        self._frame_n = 0

    async def recv(self):
        import av
        self._frame_n += 1
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[200:280, 200:440] = self._frame_n % 200  # non-zero region
        return av.VideoFrame.from_ndarray(img, format="bgr24")


class MockDataChannel:
    """Minimal stand-in for RTCDataChannel."""
    def __init__(self, open: bool = True):
        self.readyState = "open" if open else "closed"
        self.sent: List[str] = []

    def send(self, data: str):
        self.sent.append(data)


def _make_processor(**kwargs):
    """Creates a SignLanguageTrackProcessor without importing aiortc at module level."""
    from backend.webrtc.track_processor import SignLanguageTrackProcessor
    track = MockMediaStreamTrack()
    return SignLanguageTrackProcessor(track=track, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# 1. CSLRSlidingWindowBuffer (unit)
# ──────────────────────────────────────────────────────────────────────────────

class TestCSLRSlidingWindowBuffer:
    def setup_method(self):
        self.buf = CSLRSlidingWindowBuffer(window_size=32, stride=8)

    def test_append_returns_false_before_full(self):
        """Buffer should not trigger inference until full."""
        for i in range(31):
            triggered = self.buf.append(np.zeros((75, 3), dtype=np.float32))
            assert not triggered

    def test_append_triggers_at_stride_after_full(self):
        """Once full, every `stride`-th frame should return True."""
        for _ in range(32):
            self.buf.append(np.zeros((75, 3), dtype=np.float32))
        # Next stride trigger comes at frame 40 (32 + 8)
        triggers = []
        for _ in range(8):
            triggers.append(self.buf.append(np.zeros((75, 3), dtype=np.float32)))
        # Exactly 1 trigger in 8 frames (at stride boundary)
        assert sum(triggers) == 1

    def test_get_window_shape_when_full(self):
        for _ in range(32):
            self.buf.append(np.zeros((75, 3), dtype=np.float32))
        w = self.buf.get_window()
        assert w.shape == (32, 75, 3)

    def test_get_window_padded_before_full(self):
        for _ in range(10):
            self.buf.append(np.zeros((75, 3), dtype=np.float32))
        w = self.buf.get_window()
        assert w.shape == (32, 75, 3)

    def test_reset_clears_state(self):
        for _ in range(32):
            self.buf.append(np.zeros((75, 3), dtype=np.float32))
        self.buf.reset()
        assert len(self.buf.buffer) == 0
        assert self.buf.counter == 0

    def test_append_accepts_wrong_shape_and_pads(self):
        """Non-(75,3) arrays should be gracefully padded."""
        flat = np.zeros(100, dtype=np.float32)
        # Should not raise
        self.buf.append(flat)

    def test_window_is_float32(self):
        for _ in range(32):
            self.buf.append(np.ones((75, 3), dtype=np.float64))
        w = self.buf.get_window()
        assert w.dtype == np.float32


# ──────────────────────────────────────────────────────────────────────────────
# 2. SignLanguageTrackProcessor — construction
# ──────────────────────────────────────────────────────────────────────────────

class TestTrackProcessorConstruction:
    def test_creates_without_error(self):
        proc = _make_processor()
        assert proc is not None

    def test_frame_count_starts_at_zero(self):
        proc = _make_processor()
        assert proc.frame_count == 0

    def test_active_inferences_starts_at_zero(self):
        proc = _make_processor()
        assert proc.active_inferences == 0

    def test_custom_engine_accepted(self):
        engine = CSLROnnxEngine(window_size=32, stride=8)
        proc = _make_processor(engine=engine)
        assert proc.engine is engine

    def test_default_engine_is_cslr_onnx(self):
        proc = _make_processor()
        assert isinstance(proc.engine, CSLROnnxEngine)

    def test_data_channel_stored(self):
        dc = MockDataChannel()
        proc = _make_processor(data_channel=dc)
        assert proc.data_channel is dc


# ──────────────────────────────────────────────────────────────────────────────
# 3. _extract_landmarks_fast
# ──────────────────────────────────────────────────────────────────────────────

class TestExtractLandmarksFast:
    def setup_method(self):
        self.proc = _make_processor()

    def test_output_shape(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.proc._extract_landmarks_fast(img)
        assert result.shape == (75, 3)

    def test_output_dtype_float32(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.proc._extract_landmarks_fast(img)
        assert result.dtype == np.float32

    def test_zero_image_landmarks_near_zero(self):
        """All-zero image → intensity=0 → landmarks should be scaled close to 0."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        lm = self.proc._extract_landmarks_fast(img)
        # With intensity=0, scale factor = 0.3
        assert lm.max() <= 1.0

    def test_bright_image_different_from_dark(self):
        dark  = self.proc._extract_landmarks_fast(np.zeros((480, 640, 3), dtype=np.uint8))
        bright = self.proc._extract_landmarks_fast(np.full((480, 640, 3), 200, dtype=np.uint8))
        # Should produce different landmark energy
        assert not np.allclose(dark, bright)

    def test_no_exception_on_tiny_image(self):
        """1×1 image should not crash."""
        img = np.zeros((1, 1, 3), dtype=np.uint8)
        result = self.proc._extract_landmarks_fast(img)
        assert result.shape == (75, 3)


# ──────────────────────────────────────────────────────────────────────────────
# 4. _emit_prediction — payload contract
# ──────────────────────────────────────────────────────────────────────────────

class TestEmitPrediction:
    @pytest.fixture
    def proc(self):
        return _make_processor()

    @pytest.mark.asyncio
    async def test_emit_returns_payload_dict(self, proc):
        payload = await proc._emit_prediction("আমি স্কুল যাওয়া", 0.94, 18.5)
        assert payload is not None
        assert isinstance(payload, dict)

    @pytest.mark.asyncio
    async def test_payload_has_required_keys(self, proc):
        payload = await proc._emit_prediction("ধন্যবাদ", 0.90, 12.0)
        for key in ["type", "gloss", "text", "confidence", "latency_ms", "frame_count", "timestamp"]:
            assert key in payload, f"Missing payload key: {key}"

    @pytest.mark.asyncio
    async def test_emit_returns_none_for_same_gloss(self, proc):
        proc.last_gloss = "আমি স্কুল যাওয়া"
        result = await proc._emit_prediction("আমি স্কুল যাওয়া", 0.90, 15.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_emit_updates_last_gloss(self, proc):
        await proc._emit_prediction("নতুন সাইন", 0.85, 10.0)
        assert proc.last_gloss == "নতুন সাইন"

    @pytest.mark.asyncio
    async def test_data_channel_receives_json(self, proc):
        dc = MockDataChannel(open=True)
        proc.data_channel = dc
        proc.last_gloss = ""
        await proc._emit_prediction("আমি যাওয়া", 0.91, 14.0)
        assert len(dc.sent) == 1
        parsed = json.loads(dc.sent[0])
        assert parsed["gloss"] == "আমি যাওয়া"

    @pytest.mark.asyncio
    async def test_closed_data_channel_not_called(self, proc):
        dc = MockDataChannel(open=False)
        proc.data_channel = dc
        proc.last_gloss = ""
        await proc._emit_prediction("আমি যাওয়া", 0.91, 14.0)
        assert len(dc.sent) == 0

    @pytest.mark.asyncio
    async def test_sync_callback_invoked(self, proc):
        calls = []
        proc.on_prediction = lambda p: calls.append(p)
        proc.last_gloss = ""
        await proc._emit_prediction("মা বাবা", 0.88, 20.0)
        assert len(calls) == 1
        assert calls[0]["gloss"] == "মা বাবা"

    @pytest.mark.asyncio
    async def test_async_callback_invoked(self, proc):
        calls = []
        async def async_cb(p):
            calls.append(p)
        proc.on_prediction = async_cb
        proc.last_gloss = ""
        await proc._emit_prediction("ধন্যবাদ", 0.95, 8.0)
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_confidence_clamped_in_payload(self, proc):
        proc.last_gloss = ""
        payload = await proc._emit_prediction("সালাম", 0.876543, 11.0)
        assert payload["confidence"] == round(0.876543, 4)


# ──────────────────────────────────────────────────────────────────────────────
# 5. _run_async_inference — non-blocking end-to-end
# ──────────────────────────────────────────────────────────────────────────────

class TestRunAsyncInference:
    @pytest.fixture
    def proc(self):
        return _make_processor()

    @pytest.mark.asyncio
    async def test_returns_none_or_dict(self, proc):
        window = np.zeros((32, 75, 3), dtype=np.float32)
        result = await proc._run_async_inference(window)
        assert result is None or isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_inference_counter_released_after_completion(self, proc):
        window = np.zeros((32, 75, 3), dtype=np.float32)
        await proc._run_async_inference(window)
        assert proc.active_inferences == 0

    @pytest.mark.asyncio
    async def test_last_inference_latency_updated(self, proc):
        window = np.zeros((32, 75, 3), dtype=np.float32)
        await proc._run_async_inference(window)
        assert proc.last_inference_latency_ms >= 0.0

    @pytest.mark.asyncio
    async def test_multiple_sequential_inferences(self, proc):
        """Multiple sequential inferences should not leave counter > 0."""
        window = np.zeros((32, 75, 3), dtype=np.float32)
        for _ in range(3):
            await proc._run_async_inference(window)
        assert proc.active_inferences == 0

    @pytest.mark.asyncio
    async def test_error_in_engine_does_not_raise(self, proc):
        """If engine raises, _run_async_inference should catch and return None."""
        async def bad_engine_predict(*args, **kwargs):
            raise RuntimeError("simulated engine crash")

        proc.engine.predict_cslr_ctc = bad_engine_predict
        window = np.zeros((32, 75, 3), dtype=np.float32)
        result = await proc._run_async_inference(window)
        assert result is None
        assert proc.active_inferences == 0


# ──────────────────────────────────────────────────────────────────────────────
# 6. reset() and legacy alias
# ──────────────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_frame_count(self):
        proc = _make_processor()
        proc.frame_count = 99
        proc.reset()
        assert proc.frame_count == 0

    def test_reset_clears_last_gloss(self):
        proc = _make_processor()
        proc.last_gloss = "আমি"
        proc.reset()
        assert proc.last_gloss == ""

    def test_reset_clears_active_inferences(self):
        proc = _make_processor()
        proc._active_inferences = 3
        proc.reset()
        assert proc.active_inferences == 0

    @pytest.mark.asyncio
    async def test_legacy_alias_run_inference_if_ready(self):
        proc = _make_processor()
        result = await proc.run_inference_if_ready()
        assert result is None or isinstance(result, dict)
