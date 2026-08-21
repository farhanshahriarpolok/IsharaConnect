import pytest
import asyncio
import numpy as np
from unittest.mock import MagicMock, AsyncMock

from backend.webrtc.track_processor import SignLanguageTrackProcessor
from core_engine.inference.cslr_engine import IsharaInferenceEngine

@pytest.mark.asyncio
async def test_sign_language_track_processor():
    # Mock track
    mock_track = AsyncMock()
    
    # Mock video frame
    mock_frame = MagicMock()
    mock_frame.to_ndarray.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
    mock_track.recv.return_value = mock_frame
    
    # Mock data channel
    mock_data_channel = MagicMock()
    mock_data_channel.readyState = "open"
    
    # Mock engine
    engine = IsharaInferenceEngine()
    engine.predict_cslr_ctc = AsyncMock(return_value="আমি স্কুল যাওয়া")
    engine.translate_gloss_to_text = AsyncMock(return_value="আমি স্কুলে যাচ্ছি।")
    
    # Init processor
    processor = SignLanguageTrackProcessor(track=mock_track, data_channel=mock_data_channel, engine=engine)
    
    # Push 31 frames (window size is 32, stride is 8)
    for i in range(31):
        frame = await processor.recv()
        assert frame == mock_frame
        assert engine.predict_cslr_ctc.call_count == 0
        
    # Push 32nd frame (should trigger inference)
    frame = await processor.recv()
    assert frame == mock_frame
    
    # Allow asyncio tasks to run
    await asyncio.sleep(0.05)
    
    # Verify inference was called
    assert engine.predict_cslr_ctc.call_count == 1
    assert engine.translate_gloss_to_text.call_count == 1
    
    # Verify data channel send
    assert mock_data_channel.send.call_count == 1
    payload_str = mock_data_channel.send.call_args[0][0]
    import json
    payload = json.loads(payload_str)
    assert payload["gloss"] == "আমি স্কুল যাওয়া"
    assert payload["text"] == "আমি স্কুলে যাচ্ছি।"
