import pytest
import numpy as np
import asyncio
from core_engine.inference.cslr_engine import SlidingWindowBuffer, IsharaInferenceEngine

def test_sliding_window_buffer():
    buffer = SlidingWindowBuffer(window_size=4, stride=2)
    
    # 1. Fill buffer up to window_size - 1
    assert not buffer.append(np.array([1, 2]))
    assert not buffer.append(np.array([3, 4]))
    assert not buffer.append(np.array([5, 6]))
    
    # 2. Window is full (counter = 4), should trigger because 4 % 2 == 0
    assert buffer.append(np.array([7, 8]))
    
    # Check buffer content
    window = buffer.get_window()
    assert window.shape == (4, 2)
    assert np.array_equal(window[0], [1, 2])
    
    # 3. Add one more item (counter = 5), shouldn't trigger because 5 % 2 != 0
    assert not buffer.append(np.array([9, 10]))
    
    # 4. Add another (counter = 6), should trigger because 6 % 2 == 0
    assert buffer.append(np.array([11, 12]))
    
    # Buffer should drop oldest elements
    window = buffer.get_window()
    assert window.shape == (4, 2)
    assert np.array_equal(window[0], [5, 6])
    assert np.array_equal(window[-1], [11, 12])

@pytest.mark.asyncio
async def test_ishara_inference_engine():
    engine = IsharaInferenceEngine()
    
    # Test CTC mock
    dummy_data = np.random.randn(32, 75, 3)
    gloss = await engine.predict_cslr_ctc(dummy_data)
    assert gloss == "আমি স্কুল যাওয়া"
    
    # Test Gloss to Text translation
    translated = await engine.translate_gloss_to_text("আমি স্কুল যাওয়া")
    assert translated == "আমি স্কুলে যাচ্ছি।"
    
    # Test unknown gloss translation fallback
    fallback = await engine.translate_gloss_to_text("অজানা গ্লস")
    assert fallback == "অজানা গ্লস"
