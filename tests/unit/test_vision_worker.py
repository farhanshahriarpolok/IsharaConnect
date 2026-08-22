"""Unit tests for the Async Vision Pipeline Worker and Bounded Worker Pool."""

import asyncio
import cv2
import numpy as np
import pytest

from backend.workers.vision_worker import (
    AsyncVisionWorker,
    AsyncVisionWorkerPool,
    BoundedClientQueue,
)


@pytest.mark.asyncio
async def test_async_vision_worker_numpy_frame():
    """Test async frame processing using a synthetic numpy image."""
    worker = AsyncVisionWorker(max_workers=2)
    synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    result = await worker.process_frame_async(synthetic_frame)
    assert isinstance(result, dict)
    assert "status" in result
    assert "processing_time_ms" in result
    assert result["processing_time_ms"] >= 0.0
    
    worker.close()


@pytest.mark.asyncio
async def test_async_vision_worker_encoded_bytes():
    """Test async frame processing using raw JPEG encoded bytes."""
    worker = AsyncVisionWorker(max_workers=2)
    synthetic_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", synthetic_frame)
    raw_bytes = encoded.tobytes()
    
    result = await worker.process_frame_async(raw_bytes)
    assert isinstance(result, dict)
    assert result["status"] in ["no_hands", "idle", "success"]
    assert worker.total_frames_processed >= 0
    
    worker.close()


def test_bounded_client_queue_stale_frame_dropping():
    """Test LIFO stale frame dropping policy in bounded queue."""
    queue = BoundedClientQueue(max_queue_size=2)
    
    # Submit 5 consecutive frames
    for i in range(5):
        frame = np.full((100, 100, 3), fill_value=i, dtype=np.uint8)
        queue.push_frame(frame)
        
    assert queue.submitted_frames_count == 5
    assert queue.dropped_frames_count == 3
    
    # Popping should yield the freshest frame (value 4)
    latest_frame = queue.pop_frame()
    assert latest_frame is not None
    assert latest_frame[0, 0, 0] == 4


@pytest.mark.asyncio
async def test_async_vision_worker_pool():
    """Test AsyncVisionWorkerPool frame submission and processing."""
    pool = AsyncVisionWorkerPool(num_workers=2, max_client_queue_size=2)
    client_id = "test_client_007"
    
    # Submit 3 frames (1 dropped)
    for i in range(3):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        pool.submit_frame(client_id, frame)
        
    stats = pool.get_stats()
    assert stats["total_dropped_frames"] == 1
    assert stats["active_client_queues"] == 1
    
    # Process latest frame
    res = await pool.process_client_latest_frame(client_id)
    assert res is not None
    assert "status" in res
    
    # Direct processing
    frame_direct = np.zeros((120, 160, 3), dtype=np.uint8)
    direct_res = await pool.process_direct_async(frame_direct)
    assert direct_res is not None
    
    pool.close()


@pytest.mark.asyncio
async def test_async_vision_worker_invalid_input():
    """Test error handling with invalid frame data."""
    worker = AsyncVisionWorker()
    invalid_data = 12345  # Not numpy array or bytes
    
    res = await worker.process_frame_async(invalid_data)
    assert res["status"] == "error"
    assert "Unsupported frame input" in res["error"]
    
    worker.close()
