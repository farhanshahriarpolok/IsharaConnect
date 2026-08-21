"""Unit tests for Video-to-Landmark Ingestion Pipeline."""

import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest
import numpy as np

from dataset.tools.video_dataset_converter import (
    resample_temporal_sequence,
    extract_151d_vector,
    process_video,
    convert_video_dataset
)


def test_resample_temporal_sequence_shapes():
    """Verify temporal resampling to exactly target_length."""
    # 1. Upsampling from 12 frames to 30 frames
    short_seq = np.random.randn(12, 151).astype(np.float32)
    resampled = resample_temporal_sequence(short_seq, target_length=30)
    assert resampled.shape == (30, 151)

    # 2. Downsampling from 75 frames to 30 frames
    long_seq = np.random.randn(75, 151).astype(np.float32)
    resampled = resample_temporal_sequence(long_seq, target_length=30)
    assert resampled.shape == (30, 151)

    # 3. Exact match length unchanged
    exact_seq = np.random.randn(30, 151).astype(np.float32)
    resampled = resample_temporal_sequence(exact_seq, target_length=30)
    assert resampled.shape == (30, 151)
    np.testing.assert_array_almost_equal(resampled, exact_seq)


def test_resample_edge_cases():
    """Verify resampling on single frame and empty sequences."""
    # Single frame tiled to 30 frames
    single = np.ones((1, 151), dtype=np.float32) * 3.5
    res = resample_temporal_sequence(single, target_length=30)
    assert res.shape == (30, 151)
    assert np.all(res == 3.5)

    # Empty array returns zero matrix of target length
    empty = np.array([], dtype=np.float32)
    res_empty = resample_temporal_sequence(empty, target_length=30)
    assert res_empty.shape == (30, 151)
    assert np.all(res_empty == 0.0)


def test_extract_151d_vector():
    """Verify extraction of 151-D spatial vector from features dictionary."""
    mock_features = {
        "normalized_landmarks": np.random.randn(42, 3).astype(np.float32),
        "touch_matrix": np.full((5, 5), float("inf"), dtype=np.float32)
    }

    vec = extract_151d_vector(mock_features)
    assert vec.shape == (151,)
    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()


@patch("cv2.VideoCapture")
def test_process_video_mock(mock_video_capture):
    """Verify process_video with mocked VideoCapture frames."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    # 10 frames of dummy images
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap.read.side_effect = [(True, dummy_frame)] * 10 + [(False, None)]
    mock_video_capture.return_value = mock_cap

    mock_engine = MagicMock()
    mock_engine.extract_spatial_features.return_value = {
        "normalized_landmarks": np.zeros((42, 3), dtype=np.float32),
        "touch_matrix": np.zeros((5, 5), dtype=np.float32)
    }

    seq = process_video("test_sign.mp4", engine=mock_engine, target_frames=30)
    assert seq is not None
    assert seq.shape == (30, 151)
    assert mock_engine.extract_spatial_features.call_count == 10


def test_convert_video_dataset_pipeline():
    """Verify batch conversion directory scanning and output saving."""
    with tempfile.TemporaryDirectory() as tmp_video_dir, \
         tempfile.TemporaryDirectory() as tmp_out_dir:

        # Create dummy class folders and files
        sign_dir = os.path.join(tmp_video_dir, "dhonnobad")
        os.makedirs(sign_dir, exist_ok=True)
        dummy_vid = os.path.join(sign_dir, "sample_01.mp4")
        with open(dummy_vid, "wb") as f:
            f.write(b"dummy video bytes")

        with patch("dataset.tools.video_dataset_converter.process_video") as mock_proc:
            mock_proc.return_value = np.zeros((30, 151), dtype=np.float32)

            summary = convert_video_dataset(tmp_video_dir, tmp_out_dir, target_frames=30)
            assert summary["processed"] == 1
            assert summary["skipped"] == 0
            assert len(summary["output_files"]) == 1

            saved_file = summary["output_files"][0]
            assert os.path.exists(saved_file)
            loaded_data = np.load(saved_file)
            assert loaded_data.shape == (30, 151)
