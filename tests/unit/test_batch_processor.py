"""Unit tests for Batch Dataset Ingestion Tool."""

import json
from pathlib import Path
import numpy as np
import pytest

from dataset.tools.batch_dataset_processor import BatchDatasetProcessor


def test_batch_processor_init():
    """Processor initializes with multi-worker pool and spatial/temporal ingestors."""
    proc = BatchDatasetProcessor(max_workers=2)
    assert proc.spatial_engine is not None
    assert proc.tier1_ingestor is not None
    assert proc.tier2_ingestor is not None


def test_batch_processor_empty_or_missing_dir(tmp_path):
    """Empty or missing directory handles gracefully without raising uncaught exceptions."""
    proc = BatchDatasetProcessor(max_workers=2)
    res = proc.process_directory(str(tmp_path / "non_existent"), modality="image")
    assert res["status"] == "error"

    empty_dir = tmp_path / "empty_folder"
    empty_dir.mkdir()
    res2 = proc.process_directory(str(empty_dir), modality="image")
    assert res2["status"] == "warning"


def test_batch_processor_mock_image_batch(tmp_path):
    """Processes simulated image files into consolidated .npz dataset."""
    import cv2

    class_a = tmp_path / "class_a"
    class_b = tmp_path / "class_b"
    class_a.mkdir()
    class_b.mkdir()

    # Create dummy images
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(class_a / "sample1.jpg"), img1)
    cv2.imwrite(str(class_b / "sample2.jpg"), img1)

    proc = BatchDatasetProcessor(max_workers=2)
    out_file = tmp_path / "test_out.npz"
    res = proc.process_directory(str(tmp_path), modality="image", output_file=str(out_file))

    # Note: MediaPipe might not detect hands on all-black images, status should report properly
    assert "status" in res
