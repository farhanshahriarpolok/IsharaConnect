"""Unit tests for Tier 3: BdSL Continuous Sign Language Recognition (CSLR) Pipeline."""

import numpy as np
import pytest
from pathlib import Path

from dataset.ingestors.tier3_cslr_ingestor import Tier3CSLRIngestor


def test_tier3_manifest_and_vocab():
    """Tier 3 ingestor initializes with non-empty CTC gloss vocabulary."""
    ingestor = Tier3CSLRIngestor()
    assert ingestor.manifest_data is not None
    assert "<blank>" in ingestor.gloss_vocab
    assert ingestor.gloss_vocab["<blank>"] == 0
    assert len(ingestor.gloss_vocab) > 2


def test_tier3_ctc_encoding_and_decoding():
    """Encodes gloss sequence to CTC tokens and decodes back cleanly."""
    ingestor = Tier3CSLRIngestor()
    glosses = ["AMI", "DAKTAR", "SAHAJJO"]
    encoded = ingestor.encode_ctc_targets(glosses)
    assert len(encoded) == 3
    assert all(isinstance(x, int) for x in encoded)

    # CTC collapse decoding (with blanks and repeats)
    raw_ctc_stream = [0, encoded[0], encoded[0], 0, encoded[1], 0, encoded[2], encoded[2]]
    decoded = ingestor.decode_ctc_targets(raw_ctc_stream)
    assert decoded == ["AMI", "DAKTAR", "SAHAJJO"]


def test_tier3_generate_mock_and_validate(tmp_path):
    """Generates continuous signing landmark streams with CTC ground truths."""
    ingestor = Tier3CSLRIngestor()
    out_dir = tmp_path / "tier3_test"
    seqs, targets, save_path = ingestor.generate_mock_dataset(num_clips=5, output_dir=str(out_dir))

    assert len(seqs) == 5
    assert len(targets) == 5
    assert seqs[0].shape[1] == 151

    val = ingestor.validate(save_path)
    assert val["valid"] is True
    assert val["num_clips"] == 5

    stats = ingestor.get_statistics(save_path)
    assert stats["clips_count"] == 5
    assert stats["feature_dimension"] == 151
