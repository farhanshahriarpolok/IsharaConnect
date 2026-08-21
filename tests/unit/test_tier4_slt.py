"""Unit tests for Tier 4: BdSL Sign Language Translation (SLT) Pipeline."""

import numpy as np
import pytest
from pathlib import Path

from dataset.ingestors.tier4_slt_ingestor import Tier4SLTIngestor


def test_tier4_vocabularies():
    """Tier 4 ingestor builds valid source and target vocabularies."""
    ingestor = Tier4SLTIngestor()
    assert len(ingestor.gloss_to_id) > 4
    assert len(ingestor.text_to_id) > 4
    assert "<pad>" in ingestor.gloss_to_id
    assert "<sos>" in ingestor.text_to_id


def test_tier4_tokenization_sequence():
    """Tokenizes glosses and Bengali text into bounded, padded matrices."""
    ingestor = Tier4SLTIngestor()
    gloss_tokens = ["AMI", "DAKTAR", "SAHAJJO"]
    enc = ingestor.tokenize_sequence(gloss_tokens, is_source=True, max_len=16)

    assert len(enc) == 16
    assert enc[0] == ingestor.gloss_to_id["<sos>"]
    # Check that sequence ends before max_len
    assert enc[-1] == 0  # padded with <pad> = 0


def test_tier4_generate_mock_and_validate(tmp_path):
    """Generates parallel translation corpus matrices and validates schemas."""
    ingestor = Tier4SLTIngestor()
    out_dir = tmp_path / "tier4_test"
    src, tgt, save_path = ingestor.generate_mock_corpus(num_pairs=15, output_dir=str(out_dir))

    assert src.shape == (15, 32)
    assert tgt.shape == (15, 32)

    val = ingestor.validate(save_path)
    assert val["valid"] is True
    assert val["num_pairs"] == 15
    assert val["max_sequence_len"] == 32

    stats = ingestor.get_statistics(save_path)
    assert stats["parallel_pairs_count"] == 15
