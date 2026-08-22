"""Tests for Ban-Sign-Sent-9K ingestion pipeline and processor.

Coverage:
  - BanSignSentIngestor: manifest generation, .npy keypoints, vocab file, statistics
  - BanSignSentProcessor: normalizer, resample, group extraction, vocabulary coverage
  - MasterBdSLLexicon integration via expand_with_bornildb_vocab
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.harvest_bansignsent_corpus import (
    BanSignSentIngestor,
    BANSIGNSENT_METADATA,
    BANSIGNSENT_SENTENCES,
)
from core_engine.dataset.bansignsent_processor import (
    BanSignSentProcessor,
    BanSignSentNormalizer,
    LANDMARK_GROUPS,
    SENTENCE_PATTERNS,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("bansignsent_processed")


@pytest.fixture(scope="module")
def ingestor(tmp_dir):
    return BanSignSentIngestor(
        output_dir=tmp_dir,
        raw_dir=tmp_dir / "raw",
        samples=30,
    )


@pytest.fixture(scope="module")
def ingested_stats(ingestor):
    return ingestor.ingest()


@pytest.fixture(scope="module")
def processor(tmp_dir, ingested_stats):
    return BanSignSentProcessor(processed_dir=tmp_dir)


# ──────────────────────────────────────────────────────────────────────────────
# 1. BanSignSentIngestor — basic contract
# ──────────────────────────────────────────────────────────────────────────────

class TestBanSignSentIngestor:
    def test_returns_stats_dict(self, ingested_stats):
        assert isinstance(ingested_stats, dict)
        assert "total_samples" in ingested_stats
        assert "split_counts" in ingested_stats

    def test_sample_count(self, ingested_stats):
        assert ingested_stats["total_samples"] == 30

    def test_split_manifests_exist(self, tmp_dir):
        for split in ["train", "val", "test"]:
            assert (tmp_dir / f"manifest_{split}.json").exists()

    def test_manifest_has_metadata_key(self, tmp_dir):
        with open(tmp_dir / "manifest_train.json", encoding="utf-8") as f:
            data = json.load(f)
        assert "metadata" in data and "samples" in data

    def test_sample_required_fields(self, tmp_dir):
        with open(tmp_dir / "manifest_train.json", encoding="utf-8") as f:
            data = json.load(f)
        sample = data["samples"][0]
        for field in ["sample_id", "split", "gloss_sequence", "keypoints_path",
                      "duration_frames", "sentence_text", "sentence_gloss"]:
            assert field in sample, f"Missing field: {field}"

    def test_keypoints_npy_dir_exists(self, tmp_dir):
        kp_dir = tmp_dir / "keypoints"
        assert kp_dir.exists()
        assert len(list(kp_dir.glob("*.npy"))) > 0

    def test_vocabulary_json_written(self, tmp_dir):
        assert (tmp_dir / "bansignsent_vocabulary.json").exists()

    def test_vocabulary_are_strings(self, tmp_dir):
        with open(tmp_dir / "bansignsent_vocabulary.json", encoding="utf-8") as f:
            vocab = json.load(f)
        assert all(isinstance(t, str) and t for t in vocab)

    def test_statistics_json_written(self, tmp_dir):
        assert (tmp_dir / "dataset_statistics.json").exists()

    def test_avg_duration_positive(self, ingested_stats):
        assert ingested_stats["avg_duration_frames"] > 0

    def test_vocabulary_size_positive(self, ingested_stats):
        assert ingested_stats["vocabulary_size"] > 0

    def test_get_statistics_reads_cache(self, ingestor, ingested_stats):
        cached = ingestor.get_statistics()
        assert cached.get("total_samples") == ingested_stats["total_samples"]

    def test_source_tag_correct(self, tmp_dir):
        with open(tmp_dir / "manifest_train.json", encoding="utf-8") as f:
            data = json.load(f)
        sample = data["samples"][0]
        assert sample.get("source") == "Ban-Sign-Sent-9K"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Keypoint data integrity
# ──────────────────────────────────────────────────────────────────────────────

class TestBanSignSentKeypoints:
    def test_npy_shape_T75_3(self, tmp_dir):
        for npy in list((tmp_dir / "keypoints").glob("*.npy"))[:10]:
            arr = np.load(str(npy))
            assert arr.ndim == 3, f"{npy.name}: expected 3D"
            assert arr.shape[1] == 75
            assert arr.shape[2] == 3

    def test_xy_clamped_0_1(self, tmp_dir):
        for npy in list((tmp_dir / "keypoints").glob("*.npy"))[:5]:
            arr = np.load(str(npy)).astype(np.float32)
            assert arr[:, :, 0].max() <= 1.01
            assert arr[:, :, 0].min() >= -0.01

    def test_not_all_zeros(self, tmp_dir):
        non_zero = sum(
            1 for npy in list((tmp_dir / "keypoints").glob("*.npy"))[:10]
            if np.any(np.load(str(npy)) != 0)
        )
        assert non_zero > 0

    def test_gloss_boundaries_count_matches_sequence(self, tmp_dir):
        with open(tmp_dir / "manifest_train.json", encoding="utf-8") as f:
            data = json.load(f)
        for sample in data["samples"][:5]:
            seq = sample.get("gloss_sequence", [])
            bounds = sample.get("gloss_boundaries", [])
            assert len(bounds) == len(seq)


# ──────────────────────────────────────────────────────────────────────────────
# 3. BanSignSentNormalizer
# ──────────────────────────────────────────────────────────────────────────────

class TestBanSignSentNormalizer:
    def setup_method(self):
        self.norm = BanSignSentNormalizer()

    def test_preserves_shape(self):
        arr = np.random.rand(50, 75, 3).astype(np.float32)
        assert self.norm.normalize(arr).shape == (50, 75, 3)

    def test_reference_node_zeroed(self):
        arr = np.ones((20, 75, 3), dtype=np.float32)
        out = self.norm.normalize(arr)
        np.testing.assert_allclose(out[:, 0, :], 0.0, atol=1e-5)

    def test_returns_float32(self):
        arr = np.random.rand(10, 75, 3).astype(np.float64)
        assert self.norm.normalize(arr).dtype == np.float32

    def test_invalid_shape_passthrough(self):
        arr = np.zeros((10, 3))
        out = self.norm.normalize(arr)
        assert out.shape == (10, 3)

    def test_batch_normalize_count(self):
        samples = [np.random.rand(np.random.randint(20, 60), 75, 3).astype(np.float32) for _ in range(6)]
        results = self.norm.batch_normalize(samples)
        assert len(results) == 6


# ──────────────────────────────────────────────────────────────────────────────
# 4. BanSignSentProcessor
# ──────────────────────────────────────────────────────────────────────────────

class TestBanSignSentProcessor:
    def setup_method(self):
        self.proc = BanSignSentProcessor()

    def test_load_manifest_missing_returns_empty(self):
        p = BanSignSentProcessor(processed_dir=Path("/nonexistent/xyz"))
        assert p.load_manifest("train") == []

    def test_resample_to_32_always(self):
        for T in [10, 32, 100, 240]:
            arr = np.random.rand(T, 75, 3).astype(np.float32)
            out = self.proc.resample_to_window(arr, window_size=32)
            assert out.shape == (32, 75, 3)

    def test_resample_exact_passthrough(self):
        arr = np.random.rand(32, 75, 3).astype(np.float32)
        np.testing.assert_array_equal(self.proc.resample_to_window(arr, 32), arr)

    def test_extract_group_pose(self):
        arr = np.random.rand(30, 75, 3).astype(np.float32)
        assert self.proc.extract_group(arr, "pose_upper").shape == (30, 22, 3)

    def test_extract_group_right_hand(self):
        arr = np.random.rand(30, 75, 3).astype(np.float32)
        assert self.proc.extract_group(arr, "right_hand").shape == (30, 21, 3)

    def test_extract_group_face(self):
        arr = np.random.rand(30, 75, 3).astype(np.float32)
        assert self.proc.extract_group(arr, "face_contour").shape == (30, 11, 3)

    def test_get_sentence_patterns_not_empty(self):
        patterns = self.proc.get_sentence_patterns()
        assert len(patterns) > 0
        assert all(isinstance(p, tuple) and len(p) == 2 for p in patterns)

    def test_vocabulary_coverage_after_ingestion(self, processor):
        result = processor.compute_vocabulary_coverage(splits=["train"])
        assert isinstance(result, dict)
        assert result.get("total_samples", 0) > 0

    def test_avg_sentence_length_positive(self, processor):
        result = processor.compute_vocabulary_coverage(splits=["train"])
        assert result["avg_sentence_length"] > 0

    def test_extract_gloss_boundaries_from_sample(self, tmp_dir):
        with open(tmp_dir / "manifest_train.json", encoding="utf-8") as f:
            sample = json.load(f)["samples"][0]
        bounds = self.proc.extract_gloss_boundaries(sample)
        assert isinstance(bounds, list)
        assert all("gloss" in b and "start_frame" in b for b in bounds)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Lexicon expansion via Ban-Sign-Sent
# ──────────────────────────────────────────────────────────────────────────────

class TestBanSignSentLexiconExpansion:
    def test_expand_master_lexicon_returns_dict(self):
        proc = BanSignSentProcessor()
        result = proc.expand_master_lexicon(["নতুন_শব্দ_বানান", "পরীক্ষা_শব্দ"])
        assert "added" in result and "already_present" in result

    def test_expanded_tokens_queryable(self):
        from core_engine.nlp.master_lexicon import MasterBdSLLexicon
        lex = MasterBdSLLexicon()
        unique = ["বিশেষ_বানস্তান_টোকেন_99"]
        lex.expand_with_bornildb_vocab(unique)
        found = lex.get_sign_by_gloss(unique[0])
        assert found is not None

    def test_sentence_level_vocab_coverage(self):
        """Sentence gloss tokens from corpus should be expandable."""
        tokens = list({tok for _, gloss in BANSIGNSENT_SENTENCES for tok in gloss.split()})
        proc = BanSignSentProcessor()
        result = proc.expand_master_lexicon(tokens)
        assert result["total_known"] >= result["added"]


# ──────────────────────────────────────────────────────────────────────────────
# 6. Metadata constants sanity checks
# ──────────────────────────────────────────────────────────────────────────────

class TestBanSignSentMetadata:
    def test_total_clips_matches(self):
        assert BANSIGNSENT_METADATA["total_clips"] == 9610

    def test_split_ratios_sum_to_1(self):
        s = BANSIGNSENT_METADATA["splits"]
        assert abs(sum(s.values()) - 1.0) < 1e-6

    def test_sentences_list_not_empty(self):
        assert len(BANSIGNSENT_SENTENCES) > 0
        assert all(len(s) == 2 for s in BANSIGNSENT_SENTENCES)

    def test_sentence_gloss_not_empty(self):
        for text, gloss in BANSIGNSENT_SENTENCES:
            assert text.strip() and gloss.strip()

    def test_landmark_groups_cover_75_nodes(self):
        covered = set()
        for start, end in LANDMARK_GROUPS.values():
            covered.update(range(start, end))
        assert covered == set(range(75)), "Landmark groups must cover exactly nodes 0-74."

    def test_sentence_patterns_have_descriptions(self):
        for pattern, desc in SENTENCE_PATTERNS:
            assert pattern and desc
