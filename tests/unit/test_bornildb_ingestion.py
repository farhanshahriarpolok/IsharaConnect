"""Tests for BornilDB v1.0 ingestion, keypoint processing, and benchmark evaluation pipeline.

Coverage:
  - BornilDBIngestor: manifest generation, keypoint .npy creation, statistics output
  - BornilDBProcessor: landmark normalization, window resampling, vocabulary coverage
  - CSLRBenchmarkEvaluator: WER, CER metric computation
  - MasterBdSLLexicon: expand_with_bornildb_vocab, get_bornildb_coverage
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download_and_ingest_bornildb import BornilDBIngestor, BORNILDB_GLOSS_VOCAB
from core_engine.dataset.bornildb_processor import BornilDBProcessor, LandmarkNormalizer, LANDMARK_GROUPS
from core_engine.inference.cslr_benchmark_evaluator import (
    CSLRBenchmarkEvaluator,
    compute_wer,
    compute_cer,
    _levenshtein_distance,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tmp_processed_dir(tmp_path_factory):
    """Provides a temporary processed directory for ingestion tests."""
    return tmp_path_factory.mktemp("bornildb_processed")


@pytest.fixture(scope="module")
def ingestor(tmp_processed_dir):
    """Returns a BornilDBIngestor instance targeting a temp directory."""
    return BornilDBIngestor(
        output_dir=tmp_processed_dir,
        raw_dir=tmp_processed_dir / "raw",
        samples=30
    )


@pytest.fixture(scope="module")
def ingested_stats(ingestor):
    """Runs ingestion once and caches stats for module-level reuse."""
    return ingestor.ingest()


@pytest.fixture(scope="module")
def processor(tmp_processed_dir, ingested_stats):
    """Returns a BornilDBProcessor pointing at the temp processed dir (after ingestion)."""
    return BornilDBProcessor(processed_dir=tmp_processed_dir)



# ──────────────────────────────────────────────────────────────────────────────
# 1. BornilDBIngestor — Basic Contract
# ──────────────────────────────────────────────────────────────────────────────

class TestBornilDBIngestor:
    def test_ingest_returns_stats_dict(self, ingested_stats):
        """Ingestion must return a dict with expected keys."""
        assert isinstance(ingested_stats, dict)
        assert "total_samples" in ingested_stats
        assert "split_counts" in ingested_stats
        assert "vocabulary_size" in ingested_stats

    def test_sample_count(self, ingested_stats):
        """Total samples should equal the requested amount."""
        assert ingested_stats["total_samples"] == 30

    def test_split_manifests_exist(self, tmp_processed_dir):
        """All three split manifest JSON files must be written to disk."""
        for split in ["train", "val", "test"]:
            assert (tmp_processed_dir / f"manifest_{split}.json").exists()

    def test_manifest_structure(self, tmp_processed_dir):
        """Train manifest must contain 'metadata' and 'samples' keys."""
        with open(tmp_processed_dir / "manifest_train.json", encoding="utf-8") as f:
            data = json.load(f)
        assert "metadata" in data
        assert "samples" in data
        assert len(data["samples"]) > 0

    def test_sample_fields(self, tmp_processed_dir):
        """Each sample must have required schema fields."""
        with open(tmp_processed_dir / "manifest_train.json", encoding="utf-8") as f:
            data = json.load(f)
        sample = data["samples"][0]
        for field in ["sample_id", "split", "gloss_sequence", "keypoints_path", "duration_frames"]:
            assert field in sample, f"Missing field: {field}"

    def test_keypoints_npy_files_created(self, tmp_processed_dir):
        """At least one .npy keypoint file should be created in the keypoints subdir."""
        kp_dir = tmp_processed_dir / "keypoints"
        assert kp_dir.exists()
        npy_files = list(kp_dir.glob("*.npy"))
        assert len(npy_files) > 0, "No .npy keypoint files created."

    def test_vocabulary_json_created(self, tmp_processed_dir):
        """A bornildb_vocabulary.json file must be written."""
        assert (tmp_processed_dir / "bornildb_vocabulary.json").exists()

    def test_vocabulary_tokens_are_strings(self, tmp_processed_dir):
        """All vocabulary entries must be non-empty strings."""
        with open(tmp_processed_dir / "bornildb_vocabulary.json", encoding="utf-8") as f:
            vocab = json.load(f)
        assert isinstance(vocab, list)
        assert all(isinstance(t, str) and len(t) > 0 for t in vocab)

    def test_statistics_json_created(self, tmp_processed_dir):
        """dataset_statistics.json must be written."""
        assert (tmp_processed_dir / "dataset_statistics.json").exists()

    def test_statistics_avg_duration_positive(self, ingested_stats):
        """Average frame duration must be strictly positive."""
        assert ingested_stats["avg_duration_frames"] > 0

    def test_get_statistics_reads_from_cache(self, ingestor, ingested_stats):
        """get_statistics() must return the cached result without re-running ingestion."""
        cached = ingestor.get_statistics()
        assert cached.get("total_samples") == ingested_stats["total_samples"]


# ──────────────────────────────────────────────────────────────────────────────
# 2. BornilDBIngestor — Keypoint Data Integrity
# ──────────────────────────────────────────────────────────────────────────────

class TestKeyPointDataIntegrity:
    def test_keypoints_shape_T75_3(self, tmp_processed_dir):
        """Every .npy file must have shape (T, 75, 3)."""
        kp_dir = tmp_processed_dir / "keypoints"
        for npy_path in list(kp_dir.glob("*.npy"))[:10]:
            arr = np.load(str(npy_path))
            assert arr.ndim == 3, f"{npy_path.name}: expected 3D, got {arr.ndim}D"
            assert arr.shape[1] == 75, f"{npy_path.name}: expected 75 nodes, got {arr.shape[1]}"
            assert arr.shape[2] == 3, f"{npy_path.name}: expected 3 coords, got {arr.shape[2]}"

    def test_xy_coordinates_clamped(self, tmp_processed_dir):
        """x/y coordinates must be in [0, 1]."""
        kp_dir = tmp_processed_dir / "keypoints"
        for npy_path in list(kp_dir.glob("*.npy"))[:5]:
            arr = np.load(str(npy_path)).astype(np.float32)
            assert arr[:, :, 0].min() >= -0.01
            assert arr[:, :, 0].max() <= 1.01
            assert arr[:, :, 1].min() >= -0.01
            assert arr[:, :, 1].max() <= 1.01

    def test_keypoints_not_all_zeros(self, tmp_processed_dir):
        """Keypoints should not be trivially all-zero."""
        kp_dir = tmp_processed_dir / "keypoints"
        npy_files = list(kp_dir.glob("*.npy"))
        non_zero_count = 0
        for npy_path in npy_files[:10]:
            arr = np.load(str(npy_path))
            if np.any(arr != 0):
                non_zero_count += 1
        assert non_zero_count > 0, "All keypoint files appear to be all-zero."


# ──────────────────────────────────────────────────────────────────────────────
# 3. LandmarkNormalizer
# ──────────────────────────────────────────────────────────────────────────────

class TestLandmarkNormalizer:
    def setup_method(self):
        self.normalizer = LandmarkNormalizer(reference_node=0, scale_ref_pair=(11, 12))

    def test_normalize_preserves_shape(self):
        """Normalization must not change the array shape."""
        arr = np.random.rand(60, 75, 3).astype(np.float32)
        out = self.normalizer.normalize(arr)
        assert out.shape == arr.shape

    def test_normalize_root_node_is_near_zero(self):
        """After normalization, the reference node should be at ~origin each frame."""
        arr = np.ones((30, 75, 3), dtype=np.float32)
        out = self.normalizer.normalize(arr)
        # Reference node should be centered at (0, 0, 0)
        root_vals = out[:, self.normalizer.reference_node, :]
        assert np.allclose(root_vals, 0.0, atol=1e-5), "Root node not at origin post-normalization."

    def test_normalize_returns_float32(self):
        """Output dtype should be float32."""
        arr = np.random.rand(10, 75, 3).astype(np.float64)
        out = self.normalizer.normalize(arr)
        assert out.dtype == np.float32

    def test_batch_normalize_correct_count(self):
        """batch_normalize must return same number of samples."""
        samples = [np.random.rand(np.random.randint(30, 80), 75, 3).astype(np.float32) for _ in range(5)]
        results = self.normalizer.batch_normalize(samples)
        assert len(results) == len(samples)

    def test_normalize_invalid_shape_passthrough(self):
        """Incorrectly shaped arrays should pass through without error."""
        arr = np.random.rand(10, 5).astype(np.float32)
        out = self.normalizer.normalize(arr)
        assert out.shape == arr.shape


# ──────────────────────────────────────────────────────────────────────────────
# 4. BornilDBProcessor — Resampling & Group Extraction
# ──────────────────────────────────────────────────────────────────────────────

class TestBornilDBProcessor:
    def setup_method(self):
        self.processor = BornilDBProcessor()

    def test_build_feature_window_output_shape(self):
        """build_feature_window must return (32, 75, 3) regardless of input T."""
        for T in [10, 32, 100, 250]:
            arr = np.random.rand(T, 75, 3).astype(np.float32)
            window = self.processor.build_feature_window(arr, window_size=32)
            assert window.shape == (32, 75, 3), f"T={T}: Expected (32, 75, 3), got {window.shape}"

    def test_build_feature_window_exact_passthrough(self):
        """When T==window_size, output should be identical to input."""
        arr = np.random.rand(32, 75, 3).astype(np.float32)
        window = self.processor.build_feature_window(arr, window_size=32)
        np.testing.assert_array_equal(window, arr)

    def test_extract_landmark_group_pose(self):
        """pose_upper group should return nodes 0-21 (22 nodes)."""
        arr = np.random.rand(30, 75, 3).astype(np.float32)
        pose = self.processor.extract_landmark_group(arr, "pose_upper")
        assert pose.shape == (30, 22, 3)

    def test_extract_landmark_group_right_hand(self):
        """right_hand group should return 21 nodes."""
        arr = np.random.rand(30, 75, 3).astype(np.float32)
        rh = self.processor.extract_landmark_group(arr, "right_hand")
        assert rh.shape == (30, 21, 3)

    def test_extract_landmark_group_face(self):
        """face_contour group should return 11 nodes."""
        arr = np.random.rand(30, 75, 3).astype(np.float32)
        face = self.processor.extract_landmark_group(arr, "face_contour")
        assert face.shape == (30, 11, 3)

    def test_load_manifest_missing_returns_empty(self):
        """load_manifest for nonexistent split should return []."""
        proc = BornilDBProcessor(processed_dir=Path("/nonexistent/path"))
        result = proc.load_manifest("train")
        assert result == []

    def test_compute_vocabulary_coverage_returns_dict(self, processor, ingested_stats):
        """compute_vocabulary_coverage must return a dict with expected keys."""
        result = processor.compute_vocabulary_coverage(splits=["train"])
        assert isinstance(result, dict)
        for key in ["total_samples", "vocabulary_size", "avg_glosses_per_sentence"]:
            assert key in result, f"Missing key: {key}"

    def test_avg_glosses_per_sentence_positive(self, processor, ingested_stats):
        """Average glosses per sentence must be > 0 after ingestion."""
        result = processor.compute_vocabulary_coverage(splits=["train"])
        assert result.get("avg_glosses_per_sentence", 0) > 0



# ──────────────────────────────────────────────────────────────────────────────
# 5. WER / CER Metric Functions
# ──────────────────────────────────────────────────────────────────────────────

class TestWERCERMetrics:
    @pytest.mark.parametrize("ref,hyp,expected_edits", [
        (["a", "b", "c"], ["a", "b", "c"], 0),
        (["a", "b", "c"], ["a", "x", "c"], 1),
        (["a", "b", "c"], ["a", "b"], 1),
        (["a"], ["a", "b", "c"], 2),
        ([], [], 0),
        (["a"], [], 1),
        ([], ["a"], 1),
    ])
    def test_levenshtein_distance(self, ref, hyp, expected_edits):
        assert _levenshtein_distance(ref, hyp) == expected_edits

    def test_wer_perfect_match(self):
        assert compute_wer("আমি স্কুল যাওয়া", "আমি স্কুল যাওয়া") == 0.0

    def test_wer_all_wrong(self):
        wer = compute_wer("আমি স্কুল যাওয়া", "তুমি বাড়ি আসা")
        assert 0.0 < wer <= 1.0

    def test_wer_empty_reference(self):
        assert compute_wer("", "") == 0.0
        assert compute_wer("", "কিছু") == 1.0

    def test_wer_empty_hypothesis(self):
        wer = compute_wer("আমি যাই", "")
        assert wer > 0.0

    def test_cer_perfect_match(self):
        assert compute_cer("ধন্যবাদ", "ধন্যবাদ") == 0.0

    def test_cer_partially_correct(self):
        cer = compute_cer("ধন্যবাদ", "ধন্যবা")
        assert 0.0 < cer < 1.0

    def test_cer_empty_ref_non_empty_hyp(self):
        assert compute_cer("", "x") == 1.0

    def test_cer_both_empty(self):
        assert compute_cer("", "") == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 6. CSLRBenchmarkEvaluator — Integration Tests (no real ONNX model)
# ──────────────────────────────────────────────────────────────────────────────

class TestCSLRBenchmarkEvaluator:
    """Tests using synthetic sample dicts (no ONNX model required)."""

    @pytest.fixture
    def evaluator(self):
        """Returns evaluator with default (mock-tolerant) engine."""
        return CSLRBenchmarkEvaluator(window_size=32, stride=8)

    def test_evaluate_from_manifest_missing_path(self, evaluator):
        """Missing manifest path should trigger synthetic evaluation, not crash."""
        result = evaluator.evaluate_from_manifest("/nonexistent/manifest_test.json", max_samples=5)
        assert isinstance(result, dict)
        assert "wer" in result
        assert "cer" in result

    def test_wer_cer_in_valid_range(self, evaluator):
        """WER and CER must be in [0, ∞); typically [0, 1] or slightly above for bad preds."""
        result = evaluator.evaluate_from_manifest("/nonexistent/manifest_test.json", max_samples=5)
        assert result["wer"] >= 0.0
        assert result["cer"] >= 0.0

    def test_evaluate_sample_batch_returns_expected_keys(self, evaluator):
        """evaluate_sample_batch must return dict with wer, cer, frame_accuracy."""
        dummy = [
            {
                "sample_id": "test_0",
                "sentence_text": "আমি স্কুল যাওয়া",
                "duration_frames": 32,
                "keypoints_path": None
            }
        ]
        result = evaluator.evaluate_sample_batch(dummy)
        assert isinstance(result, dict)
        for key in ["wer", "cer", "frame_accuracy", "evaluated_samples"]:
            assert key in result, f"Missing result key: {key}"

    def test_evaluated_sample_count_matches(self, evaluator):
        """evaluated_samples in result should equal input batch size."""
        batch = [
            {"sample_id": f"s{i}", "sentence_text": "ধন্যবাদ", "duration_frames": 32, "keypoints_path": None}
            for i in range(4)
        ]
        result = evaluator.evaluate_sample_batch(batch)
        assert result["evaluated_samples"] == 4


# ──────────────────────────────────────────────────────────────────────────────
# 7. MasterBdSLLexicon — BornilDB Vocabulary Expansion
# ──────────────────────────────────────────────────────────────────────────────

class TestLexiconBornilDBExpansion:
    @pytest.fixture(autouse=True)
    def fresh_lexicon(self):
        """Creates a fresh MasterBdSLLexicon with empty backing for each test."""
        from core_engine.nlp.master_lexicon import MasterBdSLLexicon
        self.lexicon = MasterBdSLLexicon()

    def test_expand_new_tokens_added(self):
        """expand_with_bornildb_vocab should add new tokens not in lexicon."""
        unique_new = ["নতুন_শব্দ_১", "নতুন_শব্দ_২", "নতুন_শব্দ_৩"]
        result = self.lexicon.expand_with_bornildb_vocab(unique_new)
        assert result["added"] == 3

    def test_expand_duplicate_tokens_not_double_counted(self):
        """Tokens already in lexicon should not be added again."""
        # Common words that may already be present
        tokens = ["ধন্যবাদ", "__definitely_not_in_lexicon_xyz__"]
        result = self.lexicon.expand_with_bornildb_vocab(tokens)
        total = result["added"] + result["already_present"]
        assert total == len(tokens)

    def test_expand_returns_new_token_list(self):
        """result['new_tokens'] should contain the actually added tokens."""
        new = ["বিশেষ_নতুন_১", "বিশেষ_নতুন_২"]
        result = self.lexicon.expand_with_bornildb_vocab(new)
        assert set(result["new_tokens"]).issubset(set(new))

    def test_expand_empty_list_noop(self):
        """Expanding with empty list should be a no-op."""
        initial_count = len(self.lexicon.signs_by_bn)
        result = self.lexicon.expand_with_bornildb_vocab([])
        assert result["added"] == 0
        assert len(self.lexicon.signs_by_bn) == initial_count

    def test_expand_added_entries_queryable(self):
        """Entries added via expand_with_bornildb_vocab should be retrievable."""
        token = "পরীক্ষা_টোকেন_999"
        self.lexicon.expand_with_bornildb_vocab([token])
        found = self.lexicon.get_sign_by_gloss(token)
        assert found is not None, "Expanded token not retrievable via get_sign_by_gloss."
        assert found.get("source") == "BornilDB_v1.0"

    def test_get_bornildb_coverage_all_known(self):
        """Coverage should be 100% if all tokens are in lexicon (after expansion)."""
        tokens = ["টোকেন_এক", "টোকেন_দুই"]
        self.lexicon.expand_with_bornildb_vocab(tokens)
        cov = self.lexicon.get_bornildb_coverage(tokens)
        assert cov["coverage_pct"] == 100.0

    def test_get_bornildb_coverage_none_known(self):
        """Coverage should be 0% for completely unknown tokens."""
        tokens = ["__unknown_x1__", "__unknown_x2__"]
        cov = self.lexicon.get_bornildb_coverage(tokens)
        assert cov["covered"] == 0
        assert cov["coverage_pct"] == 0.0

    def test_get_bornildb_coverage_partial(self):
        """Coverage should be between 0% and 100% for partial matches."""
        known = ["পরিচিত_শব্দ"]
        unknown = ["অজানা_শব্দ_1", "অজানা_শব্দ_2"]
        self.lexicon.expand_with_bornildb_vocab(known)
        cov = self.lexicon.get_bornildb_coverage(known + unknown)
        assert 0.0 < cov["coverage_pct"] < 100.0

    def test_coverage_missing_tokens_list(self):
        """missing_tokens should list at most 50 items."""
        tokens = [f"unknown_{i}" for i in range(60)]
        cov = self.lexicon.get_bornildb_coverage(tokens)
        assert len(cov["missing_tokens"]) <= 50
