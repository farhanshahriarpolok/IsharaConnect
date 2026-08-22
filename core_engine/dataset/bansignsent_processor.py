"""Ban-Sign-Sent-9K Keypoint Matrix Extraction & Manifest Processor.

Loads Ban-Sign-Sent-9K manifests, normalizes T×75×3 landmark sequences,
provides temporal resampling, vocabulary coverage analysis, and sentence-aligned
gloss boundary extraction for CSLR training.

Landmark node layout (75 nodes, identical to BornilDB schema):
  - Pose upper body : nodes  0-21  (22 nodes)
  - Right hand      : nodes 22-42  (21 nodes)
  - Left hand       : nodes 43-63  (21 nodes)
  - Face contour    : nodes 64-74  (11 nodes)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed" / "bansignsent"

# Re-use landmark group boundaries from BornilDB processor schema
LANDMARK_GROUPS = {
    "pose_upper":   (0,  22),
    "right_hand":   (22, 43),
    "left_hand":    (43, 64),
    "face_contour": (64, 75),
}

# Sentence pattern registry for MasterBdSLLexicon expansion
SENTENCE_PATTERNS = [
    ("কর্তা ক্রিয়া বিভাগ যাওয়া",   "Subject-Verb-Location-Movement"),
    ("কর্তা বস্তু ক্রিয়া",          "Subject-Object-Verb (SOV default)"),
    ("প্রশ্ন বাক্য চোখের ভ্রু উপরে", "WH-Question with raised eyebrow NMM"),
    ("নেতিবাচক মাথা দুলানো",         "Negation with headshake NMM"),
    ("বিশেষণ বিশেষ্য",               "Adjective-Noun classifier"),
    ("সংখ্যা বিশেষ্য",               "Numeral-Noun"),
    ("সময় স্থান ক্রিয়া",            "Tense-Location-Verb (topic first)"),
]


class BanSignSentNormalizer:
    """Scale-invariant 3D landmark normalizer for Ban-Sign-Sent-9K keypoints.

    Uses the same root-centering + inter-shoulder scaling strategy as
    BornilDBProcessor.LandmarkNormalizer for consistent feature representation
    across both corpora.
    """

    def __init__(
        self,
        reference_node: int = 0,
        scale_ref_pair: Tuple[int, int] = (11, 12),
    ):
        self.reference_node = reference_node
        self.scale_ref_pair = scale_ref_pair

    def normalize(self, landmarks_TN3: np.ndarray) -> np.ndarray:
        """Root-centers and scale-normalizes a (T, 75, 3) landmark array."""
        if landmarks_TN3.ndim != 3 or landmarks_TN3.shape[2] != 3:
            return landmarks_TN3.astype(np.float32)

        T, N, _ = landmarks_TN3.shape
        out = landmarks_TN3.astype(np.float32).copy()
        n_l, n_r = [min(n, N - 1) for n in self.scale_ref_pair]

        for t in range(T):
            root = out[t, self.reference_node, :].copy()
            out[t] -= root
            dist = float(np.linalg.norm(out[t, n_r, :2] - out[t, n_l, :2]))
            if dist > 1e-6:
                out[t] /= dist

        return out

    def batch_normalize(self, samples: List[np.ndarray]) -> List[np.ndarray]:
        """Normalizes a list of variable-length landmark arrays."""
        return [self.normalize(s) for s in samples]


class BanSignSentProcessor:
    """Loads and processes Ban-Sign-Sent-9K manifests for CSLR model training."""

    def __init__(
        self,
        processed_dir: Optional[Path] = None,
        normalizer: Optional[BanSignSentNormalizer] = None,
    ):
        self.processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
        self.normalizer = normalizer or BanSignSentNormalizer()

    def load_manifest(self, split: str = "train") -> List[Dict[str, Any]]:
        """Loads split manifest. Returns empty list if not found."""
        mp = self.processed_dir / f"manifest_{split}.json"
        if not mp.exists():
            logger.warning("Manifest not found: %s", mp)
            return []
        with open(mp, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("samples", [])

    def load_keypoints(self, sample: Dict[str, Any]) -> Optional[np.ndarray]:
        """Loads and normalizes T×75×3 keypoints for a sample. Handles abs/rel paths."""
        kp_str = sample.get("keypoints_path", "")
        if not kp_str:
            return None
        kp = Path(kp_str)
        if not kp.is_absolute():
            kp = PROJECT_ROOT / kp
        if not kp.exists():
            logger.debug("Keypoints not found: %s", kp)
            return None
        arr = np.load(str(kp)).astype(np.float32)
        if arr.ndim == 3 and arr.shape[1:] == (75, 3):
            return self.normalizer.normalize(arr)
        return arr

    def resample_to_window(self, arr: np.ndarray, window_size: int = 32) -> np.ndarray:
        """Temporally resamples T×75×3 to exactly window_size frames via linear interp."""
        T = arr.shape[0]
        if T == window_size:
            return arr
        idx = np.linspace(0, T - 1, window_size)
        i_int = idx.astype(int)
        i_frac = (idx - i_int).reshape(-1, 1, 1)
        i_next = np.minimum(i_int + 1, T - 1)
        return (arr[i_int] * (1 - i_frac) + arr[i_next] * i_frac).astype(np.float32)

    def extract_group(self, arr: np.ndarray, group: str) -> np.ndarray:
        """Extracts a landmark group slice (T, N_group, 3)."""
        start, end = LANDMARK_GROUPS.get(group, (0, 75))
        return arr[:, start:end, :]

    def extract_gloss_boundaries(
        self, sample: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Returns the gloss boundary list from a manifest sample."""
        return sample.get("gloss_boundaries", [])

    def compute_vocabulary_coverage(
        self, splits: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyzes vocabulary distribution across splits."""
        if splits is None:
            splits = ["train", "val", "test"]

        vocab_count: Dict[str, int] = {}
        total_samples = 0
        total_glosses = 0
        sentence_lengths: List[int] = []
        transition_count: Dict[str, int] = {}

        for split in splits:
            for sample in self.load_manifest(split):
                total_samples += 1
                seq = sample.get("gloss_sequence", [])
                total_glosses += len(seq)
                sentence_lengths.append(len(seq))
                for tok in seq:
                    vocab_count[tok] = vocab_count.get(tok, 0) + 1
                for i in range(len(seq) - 1):
                    k = f"{seq[i]}→{seq[i+1]}"
                    transition_count[k] = transition_count.get(k, 0) + 1

        sorted_vocab = sorted(vocab_count.items(), key=lambda x: x[1], reverse=True)
        top_transitions = sorted(transition_count.items(), key=lambda x: x[1], reverse=True)[:20]

        return {
            "total_samples": total_samples,
            "total_glosses": total_glosses,
            "vocabulary_size": len(vocab_count),
            "avg_sentence_length": total_glosses / max(total_samples, 1),
            "max_sentence_length": max(sentence_lengths, default=0),
            "min_sentence_length": min(sentence_lengths, default=0),
            "top_30_glosses": sorted_vocab[:30],
            "class_distribution": dict(sorted_vocab),
            "top_20_transitions": dict(top_transitions),
        }

    def expand_master_lexicon(self, vocab_tokens: List[str]) -> Dict[str, Any]:
        """Expands the MasterBdSLLexicon with Ban-Sign-Sent gloss tokens."""
        from core_engine.nlp.master_lexicon import MasterBdSLLexicon
        lex = MasterBdSLLexicon()
        result = lex.expand_with_bornildb_vocab(vocab_tokens)
        logger.info(
            "MasterLexicon expanded: +%d tokens (%d already present)",
            result["added"], result["already_present"]
        )
        return result

    def get_sentence_patterns(self) -> List[Tuple[str, str]]:
        """Returns the registered BdSL sentence structural patterns."""
        return SENTENCE_PATTERNS
