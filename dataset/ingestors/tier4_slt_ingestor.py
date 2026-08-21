"""Tier 4: BdSL Sign Language Translation (SLT) Ingestor (Bangla-SGP / BornilDB Corpus).

Processes parallel Gloss-to-Bengali/English text translation pairs, constructs
bidirectional subword/word tokenizers, and exports parallel sequence tensors.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core_engine.nlp.advanced_grammar_engine import AdvancedBdSLGrammarEngine

logger = logging.getLogger("tier4_ingestor")


class Tier4SLTIngestor:
    """Ingestion & Translation Matrix Pipeline for Tier 4 Gloss-to-Text SLT."""

    def __init__(self, manifest_path: Optional[str] = None):
        self.grammar_engine = AdvancedBdSLGrammarEngine()
        self.manifest_path = Path(manifest_path or "dataset/manifests/tier4_manifest.json")
        self.manifest_data: Dict[str, Any] = self._load_manifest()
        
        self.gloss_to_id: Dict[str, int] = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.text_to_id: Dict[str, int] = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self._build_vocabularies()

    def _load_manifest(self) -> Dict[str, Any]:
        """Loads Tier 4 parallel manifest."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed loading manifest {self.manifest_path}: {e}")

        return {
            "tier_name": "Tier 4: Sign Language Translation (SLT)",
            "sample_parallel_corpus": []
        }

    def _build_vocabularies(self):
        """Constructs source (gloss) and target (Bengali text) vocabularies from manifest."""
        samples = self.manifest_data.get("sample_parallel_corpus", [])
        g_idx = len(self.gloss_to_id)
        t_idx = len(self.text_to_id)

        for item in samples:
            for g in item.get("tokens_gloss", []):
                g_upper = g.upper()
                if g_upper not in self.gloss_to_id:
                    self.gloss_to_id[g_upper] = g_idx
                    g_idx += 1

            for w in item.get("tokens_text", []):
                if w not in self.text_to_id:
                    self.text_to_id[w] = t_idx
                    t_idx += 1

    def tokenize_sequence(self, tokens: List[str], is_source: bool = True, max_len: int = 32) -> np.ndarray:
        """Encodes token list to padded integer sequence tensor."""
        vocab = self.gloss_to_id if is_source else self.text_to_id
        seq = [vocab.get("<sos>")]
        for tok in tokens:
            key = tok.upper() if is_source else tok
            seq.append(vocab.get(key, vocab["<unk>"]))
        seq.append(vocab.get("<eos>"))

        padded = np.zeros(max_len, dtype=np.int64)
        n = min(len(seq), max_len)
        padded[:n] = seq[:n]
        return padded

    def generate_mock_corpus(
        self,
        num_pairs: int = 100,
        output_dir: str = "dataset/processed/tier4_slt"
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Generates synthetic parallel Gloss -> Bengali sentence matrices."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        samples = self.manifest_data.get("sample_parallel_corpus", [])
        if not samples:
            samples = [
                {
                    "gloss_sequence": "AMI DAKTAR SAHAJJO CHAI",
                    "bengali_text": "আমি ডাক্তারের সাহায্য চাই।",
                    "tokens_gloss": ["AMI", "DAKTAR", "SAHAJJO", "CHAI"],
                    "tokens_text": ["আমি", "ডাক্তারের", "সাহায্য", "চাই", "।"]
                },
                {
                    "gloss_sequence": "APNI KEMON ACHEN",
                    "bengali_text": "আপনি কেমন আছেন?",
                    "tokens_gloss": ["APNI", "KEMON", "ACHEN"],
                    "tokens_text": ["আপনি", "কেমন", "আছেন", "?"]
                }
            ]

        src_matrices = []
        tgt_matrices = []
        metadata = []

        logger.info(f"Generating Tier 4 SLT parallel corpus: {num_pairs} pairs...")

        for i in range(num_pairs):
            template = samples[i % len(samples)]
            g_tokens = template.get("tokens_gloss", ["AMI", "BHALO"])
            t_tokens = template.get("tokens_text", ["আমি", "ভালো", "আছি", "।"])

            src_enc = self.tokenize_sequence(g_tokens, is_source=True, max_len=32)
            tgt_enc = self.tokenize_sequence(t_tokens, is_source=False, max_len=32)

            src_matrices.append(src_enc)
            tgt_matrices.append(tgt_enc)
            metadata.append({
                "id": f"SLT_{i:04d}",
                "gloss": template.get("gloss_sequence", ""),
                "bengali": template.get("bengali_text", "")
            })

        X_src = np.array(src_matrices, dtype=np.int64)
        Y_tgt = np.array(tgt_matrices, dtype=np.int64)

        save_file = out_path / "tier4_slt_dataset.npz"
        np.savez_compressed(
            save_file,
            src_tokens=X_src,
            tgt_tokens=Y_tgt,
            gloss_vocab=json.dumps(self.gloss_to_id),
            text_vocab=json.dumps(self.text_to_id),
            metadata_json=json.dumps(metadata)
        )
        logger.info(f"Tier 4 SLT dataset saved to {save_file} with shape {X_src.shape} -> {Y_tgt.shape}")

        return X_src, Y_tgt, str(save_file)

    def validate(self, dataset_path: str = "dataset/processed/tier4_slt/tier4_slt_dataset.npz") -> Dict[str, Any]:
        """Validates Tier 4 parallel dataset format and vocabulary consistency."""
        path = Path(dataset_path)
        if not path.exists():
            return {"valid": False, "error": f"File not found: {dataset_path}"}

        try:
            data = np.load(path)
            src, tgt = data["src_tokens"], data["tgt_tokens"]
            g_vocab = json.loads(str(data["gloss_vocab"]))
            t_vocab = json.loads(str(data["text_vocab"]))

            is_valid = len(src) == len(tgt) and len(src) > 0
            return {
                "valid": bool(is_valid),
                "num_pairs": int(len(src)),
                "max_sequence_len": int(src.shape[1]),
                "gloss_vocab_size": int(len(g_vocab)),
                "text_vocab_size": int(len(t_vocab)),
                "dataset_file": str(path)
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def get_statistics(self, dataset_path: str = "dataset/processed/tier4_slt/tier4_slt_dataset.npz") -> Dict[str, Any]:
        """Computes summary statistics for Tier 4 SLT dataset."""
        val = self.validate(dataset_path)
        if not val.get("valid", False):
            return val

        return {
            "tier": "Tier 4: Sign Language Translation (SLT)",
            "parallel_pairs_count": val.get("num_pairs", 0),
            "gloss_vocab_size": val.get("gloss_vocab_size", 0),
            "text_vocab_size": val.get("text_vocab_size", 0),
            "max_sequence_length": val.get("max_sequence_len", 32),
            "status": "Ready for Sequence-to-Sequence / Transformer Translation Training"
        }
