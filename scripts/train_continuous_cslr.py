"""Production Training Script for Continuous Sign Language Recognition (CSLR) with CTC.

Trains BiLSTM-CTC temporal alignment models on continuous BdSL signing landmark streams (Tier 3).
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dataset.ingestors.tier3_cslr_ingestor import Tier3CSLRIngestor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("train_continuous_cslr")


class ContinuousCSLRModel(nn.Module):
    """BiLSTM temporal sequence encoder with CTC output projection layer."""

    def __init__(self, input_dim: int = 151, hidden_dim: int = 128, vocab_size: int = 30):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 151)
        out, _ = self.lstm(x)  # (B, T, 256)
        logits = self.fc(out)  # (B, T, vocab_size)
        return logits.log_softmax(2)


def train_cslr_model(
    dataset_path: str = "dataset/processed/tier3_cslr/tier3_cslr_dataset.npz",
    epochs: int = 3,
    lr: float = 1e-3,
    output_dir: str = "models"
) -> Dict[str, Any]:
    """Trains continuous CSLR model with PyTorch CTCLoss."""
    data_file = Path(dataset_path)
    if not data_file.exists():
        logger.info(f"Dataset not found at {data_file}. Generating mock Tier 3 CSLR dataset...")
        ingestor = Tier3CSLRIngestor()
        ingestor.generate_mock_dataset()

    data = np.load(data_file, allow_pickle=True)
    seqs, targets = data["sequences"], data["targets"]
    vocab = json.loads(str(data["vocab_json"]))
    vocab_size = max(len(vocab), 20)

    logger.info(f"Loaded {len(seqs)} continuous clips with vocab size {vocab_size}.")

    model = ContinuousCSLRModel(input_dim=151, hidden_dim=128, vocab_size=vocab_size)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for i in range(len(seqs)):
            stream = torch.tensor(seqs[i], dtype=torch.float32).unsqueeze(0)  # (1, T, 151)
            target = torch.tensor(targets[i], dtype=torch.long)
            
            optimizer.zero_grad()
            log_probs = model(stream)  # (1, T, vocab_size)
            # CTC loss expects (T, N, C)
            log_probs = log_probs.permute(1, 0, 2)
            input_lengths = torch.tensor([log_probs.size(0)], dtype=torch.long)
            target_lengths = torch.tensor([len(target)], dtype=torch.long)

            loss = ctc_loss(log_probs, target.unsqueeze(0), input_lengths, target_lengths)
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        avg_loss = epoch_loss / len(seqs)
        logger.info(f"Epoch [{epoch}/{epochs}] - CTC Loss: {avg_loss:.4f}")

    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pt_file = ckpt_dir / "bdsl_continuous_cslr.pt"
    torch.save(model.state_dict(), pt_file)
    logger.info(f"Saved Continuous CSLR model to {pt_file}")

    return {
        "status": "success",
        "epochs": epochs,
        "final_loss": avg_loss,
        "checkpoint": str(pt_file)
    }


def main():
    parser = argparse.ArgumentParser(description="Train Continuous CSLR Model with CTC")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    res = train_cslr_model(epochs=args.epochs)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
