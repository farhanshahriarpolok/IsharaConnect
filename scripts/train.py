"""Training script for Dynamic Bi-LSTM BdSL Sequence Classifier."""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import f1_score
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_engine.inference.model import BdSLSequenceClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train")


def generate_synthetic_baseline(num_classes: int, samples_per_class: int = 50, seq_len: int = 30, feature_dim: int = 126):
    """Generate synthetic baseline dataset if real data is absent."""
    logger.info("Generating synthetic baseline dataset with %d classes...", num_classes)
    X = []
    y = []
    
    for c in range(num_classes):
        # Base trajectory for this class
        base_trajectory = np.linspace(0, 1, seq_len).reshape(-1, 1) * np.random.randn(1, feature_dim)
        
        for _ in range(samples_per_class):
            # Add Gaussian noise
            noise = np.random.normal(0, 0.05, (seq_len, feature_dim))
            # Kinematic smoothing (moving average)
            noisy_traj = base_trajectory + noise
            smoothed_traj = np.zeros_like(noisy_traj)
            for i in range(feature_dim):
                smoothed_traj[:, i] = np.convolve(noisy_traj[:, i], np.ones(3)/3, mode='same')
                
            X.append(smoothed_traj)
            y.append(c)
            # Synthetic signers (modulo 5 groups)
            groups.append(len(X) % 5)
            
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), np.array(groups, dtype=np.int64)


def load_dataset(dataset_dir: str, num_classes: int):
    """Load dataset from .npy files in dataset_dir."""
    dataset_path = Path(dataset_dir)
    X = []
    y = []
    groups = []
    
    # Check if there are real .npy files
    npy_files = list(dataset_path.glob("**/*.npy"))
    if not npy_files:
        logger.warning("No real data found in %s. Using synthetic baseline.", dataset_dir)
        return generate_synthetic_baseline(num_classes)
        
    logger.info("Found %d real .npy sequences in %s", len(npy_files), dataset_dir)
    
        try:
            # f.parent.name should be the class ID
            label = int(f.parent.name)
            seq = np.load(f)
            
            # Parse signer_id from filename: <signer_id>_<slug>_<timestamp>.npy
            parts = f.stem.split("_")
            signer_id = 0
            if len(parts) >= 3 and parts[0].isdigit():
                signer_id = int(parts[0])
            
            # seq shape should be (30, 128) or similar
            X.append(seq)
            y.append(label)
            groups.append(signer_id)
        except Exception as e:
            logger.warning("Failed to load %s: %s", f, e)
            
    if not X:
        logger.warning("No valid sequences loaded. Falling back to synthetic.")
        return generate_synthetic_baseline(num_classes)
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), np.array(groups, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BdSL Landmark Recognition Models")
    parser.add_argument("--dataset-dir", type=str, default="dataset/raw_landmarks", help="Path to preprocessed dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="models/checkpoints", help="Directory to save model checkpoints")

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    labels_file = Path("dataset/labels.json")
    if not labels_file.exists():
        logger.error("Labels file not found!")
        return
        
    with open(labels_file, "r", encoding="utf-8") as f:
        labels_data = json.load(f)
    num_classes = len(labels_data.get("signs", []))
    
    X, y, groups = load_dataset(args.dataset_dir, num_classes)
    
    # Ensure input matches 128 dimensions expected by model
    if X.shape[-1] != 128:
        X_padded = np.zeros((X.shape[0], X.shape[1], 128), dtype=np.float32)
        X_padded[:, :, :X.shape[-1]] = X
        if X.shape[-1] == 126:
            X_padded[:, :, 126:] = 1.0 # Set presence flags
        X = X_padded

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    unique_groups = np.unique(groups)
    if len(unique_groups) < 5:
        logger.warning("Fewer than 5 signers found. Falling back to synthetic group assignment for GroupKFold.")
        groups = np.arange(len(y)) % 5
        
    gkf = GroupKFold(n_splits=5)
    
    best_macro_f1 = -1.0
    best_model_path = Path(args.output_dir) / "bdsl_model_best.pth"
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        logger.info(f"--- Starting Fold {fold+1} ---")
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        model = BdSLSequenceClassifier(input_dim=128, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        fold_best_val_loss = float('inf')
        
        for epoch in range(args.epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
            scheduler.step()
            
            # Validation
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
                    
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())
                    
            val_loss /= len(val_loader.dataset)
            macro_f1 = f1_score(all_targets, all_preds, average='macro')
            
            if epoch == args.epochs - 1:
                logger.info(f"Fold {fold+1} Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Macro F1: {macro_f1:.4f}")
            
            # Save if this is the best globally
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"New Global Best Model saved (Fold {fold+1}, Macro F1: {macro_f1:.4f})")
                
    logger.info("Training complete. Best model Macro F1: %.4f saved to %s", best_macro_f1, best_model_path)


if __name__ == "__main__":
    main()
