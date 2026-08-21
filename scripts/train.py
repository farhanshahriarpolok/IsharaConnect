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
from sklearn.model_selection import train_test_split
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
            
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def load_dataset(dataset_dir: str, num_classes: int):
    """Load dataset from .npy files in dataset_dir."""
    dataset_path = Path(dataset_dir)
    X = []
    y = []
    
    # Check if there are real .npy files
    npy_files = list(dataset_path.glob("**/*.npy"))
    if not npy_files:
        logger.warning("No real data found in %s. Using synthetic baseline.", dataset_dir)
        return generate_synthetic_baseline(num_classes)
        
    logger.info("Found %d real .npy sequences in %s", len(npy_files), dataset_dir)
    
    for f in npy_files:
        try:
            # f.parent.name should be the class ID
            label = int(f.parent.name)
            seq = np.load(f)
            # seq shape should be (30, 128) or similar
            X.append(seq)
            y.append(label)
        except Exception as e:
            logger.warning("Failed to load %s: %s", f, e)
            
    if not X:
        logger.warning("No valid sequences loaded. Falling back to synthetic.")
        return generate_synthetic_baseline(num_classes)
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


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
    
    X, y = load_dataset(args.dataset_dir, num_classes)
    
    # Ensure input matches 128 dimensions expected by model
    if X.shape[-1] != 128:
        X_padded = np.zeros((X.shape[0], X.shape[1], 128), dtype=np.float32)
        X_padded[:, :, :X.shape[-1]] = X
        if X.shape[-1] == 126:
            X_padded[:, :, 126:] = 1.0 # Set presence flags
        X = X_padded

    classes, counts = np.unique(y, return_counts=True)
    if any(c < 2 for c in counts):
        logger.warning("Some classes have fewer than 2 samples. Disabling stratified split.")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = BdSLSequenceClassifier(input_dim=128, num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_loss = float('inf')
    best_model_path = Path(args.output_dir) / "bdsl_model_best.pth"
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        scheduler.step()
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        class_correct = {i: 0 for i in range(num_classes)}
        class_total = {i: 0 for i in range(num_classes)}
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == batch_y).item()
                
                for p, t in zip(preds, batch_y):
                    class_correct[t.item()] += (p == t).item()
                    class_total[t.item()] += 1
                
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
        
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            # Print class-wise accuracy for last epoch
            if epoch == args.epochs - 1:
                logger.info("Class-wise Accuracy:")
                for i in range(num_classes):
                    if class_total[i] > 0:
                        logger.info(f"  Class {i} ({labels_data['signs'][i]['slug'] if i < len(labels_data['signs']) else 'unknown'}): {class_correct[i]/class_total[i]:.2f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            
    logger.info("Training complete. Best model saved to %s", best_model_path)


if __name__ == "__main__":
    main()
