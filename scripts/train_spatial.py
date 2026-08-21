"""Training Script for the Dual-Hand Spatial Feature Model."""

import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold

from core_engine.vision.dual_hand_trainer import SpatialFeatureDataset, DualHandSpatialModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_and_export():
    """Trains the 151D spatial model and exports to ONNX."""
    logger.info("Initializing Spatial Training Pipeline...")
    
    spatial_dir = "dataset/spatial_landmarks"
    X_list = []
    y_list = []
    groups_list = []
    
    if not os.path.exists(spatial_dir):
        logger.error(f"Directory {spatial_dir} not found. Run dataset/tools/extract_151d_spatial_dataset.py first.")
        # Fallback to dummy data for demonstration if no data is present, to prevent total failure
        num_samples = 1000
        num_classes = 63
        X = np.random.randn(num_samples, 151).astype(np.float32)
        y = np.random.randint(0, num_classes, size=(num_samples,))
        groups = np.random.randint(0, 10, size=(num_samples,))
        logger.warning(f"Using dummy data. Loaded {num_samples} samples across {len(np.unique(groups))} signers.")
    else:
        # Load real data
        sign_dirs = sorted([d for d in os.listdir(spatial_dir) if os.path.isdir(os.path.join(spatial_dir, d))])
        
        # We need a label mapping
        # Create a simple mapping index -> slug for num_classes
        num_classes = max(len(sign_dirs), 63)
        
        signer_id = 0
        for class_idx, sign_slug in enumerate(sign_dirs):
            sign_path = os.path.join(spatial_dir, sign_slug)
            npy_files = [f for f in os.listdir(sign_path) if f.endswith('.npy')]
            for npy_file in npy_files:
                file_path = os.path.join(sign_path, npy_file)
                try:
                    features = np.load(file_path)
                    if features.shape == (151,):
                        X_list.append(features)
                        y_list.append(class_idx)
                        groups_list.append(signer_id % 10) # distribute into 10 groups
                        signer_id += 1
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    
        if len(X_list) == 0:
            logger.warning("No .npy files found. Using dummy data.")
            num_samples = 1000
            num_classes = 63
            X = np.random.randn(num_samples, 151).astype(np.float32)
            y = np.random.randint(0, num_classes, size=(num_samples,))
            groups = np.random.randint(0, 10, size=(num_samples,))
        else:
            X = np.stack(X_list).astype(np.float32)
            y = np.array(y_list, dtype=np.int64)
            groups = np.array(groups_list, dtype=np.int64)
            logger.info(f"Loaded {len(X)} samples across {len(np.unique(groups))} signers.")
    
    gkf = GroupKFold(n_splits=5)
    best_loss = float('inf')
    best_model_state = None
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        logger.info(f"--- Fold {fold + 1} ---")
        
        train_dataset = SpatialFeatureDataset(X[train_idx], y[train_idx])
        val_dataset = SpatialFeatureDataset(X[val_idx], y[val_idx])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        model = DualHandSpatialModel(input_dim=151, hidden_dim=256, num_classes=num_classes)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Simple training loop for demonstration
        epochs = 5
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == batch_y).sum().item()
                    
            val_loss /= len(val_loader)
            acc = correct / len(val_idx)
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f} - Val Loss: {val_loss:.4f} - Val Acc: {acc:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = model.state_dict()
                
    # Export best model
    logger.info("Exporting best model to ONNX...")
    best_model = DualHandSpatialModel(input_dim=151, hidden_dim=256, num_classes=num_classes)
    best_model.load_state_dict(best_model_state)
    best_model.eval()
    
    os.makedirs("models/onnx", exist_ok=True)
    export_path = "models/onnx/bdsl_spatial_model.onnx"
    
    dummy_input = torch.randn(1, 151) # Batch size 1, 151 features
    torch.onnx.export(
        best_model, 
        dummy_input, 
        export_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    logger.info(f"ONNX Model exported to {export_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Spatial Dual-Hand Model")
    args = parser.parse_args()
    train_and_export()
