import torch
import numpy as np
from core_engine.vision.dual_hand_trainer import SpatialFeatureDataset, DualHandSpatialModel

def test_spatial_dataset():
    # 10 samples, 151 features
    features = np.random.randn(10, 151)
    labels = np.random.randint(0, 63, size=(10,))
    
    dataset = SpatialFeatureDataset(features, labels)
    assert len(dataset) == 10
    
    feat, label = dataset[0]
    assert feat.shape == (151,)
    assert label.item() in labels

def test_dual_hand_model():
    model = DualHandSpatialModel(input_dim=151, hidden_dim=64, num_classes=63)
    
    # Test flat input
    x_flat = torch.randn(8, 151)
    out_flat = model(x_flat)
    assert out_flat.shape == (8, 63)
    
    # Test temporal input (N, seq_len, 151)
    x_temp = torch.randn(8, 30, 151)
    out_temp = model(x_temp)
    assert out_temp.shape == (8, 63)
