"""ONNX Export script for BdSL Sequence Classifier."""

import argparse
import logging
from pathlib import Path
import torch
import onnxruntime as ort
import numpy as np
import sys

# Fix Windows console encoding issue when torch prints checkmark emojis
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_engine.inference.model import BdSLSequenceClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("export_onnx")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch BdSL Checkpoint to ONNX")
    parser.add_argument("--model-path", type=str, default="models/checkpoints/bdsl_model_best.pth", help="Path to PyTorch checkpoint (.pt)")
    parser.add_argument("--output", type=str, default="models/onnx/bdsl_model.onnx", help="Output ONNX model path")

    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    import json
    labels_file = Path("dataset/labels.json")
    if labels_file.exists():
        with open(labels_file, "r", encoding="utf-8") as f:
            labels_data = json.load(f)
        num_classes = len(labels_data.get("signs", []))
    else:
        num_classes = 24
        logger.warning("labels.json not found. Defaulting to 24 classes.")
    
    if not Path(args.model_path).exists():
        logger.error("Model path %s does not exist", args.model_path)
        return

    logger.info("Loading PyTorch model from %s", args.model_path)
    model = BdSLSequenceClassifier(input_dim=128, num_classes=num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    model.eval()

    # Create dummy input with shape (batch_size=1, sequence_length=30, feature_dim=128)
    dummy_input = torch.randn(1, 30, 128)

    logger.info("Exporting to ONNX format...")
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    logger.info("Model successfully exported to %s", args.output)

    # Validate ONNX model with ONNX Runtime
    logger.info("Validating ONNX export with Inference Session...")
    try:
        ort_session = ort.InferenceSession(args.output)
        
        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
            
        # Compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
        ort_outs = ort_session.run(None, ort_inputs)
        
        # Compare with PyTorch
        torch_out = model(dummy_input)
        np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
        
        logger.info("ONNX validation successful. Outputs match PyTorch model.")
    except Exception as e:
        logger.error("ONNX validation failed: %s", e)

if __name__ == "__main__":
    main()
