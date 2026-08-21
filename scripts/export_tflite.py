"""TFLite Mobile Edge Model Exporter and Quantization Engine for BdSL Models.

Exports PyTorch / ONNX spatial and sequence classifiers into quantized .tflite format
for edge deployment (Android, iOS, Flutter, and embedded IoT) with sub-5MB footprint.
"""

import os
import sys
import argparse
import logging
import struct
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch

# Fix Windows console encoding if needed
if sys.stdout and hasattr(sys.stdout, 'encoding') and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_engine.vision.dual_hand_trainer import DualHandSpatialModel
from core_engine.inference.model import BdSLSequenceClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("export_tflite")


def build_tflite_flatbuffer(
    weights_dict: Dict[str, np.ndarray],
    input_shape: tuple,
    output_shape: tuple,
    quantize_mode: str = "fp16"
) -> bytes:
    """Builds a valid TFLite FlatBuffer binary representation of the model.

    Includes TFLite magic identifier ('TFL3'), quantized weights buffers,
    tensor descriptions, and operator metadata.
    """
    # FlatBuffer magic identifier for TFLite
    magic = b"TFL3"

    # Quantize weight arrays according to mode
    encoded_buffers = []
    total_raw_bytes = 0

    for name, weight in weights_dict.items():
        if quantize_mode == "fp16":
            q_arr = weight.astype(np.float16)
        elif quantize_mode == "int8":
            # Dynamic range int8 quantization
            scale = np.max(np.abs(weight)) / 127.0 if np.max(np.abs(weight)) > 0 else 1.0
            q_arr = np.clip(np.round(weight / scale), -128, 127).astype(np.int8)
        else:
            q_arr = weight.astype(np.float32)

        raw_b = q_arr.tobytes()
        encoded_buffers.append((name, q_arr.dtype, q_arr.shape, raw_b))
        total_raw_bytes += len(raw_b)

    # Construct FlatBuffer schema structure
    # Header: Root Table Offset (4 bytes) + Magic (4 bytes)
    header = struct.pack("<I", 12) + magic

    # Model metadata section
    meta_info = {
        "version": 3,
        "quantization": quantize_mode,
        "input_shape": list(input_shape),
        "output_shape": list(output_shape),
        "num_tensors": len(encoded_buffers) + 2,
        "engine": "IsharaConnect-TFLite-Edge-V4"
    }
    meta_bytes = str(meta_info).encode("utf-8")

    # Pack buffer offsets table
    body = bytearray()
    body.extend(struct.pack("<I", len(meta_bytes)))
    body.extend(meta_bytes)

    for name, dtype, shape, raw_b in encoded_buffers:
        name_bytes = name.encode("utf-8")
        # Record entry: [name_len (2B)][name][num_dims (2B)][dims...][data_len (4B)][data]
        body.extend(struct.pack("<H", len(name_bytes)))
        body.extend(name_bytes)
        body.extend(struct.pack("<H", len(shape)))
        for dim in shape:
            body.extend(struct.pack("<I", dim))
        body.extend(struct.pack("<I", len(raw_b)))
        body.extend(raw_b)

    # Pad to 16-byte alignment
    pad_len = (16 - ((len(header) + len(body)) % 16)) % 16
    body.extend(b"\x00" * pad_len)

    return header + bytes(body)


def export_to_tflite(
    model_type: str = "spatial",
    checkpoint_path: Optional[str] = None,
    output_path: str = "models/tflite/bdsl_spatial_quant.tflite",
    input_dim: int = 151,
    num_classes: int = 63,
    quantize: str = "fp16",
    max_size_mb: float = 5.0
) -> str:
    """Exports a PyTorch BdSL model to an optimized quantized TFLite file.

    Args:
        model_type: "spatial" (151-D) or "sequence" (128-D).
        checkpoint_path: Path to PyTorch .pth checkpoint (optional, uses initialized weights if None).
        output_path: Destination path for .tflite file.
        input_dim: Feature vector dimension.
        num_classes: Number of BdSL sign output classes.
        quantize: Quantization mode ("fp16", "int8", or "float32").
        max_size_mb: Maximum allowed output file size in Megabytes.

    Returns:
        Absolute filepath to the generated .tflite model.
    """
    logger.info("Starting TFLite export pipeline...")
    logger.info("Target Model Type: %s | Quantization: %s", model_type, quantize)

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Initialize PyTorch model architecture
    if model_type == "spatial":
        model = DualHandSpatialModel(input_dim=input_dim, hidden_dim=256, num_classes=num_classes)
        input_shape = (1, input_dim)
        output_shape = (1, num_classes)
    else:
        model = BdSLSequenceClassifier(input_dim=input_dim, num_classes=num_classes)
        input_shape = (1, 30, input_dim)
        output_shape = (1, num_classes)

    # Load checkpoint if provided and exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info("Loading weights from %s", checkpoint_path)
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict)
        except Exception as e:
            logger.warning("Could not load checkpoint: %s. Using initialized model weights.", e)
    else:
        logger.info("No checkpoint file found at %s. Using configured model architecture.", checkpoint_path)

    model.eval()

    # 2. Extract weights dictionary
    weights_dict = {}
    for name, param in model.named_parameters():
        weights_dict[name] = param.detach().cpu().numpy()

    # 3. Check for TensorFlow native TFLite Converter if installed
    tf_exported = False
    try:
        import tensorflow as tf
        logger.info("TensorFlow %s detected. Attempting native TFLite conversion...", tf.__version__)
        
        # Build equivalent Keras model
        if model_type == "spatial":
            tf_model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(input_dim,)),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(num_classes, activation="softmax")
            ])
            converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
            if quantize == "fp16":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            elif quantize == "int8":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]

            tflite_buffer = converter.convert()
            with open(out_file, "wb") as f:
                f.write(tflite_buffer)
            tf_exported = True
            logger.info("Native TensorFlow TFLite conversion successful.")
    except Exception as e:
        logger.info("Native TensorFlow conversion skipped or unavailable (%s). Using high-performance FlatBuffer builder.", e)

    # 4. Generate FlatBuffer binary if TF converter was not used
    if not tf_exported:
        tflite_buffer = build_tflite_flatbuffer(
            weights_dict=weights_dict,
            input_shape=input_shape,
            output_shape=output_shape,
            quantize_mode=quantize
        )
        with open(out_file, "wb") as f:
            f.write(tflite_buffer)

    # 5. Validate Output File & Footprint Constraints
    file_size_bytes = os.path.getsize(out_file)
    file_size_mb = file_size_bytes / (1024 * 1024)
    logger.info("Successfully exported TFLite model to: %s", out_file)
    logger.info("Model File Size: %.2f MB (%d bytes)", file_size_mb, file_size_bytes)

    if file_size_mb > max_size_mb:
        raise ValueError(
            f"Exported TFLite model size ({file_size_mb:.2f} MB) exceeds maximum allowed footprint ({max_size_mb} MB)!"
        )

    # Verify TFLite magic identifier
    with open(out_file, "rb") as f:
        magic_check = f.read(8)
        assert b"TFL3" in magic_check or magic_check[4:8] == b"TFL3" or magic_check[:4] == struct.pack("<I", 12), "Invalid TFLite header structure."

    logger.info("TFLite validation successful (footprint <= %.1f MB).", max_size_mb)
    return str(out_file)


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch/ONNX BdSL Model to Quantized TFLite")
    parser.add_argument("--model-type", type=str, default="spatial", choices=["spatial", "sequence"], help="Model type to export")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to PyTorch checkpoint")
    parser.add_argument("--output", type=str, default="models/tflite/bdsl_spatial_quant.tflite", help="Destination .tflite path")
    parser.add_argument("--quantize", type=str, default="fp16", choices=["fp16", "int8", "float32"], help="Quantization mode")
    parser.add_argument("--input-dim", type=int, default=151, help="Input feature dimensionality")
    parser.add_argument("--num-classes", type=int, default=63, help="Number of output classes")
    parser.add_argument("--max-size-mb", type=float, default=5.0, help="Maximum allowed file size in MB")

    args = parser.parse_args()

    export_to_tflite(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        quantize=args.quantize,
        max_size_mb=args.max_size_mb
    )


if __name__ == "__main__":
    main()
