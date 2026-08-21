"""Benchmark Script for ONNX Model Inference."""

import json
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import onnxruntime as ort
from sklearn.metrics import classification_report, confusion_matrix
import tracemalloc

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("benchmark")

def load_dataset(dataset_dir: str):
    dataset_path = Path(dataset_dir)
    X = []
    y = []
    
    npy_files = list(dataset_path.glob("**/*.npy"))
    if not npy_files:
        logger.warning("No real data found in %s. Using synthetic for benchmark.", dataset_dir)
        # Synthetic data
        for c in range(5):
            for _ in range(20):
                seq = np.random.randn(30, 128).astype(np.float32)
                X.append(seq)
                y.append(c)
        return np.array(X), np.array(y)

    for f in npy_files:
        try:
            label = int(f.parent.name)
            seq = np.load(f)
            if seq.shape[-1] != 128:
                padded = np.zeros((30, 128), dtype=np.float32)
                padded[:, :seq.shape[-1]] = seq
                seq = padded
            X.append(seq)
            y.append(label)
        except:
            continue
            
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def run_benchmark(model_path: str, dataset_dir: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if not Path(model_path).exists():
        logger.error("ONNX model %s not found.", model_path)
        return
        
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    logger.info("Loading dataset...")
    X, y_true = load_dataset(dataset_dir)
    logger.info("Dataset loaded. Shape: %s", X.shape)
    
    latencies = []
    y_pred = []
    
    logger.info("Starting Inference Benchmark...")
    
    tracemalloc.start()
    
    # Run predictions
    for i in range(len(X)):
        input_data = np.expand_dims(X[i], axis=0) # (1, 30, 128)
        
        start_t = time.perf_counter()
        ort_outs = session.run(None, {input_name: input_data})
        end_t = time.perf_counter()
        
        latencies.append((end_t - start_t) * 1000.0) # ms
        
        logits = ort_outs[0][0]
        pred_class = int(np.argmax(logits))
        y_pred.append(pred_class)
        
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Calculate Latency Statistics
    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    fps = 1000.0 / np.mean(latencies)
    
    # Metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Save Confusion Matrix
    cm_path = Path(output_dir) / "confusion_matrix.json"
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump(cm.tolist(), f)
        
    # Write Markdown Report
    report_path = Path(output_dir) / "model_evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Model Evaluation Report\n\n")
        f.write("## Latency Profile\n")
        f.write(f"- **p50 Latency:** {p50:.2f} ms\n")
        f.write(f"- **p95 Latency:** {p95:.2f} ms\n")
        f.write(f"- **p99 Latency:** {p99:.2f} ms\n")
        f.write(f"- **Effective Throughput:** {fps:.2f} FPS\n")
        f.write(f"- **Peak Memory Usage:** {peak / 1024 / 1024:.2f} MB\n\n")
        
        f.write("## Overall Metrics\n")
        f.write(f"- **Accuracy:** {report.get('accuracy', 0):.4f}\n")
        macro = report.get('macro avg', {})
        f.write(f"- **Macro Precision:** {macro.get('precision', 0):.4f}\n")
        f.write(f"- **Macro Recall:** {macro.get('recall', 0):.4f}\n")
        f.write(f"- **Macro F1-Score:** {macro.get('f1-score', 0):.4f}\n\n")
        
        f.write("## Per-Class Accuracy\n")
        f.write("| Class ID | Precision | Recall | F1-Score | Support |\n")
        f.write("|----------|-----------|--------|----------|---------|\n")
        for key, metrics in report.items():
            if key.isdigit():
                f.write(f"| {key} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1-score']:.4f} | {metrics['support']} |\n")
                
    logger.info("Benchmark complete. Report saved to %s", report_path)
    logger.info("Confusion matrix saved to %s", cm_path)
    
    # Print summary
    logger.info("=== SUMMARY ===")
    logger.info("F1 (Macro): %.4f", macro.get('f1-score', 0))
    logger.info("p95 Latency: %.2f ms", p95)
    logger.info("Throughput: %.2f FPS", fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/onnx/bdsl_model.onnx")
    parser.add_argument("--dataset", type=str, default="dataset/raw_landmarks")
    parser.add_argument("--output", type=str, default="docs/benchmarks")
    args = parser.parse_args()
    
    run_benchmark(args.model, args.dataset, args.output)
