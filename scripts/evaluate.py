"""Model evaluation and metrics script for IsharaConnect."""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BdSL Model Accuracy & Latency")
    parser.add_argument("--model-path", type=str, required=True, help="Path to ONNX or PyTorch model")
    parser.add_argument("--test-dir", type=str, default="dataset/processed/test", help="Path to test dataset")
    parser.add_argument("--benchmark-latency", action="store_true", help="Benchmark per-frame inference latency")

    args = parser.parse_args()
    logger.info("Evaluating model %s on test dataset %s", args.model_path, args.test_dir)


if __name__ == "__main__":
    main()
