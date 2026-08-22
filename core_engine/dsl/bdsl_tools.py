"""BdSL Parametric DSL and Landmark Retrieval Tools for Antigravity AI Agent.

Provides agent tools:
1. `get_sign_dsl_tool(gloss: str)`: Look up BdSL Parametric DSL, handshape, trajectory, orientation, and grammar metadata.
2. `get_sign_landmarks_tool(sign_name: str)`: Retrieve frame-by-frame 2D/3D skeletal landmark coordinates for rendering.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Resolve Path configurations with multi-fallback support
BASE_DIR = Path(__file__).resolve().parent.parent.parent

CANDIDATE_DICT_PATHS = [
    BASE_DIR / "data" / "bdsl_dictionary.json",
    BASE_DIR / "dataset" / "bdsl_dictionary.json",
    BASE_DIR / "dataset" / "labels.json",
]

CANDIDATE_LANDMARK_DIRS = [
    BASE_DIR / "data" / "landmarks",
    BASE_DIR / "dataset" / "landmarks",
    BASE_DIR / "dataset" / "spatial_landmarks",
]

DICTIONARY_PATH = next((p for p in CANDIDATE_DICT_PATHS if p.exists()), BASE_DIR / "data" / "bdsl_dictionary.json")
LANDMARKS_DIR = next((p for p in CANDIDATE_LANDMARK_DIRS if p.exists()), BASE_DIR / "data" / "landmarks")


def load_bdsl_dictionary() -> Dict[str, Any]:
    """Loads bdsl_dictionary.json or dataset/labels.json into memory."""
    for p in CANDIDATE_DICT_PATHS:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # If loaded from labels.json format with 'signs' array, index by slug and label_bn
                    if isinstance(data, dict) and "signs" in data and isinstance(data["signs"], list):
                        indexed = {}
                        for sign in data["signs"]:
                            slug = sign.get("slug")
                            bn = sign.get("label_bn")
                            if slug:
                                indexed[slug] = sign
                            if bn:
                                indexed[bn] = sign
                        return indexed
                    return data
            except Exception as e:
                logger.warning("Failed to load dictionary from %s: %s", p, e)
    return {}


BDSL_DB: Dict[str, Any] = load_bdsl_dictionary()


def get_sign_dsl_tool(gloss: str) -> Dict[str, Any]:
    """
    Look up the BdSL Parametric DSL and metadata for a given sign/gloss.

    Args:
        gloss: Bengali text (e.g. 'ধন্যবাদ', 'আমি') or English slug (e.g. 'dhonnobad', 'ami')

    Returns:
        Structured dictionary containing sign DSL configuration, handshape, motion, and metadata.
    """
    clean_gloss = gloss.strip() if gloss else ""
    sign_data = BDSL_DB.get(clean_gloss)
    if not sign_data:
        # Case-insensitive or stripped lookup
        for k, v in BDSL_DB.items():
            if k.lower() == clean_gloss.lower():
                sign_data = v
                break

    if not sign_data:
        return {"status": "not_found", "message": f"Sign '{gloss}' not in dictionary"}

    return {"status": "success", "gloss": gloss, "data": sign_data}


def get_sign_landmarks_tool(sign_name: str) -> Dict[str, Any]:
    """
    Retrieve frame-by-frame 2D/3D skeletal landmark coordinates for rendering.

    Args:
        sign_name: Sign slug or name (e.g. 'dhonnobad')

    Returns:
        Structured landmark frame list or error status.
    """
    slug = "".join(c if c.isalnum() or c in "_-" else "_" for c in sign_name).strip("_")

    for l_dir in CANDIDATE_LANDMARK_DIRS:
        if not l_dir.exists():
            continue

        # 1. Direct JSON file match
        file_path = l_dir / f"{slug}.json"
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    landmarks_data = json.load(f)
                return {"status": "success", "sign": slug, "landmarks": landmarks_data}
            except Exception as e:
                logger.warning("Error reading %s: %s", file_path, e)

        # 2. Directory match (e.g., spatial_landmarks/<slug>/sample_0000.npy)
        slug_dir = l_dir / slug
        if slug_dir.is_dir():
            npy_files = list(slug_dir.glob("*.npy"))
            if npy_files:
                try:
                    import numpy as np
                    sample_file = npy_files[0]
                    arr = np.load(sample_file)
                    return {
                        "status": "success",
                        "sign": slug,
                        "landmarks": {
                            "sign": slug,
                            "source_file": sample_file.name,
                            "shape": list(arr.shape),
                            "frames": arr.tolist()
                        }
                    }
                except Exception as e:
                    logger.warning("Error reading npy landmarks for %s: %s", slug, e)

    return {"status": "not_found", "message": f"Landmark file for '{sign_name}' not found"}
