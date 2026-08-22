"""In-App Golden Template Sign Calibrator Tool (Sprint 35).

Allows developers and administrators to calibrate golden ground-truth anatomical
specifications (body anchors, articulator modes, motion trajectories, finger flexes)
and update `dataset/lexicon/master_bdsl_lexicon.json`.

Usage:
    python scripts/calibrate_golden_sign.py --inspect-all
    python scripts/calibrate_golden_sign.py --sign ma --anchor CHEEK_RIGHT --articulator INDEX_TIP --motion TAP_TWICE
    python scripts/calibrate_golden_sign.py --validate-all
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core_engine.nlp.master_lexicon import master_lexicon, MasterBdSLLexicon

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GoldenCalibrator")

LEXICON_JSON_PATH = PROJECT_ROOT / "dataset" / "lexicon" / "master_bdsl_lexicon.json"


def load_master_json() -> Dict[str, Any]:
    """Loads master_bdsl_lexicon.json."""
    if not LEXICON_JSON_PATH.exists():
        raise FileNotFoundError(f"Lexicon file not found at {LEXICON_JSON_PATH}")
    with open(LEXICON_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_master_json(data: Dict[str, Any]) -> None:
    """Saves updated master_bdsl_lexicon.json with UTF-8 encoding and formatting."""
    with open(LEXICON_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Successfully saved updated lexicon to {LEXICON_JSON_PATH}")


def inspect_all_signs() -> None:
    """Prints a formatted summary table of all calibrated signs in the lexicon."""
    lexicon = MasterBdSLLexicon()
    signs = lexicon.get_all_signs()
    print("\n" + "=" * 90)
    print(f" 📐 ISHARACONNECT MASTER BDSL LEXICON CALIBRATION MATRIX ({len(signs)} Signs)")
    print("=" * 90)
    print(f"{'Slug':<15} {'Bangla':<10} {'Hand':<12} {'Body Anchor':<16} {'Articulator':<18} {'Motion':<15}")
    print("-" * 90)

    for s in signs:
        spec = lexicon.get_articulatory_spec(s.get("slug", ""))
        slug = spec.get("slug", "")
        bn = spec.get("label_bn", "")
        hand = spec.get("required_hand", "RIGHT_ONLY")
        anchor = spec.get("target_body_anchor", "NEUTRAL_SPACE")
        art = spec.get("articulator_type", "AUTO")
        motion = spec.get("motion_type", "STATIC_HOLD")
        print(f"{slug:<15} {bn:<10} {hand:<12} {anchor:<16} {art:<18} {motion:<15}")

    print("=" * 90 + "\n")


def calibrate_sign(
    slug: str,
    target_anchor: Optional[str] = None,
    articulator_type: Optional[str] = None,
    motion_type: Optional[str] = None,
    required_hand: Optional[str] = None
) -> bool:
    """Calibrates and updates ground-truth metadata for a single sign in master_bdsl_lexicon.json."""
    data = load_master_json()
    signs = data.get("signs", [])

    found = False
    for sign in signs:
        if sign.get("slug") == slug:
            found = True
            if target_anchor:
                sign.setdefault("contact_physics", {})["body_anchor"] = target_anchor.upper()
                sign["target_body_anchor"] = target_anchor.upper()
            if articulator_type:
                sign["articulator_type"] = articulator_type.upper()
            if motion_type:
                sign["motion_type"] = motion_type.upper()
            if required_hand:
                sign["required_hand"] = required_hand.upper()
                sign["handedness"] = "dual" if "DUAL" in required_hand.upper() else "single"

            logger.info(f"Calibrated sign '{slug}': anchor={target_anchor}, art={articulator_type}, motion={motion_type}, hand={required_hand}")
            break

    if not found:
        logger.warning(f"Sign '{slug}' not found in master_bdsl_lexicon.json.")
        return False

    save_master_json(data)
    return True


def validate_all_calibrations() -> bool:
    """Validates that all signs have non-empty anchors, articulators, handedness, and instructions."""
    lexicon = MasterBdSLLexicon()
    signs = lexicon.get_all_signs()
    errors: List[str] = []

    for s in signs:
        slug = s.get("slug", "")
        spec = lexicon.get_articulatory_spec(slug)

        if not spec.get("target_body_anchor"):
            errors.append(f"Sign '{slug}' missing target_body_anchor")
        if not spec.get("articulator_type"):
            errors.append(f"Sign '{slug}' missing articulator_type")
        if not spec.get("required_hand"):
            errors.append(f"Sign '{slug}' missing required_hand")
        if not spec.get("instructions_bn") or not isinstance(spec.get("instructions_bn"), dict):
            errors.append(f"Sign '{slug}' missing instructions_bn")

    if errors:
        logger.error(f"Validation failed with {len(errors)} issues:\n" + "\n".join(errors))
        return False

    logger.info(f"✅ All {len(signs)} signs validated successfully with complete articulatory calibration!")
    return True


def main():
    parser = argparse.ArgumentParser(description="IsharaConnect Golden Sign Calibrator Tool")
    parser.add_argument("--inspect-all", action="store_true", help="Print table of all calibrated signs")
    parser.add_argument("--validate-all", action="store_true", help="Validate all sign calibration specifications")
    parser.add_argument("--sign", type=str, help="Slug of sign to calibrate (e.g. 'ma', 'baba')")
    parser.add_argument("--anchor", type=str, help="Target body anchor (e.g. 'CHEEK_RIGHT', 'UPPER_LIP', 'FOREHEAD')")
    parser.add_argument("--articulator", type=str, help="Articulator type (e.g. 'INDEX_TIP', 'THUMB_INDEX_PINCH')")
    parser.add_argument("--motion", type=str, help="Motion type (e.g. 'TAP_TWICE', 'PULL_RIGHT', 'STATIC_HOLD')")
    parser.add_argument("--hand", type=str, help="Required hand (e.g. 'RIGHT_ONLY', 'DUAL_HAND')")

    args = parser.parse_args()

    if args.inspect_all:
        inspect_all_signs()
    elif args.validate_all:
        valid = validate_all_calibrations()
        sys.exit(0 if valid else 1)
    elif args.sign:
        success = calibrate_sign(
            slug=args.sign,
            target_anchor=args.anchor,
            articulator_type=args.articulator,
            motion_type=args.motion,
            required_hand=args.hand
        )
        sys.exit(0 if success else 1)
    else:
        inspect_all_signs()


if __name__ == "__main__":
    main()
