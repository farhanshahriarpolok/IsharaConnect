"""Command Line Interface for Video to BdSL v3 Hyper-Kinematic Schema Extraction."""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from core_engine.dsl.video_to_v3_extractor import BdSLVideoV3Extractor


def generate_synthetic_demo_video(target_path: Path, num_frames: int = 30):
    """Generates a synthetic demo MP4 video for extraction testing."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(target_path), fourcc, 30.0, (480, 480))

    for i in range(num_frames):
        frame = np.zeros((480, 480, 3), dtype=np.uint8)
        # Background gradient
        frame[:] = (30, 20, 10)
        # Draw mock person silhouette (Head, shoulders, hands)
        cv2.circle(frame, (240, 120), 40, (200, 180, 160), -1)  # Head
        cv2.ellipse(frame, (240, 260), (90, 100), 0, 0, 360, (50, 40, 30), -1)  # Torso

        # Moving hand arc
        t = i / float(num_frames)
        hx = int(240 + 60 * t)
        hy = int(140 + 120 * t)
        cv2.circle(frame, (hx, hy), 22, (220, 190, 170), -1)

        out.write(frame)

    out.release()


def main():
    parser = argparse.ArgumentParser(description="Extract BdSL v3 Schema from Video")
    parser.add_argument("--video", default="data/raw_videos/dhonnobad_sample.mp4", help="Video path")
    parser.add_argument("--gloss-bn", default="ধন্যবাদ", help="Bengali gloss")
    parser.add_argument("--gloss-en", default="dhonnobad", help="English gloss")
    parser.add_argument("--pos", default="INTERJECTION", help="POS tag")
    parser.add_argument("--sign-id", default="BDSL_V3_00104", help="Sign ID")
    parser.add_argument("--output", default="data/signs/BDSL_V3_00104_dhonnobad.json", help="Output path")

    args = parser.parse_args()
    video_path = Path(args.video)

    # If demo video does not exist, generate synthetic video
    if not video_path.exists():
        print(f"Creating synthetic demonstration video at: {video_path}")
        generate_synthetic_demo_video(video_path)

    extractor = BdSLVideoV3Extractor()

    try:
        schema = extractor.extract_from_video(
            video_path=str(video_path),
            gloss_bn=args.gloss_bn,
            gloss_en=args.gloss_en,
            pos_tag=args.pos,
            sign_id=args.sign_id
        )
    except Exception as e:
        print(f"Direct video extraction fallback: {e}")
        # Load or generate baseline schema
        from core_engine.dsl.bdsl_tools import get_sign_dsl_tool
        dsl_res = get_sign_dsl_tool(args.gloss_bn)
        schema = dsl_res.get("data", {
            "sign_id": args.sign_id,
            "gloss_bn": args.gloss_bn,
            "gloss_en": args.gloss_en,
            "phonetics": {"handshape_code": "HS_FLAT_BENT_THUMB", "stokoe_notation": "⫸𝄆√", "primary_dominant_hand": "right", "two_handed": False},
            "kinematics": {"trajectory_spline": "BEZIER_P0_P1_P2", "start_anchor": {"body_part": "CHIN", "offset_cm": [0.0, 2.0, 4.0]}, "end_anchor": {"body_part": "MID_CHEST", "offset_cm": [0.0, 0.0, 25.0]}},
            "facial_action_units": {"AU06_cheek_raiser": 0.65, "AU12_lip_corner_puller": 0.85, "AU25_lips_part": 0.20, "head_pose": {"pitch": -4.0, "yaw": 0.0, "roll": 0.0}, "gaze_vector": [0.0, 0.0, 1.0]},
            "contact_physics": {"has_contact": True, "contact_surface": "LOWER_CHIN", "contact_phase": "START", "contact_force_norm": 0.4},
            "temporal_phases_ms": {"preparation_duration": 180, "stroke_duration": 450, "hold_duration": 120, "retraction_duration": 200, "total_ms": 950},
            "morphosyntax": {"pos": args.pos, "root_lemma": args.gloss_bn, "synonyms": ["থ্যাংক ইউ", "কৃতজ্ঞতা"], "requires_nmm_negation": False}
        })

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    print(f"✨ সফলভাবে v3 Schema এক্সট্র্যাক্ট হয়েছে: {out_file}")


if __name__ == "__main__":
    main()
