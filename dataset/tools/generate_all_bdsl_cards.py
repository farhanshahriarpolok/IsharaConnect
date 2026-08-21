"""Clean, High-Fidelity Anatomical Hand Gesture Visual Card Generator for BdSL.

Renders clean, distraction-free SVG visual cards focusing 100% on high-contrast,
organic anatomical hand silhouettes, distinct multi-segment phalanges, glowing
knuckle nodes, and neon direction arrows.
"""

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("card_generator")

LABELS_FILE = Path("dataset/labels.json")
OUTPUT_DIR = Path("dataset/visual_cards")


def get_finger_state_for_sign(sign: Dict) -> Tuple[List[bool], str]:
    """Determines finger extension pattern [Thumb, Index, Middle, Ring, Pinky] and pose hint."""
    slug = sign.get("slug", "").lower()
    sign_id = sign.get("id", 0)

    specific_patterns = {
        # Vowels
        "a": ([True, False, False, False, False], "Fist with Thumb Up"),
        "aa": ([True, True, False, False, False], "Thumb and Index Open"),
        "i": ([False, False, False, False, True], "Pinky Raised"),
        "u": ([False, True, True, False, False], "Index and Middle V-Shape"),
        "e": ([True, True, True, False, False], "Three Fingers Open"),
        "o": ([True, True, True, True, True], "Curved O-Shape"),
        
        # Digits
        "ek": ([False, True, False, False, False], "Index Extended"),
        "dui": ([False, True, True, False, False], "Index and Middle Extended"),
        "tin": ([False, True, True, True, False], "Three Fingers Extended"),
        
        # Consonants
        "ka": ([False, True, True, False, False], "Curved Fingers Crossed"),
        "kha": ([True, True, False, False, False], "Thumb and Index Hook"),
        "ga": ([False, True, False, False, False], "Index Pointing Down"),
        "gha": ([False, True, True, True, True], "Four Fingers Spread"),
        "cha": ([True, False, False, False, False], "Fist Thumb Lateral"),
        "ja": ([False, True, True, False, False], "Index and Middle Spread"),
        "ta": ([True, True, False, False, False], "L-Shape Flat"),
        "da": ([False, True, False, False, False], "Bent Index Hook"),
        "pa": ([True, True, True, True, True], "Open Palm Forward"),
        "bha": ([False, True, True, True, False], "Three Fingers"),
        "ma": ([False, False, False, False, False], "Full Fist Closed"),
        
        # Dynamic Vocabulary
        "dhonnobad": ([True, True, True, True, True], "Flat Palm Forward Motion"),
        "shahajjo": ([True, True, True, True, True], "Dual Palm Supporting Lift"),
        "shagotom": ([True, True, True, True, True], "Both Palms Open Inward"),
        "kemon_achen": ([False, True, True, False, False], "Dual Waving Gestures"),
        "bhalo": ([True, False, False, False, False], "Thumbs Up"),
        "ami": ([False, True, False, False, False], "Index Pointing Chest"),
        "apni": ([False, True, False, False, False], "Index Pointing Outward")
    }

    if slug in specific_patterns:
        return specific_patterns[slug]

    pattern = [
        bool((sign_id >> 0) & 1),
        bool((sign_id >> 1) & 1),
        bool((sign_id >> 2) & 1),
        bool((sign_id >> 3) & 1),
        bool((sign_id >> 4) & 1),
    ]
    if not any(pattern):
        pattern[1] = True

    return pattern, "Standard BdSL Gesture"


def generate_anatomical_hand_svg(cx: float, cy: float, ext_pattern: List[bool], is_right: bool = True, scale: float = 1.0) -> str:
    """Generates realistic anatomical hand SVG elements with organic palm and 5 segmented fingers."""
    elements = []
    
    # Palm Center and Dimensions
    palm_w = 46 * scale
    palm_h = 50 * scale
    wrist_y = cy + palm_h * 0.55
    knuckle_y = cy - palm_h * 0.45

    # Palette
    palm_fill = "#1E293B"
    skin_contour = "#38BDF8"
    joint_color = "#10B981"
    knuckle_fill = "#0F172A"

    sign_dir = 1.0 if is_right else -1.0
    
    p_wrist_left = f"{cx - 18 * scale * sign_dir:.1f},{wrist_y:.1f}"
    p_wrist_right = f"{cx + 18 * scale * sign_dir:.1f},{wrist_y:.1f}"
    p_outer = f"{cx + 26 * scale * sign_dir:.1f},{cy:.1f}"
    p_pinky_k = f"{cx + 20 * scale * sign_dir:.1f},{knuckle_y:.1f}"
    p_mid_k = f"{cx:.1f},{knuckle_y - 4 * scale:.1f}"
    p_index_k = f"{cx - 18 * scale * sign_dir:.1f},{knuckle_y:.1f}"
    p_thumb_base = f"{cx - 28 * scale * sign_dir:.1f},{cy + 6 * scale:.1f}"

    # 1. Palm Silhouette
    palm_path = f"M {p_wrist_left} Q {cx:.1f},{wrist_y + 4 * scale:.1f} {p_wrist_right} Q {p_outer} {p_pinky_k} Q {p_mid_k} {p_index_k} Q {p_thumb_base} {p_wrist_left} Z"
    elements.append(f'<path d="{palm_path}" fill="{palm_fill}" stroke="{skin_contour}" stroke-width="2.2" stroke-linejoin="round" />')

    # Palm Crease
    crease_start = f"{cx - 15 * scale * sign_dir:.1f},{cy - 5 * scale:.1f}"
    crease_end = f"{cx + 12 * scale * sign_dir:.1f},{cy + 15 * scale:.1f}"
    elements.append(f'<path d="M {crease_start} Q {cx:.1f},{cy + 10 * scale:.1f} {crease_end}" fill="none" stroke="rgba(56, 189, 248, 0.35)" stroke-width="1.5" stroke-linecap="round" />')

    # Wrist Baseline
    elements.append(f'<line x1="{cx - 16 * scale * sign_dir:.1f}" y1="{wrist_y + 6 * scale:.1f}" x2="{cx + 16 * scale * sign_dir:.1f}" y2="{wrist_y + 6 * scale:.1f}" stroke="rgba(148, 163, 184, 0.4)" stroke-width="2" stroke-dasharray="3,3" />')

    # 2. 5 Multi-segmented Fingers
    fingers_config = [
        (-42, -24, 34, 0),  # Thumb
        (-18, -16, 44, 1),  # Index
        (0, -1, 50, 2),     # Middle
        (18, 14, 45, 3),    # Ring
        (36, 24, 36, 4)     # Pinky
    ]

    for deg_offset, x_offset, length, f_idx in fingers_config:
        is_extended = ext_pattern[f_idx]
        actual_deg = deg_offset * sign_dir
        base_x = cx + x_offset * scale * sign_dir
        base_y = cy - (10 * scale if f_idx == 0 else palm_h * 0.4)

        rad = math.radians(actual_deg - 90)
        curr_len = length * scale if is_extended else 16 * scale

        j1_x = base_x + (curr_len * 0.45) * math.cos(rad)
        j1_y = base_y + (curr_len * 0.45) * math.sin(rad)
        
        tip_x = base_x + curr_len * math.cos(rad)
        tip_y = base_y + curr_len * math.sin(rad)

        if is_extended:
            finger_w = 6.5 * scale
            elements.append(f'<line x1="{base_x:.1f}" y1="{base_y:.1f}" x2="{tip_x:.1f}" y2="{tip_y:.1f}" stroke="{skin_contour}" stroke-width="{finger_w:.1f}" stroke-linecap="round" />')
            elements.append(f'<line x1="{base_x:.1f}" y1="{base_y:.1f}" x2="{tip_x:.1f}" y2="{tip_y:.1f}" stroke="{palm_fill}" stroke-width="{finger_w - 2.5:.1f}" stroke-linecap="round" />')
            elements.append(f'<circle cx="{j1_x:.1f}" cy="{j1_y:.1f}" r="{2.5 * scale:.1f}" fill="{joint_color}" />')
            elements.append(f'<circle cx="{tip_x:.1f}" cy="{tip_y:.1f}" r="{3.5 * scale:.1f}" fill="{skin_contour}" stroke="#FFFFFF" stroke-width="1" />')
        else:
            curl_x = base_x + (10 * scale) * math.cos(rad)
            curl_y = base_y + (10 * scale) * math.sin(rad)
            elements.append(f'<circle cx="{curl_x:.1f}" cy="{curl_y:.1f}" r="{5.5 * scale:.1f}" fill="{knuckle_fill}" stroke="rgba(148, 163, 184, 0.6)" stroke-width="2" />')
            elements.append(f'<circle cx="{curl_x:.1f}" cy="{curl_y:.1f}" r="{2.0 * scale:.1f}" fill="#64748B" />')

    return "\n    ".join(elements)


def generate_svg_card(sign: Dict) -> str:
    """Generates a distraction-free, high-fidelity anatomical BdSL hand gesture card."""
    slug = sign.get("slug", "sign")
    hands = sign.get("hands", 1)
    is_dynamic = "dynamic" in sign.get("motion_type", "").lower() or "dynamic" in sign.get("category", "").lower()

    ext_pattern, _ = get_finger_state_for_sign(sign)
    is_dual = hands == 2

    width, height = 280, 220
    
    if is_dual:
        left_hand_svg = generate_anatomical_hand_svg(cx=80, cy=125, ext_pattern=ext_pattern, is_right=False, scale=0.92)
        right_hand_svg = generate_anatomical_hand_svg(cx=200, cy=125, ext_pattern=ext_pattern, is_right=True, scale=0.92)
        hands_markup = left_hand_svg + "\n    " + right_hand_svg
    else:
        hands_markup = generate_anatomical_hand_svg(cx=140, cy=120, ext_pattern=ext_pattern, is_right=True, scale=1.15)

    motion_markup = ""
    if is_dynamic:
        motion_markup = '''
    <!-- Dynamic Directional Trajectory -->
    <path d="M 225 90 Q 255 120 225 150" fill="none" stroke="#F59E0B" stroke-width="3.5" stroke-linecap="round" stroke-dasharray="6,4" />
    <polygon points="220,155 232,150 224,140" fill="#F59E0B" />
        '''

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <defs>
    <linearGradient id="cardBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0F172A" />
      <stop offset="100%" stop-color="#181825" />
    </linearGradient>
    <linearGradient id="borderGlow" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#06B6D4" />
      <stop offset="50%" stop-color="#38BDF8" />
      <stop offset="100%" stop-color="#818CF8" />
    </linearGradient>
  </defs>

  <!-- Clean Container Frame -->
  <rect width="{width - 4}" height="{height - 4}" x="2" y="2" rx="14" fill="url(#cardBg)" stroke="url(#borderGlow)" stroke-width="2" />
  
  <!-- Subtle Handedness Chip -->
  <rect x="12" y="12" width="70" height="18" rx="5" fill="#1E293B" stroke="rgba(56, 189, 248, 0.3)" stroke-width="1" />
  <text x="47" y="24" fill="#38BDF8" font-family="'Segoe UI', Arial, sans-serif" font-size="9" font-weight="bold" text-anchor="middle">{'Dual Hand' if is_dual else 'Single Hand'}</text>

  <!-- High-Fidelity Anatomical Hand Model -->
  <g id="hand_anatomy">
    {hands_markup}
    {motion_markup}
  </g>
</svg>'''
    return svg


def generate_all_cards(output_dir: Path = OUTPUT_DIR) -> int:
    """Generates clean SVG visual cards for all 63 signs in labels.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not LABELS_FILE.exists():
        logger.error(f"Labels file not found at {LABELS_FILE}")
        return 0

    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    signs = data.get("signs", [])
    count = 0

    logger.info(f"Generating clean anatomical cards for {len(signs)} BdSL signs into {output_dir}...")

    for sign in signs:
        slug = sign.get("slug")
        if not slug:
            continue

        svg_content = generate_svg_card(sign)
        svg_file = output_dir / f"{slug}.svg"
        with open(svg_file, "w", encoding="utf-8") as f:
            f.write(svg_content)
        count += 1

    logger.info(f"Successfully generated {count} clean anatomical BdSL visual cards!")
    return count


if __name__ == "__main__":
    generate_all_cards()
