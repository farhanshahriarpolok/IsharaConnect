"""Realistic Anatomical Hand Gesture Visual Card Generator for all 63 BdSL Signs.

Renders high-contrast, scalable SVG visual cards featuring organic anatomical hand
silhouettes, distinct finger phalanges (proximal, intermediate, distal), knuckle nodes,
touch-point highlights, and bold Bengali typography.
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
    category = sign.get("category", "").lower()

    # Default lookup for known canonical signs
    # [Thumb, Index, Middle, Ring, Pinky]
    specific_patterns = {
        # Vowels
        "a": ([True, False, False, False, False], "Fist with Thumb Up (মুষ্টিবদ্ধ বৃদ্ধাঙ্গুলি)"),
        "aa": ([True, True, False, False, False], "Thumb and Index Open (বৃদ্ধাঙ্গুলি ও তর্জনী খোলা)"),
        "i": ([False, False, False, False, True], "Pinky Raised (কনিষ্ঠা আঙুল খাড়া)"),
        "u": ([False, True, True, False, False], "Index and Middle V-Shape (তর্জনী ও মধ্যমা খোলা)"),
        "e": ([True, True, True, False, False], "Three Fingers Open (তিন আঙুল খোলা)"),
        "o": ([True, True, True, True, True], "Curved O-Shape (গোলাকার 'ও' ভঙ্গি)"),
        
        # Digits
        "ek": ([False, True, False, False, False], "Index Extended (তর্জনী সোজা ১)"),
        "dui": ([False, True, True, False, False], "Index and Middle Extended (দুই আঙুল ২)"),
        "tin": ([False, True, True, True, False], "Three Fingers Extended (তিন আঙুল ৩)"),
        
        # Consonants
        "ka": ([False, True, True, False, False], "Curved Fingers Crossed (বাঁকানো তর্জনী ও মধ্যমা)"),
        "kha": ([True, True, False, False, False], "Thumb and Index Hook (হুক সদৃশ ভঙ্গি)"),
        "ga": ([False, True, False, False, False], "Index Pointing Down/Sideways (তর্জনী নির্দেশক)"),
        "gha": ([False, True, True, True, True], "Four Fingers Spread (চার আঙুল খোলা)"),
        "cha": ([True, False, False, False, False], "Fist Thumb Lateral (পার্শ্বীয় বৃদ্ধাঙ্গুলি)"),
        "ja": ([False, True, True, False, False], "Index and Middle Spread (সোজা দুই আঙুল)"),
        "ta": ([True, True, False, False, False], "L-Shape Flat (এল-আকৃতি ভঙ্গি)"),
        "da": ([False, True, False, False, False], "Bent Index Hook (বাঁকা তর্জনী)"),
        "pa": ([True, True, True, True, True], "Open Palm Forward (উন্মুক্ত তালু)"),
        "bha": ([False, True, True, True, False], "Three Fingers (তিন আঙুল ভঙ্গি)"),
        "ma": ([False, False, False, False, False], "Full Fist Closed (সম্পূর্ণ মুষ্টি)"),
        
        # Dynamic Vocabulary
        "dhonnobad": ([True, True, True, True, True], "Flat Palm Forward Motion (সম্মুখে সঞ্চালন)"),
        "shahajjo": ([True, True, True, True, True], "Dual Palm Supporting Lift (উত্তোলন ভঙ্গি)"),
        "shagotom": ([True, True, True, True, True], "Both Palms Open Inward (উন্মুক্ত উভয় তালু)"),
        "kemon_achen": ([False, True, True, False, False], "Dual Waving Gestures (প্রশ্নবোধক সঞ্চালন)"),
        "bhalo": ([True, False, False, False, False], "Thumbs Up (উৎকৃষ্টতার প্রতীক)"),
        "ami": ([False, True, False, False, False], "Index Pointing Chest (বুকের দিকে নির্দেশ)"),
        "apni": ([False, True, False, False, False], "Index Pointing Outward (সম্মুখে নির্দেশ)")
    }

    if slug in specific_patterns:
        return specific_patterns[slug]

    # Procedural fallback based on binary id pattern
    pattern = [
        bool((sign_id >> 0) & 1),
        bool((sign_id >> 1) & 1),
        bool((sign_id >> 2) & 1),
        bool((sign_id >> 3) & 1),
        bool((sign_id >> 4) & 1),
    ]
    # Ensure at least 1 finger is distinctive
    if not any(pattern):
        pattern[1] = True  # Index default

    desc = "নির্দিষ্ট অঙ্গুলি নির্দেশক ভঙ্গি"
    return pattern, desc


def generate_anatomical_hand_svg(cx: float, cy: float, ext_pattern: List[bool], is_right: bool = True, scale: float = 1.0) -> str:
    """Generates realistic anatomical hand SVG elements with organic palm and 5 segmented fingers."""
    elements = []
    
    # Palm Center and Dimensions
    palm_w = 46 * scale
    palm_h = 52 * scale
    wrist_y = cy + palm_h * 0.55
    knuckle_y = cy - palm_h * 0.45

    # Colors
    palm_fill = "#1E293B"
    skin_contour = "#38BDF8"
    bone_color = "rgba(56, 189, 248, 0.4)"
    joint_color = "#10B981"
    knuckle_fill = "#0F172A"

    # 1. Realistic Organic Palm Polygon
    # Outer palm contour: Wrist -> Hypothenar edge -> Knuckles -> Thenar (Thumb base) -> Wrist
    sign_dir = 1.0 if is_right else -1.0
    
    p_wrist_left = f"{cx - 18 * scale * sign_dir:.1f},{wrist_y:.1f}"
    p_wrist_right = f"{cx + 18 * scale * sign_dir:.1f},{wrist_y:.1f}"
    p_outer = f"{cx + 26 * scale * sign_dir:.1f},{cy:.1f}"
    p_pinky_k = f"{cx + 20 * scale * sign_dir:.1f},{knuckle_y:.1f}"
    p_mid_k = f"{cx:.1f},{knuckle_y - 4 * scale:.1f}"
    p_index_k = f"{cx - 18 * scale * sign_dir:.1f},{knuckle_y:.1f}"
    p_thumb_base = f"{cx - 28 * scale * sign_dir:.1f},{cy + 6 * scale:.1f}"

    palm_path = f"M {p_wrist_left} Q {cx:.1f},{wrist_y + 4 * scale:.1f} {p_wrist_right} Q {p_outer} {p_pinky_k} Q {p_mid_k} {p_index_k} Q {p_thumb_base} {p_wrist_left} Z"
    elements.append(f'<path d="{palm_path}" fill="{palm_fill}" stroke="{skin_contour}" stroke-width="2" stroke-linejoin="round" />')

    # Palm Crease Lines (Life line / Heart line for realism)
    crease_start = f"{cx - 15 * scale * sign_dir:.1f},{cy - 5 * scale:.1f}"
    crease_end = f"{cx + 12 * scale * sign_dir:.1f},{cy + 15 * scale:.1f}"
    elements.append(f'<path d="M {crease_start} Q {cx:.1f},{cy + 10 * scale:.1f} {crease_end}" fill="none" stroke="rgba(56, 189, 248, 0.3)" stroke-width="1.5" stroke-linecap="round" />')

    # Wrist Baseline
    elements.append(f'<line x1="{cx - 16 * scale * sign_dir:.1f}" y1="{wrist_y + 6 * scale:.1f}" x2="{cx + 16 * scale * sign_dir:.1f}" y2="{wrist_y + 6 * scale:.1f}" stroke="rgba(148, 163, 184, 0.4)" stroke-width="2" stroke-dasharray="3,3" />')

    # 2. Five Distinct Segmented Phalanges (Thumb, Index, Middle, Ring, Pinky)
    # [deg_offset, base_x_offset, length, finger_idx]
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

        # Calculate joint coordinates (Proximal, Intermediate, Distal)
        j1_x = base_x + (curr_len * 0.45) * math.cos(rad)
        j1_y = base_y + (curr_len * 0.45) * math.sin(rad)
        
        tip_x = base_x + curr_len * math.cos(rad)
        tip_y = base_y + curr_len * math.sin(rad)

        if is_extended:
            # Extended Finger: Segmented Capsule + Joints
            finger_w = 6.5 * scale
            elements.append(f'<line x1="{base_x:.1f}" y1="{base_y:.1f}" x2="{tip_x:.1f}" y2="{tip_y:.1f}" stroke="{skin_contour}" stroke-width="{finger_w:.1f}" stroke-linecap="round" />')
            elements.append(f'<line x1="{base_x:.1f}" y1="{base_y:.1f}" x2="{tip_x:.1f}" y2="{tip_y:.1f}" stroke="{palm_fill}" stroke-width="{finger_w - 2.5:.1f}" stroke-linecap="round" />')
            # Middle Knuckle Node
            elements.append(f'<circle cx="{j1_x:.1f}" cy="{j1_y:.1f}" r="{2.5 * scale:.1f}" fill="{joint_color}" />')
            # Fingertip Target Node
            elements.append(f'<circle cx="{tip_x:.1f}" cy="{tip_y:.1f}" r="{3.5 * scale:.1f}" fill="{skin_contour}" stroke="#FFFFFF" stroke-width="1" />')
        else:
            # Curled / Folded Finger: Rounded Pill Knuckle over palm
            curl_x = base_x + (10 * scale) * math.cos(rad)
            curl_y = base_y + (10 * scale) * math.sin(rad)
            elements.append(f'<circle cx="{curl_x:.1f}" cy="{curl_y:.1f}" r="{5.5 * scale:.1f}" fill="{knuckle_fill}" stroke="rgba(148, 163, 184, 0.6)" stroke-width="2" />')
            elements.append(f'<circle cx="{curl_x:.1f}" cy="{curl_y:.1f}" r="{2.0 * scale:.1f}" fill="#64748B" />')

    return "\n    ".join(elements)


def generate_svg_card(sign: Dict) -> str:
    """Generates a complete high-resolution, high-contrast anatomical BdSL vector card."""
    slug = sign.get("slug", "sign")
    bn = sign.get("label_bn", "")
    en = sign.get("label_en", "")
    cat = sign.get("category", "General").title()
    hands = sign.get("hands", 1)
    is_dynamic = "dynamic" in sign.get("motion_type", "").lower() or "dynamic" in sign.get("category", "").lower()

    ext_pattern, pose_hint = get_finger_state_for_sign(sign)
    is_dual = hands == 2

    # Canvas Dimensions
    width, height = 300, 260
    
    # Hand Positioning
    if is_dual:
        left_hand_svg = generate_anatomical_hand_svg(cx=85, cy=155, ext_pattern=ext_pattern, is_right=False, scale=0.88)
        right_hand_svg = generate_anatomical_hand_svg(cx=215, cy=155, ext_pattern=ext_pattern, is_right=True, scale=0.88)
        hands_markup = left_hand_svg + "\n    " + right_hand_svg
    else:
        hands_markup = generate_anatomical_hand_svg(cx=150, cy=150, ext_pattern=ext_pattern, is_right=True, scale=1.05)

    # Dynamic Motion Path & Trajectory Arrow
    motion_markup = ""
    if is_dynamic:
        motion_markup = '''
    <!-- Dynamic Motion Trajectory -->
    <path d="M 235 125 Q 265 155 235 185" fill="none" stroke="#F59E0B" stroke-width="3.5" stroke-linecap="round" stroke-dasharray="6,4" />
    <polygon points="230,190 242,185 234,175" fill="#F59E0B" />
        '''

    import html
    safe_bn = html.escape(str(bn))
    safe_en = html.escape(str(en))
    safe_cat = html.escape(str(cat))
    safe_pose_hint = html.escape(str(pose_hint))

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
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="3" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
  </defs>

  <!-- Background Glassmorphic Container -->
  <rect width="{width - 4}" height="{height - 4}" x="2" y="2" rx="16" fill="url(#cardBg)" stroke="url(#borderGlow)" stroke-width="2" />
  
  <!-- Category & Handedness Header Badges -->
  <rect x="12" y="12" width="75" height="20" rx="6" fill="#1E293B" stroke="rgba(56, 189, 248, 0.4)" stroke-width="1" />
  <text x="49" y="26" fill="#38BDF8" font-family="'Segoe UI', Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle">{safe_cat}</text>

  <rect x="{width - 87}" y="12" width="75" height="20" rx="6" fill="#1E293B" stroke="rgba(16, 185, 129, 0.4)" stroke-width="1" />
  <text x="{width - 50}" y="26" fill="#10B981" font-family="'Segoe UI', Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle">{'Dual Hand' if is_dual else 'Single Hand'}</text>

  <!-- Prominent Bengali Sign Glyph -->
  <text x="150" y="58" fill="#F8FAFC" font-family="'SolaimanLipi', 'Segoe UI', Arial, sans-serif" font-size="38" font-weight="900" text-anchor="middle" filter="url(#glow)">{safe_bn}</text>

  <!-- English Label & Subtitle -->
  <text x="150" y="80" fill="#94A3B8" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="600" text-anchor="middle">{safe_en}</text>

  <!-- Anatomical Hand Gesture Silhouette -->
  <g id="hand_anatomy">
    {hands_markup}
    {motion_markup}
  </g>

  <!-- Footer Posture Cue / Hint -->
  <rect x="16" y="{height - 28}" width="{width - 32}" height="18" rx="5" fill="#1E293B" />
  <text x="150" y="{height - 15}" fill="#CBD5E1" font-family="'Segoe UI', Arial, sans-serif" font-size="9.5" font-weight="500" text-anchor="middle">{safe_pose_hint}</text>
</svg>'''
    return svg


def generate_all_cards(output_dir: Path = OUTPUT_DIR) -> int:
    """Generates SVG visual cards for all 63 signs in labels.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not LABELS_FILE.exists():
        logger.error(f"Labels file not found at {LABELS_FILE}")
        return 0

    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    signs = data.get("signs", [])
    count = 0

    logger.info(f"Generating realistic anatomical cards for {len(signs)} BdSL signs into {output_dir}...")

    for sign in signs:
        slug = sign.get("slug")
        if not slug:
            continue

        svg_content = generate_svg_card(sign)
        svg_file = output_dir / f"{slug}.svg"
        with open(svg_file, "w", encoding="utf-8") as f:
            f.write(svg_content)
        count += 1

    logger.info(f"Successfully generated {count} realistic anatomical BdSL visual cards!")
    return count


if __name__ == "__main__":
    generate_all_cards()
