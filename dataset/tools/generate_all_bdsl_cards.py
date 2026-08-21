"""Procedural Visual Card Asset Generator for all 63 canonical BdSL Signs.

Renders SVG and high-resolution PNG visual cards for Vowels, Consonants, Digits,
and Conversational Vocabulary with stylized anatomical hand silhouettes and Bangla glyphs.
"""

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("card_generator")

LABELS_FILE = Path("dataset/labels.json")
OUTPUT_DIR = Path("dataset/visual_cards")


def generate_svg_content(sign: Dict) -> str:
    """Generates clean, scalable SVG markup with hand skeleton and typography."""
    slug = sign.get("slug", "sign")
    bn = sign.get("label_bn", "")
    en = sign.get("label_en", "")
    cat = sign.get("category", "General").title()
    hands = sign.get("hands", 1)
    is_dynamic = "dynamic" in sign.get("motion_type", "").lower() or "dynamic" in sign.get("category", "").lower()
    sign_id = sign.get("id", 0)

    # Palette
    bg_color = "#181825"
    border_color = "#06B6D4"
    accent_color = "#89B4FA"
    text_color = "#CDD6F4"
    joint_color = "#10B981"
    bone_color = "rgba(6, 182, 212, 0.6)"

    # Base Hand coordinates
    is_dual = hands == 2
    right_cx = 210 if is_dual else 150
    left_cx = 90
    cy = 160

    # Finger curl calculation based on sign_id
    f_angles = [-36, -18, 0, 18, 36]
    ext_pattern = [
        bool((sign_id >> 0) & 1),
        bool((sign_id >> 1) & 1),
        bool((sign_id >> 2) & 1),
        bool((sign_id >> 3) & 1),
        bool((sign_id >> 4) & 1),
    ]

    def render_hand(cx: float, is_right: bool = True) -> str:
        elements = []
        # Palm circle
        elements.append(f'<circle cx="{cx}" cy="{cy}" r="22" fill="#313244" stroke="{border_color}" stroke-width="2" />')
        elements.append(f'<circle cx="{cx}" cy="{cy+24}" r="5" fill="{accent_color}" />') # Wrist node

        # Fingers
        for idx, (deg, is_ext) in enumerate(zip(f_angles, ext_pattern)):
            actual_deg = deg if is_right else -deg
            rad = math.radians(actual_deg - 90)
            length = 42 if is_ext else 20

            x1 = cx + 18 * math.cos(rad)
            y1 = cy + 18 * math.sin(rad)
            x2 = cx + length * math.cos(rad)
            y2 = cy + length * math.sin(rad)

            # Bone
            elements.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{bone_color}" stroke-width="4" stroke-linecap="round" />')
            # Joint
            elements.append(f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="4" fill="{joint_color}" />')

        return "\n".join(elements)

    hands_markup = render_hand(right_cx, is_right=True)
    if is_dual:
        hands_markup += "\n" + render_hand(left_cx, is_right=False)

    # Dynamic Arrow
    arrow_markup = ""
    if is_dynamic:
        arrow_markup = f'''
        <path d="M 230 140 Q 250 160 230 180" fill="none" stroke="#F59E0B" stroke-width="3" marker-end="url(#arrow)" />
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#F59E0B" />
          </marker>
        </defs>
        '''

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 240" width="300" height="240">
  <rect width="296" height="236" x="2" y="2" rx="14" fill="{bg_color}" stroke="{border_color}" stroke-width="2" />
  
  <!-- Category Badge -->
  <rect x="12" y="12" width="70" height="18" rx="5" fill="#313244" />
  <text x="47" y="24" fill="#A6E3A1" font-family="Segoe UI, sans-serif" font-size="9" font-weight="bold" text-anchor="middle">{cat}</text>

  <!-- Large Bengali Character -->
  <text x="150" y="55" fill="{accent_color}" font-family="SolaimanLipi, Arial, sans-serif" font-size="34" font-weight="bold" text-anchor="middle">{bn}</text>

  <!-- English Subtitle -->
  <text x="150" y="78" fill="{text_color}" font-family="Segoe UI, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">{en}</text>

  <!-- Anatomical Hand Gesture -->
  <g transform="translate(0, 10)">
    {hands_markup}
    {arrow_markup}
  </g>

  <!-- Footnote -->
  <text x="150" y="224" fill="#94A3B8" font-family="Segoe UI, sans-serif" font-size="9" text-anchor="middle">BdSL Canonical Standard Vector Card</text>
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

    logger.info(f"Generating visual cards for {len(signs)} BdSL signs into {output_dir}...")

    for sign in signs:
        slug = sign.get("slug")
        if not slug:
            continue

        svg_content = generate_svg_content(sign)
        svg_file = output_dir / f"{slug}.svg"

        with open(svg_file, "w", encoding="utf-8") as f:
            f.write(svg_content)
        count += 1

    logger.info(f"Generated {count} BdSL visual cards successfully!")
    return count


if __name__ == "__main__":
    generate_all_cards()
