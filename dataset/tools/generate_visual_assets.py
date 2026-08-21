"""Visual Sign Card & Vector Anatomy Asset Generator."""

import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_svg_card(sign_data: dict, output_path: Path):
    """Generates an SVG visual flashcard for a specific sign."""
    slug = sign_data.get("slug", "unknown")
    bn_label = sign_data.get("label_bn", "অজানা")
    en_label = sign_data.get("label_en", "Unknown")
    desc = sign_data.get("description", "")
    motion = sign_data.get("motion_type", "static")
    handedness = sign_data.get("handedness", "single")
    
    # Very basic geometric SVG template representing the hand posture/card
    svg_template = f"""<svg width="400" height="500" xmlns="http://www.w3.org/2000/svg">
    <!-- Background -->
    <rect width="100%" height="100%" rx="15" fill="#1E1E2E"/>
    
    <!-- Header Text -->
    <text x="200" y="50" font-family="Arial, sans-serif" font-size="28" font-weight="bold" fill="#89B4FA" text-anchor="middle">{bn_label}</text>
    <text x="200" y="80" font-family="Arial, sans-serif" font-size="18" fill="#CDD6F4" text-anchor="middle">{en_label}</text>
    
    <!-- Central Geometric Hand Stand-in -->
    <circle cx="200" cy="220" r="80" fill="#313244" stroke="#F9E2AF" stroke-width="4"/>
    <rect x="190" y="140" width="20" height="60" rx="10" fill="#A6E3A1"/> <!-- Finger stand-in -->
    <circle cx="200" cy="140" r="10" fill="#F38BA8"/> <!-- Touch point -->
    
    <!-- Info Footer -->
    <rect x="20" y="380" width="360" height="100" rx="10" fill="#181825"/>
    <text x="40" y="410" font-family="Arial, sans-serif" font-size="14" fill="#CDD6F4" font-weight="bold">Type: {motion.title()} | {handedness.title()} Handed</text>
    
    <!-- Word-wrap isn't native to SVG 1.1 text, so we split description -->
    <text x="40" y="440" font-family="Arial, sans-serif" font-size="12" fill="#A6ADC8">
        {desc[:50]}
    </text>
    <text x="40" y="460" font-family="Arial, sans-serif" font-size="12" fill="#A6ADC8">
        {desc[50:100]}
    </text>
</svg>
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_template)

def main():
    root_dir = Path(__file__).resolve().parent.parent.parent
    labels_file = root_dir / "dataset" / "labels.json"
    output_dir = root_dir / "dataset" / "visual_cards"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not labels_file.exists():
        logger.error(f"Labels file not found at {labels_file}")
        return
        
    with open(labels_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    signs = data.get("signs", [])
    logger.info(f"Generating visual assets for {len(signs)} signs...")
    
    for sign in signs:
        slug = sign.get("slug", "unknown")
        out_file = output_dir / f"{slug}.svg"
        generate_svg_card(sign, out_file)
        
    logger.info(f"Successfully generated {len(signs)} SVG cards in {output_dir}")

if __name__ == "__main__":
    main()
