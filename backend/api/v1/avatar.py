"""FastAPI WebSocket Avatar Streaming Router for Continuous Sign Synthesis."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core_engine.dsl import (
    load_bdsl_dictionary,
    MultiSignSequenceBlender,
    BdSLV3SignSpec,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["Avatar Streaming"])
blender = MultiSignSequenceBlender(fps=60, transition_ms=150)
dictionary = load_bdsl_dictionary()

SIGNS_DIR = Path(__file__).resolve().parents[3] / "data" / "signs"


def resolve_sign_v3_spec(token: str) -> Optional[Dict[str, Any]]:
    """Resolves v3 sign specification from dictionary or signs folder."""
    clean_token = token.strip() if token else ""
    if not clean_token:
        return None

    # 1. Direct dictionary check
    sign_meta = dictionary.get(clean_token)
    if sign_meta:
        if "v3_spec" in sign_meta:
            return sign_meta["v3_spec"]
        if "phonetics" in sign_meta and "kinematics" in sign_meta:
            return sign_meta

    # 2. Check signs directory for matched JSON file
    if SIGNS_DIR.exists():
        for json_file in SIGNS_DIR.glob("*.json"):
            stem = json_file.stem.lower()
            if clean_token.lower() in stem:
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if "phonetics" in data and "kinematics" in data:
                            return data
                except Exception as e:
                    logger.warning("Error reading sign spec from %s: %s", json_file, e)

    return None


@router.websocket("/avatar_stream")
async def websocket_avatar_stream(websocket: WebSocket):
    """
    ক্লায়েন্ট থেকে বাংলা বাক্য গ্রহণ করে তাৎক্ষণিক কো-আর্টিকুলেটেড 
    60 FPS ল্যান্ডমার্ক ও FACS ফ্রেম স্ট্রিম পুশ করে।
    """
    await websocket.accept()
    try:
        while True:
            raw_data = await websocket.receive_text()
            payload = json.loads(raw_data)
            gloss_tokens = payload.get("glosses", [])

            # ১. ডিকশনারি / সাইন স্টোর থেকে v3 স্পেক সংগ্রহ
            sign_specs = []
            for token in gloss_tokens:
                spec = resolve_sign_v3_spec(token)
                if spec:
                    sign_specs.append(spec)

            if not sign_specs:
                await websocket.send_json({"status": "error", "message": "No valid v3 signs found."})
                continue

            # ২. সম্পূর্ণ বাক্যের জন্য মসৃণ ট্র্যাজেক্টরি ফ্রেম ব্লেন্ডিং
            continuous_stream = blender.blend_sentence_stream(sign_specs)

            # ৩. ফ্রেমগুলো রিয়েল-টাইমে ক্লায়েন্টে পুশ
            await websocket.send_json({
                "status": "start_stream",
                "total_frames": len(continuous_stream),
                "fps": 60
            })

            for frame in continuous_stream:
                await websocket.send_json({
                    "status": "frame",
                    "data": frame
                })
                # ৬০ এফপিএস টাইমিং সিঙ্ক (~16.6ms)
                await asyncio.sleep(1 / 60.0)

            await websocket.send_json({"status": "end_stream"})

    except WebSocketDisconnect:
        logger.info("Avatar stream client disconnected.")
