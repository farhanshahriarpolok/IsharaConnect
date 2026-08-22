"""FastAPI Main Application for IsharaConnect Backend."""

import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import asyncio
from pydantic import BaseModel
from aiortc import RTCPeerConnection, RTCSessionDescription
import numpy as np

from backend.webrtc.track_processor import SignLanguageTrackProcessor
from core_engine.inference.cslr_engine import IsharaInferenceEngine, SlidingWindowBuffer

from backend.api.routes import router as api_router
from backend.api.admin_routes import router as admin_router
from backend.routers.room_router import router as room_router
from backend.websockets.room_manager import manager, ClientType
from backend.core.config import settings

# Mega Sprint 19 new routers
from backend.api.v1.auth import router as auth_router
from backend.api.v1.users import router as users_router
from backend.api.v1.progress import router as progress_router
from backend.api.v1.exams import router as exams_router
from backend.api.v1.certificates import router as certificates_router
from backend.api.v1.rooms import router as rooms_router
from backend.api.v1.nlp import router as nlp_router
from backend.api.v1.avatar import router as avatar_router
from backend.db.session import get_async_db
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Request, Depends, Query
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="IsharaConnect Real-Time Backend",
    description="Real-Time Bangla Sign Language Two-Way Communication Platform",
    version="1.0.0"
)

# CORS middleware for local/web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(admin_router, prefix="/api/v1")
app.include_router(room_router)

from backend.routers.dashboard import router as dashboard_router

# Include Mega Sprint 19, 22, 23, 36 & Avatar Streaming routers
app.include_router(auth_router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(users_router, prefix="/api/v1/users", tags=["Users"])
app.include_router(progress_router, prefix="/api/v1/progress", tags=["Progress"])
app.include_router(exams_router, prefix="/api/v1/exams", tags=["Exams"])
app.include_router(certificates_router, prefix="/api/v1/certificates", tags=["Certificates"])
app.include_router(rooms_router, prefix="/api/v1/rooms", tags=["Rooms"])
app.include_router(nlp_router, prefix="/api/v1/nlp", tags=["NLP Translation"])
app.include_router(avatar_router)
app.include_router(dashboard_router)

# Public Web Verification Endpoints (QR Code Target)
@app.get("/verify/{cert_hash}", response_class=HTMLResponse, tags=["Public Verification"])
async def public_verify_page(
    request: Request,
    cert_hash: str,
    hash: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_async_db)
):
    from backend.api.v1.certificates import verify_certificate_html
    return await verify_certificate_html(request, cert_hash, hash=hash, db=db)


@app.get("/admin/verify-certificate/{cert_id}", response_class=HTMLResponse, tags=["Public Verification"])
async def admin_verify_page_alias(
    request: Request,
    cert_id: str,
    hash: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_async_db)
):
    from backend.api.v1.certificates import verify_certificate_html
    return await verify_certificate_html(request, cert_id, hash=hash, db=db)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

engine = IsharaInferenceEngine()
pcs = set()

class WebRTCOffer(BaseModel):
    sdp: str
    type: str

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def serve_root():
    index_path = TEMPLATES_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>IsharaConnect Backend Running</h1>")


@app.get("/player", response_class=HTMLResponse)
async def serve_skeleton_player():
    p = TEMPLATES_DIR / "skeleton_player.html"
    if p.exists():
        return FileResponse(p)
    return HTMLResponse("<h1>Skeleton Player</h1>")


@app.get("/avatar", response_class=HTMLResponse)
async def serve_avatar_viewport():
    p = TEMPLATES_DIR / "avatar_viewport.html"
    if p.exists():
        return FileResponse(p)
    return HTMLResponse("<h1>Avatar Viewport</h1>")


@app.get("/health")
async def health_check():
    """Ultra-fast root health endpoint for launcher and health probes."""
    return {"status": "healthy", "service": "IsharaConnect-Backend"}


# ----------------- WebRTC Endpoint -----------------
@app.post("/offer")
async def webrtc_offer(offer_data: WebRTCOffer):
    offer = RTCSessionDescription(sdp=offer_data.sdp, type=offer_data.type)
    pc = RTCPeerConnection()
    pcs.add(pc)

    data_channel = None

    @pc.on("datachannel")
    def on_datachannel(channel):
        nonlocal data_channel
        data_channel = channel

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            # ভিডিও ট্র্যাকে ইনফারেন্স পাইপলাইন যুক্ত করা
            processor = SignLanguageTrackProcessor(track, data_channel, engine)
            pc.addTrack(processor)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


# ----------------- WebSocket Landmark Endpoint (Low-Bandwidth Mode) -----------------
@app.websocket("/ws/landmarks")
async def websocket_landmark_stream(websocket: WebSocket):
    """
    ক্লায়েন্ট সাইড থেকে যদি MediaPipe ল্যান্ডমার্ক প্রসেস করে (JSON/Binary),
    তবে এই এন্ডপয়েন্ট ব্যবহার করে সার্ভারের GPU কস্ট ৮০% কমানো যায়।
    """
    await websocket.accept()
    buffer = SlidingWindowBuffer(window_size=32, stride=8)
    last_gloss = ""

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            # ল্যান্ডমার্ক অ্যারে কনভার্সন (Shape: 75, 3)
            landmarks = np.array(payload["landmarks"], dtype=np.float32)

            if buffer.append(landmarks):
                window = buffer.get_window()
                gloss = await engine.predict_cslr_ctc(window)
                
                if gloss and gloss != last_gloss:
                    last_gloss = gloss
                    text = await engine.translate_gloss_to_text(gloss)
                    await websocket.send_json({
                        "status": "success",
                        "gloss": gloss,
                        "text": text
                    })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")

@app.on_event("shutdown")
async def on_shutdown():
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

