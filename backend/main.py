"""FastAPI Main Application for IsharaConnect Backend."""

import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import os

from backend.api.routes import router as api_router
from backend.api.admin_routes import router as admin_router
from backend.websockets.room_manager import manager, ClientType
from backend.config import settings

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

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def serve_root():
    index_path = TEMPLATES_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>IsharaConnect Backend Running</h1>")


@app.get("/health")
async def health_check():
    """Ultra-fast root health endpoint for launcher and health probes."""
    return {"status": "healthy", "service": "IsharaConnect-Backend"}



@app.websocket("/ws/room/{room_id}/{client_type}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, client_type: ClientType):
    """Duplex WebSocket endpoint for room communication."""
    await manager.connect(websocket, room_id, client_type)
    try:
        while True:
            # Receive text/json from client
            data = await websocket.receive_json()
            event_type = data.get("type")
            payload = data.get("data", {})
            
            if event_type == "SIGN_TRANSLATION" and client_type == ClientType.SIGNER:
                # Forward to SPEAKER with synthesized audio payload
                await manager.handle_sign_translation(room_id, websocket, payload)
                
            elif event_type == "SPEECH_TEXT" and client_type == ClientType.SPEAKER:
                # Forward to SIGNER for subtitle display
                await manager.handle_speech_text(room_id, websocket, payload)
                
            else:
                logger.warning("Unknown event type '%s' from client '%s'", event_type, client_type.value)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, room_id)
