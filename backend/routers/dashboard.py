"""Two-Way Split Communication Dashboard Router."""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from backend.config.webrtc_config import get_ice_servers, get_rtc_configuration
from core_engine.nlp.master_lexicon import master_lexicon

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Dashboard"])

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(
    request: Request,
    room_id: Optional[str] = Query("room-general"),
    client_type: Optional[str] = Query("signer")
):
    """Renders the unified 2-way split communication dashboard."""
    rtc_config = get_rtc_configuration()
    lexicon_signs = master_lexicon.all_signs()

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "request": request,
            "room_id": room_id,
            "client_type": client_type,
            "rtc_config": rtc_config,
            "signs_count": len(lexicon_signs),
            "lexicon_signs": lexicon_signs
        }
    )


@router.get("/api/v1/webrtc/config")
async def get_webrtc_config():
    """Returns dynamic ICE & COTURN server configuration for clients."""
    return get_rtc_configuration()
