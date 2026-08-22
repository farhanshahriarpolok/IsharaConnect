"""Active Rooms and Distributed Cluster Metrics API."""

from fastapi import APIRouter
from backend.websockets.room_manager import manager

router = APIRouter(tags=["Rooms"])


@router.get("/active")
async def get_active_rooms():
    """Lists all currently active rooms, signer/speaker counts, and cluster adapter type."""
    rooms = await manager.get_active_rooms()
    return {
        "active_rooms_count": len(rooms),
        "adapter_type": type(manager.adapter).__name__,
        "rooms": rooms
    }
