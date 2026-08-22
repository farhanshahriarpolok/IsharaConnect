"""Base Room Adapter and In-Memory Fallback Adapter for WebSockets."""

import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ClientType(str, Enum):
    """Client role in two-way room communication."""
    SIGNER = "signer"
    SPEAKER = "speaker"


class BaseRoomAdapter(ABC):
    """Abstract interface for WebSocket room adapters."""

    @abstractmethod
    async def connect(self, websocket: WebSocket, room_id: str, client_type: ClientType):
        """Connects a client to the specified room."""
        pass

    @abstractmethod
    async def disconnect(self, websocket: WebSocket, room_id: str):
        """Disconnects a client from the specified room."""
        pass

    @abstractmethod
    async def broadcast_message(self, room_id: str, message: dict, sender: Optional[WebSocket] = None):
        """Broadcasts message to all clients in the room except sender."""
        pass

    @abstractmethod
    async def get_active_rooms(self) -> List[Dict[str, Any]]:
        """Returns statistics for active rooms."""
        pass

    async def close(self):
        """Gracefully close adapter resources."""
        pass


class InMemoryRoomAdapter(BaseRoomAdapter):
    """Thread-safe In-Memory Room Adapter for standalone local development."""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.client_types: Dict[WebSocket, ClientType] = {}
        self.room_start_times: Dict[str, float] = {}

    async def connect(self, websocket: WebSocket, room_id: str, client_type: ClientType):
        await websocket.accept()
        self.active_connections.setdefault(room_id, []).append(websocket)
        self.client_types[websocket] = client_type
        if room_id not in self.room_start_times:
            self.room_start_times[room_id] = time.time()
        logger.info("Client (%s) joined room '%s' (In-Memory).", client_type.value, room_id)

    async def disconnect(self, websocket: WebSocket, room_id: str):
        if room_id in self.active_connections and websocket in self.active_connections[room_id]:
            self.active_connections[room_id].remove(websocket)
            if not self.active_connections[room_id]:
                del self.active_connections[room_id]
                self.room_start_times.pop(room_id, None)

        client_type = self.client_types.pop(websocket, None)
        logger.info("Client (%s) left room '%s' (In-Memory).", client_type, room_id)

    async def broadcast_message(self, room_id: str, message: dict, sender: Optional[WebSocket] = None):
        connections = list(self.active_connections.get(room_id, []))
        if not connections:
            return

        dead_conns = []
        for connection in connections:
            if connection != sender:
                try:
                    await connection.send_json(message)
                except Exception:
                    dead_conns.append(connection)

        for conn in dead_conns:
            await self.disconnect(conn, room_id)

    async def get_active_rooms(self) -> List[Dict[str, Any]]:
        rooms: List[Dict[str, Any]] = []
        for room_id, conns in self.active_connections.items():
            signers = sum(1 for ws in conns if self.client_types.get(ws) == ClientType.SIGNER)
            speakers = sum(1 for ws in conns if self.client_types.get(ws) == ClientType.SPEAKER)
            rooms.append({
                "room_id": room_id,
                "signers_count": signers,
                "speakers_count": speakers,
                "total_participants": len(conns),
                "last_active": datetime.utcnow().isoformat(),
                "created_at": datetime.utcfromtimestamp(self.room_start_times.get(room_id, time.time())).isoformat(),
                "adapter": "InMemoryRoomAdapter"
            })
        return rooms

    async def close(self):
        self.active_connections.clear()
        self.client_types.clear()
