"""WebSocket Room Manager for duplex communication."""

import json
import logging
from enum import Enum
from typing import Dict, List, Set

from fastapi import WebSocket

from core_engine.audio.tts_engine import TextToSpeechEngine

logger = logging.getLogger(__name__)


class ClientType(str, Enum):
    """Types of clients connecting to a room."""
    SIGNER = "signer"    # Deaf user transmitting landmarks/signs
    SPEAKER = "speaker"  # Hearing user transmitting speech/text


class ConnectionManager:
    """Manages WebSocket connections and room-based duplex routing."""

    def __init__(self):
        # Maps room_id -> list of active WebSockets
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # Maps WebSocket -> ClientType
        self.client_types: Dict[WebSocket, ClientType] = {}
        # Shared TTS engine for synthesizing signs into audio payload
        self.tts_engine = TextToSpeechEngine()

    async def connect(self, websocket: WebSocket, room_id: str, client_type: ClientType):
        """Accept a WebSocket connection and register it to a room."""
        await websocket.accept()
        
        if room_id not in self.active_connections:
            self.active_connections[room_id] = []
            
        self.active_connections[room_id].append(websocket)
        self.client_types[websocket] = client_type
        
        logger.info("Client %s joined room %s. Total in room: %d", 
                    client_type.value, room_id, len(self.active_connections[room_id]))

    def disconnect(self, websocket: WebSocket, room_id: str):
        """Remove a WebSocket connection from a room."""
        if room_id in self.active_connections:
            if websocket in self.active_connections[room_id]:
                self.active_connections[room_id].remove(websocket)
                
            if len(self.active_connections[room_id]) == 0:
                del self.active_connections[room_id]
                
        if websocket in self.client_types:
            client_type = self.client_types.pop(websocket)
            logger.info("Client %s left room %s.", client_type.value, room_id)

    async def broadcast_to_room(self, room_id: str, message: dict, sender: WebSocket):
        """Broadcast a message to all other clients in the room."""
        if room_id not in self.active_connections:
            return
            
        disconnected = []
        for connection in self.active_connections[room_id]:
            if connection != sender:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error("Failed to send message to client in room %s: %s", room_id, e)
                    disconnected.append(connection)
                    
        # Cleanup disconnected clients
        for conn in disconnected:
            self.disconnect(conn, room_id)

    async def handle_sign_translation(self, room_id: str, sender: WebSocket, payload: dict):
        """Process SIGN_TRANSLATION event from SIGNER and broadcast to SPEAKER."""
        text = payload.get("label_bn", "")
        if text:
            import base64
            # Synthesize audio bytes
            audio_bytes = self.tts_engine.synthesize_to_bytes(text=text, lang="bn")
            if audio_bytes:
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                payload["audio_payload_base64"] = audio_base64
                
        await self.broadcast_to_room(room_id, {"type": "SIGN_TRANSLATION", "data": payload}, sender)

    async def handle_speech_text(self, room_id: str, sender: WebSocket, payload: dict):
        """Process SPEECH_TEXT event from SPEAKER and broadcast to SIGNER."""
        await self.broadcast_to_room(room_id, {"type": "SPEECH_TEXT", "data": payload}, sender)

manager = ConnectionManager()
