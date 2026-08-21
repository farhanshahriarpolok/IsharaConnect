"""WebSocket Room Manager for duplex communication."""

import json
import logging
from enum import Enum
from typing import Dict, List, Set

from fastapi import WebSocket

from core_engine.audio.tts_engine import TextToSpeechEngine
from core_engine.nlp.gloss_translator import BdSLGlossTranslator
import time

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
        
        # Continuous NLP Engine Tracking
        self.gloss_buffers: Dict[WebSocket, List[str]] = {}
        self.last_sign_time: Dict[WebSocket, float] = {}
        self.idle_timeout = 2.0  # seconds
        
        # Shared engines
        self.tts_engine = TextToSpeechEngine()
        self.gloss_translator = BdSLGlossTranslator()

    async def connect(self, websocket: WebSocket, room_id: str, client_type: ClientType):
        """Accept a WebSocket connection and register it to a room."""
        await websocket.accept()
        
        if room_id not in self.active_connections:
            self.active_connections[room_id] = []
            
        self.active_connections[room_id].append(websocket)
        self.client_types[websocket] = client_type
        
        if client_type == ClientType.SIGNER:
            self.gloss_buffers[websocket] = []
            self.last_sign_time[websocket] = time.time()
            
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
            self.gloss_buffers.pop(websocket, None)
            self.last_sign_time.pop(websocket, None)
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
        """Process SIGN_TRANSLATION event, buffer glosses, and broadcast sentence."""
        raw_gloss = payload.get("label_bn", "")
        if not raw_gloss:
            return
            
        now = time.time()
        last_time = self.last_sign_time.get(sender, now)
        
        # Check if idle timeout exceeded
        if now - last_time > self.idle_timeout:
            self.gloss_buffers[sender] = []
            
        # Append gloss
        # Basic debounce: don't append if it's the exact same consecutive gloss within a very short window, 
        # but for NLP we just append for now
        buffer = self.gloss_buffers.setdefault(sender, [])
        if not buffer or buffer[-1] != raw_gloss:
            buffer.append(raw_gloss)
            
        self.last_sign_time[sender] = now
        
        # Translate the current sequence
        translation = self.gloss_translator.translate_gloss_sequence(buffer)
        bengali_sentence = translation["bengali_sentence"]
        
        if bengali_sentence:
            import base64
            # Synthesize audio bytes based on full sentence
            audio_bytes = self.tts_engine.synthesize_to_bytes(text=bengali_sentence, lang="bn")
            if audio_bytes:
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                payload["audio_payload_base64"] = audio_base64
                
            # Update payload text to sentence
            payload["label_bn"] = bengali_sentence
            payload["label_en"] = translation["english_sentence"]
                
        await self.broadcast_to_room(room_id, {"type": "SIGN_TRANSLATION", "data": payload}, sender)

    async def handle_speech_text(self, room_id: str, sender: WebSocket, payload: dict):
        """Process SPEECH_TEXT event from SPEAKER and broadcast to SIGNER."""
        await self.broadcast_to_room(room_id, {"type": "SPEECH_TEXT", "data": payload}, sender)

manager = ConnectionManager()
