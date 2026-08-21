"""WebSocket Room Manager with Pluggable Storage Adapters."""

import json
import logging
import time
from enum import Enum
from typing import Dict, List, Optional
from fastapi import WebSocket
from abc import ABC, abstractmethod

from core_engine.audio.tts_engine import TextToSpeechEngine
from core_engine.nlp.advanced_grammar_engine import AdvancedBdSLGrammarEngine
from backend.core.config import settings

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

logger = logging.getLogger(__name__)

class ClientType(str, Enum):
    SIGNER = "signer"
    SPEAKER = "speaker"

class BaseRoomAdapter(ABC):
    @abstractmethod
    async def connect(self, websocket: WebSocket, room_id: str, client_type: ClientType): pass
    @abstractmethod
    async def disconnect(self, websocket: WebSocket, room_id: str): pass
    @abstractmethod
    async def broadcast_message(self, room_id: str, message: dict, sender: WebSocket): pass

class InMemoryRoomAdapter(BaseRoomAdapter):
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.client_types: Dict[WebSocket, ClientType] = {}
        
    async def connect(self, websocket: WebSocket, room_id: str, client_type: ClientType):
        await websocket.accept()
        self.active_connections.setdefault(room_id, []).append(websocket)
        self.client_types[websocket] = client_type
        logger.info(f"Client {client_type.value} joined {room_id} (In-Memory).")
        
    async def disconnect(self, websocket: WebSocket, room_id: str):
        if room_id in self.active_connections and websocket in self.active_connections[room_id]:
            self.active_connections[room_id].remove(websocket)
            if not self.active_connections[room_id]:
                del self.active_connections[room_id]
        client_type = self.client_types.pop(websocket, None)
        logger.info(f"Client {client_type} left {room_id} (In-Memory).")
        
    async def broadcast_message(self, room_id: str, message: dict, sender: WebSocket):
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

class RedisRoomAdapter(BaseRoomAdapter):
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        self.pubsub = self.redis.pubsub()
        self.local_connections: Dict[str, List[WebSocket]] = {}
        self.client_types: Dict[WebSocket, ClientType] = {}
        self.subscribed_rooms = set()
        
    async def connect(self, websocket: WebSocket, room_id: str, client_type: ClientType):
        await websocket.accept()
        self.local_connections.setdefault(room_id, []).append(websocket)
        self.client_types[websocket] = client_type
        logger.info(f"Client {client_type.value} joined {room_id} (Redis).")
        
        if room_id not in self.subscribed_rooms:
            await self.pubsub.subscribe(room_id)
            self.subscribed_rooms.add(room_id)
            # In a full implementation, you'd spawn an asyncio task here to listen to self.pubsub.listen()
            # and forward to local_connections[room_id]
            
    async def disconnect(self, websocket: WebSocket, room_id: str):
        if room_id in self.local_connections and websocket in self.local_connections[room_id]:
            self.local_connections[room_id].remove(websocket)
            if not self.local_connections[room_id]:
                del self.local_connections[room_id]
                if room_id in self.subscribed_rooms:
                    await self.pubsub.unsubscribe(room_id)
                    self.subscribed_rooms.remove(room_id)
        self.client_types.pop(websocket, None)
        
    async def broadcast_message(self, room_id: str, message: dict, sender: WebSocket):
        # We publish to Redis. Local listener task would pick it up and forward to WebSockets.
        # For this prototype structure, we will just simulate local broadcast + redis publish
        try:
            await self.redis.publish(room_id, json.dumps(message))
        except Exception as e:
            logger.warning("Redis publish error: %s", e)
        
        connections = list(self.local_connections.get(room_id, []))
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

class ConnectionManager:
    def __init__(self):
        if settings.REDIS_URL and redis:
            self.adapter = RedisRoomAdapter(settings.REDIS_URL)
        else:
            self.adapter = InMemoryRoomAdapter()
            
        self.gloss_buffers: Dict[WebSocket, List[str]] = {}
        self.last_sign_time: Dict[WebSocket, float] = {}
        self.idle_timeout = 2.0
        self.tts_engine = TextToSpeechEngine()
        self.gloss_translator = AdvancedBdSLGrammarEngine()

    async def connect(self, websocket: WebSocket, room_id: str, client_type: ClientType):
        await self.adapter.connect(websocket, room_id, client_type)
        if client_type == ClientType.SIGNER:
            self.gloss_buffers[websocket] = []
            self.last_sign_time[websocket] = time.time()

    def disconnect(self, websocket: WebSocket, room_id: str):
        # Fire and forget disconnect since fastapi WebsocketDisconnect is sync
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.adapter.disconnect(websocket, room_id))
        except RuntimeError:
            pass
        self.gloss_buffers.pop(websocket, None)
        self.last_sign_time.pop(websocket, None)


    async def handle_sign_translation(self, room_id: str, sender: WebSocket, payload: dict):
        raw_gloss = payload.get("label_bn", "")
        if not raw_gloss:
            return
            
        now = time.time()
        last_time = self.last_sign_time.get(sender, now)
        if now - last_time > self.idle_timeout:
            self.gloss_buffers[sender] = []
            
        buffer = self.gloss_buffers.setdefault(sender, [])
        if not buffer or buffer[-1] != raw_gloss:
            buffer.append(raw_gloss)
            
        self.last_sign_time[sender] = now
        
        translation = self.gloss_translator.generate_natural_sentence(buffer)
        bengali_sentence = translation["bengali"]
        english_sentence = translation["english"]
        
        if bengali_sentence:
            import base64
            clean_audio_text = bengali_sentence.rstrip("।!?")
            audio_bytes = self.tts_engine.synthesize_to_bytes(text=clean_audio_text, lang="bn")
            if audio_bytes:
                payload["audio_payload_base64"] = base64.b64encode(audio_bytes).decode('utf-8')
            payload["label_bn"] = bengali_sentence
            payload["label_en"] = english_sentence
            payload["nlp_confidence"] = translation.get("confidence", 0.9)
                
        await self.adapter.broadcast_message(room_id, {"type": "SIGN_TRANSLATION", "data": payload}, sender)

    async def handle_speech_text(self, room_id: str, sender: WebSocket, payload: dict):
        await self.adapter.broadcast_message(room_id, {"type": "SPEECH_TEXT", "data": payload}, sender)

manager = ConnectionManager()
