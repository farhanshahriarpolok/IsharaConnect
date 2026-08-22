"""WebSocket Room Manager with Dynamic Distributed Redis Adapter and Auto-Fallback."""

import asyncio
import base64
import json
import logging
import time
from typing import Any, Dict, List, Optional
from fastapi import WebSocket

from core_engine.audio.tts_engine import TextToSpeechEngine
from core_engine.nlp.advanced_grammar_engine import AdvancedBdSLGrammarEngine
from backend.core.config import settings
from backend.websockets.base_adapter import (
    BaseRoomAdapter,
    ClientType,
    InMemoryRoomAdapter,
)
from backend.websockets.redis_adapter import RedisRoomAdapter

logger = logging.getLogger(__name__)


async def get_room_adapter(redis_url: Optional[str] = None) -> BaseRoomAdapter:
    """Probes Redis connection with transparent fallback to InMemoryRoomAdapter.

    Args:
        redis_url: Redis connection string (e.g. redis://localhost:6379/0).
                   Defaults to settings.REDIS_URL.

    Returns:
        RedisRoomAdapter if Redis is reachable, otherwise InMemoryRoomAdapter.
    """
    target_url = redis_url if redis_url is not None else settings.REDIS_URL

    if not target_url:
        logger.info("REDIS_URL not configured. Using InMemoryRoomAdapter.")
        return InMemoryRoomAdapter()

    try:
        adapter = RedisRoomAdapter(target_url)
        await adapter._ensure_redis_connected()
        logger.info("Successfully connected to Redis. Using distributed RedisRoomAdapter.")
        return adapter
    except Exception as e:
        logger.warning(
            "Failed to connect to Redis at '%s' (%s). Falling back to InMemoryRoomAdapter.",
            target_url, e
        )
        return InMemoryRoomAdapter()


class ConnectionManager:
    """Coordinates room duplex communication, NLP translation, and TTS synthesis."""

    def __init__(self, adapter: Optional[BaseRoomAdapter] = None):
        self.adapter: BaseRoomAdapter = adapter or InMemoryRoomAdapter()
        self.gloss_buffers: Dict[WebSocket, List[str]] = {}
        self.last_sign_time: Dict[WebSocket, float] = {}
        self.idle_timeout = 2.0
        self.tts_engine = TextToSpeechEngine()
        self.gloss_translator = AdvancedBdSLGrammarEngine()
        self._initialized = False

    async def initialize_adapter(self, redis_url: Optional[str] = None):
        """Asynchronously initializes or upgrades to Redis adapter if available."""
        if not self._initialized:
            self.adapter = await get_room_adapter(redis_url)
            self._initialized = True

    async def connect(self, websocket: WebSocket, room_id: str, client_type: ClientType):
        """Connects client and prepares tracking buffers."""
        await self.adapter.connect(websocket, room_id, client_type)
        if client_type == ClientType.SIGNER:
            self.gloss_buffers[websocket] = []
            self.last_sign_time[websocket] = time.time()

    def disconnect(self, websocket: WebSocket, room_id: str):
        """Disconnects client gracefully."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.adapter.disconnect(websocket, room_id))
        except RuntimeError:
            pass
        self.gloss_buffers.pop(websocket, None)
        self.last_sign_time.pop(websocket, None)

    async def handle_sign_translation(self, room_id: str, sender: WebSocket, payload: dict):
        """Processes gloss from signer, generates natural sentence, synthesizes voice, and broadcasts."""
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
        bengali_sentence = translation.get("bengali", raw_gloss)
        english_sentence = translation.get("english", "")

        if bengali_sentence:
            clean_audio_text = bengali_sentence.rstrip("।!?")
            audio_bytes = self.tts_engine.synthesize_to_bytes(text=clean_audio_text, lang="bn")
            if audio_bytes:
                payload["audio_payload_base64"] = base64.b64encode(audio_bytes).decode("utf-8")
            payload["label_bn"] = bengali_sentence
            payload["label_en"] = english_sentence
            payload["nlp_confidence"] = translation.get("confidence", 0.9)

        await self.adapter.broadcast_message(
            room_id,
            {"type": "SIGN_TRANSLATION", "data": payload},
            sender
        )

    async def handle_speech_text(self, room_id: str, sender: WebSocket, payload: dict):
        """Broadcasts spoken transcript to signers for real-time text/subtitle rendering."""
        await self.adapter.broadcast_message(
            room_id,
            {"type": "SPEECH_TEXT", "data": payload},
            sender
        )

    async def get_active_rooms(self) -> List[Dict[str, Any]]:
        """Fetches active room metrics from current adapter."""
        return await self.adapter.get_active_rooms()


# Global Singleton Manager
manager = ConnectionManager()
