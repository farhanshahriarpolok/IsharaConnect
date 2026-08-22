"""Distributed Redis Pub/Sub Room Adapter for Horizontally Scalable WebSockets.

Enables cross-process and cross-container real-time duplex synchronization for:
- Sign language gesture translations
- Speech vocalizations & transcripts
- Distributed room participant states with auto-expiring Redis keys (1-hour TTL)
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from fastapi import WebSocket

from backend.websockets.base_adapter import BaseRoomAdapter, ClientType

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None


class RedisRoomAdapter(BaseRoomAdapter):
    """Distributed WebSocket Room Adapter utilizing Redis Pub/Sub and Key-Value Caching."""

    CHANNEL_PREFIX = "channel:ishara:rooms"
    META_KEY_PREFIX = "ishara:room_meta"
    ROOM_TTL_SECONDS = 3600  # 1 hour inactivity TTL

    def __init__(self, redis_url: str):
        if aioredis is None:
            raise RuntimeError("redis package is required for RedisRoomAdapter. Install redis>=4.2.0.")
        
        self.redis_url = redis_url
        self.node_id = str(uuid.uuid4())[:8]
        self.redis: Optional[aioredis.Redis] = None
        self.local_connections: Dict[str, List[WebSocket]] = {}
        self.client_types: Dict[WebSocket, ClientType] = {}
        self.room_pubsubs: Dict[str, Any] = {}
        self.room_listener_tasks: Dict[str, asyncio.Task] = {}
        self.room_start_times: Dict[str, float] = {}
        self._is_connected = False

    async def _ensure_redis_connected(self):
        """Initializes async Redis client connection pool."""
        if self.redis is None:
            self.redis = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=3.0
            )
            await self.redis.ping()
            self._is_connected = True

    def _get_channel(self, room_id: str) -> str:
        return f"{self.CHANNEL_PREFIX}:{room_id}"

    def _get_meta_key(self, room_id: str) -> str:
        return f"{self.META_KEY_PREFIX}:{room_id}"

    async def connect(self, websocket: WebSocket, room_id: str, client_type: ClientType):
        """Accepts WebSocket connection and subscribes to room's Redis Pub/Sub channel."""
        await websocket.accept()
        await self._ensure_redis_connected()

        is_first_local_client = room_id not in self.local_connections or len(self.local_connections[room_id]) == 0

        self.local_connections.setdefault(room_id, []).append(websocket)
        self.client_types[websocket] = client_type

        if room_id not in self.room_start_times:
            self.room_start_times[room_id] = time.time()

        # If first local client in this worker node, subscribe to Redis channel
        if is_first_local_client:
            await self._subscribe_to_room_channel(room_id)

        # Update distributed room metrics in Redis
        await self._update_room_meta_cache(room_id)
        logger.info(
            "Node [%s]: Client (%s) joined room '%s' (Total local: %d)",
            self.node_id, client_type.value, room_id, len(self.local_connections[room_id])
        )

    async def _subscribe_to_room_channel(self, room_id: str):
        """Subscribes to Redis room channel and starts async message listening loop."""
        channel_name = self._get_channel(room_id)
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(channel_name)
        self.room_pubsubs[room_id] = pubsub

        task = asyncio.create_task(self._channel_listener_loop(room_id, pubsub))
        self.room_listener_tasks[room_id] = task
        logger.info("Node [%s]: Subscribed to Redis channel '%s'", self.node_id, channel_name)

    async def _channel_listener_loop(self, room_id: str, pubsub: Any):
        """Listens for messages on Redis channel and forwards to local WebSockets."""
        channel_name = self._get_channel(room_id)
        try:
            async for message in pubsub.listen():
                if message is None or message.get("type") != "message":
                    continue

                raw_data = message.get("data")
                if not raw_data:
                    continue

                try:
                    payload = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
                except Exception:
                    continue

                # Avoid echoing if message was published by this same local node and marked with source
                origin_node = payload.get("_node_id")
                origin_socket_id = payload.get("_sender_socket_id")

                # Forward to connected local WebSockets
                local_sockets = list(self.local_connections.get(room_id, []))
                dead_sockets = []

                # Remove internal routing metadata before delivering to clients
                clean_payload = {k: v for k, v in payload.items() if not k.startswith("_")}

                for ws in local_sockets:
                    if origin_node == self.node_id and id(ws) == origin_socket_id:
                        continue  # Skip originator
                    try:
                        await ws.send_json(clean_payload)
                    except Exception:
                        dead_sockets.append(ws)

                for dead in dead_sockets:
                    await self.disconnect(dead, room_id)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("Node [%s]: Listener loop terminated for room '%s': %s", self.node_id, room_id, e)

    async def disconnect(self, websocket: WebSocket, room_id: str):
        """Removes local client, unsubscribing from Redis when room is locally empty."""
        if room_id in self.local_connections and websocket in self.local_connections[room_id]:
            self.local_connections[room_id].remove(websocket)
            
            if len(self.local_connections[room_id]) == 0:
                del self.local_connections[room_id]
                await self._unsubscribe_from_room_channel(room_id)

        self.client_types.pop(websocket, None)
        await self._update_room_meta_cache(room_id)
        logger.info("Node [%s]: Client disconnected from room '%s'", self.node_id, room_id)

    async def _unsubscribe_from_room_channel(self, room_id: str):
        """Unsubscribes from Redis channel and cancels background listener task."""
        task = self.room_listener_tasks.pop(room_id, None)
        if task and not task.done():
            task.cancel()

        pubsub = self.room_pubsubs.pop(room_id, None)
        if pubsub and self.redis:
            try:
                channel_name = self._get_channel(room_id)
                await pubsub.unsubscribe(channel_name)
                await pubsub.close()
                logger.info("Node [%s]: Unsubscribed from Redis channel '%s'", self.node_id, channel_name)
            except Exception as e:
                logger.warning("Error unsubscribing from %s: %s", room_id, e)

    async def broadcast_message(self, room_id: str, message: dict, sender: Optional[WebSocket] = None):
        """Publishes message to Redis channel for cluster-wide distribution."""
        await self._ensure_redis_connected()

        channel_name = self._get_channel(room_id)
        distributed_payload = dict(message)
        distributed_payload["_node_id"] = self.node_id
        distributed_payload["_sender_socket_id"] = id(sender) if sender else None
        distributed_payload["_timestamp"] = time.time()

        try:
            await self.redis.publish(channel_name, json.dumps(distributed_payload))
        except Exception as e:
            logger.error("Node [%s]: Failed to publish to Redis channel '%s': %s", self.node_id, channel_name, e)

            # Local fallback broadcast if Redis publish encounters error
            local_sockets = list(self.local_connections.get(room_id, []))
            dead_sockets = []
            clean_payload = {k: v for k, v in message.items() if not k.startswith("_")}
            for ws in local_sockets:
                if ws != sender:
                    try:
                        await ws.send_json(clean_payload)
                    except Exception:
                        dead_sockets.append(ws)
            for dead in dead_sockets:
                await self.disconnect(dead, room_id)

    async def _update_room_meta_cache(self, room_id: str):
        """Updates distributed participant count and TTL expiry in Redis."""
        if not self.redis:
            return

        meta_key = self._get_meta_key(room_id)
        local_sockets = self.local_connections.get(room_id, [])
        signers = sum(1 for ws in local_sockets if self.client_types.get(ws) == ClientType.SIGNER)
        speakers = sum(1 for ws in local_sockets if self.client_types.get(ws) == ClientType.SPEAKER)

        if len(local_sockets) > 0:
            meta = {
                "room_id": room_id,
                "signers_count": signers,
                "speakers_count": speakers,
                "total_participants": len(local_sockets),
                "last_active": datetime.utcnow().isoformat(),
                "created_at": datetime.utcfromtimestamp(self.room_start_times.get(room_id, time.time())).isoformat(),
                "node_id": self.node_id
            }
            try:
                await self.redis.set(meta_key, json.dumps(meta), ex=self.ROOM_TTL_SECONDS)
            except Exception as e:
                logger.warning("Failed to update Redis room metadata for %s: %s", room_id, e)
        else:
            try:
                # Expire in 60 seconds if empty locally
                await self.redis.expire(meta_key, 60)
            except Exception:
                pass

    async def get_active_rooms(self) -> List[Dict[str, Any]]:
        """Scans Redis for active room metadata across all cluster nodes."""
        await self._ensure_redis_connected()
        active_rooms: List[Dict[str, Any]] = []

        try:
            keys = await self.redis.keys(f"{self.META_KEY_PREFIX}:*")
            for k in keys:
                raw_meta = await self.redis.get(k)
                if raw_meta:
                    try:
                        meta = json.loads(raw_meta)
                        active_rooms.append(meta)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning("Failed to fetch active rooms from Redis: %s", e)
            # Fallback to local active rooms
            for r_id, conns in self.local_connections.items():
                active_rooms.append({
                    "room_id": r_id,
                    "signers_count": sum(1 for ws in conns if self.client_types.get(ws) == ClientType.SIGNER),
                    "speakers_count": sum(1 for ws in conns if self.client_types.get(ws) == ClientType.SPEAKER),
                    "total_participants": len(conns),
                    "last_active": datetime.utcnow().isoformat(),
                    "created_at": datetime.utcfromtimestamp(self.room_start_times.get(r_id, time.time())).isoformat(),
                    "node_id": self.node_id
                })

        return active_rooms

    async def close(self):
        """Closes all subscriptions and Redis connections."""
        for room_id in list(self.room_pubsubs.keys()):
            await self._unsubscribe_from_room_channel(room_id)
        if self.redis:
            await self.redis.close()
            self.redis = None
            self._is_connected = False
