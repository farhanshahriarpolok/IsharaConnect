"""WebSocket Room Router for IsharaConnect Backend."""

import logging
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.websockets.room_manager import manager, ClientType

logger = logging.getLogger(__name__)

router = APIRouter(tags=["room-websockets"])


@router.websocket("/ws/room/{room_id}/{client_type}")
async def room_websocket_endpoint(websocket: WebSocket, room_id: str, client_type: ClientType):
    """Duplex WebSocket endpoint for room communication with keepalive support."""
    await manager.connect(websocket, room_id, client_type)
    try:
        while True:
            # Receive text/json from client
            try:
                data = await websocket.receive_json()
            except Exception as e:
                # Malformed JSON or read error
                logger.debug("Error receiving JSON from client in room %s: %s", room_id, e)
                break

            event_type = data.get("type", "").upper() if isinstance(data, dict) else ""
            payload = data.get("data", {}) if isinstance(data, dict) else {}

            if event_type == "PING":
                # Resilient keep-alive response
                try:
                    await websocket.send_json({"type": "PONG", "timestamp": time.time()})
                except Exception as e:
                    logger.debug("Failed to send PONG to client in room %s: %s", room_id, e)
                    break

            elif event_type == "SIGN_TRANSLATION" and client_type == ClientType.SIGNER:
                # Forward to SPEAKER with synthesized audio payload
                await manager.handle_sign_translation(room_id, websocket, payload)

            elif event_type == "SPEECH_TEXT" and client_type == ClientType.SPEAKER:
                # Forward to SIGNER for subtitle display
                await manager.handle_speech_text(room_id, websocket, payload)

            elif event_type in ("PONG", "HEARTBEAT"):
                # Clean keep-alive acknowledgment
                pass

            else:
                logger.warning("Unknown or invalid event type '%s' from client '%s'", event_type, client_type.value)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected cleanly for room %s (%s)", room_id, client_type.value)
    except Exception as e:
        logger.warning("Unexpected WebSocket exception in room %s (%s): %s", room_id, client_type.value, e)
    finally:
        manager.disconnect(websocket, room_id)
