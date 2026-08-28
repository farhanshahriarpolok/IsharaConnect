"""Room-Based WebRTC Peer Signaling & Duplex Real-Time Message Router.

Handles SDP Offer/Answer exchanges, ICE Candidate trickling, and room-based
broadcast routing for two-way video/signaling channels.
"""

import json
import logging
from typing import Dict, Optional, Set
from fastapi import WebSocket

logger = logging.getLogger("WebRTCManager")


class WebRTCConnectionManager:
    """রুম-ভিত্তিক WebRTC পিয়ার সিগনালিং (SDP Offer/Answer, ICE Candidates) 
    এবং দ্বি-মুখী রিয়েল-টাইম মেসেজ রাউটার।
    """

    def __init__(self):
        # room_id -> set of WebSockets
        self.rooms: Dict[str, Set[WebSocket]] = {}

    async def connect(self, room_id: str, websocket: WebSocket):
        """Accepts WebSocket connection and joins the room."""
        await websocket.accept()
        if room_id not in self.rooms:
            self.rooms[room_id] = set()
        self.rooms[room_id].add(websocket)
        logger.info("Client joined room '%s'. Total peers: %d", room_id, len(self.rooms[room_id]))

    def disconnect(self, room_id: str, websocket: WebSocket):
        """Removes WebSocket connection and cleans up empty rooms."""
        if room_id in self.rooms:
            self.rooms[room_id].discard(websocket)
            if not self.rooms[room_id]:
                del self.rooms[room_id]
            logger.info("Client left room '%s'.", room_id)

    async def broadcast_to_room(self, room_id: str, message: dict, sender: Optional[WebSocket] = None):
        """Broadcasts signaling payload to all other peers in the room."""
        if room_id not in self.rooms:
            return
        payload = json.dumps(message)
        for connection in list(self.rooms[room_id]):
            if sender is None or connection != sender:
                try:
                    await connection.send_text(payload)
                except Exception as e:
                    logger.debug("Failed to send signaling payload to peer in room '%s': %s", room_id, e)


webrtc_hub = WebRTCConnectionManager()
