"""Network worker thread for asynchronous WebSocket communication."""

import asyncio
import json
import logging
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

import websockets

logger = logging.getLogger(__name__)


class NetworkWorker(QThread):
    """Worker thread for FastAPI WebSocket duplex communication."""

    # Signals
    message_received = pyqtSignal(dict)
    connection_status = pyqtSignal(bool, str)

    def __init__(self, server_url: str, room_id: str, client_type: str):
        super().__init__()
        self.server_url = server_url
        self.room_id = room_id
        self.client_type = client_type
        
        self._is_running = True
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._loop = asyncio.new_event_loop()
        
        # We use an asyncio queue to safely send messages from the main thread
        self._send_queue = asyncio.Queue()

    def run(self):
        """Start the asyncio event loop in this thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._websocket_loop())
        
    async def _websocket_loop(self):
        """Main WebSocket connection and message loop."""
        uri = f"{self.server_url}/ws/room/{self.room_id}/{self.client_type}"
        retry_delay = 1
        
        while self._is_running:
            try:
                self.connection_status.emit(False, f"Connecting to {self.room_id}...")
                async with websockets.connect(uri) as ws:
                    self._ws = ws
                    self.connection_status.emit(True, f"Connected to {self.room_id}")
                    retry_delay = 1  # Reset retry delay on successful connection
                    
                    # Create tasks for receiving and sending
                    recv_task = asyncio.create_task(self._receive_messages(ws))
                    send_task = asyncio.create_task(self._send_messages(ws))
                    
                    done, pending = await asyncio.wait(
                        [recv_task, send_task], 
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # If one task fails/completes, cancel the other
                    for task in pending:
                        task.cancel()
                        
            except (websockets.ConnectionClosed, ConnectionRefusedError, Exception) as e:
                self._ws = None
                if self._is_running:
                    self.connection_status.emit(False, f"Disconnected: {e}. Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 10) # Exponential backoff

    async def _receive_messages(self, ws: websockets.WebSocketClientProtocol):
        """Coroutine to handle incoming messages."""
        try:
            async for message in ws:
                data = json.loads(message)
                self.message_received.emit(data)
        except websockets.ConnectionClosed:
            logger.warning("WebSocket connection closed during receive.")

    async def _send_messages(self, ws: websockets.WebSocketClientProtocol):
        """Coroutine to handle outgoing messages from queue."""
        try:
            while True:
                msg = await self._send_queue.get()
                await ws.send(json.dumps(msg))
                self._send_queue.task_done()
        except websockets.ConnectionClosed:
            logger.warning("WebSocket connection closed during send.")

    def send_sign_event(self, sign_data: dict):
        """Dispatch a sign detection event to the backend. Thread-safe."""
        payload = {
            "type": "SIGN_TRANSLATION",
            "data": sign_data
        }
        # Safely put into asyncio queue from PyQt main thread
        self._loop.call_soon_threadsafe(self._send_queue.put_nowait, payload)

    def send_speech_event(self, transcript: str):
        """Dispatch a speech transcript event to the backend. Thread-safe."""
        payload = {
            "type": "SPEECH_TEXT",
            "data": {
                "transcript": transcript,
                "is_final": True
            }
        }
        self._loop.call_soon_threadsafe(self._send_queue.put_nowait, payload)

    def stop(self):
        """Gracefully stop the network worker."""
        self._is_running = False
        if self._ws:
            self._loop.call_soon_threadsafe(asyncio.create_task, self._ws.close())
        
        # Stop the loop
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            
        self.wait()
