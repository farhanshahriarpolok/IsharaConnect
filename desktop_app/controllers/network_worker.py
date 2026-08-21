"""Network worker thread for asynchronous WebSocket communication with seamless offline fallback."""

import asyncio
import json
import logging
from typing import Any, Optional

from PyQt6.QtCore import QThread, pyqtSignal
import websockets

logger = logging.getLogger(__name__)


class NetworkWorker(QThread):
    """Worker thread for FastAPI WebSocket duplex communication with resilient offline mode."""

    # Signals
    message_received = pyqtSignal(dict)
    connection_status = pyqtSignal(bool, str)
    network_status_changed = pyqtSignal(str)

    def __init__(
        self,
        server_url: str = "ws://127.0.0.1:8000",
        room_id: str = "default_room",
        client_type: str = "signer"
    ):
        super().__init__()
        self.server_url = server_url.rstrip("/")
        self.room_id = room_id
        self.client_type = client_type

        self._is_running = True
        self._ws: Optional[Any] = None
        self._loop = asyncio.new_event_loop()

        # Queue for thread-safe cross-thread messaging
        self._send_queue: asyncio.Queue = asyncio.Queue()

        # Retry & Offline State
        self.max_retries = 3
        self.retry_count = 0
        self.is_offline = False

    def run(self):
        """Start the asyncio event loop in this worker thread."""
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._websocket_loop())
        except asyncio.CancelledError:
            pass
        except RuntimeError as e:
            logger.debug("RuntimeError in NetworkWorker loop: %s", e)
        except Exception as e:
            logger.debug("Exception in NetworkWorker loop: %s", e)
        finally:
            try:
                # Cancel pending tasks properly before closing loop
                pending = [t for t in asyncio.all_tasks(loop=self._loop) if not t.done()]
                for task in pending:
                    task.cancel()

                if pending:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            except Exception as e:
                logger.debug("Error during NetworkWorker loop cleanup: %s", e)
            finally:
                self._loop.close()

    async def _websocket_loop(self):
        """Main WebSocket connection loop with exponential backoff and offline fallback."""
        uri = f"{self.server_url}/ws/room/{self.room_id}/{self.client_type}"
        retry_delay = 1.0

        while self._is_running:
            # If offline mode reached, pause connection attempts without blocking event loop
            if self.is_offline:
                while self._is_running and self.is_offline:
                    await asyncio.sleep(1.0)
                if not self._is_running:
                    break

            try:
                self.connection_status.emit(False, f"Connecting to {self.room_id}...")
                self.network_status_changed.emit(f"Connecting to {self.room_id}...")

                async with websockets.connect(uri, open_timeout=2.0) as ws:
                    self._ws = ws
                    self.retry_count = 0
                    self.is_offline = False
                    retry_delay = 1.0

                    self.connection_status.emit(True, f"Connected to {self.room_id}")
                    self.network_status_changed.emit(f"Connected to {self.room_id}")
                    logger.info("Connected to room %s as %s", self.room_id, self.client_type)

                    # Create bidirectional communication tasks
                    recv_task = asyncio.create_task(self._receive_messages(ws))
                    send_task = asyncio.create_task(self._send_messages(ws))

                    done, pending = await asyncio.wait(
                        [recv_task, send_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    for task in pending:
                        task.cancel()

            except asyncio.CancelledError:
                raise
            except (ConnectionRefusedError, OSError, Exception) as e:
                self._ws = None
                if not self._is_running:
                    break

                self.retry_count += 1
                if self.retry_count <= self.max_retries:
                    self.connection_status.emit(
                        False,
                        f"Connection attempt {self.retry_count}/{self.max_retries} failed. Retrying in {int(retry_delay)}s..."
                    )
                    self.network_status_changed.emit(
                        f"Connecting ({self.retry_count}/{self.max_retries})..."
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2.0, 8.0)
                else:
                    # Switch cleanly to Standalone Offline Mode
                    self.is_offline = True
                    self.network_status_changed.emit("Offline / Standalone Mode (Local ML)")
                    self.connection_status.emit(False, "Offline Mode (Standalone ML Ready)")
                    logger.info(
                        "NetworkWorker: Could not reach backend server (%s). Entered Offline / Standalone Mode.",
                        e
                    )

    async def _receive_messages(self, ws: Any):
        """Coroutine to receive and dispatch messages from backend."""
        try:
            async for message in ws:
                data = json.loads(message)
                self.message_received.emit(data)
        except websockets.ConnectionClosed:
            logger.debug("WebSocket connection closed during receive.")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug("Error in _receive_messages: %s", e)

    async def _send_messages(self, ws: Any):
        """Coroutine to send outgoing messages from the queue."""
        try:
            while True:
                msg = await self._send_queue.get()
                await ws.send(json.dumps(msg))
                self._send_queue.task_done()
        except websockets.ConnectionClosed:
            logger.debug("WebSocket connection closed during send.")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug("Error in _send_messages: %s", e)

    def trigger_reconnect(self):
        """Manually triggers a reconnection attempt."""
        self.retry_count = 0
        self.is_offline = False

    def send_sign_event(self, sign_data: dict):
        """Dispatch a sign detection event to the backend. Thread-safe."""
        payload = {
            "type": "SIGN_TRANSLATION",
            "data": sign_data
        }
        self._enqueue_safe(payload)

    def send_speech_event(self, transcript: str):
        """Dispatch a speech transcript event to the backend. Thread-safe."""
        payload = {
            "type": "SPEECH_TEXT",
            "data": {
                "transcript": transcript,
                "is_final": True
            }
        }
        self._enqueue_safe(payload)

    def _enqueue_safe(self, payload: dict):
        """Safely appends to send queue while preventing memory bloat when disconnected."""
        # Bound queue size to prevent unbounded memory growth if offline
        try:
            while self._send_queue.qsize() >= 50:
                self._send_queue.get_nowait()
                self._send_queue.task_done()
        except Exception:
            pass

        if self._loop is not None and self._loop.is_running():
            try:
                self._loop.call_soon_threadsafe(self._send_queue.put_nowait, payload)
            except (RuntimeError, AttributeError) as e:
                logger.debug("NetworkWorker: Failed to queue event: %s", e)
        else:
            try:
                self._send_queue.put_nowait(payload)
            except Exception as e:
                logger.debug("NetworkWorker: Failed to queue event directly: %s", e)

    def stop(self):
        """Gracefully stop the network worker."""
        self._is_running = False

        def cancel_all():
            if self._ws:
                try:
                    asyncio.create_task(self._ws.close())
                except Exception:
                    pass
            for task in asyncio.all_tasks(loop=self._loop):
                task.cancel()

        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(cancel_all)

        self.wait()
