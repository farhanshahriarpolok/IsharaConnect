"""Entry point for the IsharaConnect Desktop Client."""

import argparse
import sys
import logging

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon

from desktop_app.ui.main_window import IsharaMainWindow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(description="IsharaConnect Desktop Client")
    parser.add_argument("--mode", type=str, choices=["signer", "speaker"], default="signer", 
                        help="Start in Signer (Deaf) or Speaker (Hearing) mode")
    parser.add_argument("--room", type=str, default="room_public_01", 
                        help="Default Room ID to connect to")
    parser.add_argument("--server", type=str, default="ws://127.0.0.1:8000", 
                        help="FastAPI WebSocket server URL")
    
    args = parser.parse_args()

    app = QApplication(sys.argv)
    
    # Optional: Set global app icon
    # app.setWindowIcon(QIcon("desktop_app/resources/icon.png"))
    
    window = IsharaMainWindow(
        mode=args.mode,
        room_id=args.room,
        server_url=args.server
    )
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
