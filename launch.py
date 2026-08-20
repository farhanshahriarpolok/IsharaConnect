"""Unified launcher for IsharaConnect Backend and Desktop Client."""

import logging
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("launcher")


def check_server_health(url: str, retries: int = 10, delay: float = 1.0) -> bool:
    """Poll the backend health endpoint until it is ready."""
    for i in range(retries):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
            
        logger.info("Waiting for backend server to start... (%d/%d)", i + 1, retries)
        time.sleep(delay)
        
    return False


def main():
    logger.info("Starting IsharaConnect Unified Launcher...")
    
    # 1. Start FastAPI Backend in background
    logger.info("Launching FastAPI backend process...")
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )
    
    # Graceful shutdown handler
    def cleanup(signum, frame):
        logger.info("Shutting down processes...")
        backend_process.terminate()
        backend_process.wait()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    # 2. Wait for backend to be healthy
    health_url = "http://127.0.0.1:8000/api/v1/health"
    if not check_server_health(health_url):
        logger.error("Backend server failed to start within the expected time. Aborting.")
        backend_process.terminate()
        sys.exit(1)
        
    logger.info("Backend server is healthy and running.")
    
    # 3. Start PyQt6 Desktop Client
    logger.info("Launching PyQt6 Desktop Client...")
    desktop_script = Path("desktop_app/main.py").resolve()
    
    try:
        # Blocking call to keep launcher alive while UI runs
        subprocess.run([sys.executable, str(desktop_script)], check=True)
    except subprocess.CalledProcessError as e:
        logger.error("Desktop client exited with error: %s", e)
    except KeyboardInterrupt:
        pass
    finally:
        # 4. Cleanup on exit
        logger.info("Cleaning up backend process...")
        backend_process.terminate()
        backend_process.wait()
        logger.info("IsharaConnect Launcher exited gracefully.")


if __name__ == "__main__":
    main()
