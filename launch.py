"""Unified launcher for IsharaConnect Backend and Desktop Client."""

import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

# Explicitly add project root to PYTHONPATH to prevent module import crashes
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("launcher")


def is_port_in_use(port: int = 8000, host: str = "127.0.0.1") -> bool:
    """Checks if a local TCP port is already open/listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def free_port(port: int = 8000):
    """Terminates orphaned processes occupying the specified port."""
    if not is_port_in_use(port):
        return

    logger.warning("Port %d is currently occupied. Attempting automated port clearance...", port)
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                check=False
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.strip().split()
                    pid = parts[-1]
                    if pid.isdigit() and int(pid) != os.getpid():
                        logger.info("Terminating process PID %s on port %d...", pid, port)
                        subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True, check=False)
                        time.sleep(0.5)
        except Exception as e:
            logger.debug("Port clearance netstat error: %s", e)
    else:
        try:
            subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, check=False)
            time.sleep(0.5)
        except Exception as e:
            logger.debug("Port clearance fuser error: %s", e)


def stream_logs(proc: subprocess.Popen, prefix: str = "[Backend]"):
    """Reads stdout lines from a subprocess and prints them in real time."""
    try:
        if proc.stdout:
            for line in iter(proc.stdout.readline, ""):
                if line:
                    print(f"\033[90m{prefix} {line.strip()}\033[0m", flush=True)
                else:
                    break
    except Exception as e:
        logger.debug("Log streaming ended: %s", e)


def check_server_health(
    urls: List[str],
    process: Optional[subprocess.Popen] = None,
    retries: int = 15,
    delay: float = 0.8
) -> bool:
    """Polls backend health endpoints until ready or process crashes."""
    for i in range(retries):
        if process and process.poll() is not None:
            logger.error("Backend process exited prematurely with code %s.", process.poll())
            return False

        for url in urls:
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=1.5) as response:
                    if response.status == 200:
                        return True
            except Exception:
                pass

        logger.info("Waiting for backend server to start... (%d/%d)", i + 1, retries)
        time.sleep(delay)

    return False


def main():
    logger.info("Starting IsharaConnect Unified Launcher...")

    # 1. Clear any orphaned processes on port 8000
    free_port(8000)

    # 2. Start FastAPI Backend with real-time log forwarding
    logger.info("Launching FastAPI backend process (uvicorn backend.main:app)...")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    backend_process = subprocess.Popen(
        [sys.executable, "-u", "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )

    # Start real-time log stream thread
    log_thread = threading.Thread(target=stream_logs, args=(backend_process,), daemon=True)
    log_thread.start()

    # Graceful shutdown handler
    def cleanup(signum=None, frame=None):
        logger.info("Shutting down IsharaConnect processes...")
        try:
            if backend_process.poll() is None:
                backend_process.terminate()
                backend_process.wait(timeout=3)
        except Exception:
            backend_process.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # 3. Wait for backend to be healthy
    health_urls = [
        "http://127.0.0.1:8000/health",
        "http://127.0.0.1:8000/api/v1/health"
    ]
    if not check_server_health(health_urls, process=backend_process):
        logger.error("Backend server failed to start within the expected time. Aborting.")
        cleanup()

    logger.info("Backend server is healthy and responding.")

    # 4. Start PyQt6 Desktop Client in foreground
    logger.info("Launching PyQt6 Desktop Client in foreground...")
    try:
        from desktop_app.main import main as desktop_main
        desktop_main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error("Desktop client exited with error: %s", e)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
