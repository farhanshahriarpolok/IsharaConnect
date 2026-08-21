#!/bin/bash
set -e

echo "[IsharaConnect Entrypoint] Starting production initialization..."

# Check if PostgreSQL database is targeted
if [[ "$DATABASE_URL" == *"postgres"* ]]; then
    echo "[IsharaConnect Entrypoint] Probing PostgreSQL database connectivity..."
    python - << 'EOF'
import os
import time
import socket
from urllib.parse import urlparse

db_url = os.getenv("DATABASE_URL", "")
try:
    if "://" in db_url:
        clean_url = "http://" + db_url.split("://", 1)[1]
        parsed = urlparse(clean_url)
        host = parsed.hostname or "db"
        port = parsed.port or 5432
    else:
        host = "db"
        port = 5432

    print(f"[IsharaConnect Entrypoint] Checking connection to {host}:{port}...")
    max_retries = 30
    for attempt in range(1, max_retries + 1):
        try:
            with socket.create_connection((host, port), timeout=2):
                print(f"[IsharaConnect Entrypoint] Successfully connected to database at {host}:{port}")
                break
        except OSError:
            if attempt % 5 == 0:
                print(f"[IsharaConnect Entrypoint] Waiting for DB (attempt {attempt}/{max_retries})...")
            time.sleep(1)
    else:
        print("[IsharaConnect Entrypoint] Warning: DB probe timeout reached. Proceeding to bootstrap.")
except Exception as e:
    print(f"[IsharaConnect Entrypoint] DB probe check encountered non-fatal error: {e}")
EOF
fi

# Run database schema migrations / initialization
echo "[IsharaConnect Entrypoint] Initializing database tables..."
python -m backend.db.init_db || echo "[IsharaConnect Entrypoint] Database initialization finished."

# Execute Uvicorn application server
echo "[IsharaConnect Entrypoint] Launching Uvicorn ASGI server..."
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 2
