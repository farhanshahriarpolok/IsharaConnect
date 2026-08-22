# IsharaConnect - Production Deployment Guide

This document outlines the deployment, containerization, reverse proxying, and packaging workflows for **IsharaConnect**.

---

## 📋 Table of Contents
1. [Docker Compose 1-Click Deployment](#1-docker-compose-1-click-deployment)
2. [Standalone Windows Executable Distribution](#2-standalone-windows-executable-distribution)
3. [Production Nginx & SSL/TLS Configuration](#3-production-nginx--ssltls-configuration)
4. [Environment Variables Reference](#4-environment-variables-reference)
5. [Health Checks & Observability](#5-health-checks--observability)

---

## 1. Docker Compose 1-Click Deployment

IsharaConnect supports multi-container orchestration with automatic Redis Pub/Sub room clustering and volume persistence.

### Prerequisites
- Docker Engine 24.0+
- Docker Compose v2+

### Running the Stack
```bash
# 1. Clone the repository
git clone https://github.com/farhanshahriarpolok/IsharaConnect.git
cd IsharaConnect

# 2. Copy and configure environment variables
cp .env.example .env

# 3. Start services in detached mode
docker-compose up -d --build
```

### Services Deployed
| Service Name | Port | Description |
| :--- | :--- | :--- |
| `backend` | `8000` | FastAPI WebRTC server, REST APIs & WebSocket Hub |
| `redis` | `6379` | Distributed WebSocket Room Event Broker |

### Verifying Container Health
```bash
docker-compose ps
curl -f http://localhost:8000/health
```

---

## 2. Standalone Windows Executable Distribution

You can package IsharaConnect as a standalone, portable Windows distribution requiring no pre-installed Python interpreter or external dependencies.

### Packaging Steps
```powershell
# 1. Activate development virtual environment
.venv\Scripts\activate

# 2. Run the packaging build script
python scripts/build_windows_exe.py

# 3. Verify the generated executable bundle
.\dist\IsharaConnect\IsharaConnect.exe
```

The output folder `dist/IsharaConnect/` contains all required ONNX models, MediaPipe assets, WebRTC HTML templates, and SQLite database engines.

---

## 3. Production Nginx & SSL/TLS Configuration

When exposing IsharaConnect to public networks, place an Nginx reverse proxy in front of the Uvicorn application to handle SSL termination, static file caching, and WebSocket connection upgrades.

### Example Nginx Virtual Host (`/etc/nginx/sites-available/isharaconnect.conf`)
```nginx
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}

server {
    listen 80;
    server_name ishara.example.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name ishara.example.com;

    # SSL Certificates
    ssl_certificate /etc/letsencrypt/live/ishara.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/ishara.example.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Static Assets & Templates
    location /static/ {
        alias /app/backend/static/;
        expires 7d;
        add_header Cache-Control "public, max-age=604800, immutable";
    }

    # REST APIs & Certificate Verification
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket Real-Time Communication Hub
    location /ws/ {
        proxy_pass http://127.0.0.1:8000/ws/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}
```

---

## 4. Environment Variables Reference

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `ENVIRONMENT` | `production` | Environment mode (`development` / `production` / `testing`) |
| `SECRET_KEY` | *(Must be generated)* | Cryptographic key for JWT sessions & certificate hashes |
| `DATABASE_URL` | `sqlite+aiosqlite:///./backend/isharaconnect.db` | SQLAlchemy Async database connection URI |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URI for distributed WebSocket Pub/Sub |
| `VISION_WORKER_CONCURRENCY` | `4` | Number of worker processes in async vision inference pool |
| `LOG_LEVEL` | `INFO` | Application log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## 5. Health Checks & Observability

### Endpoint Diagnostics
- **System Health**: `GET /health` $\to$ Returns uptime, DB connectivity status, and Redis cluster status.
- **Active WebSocket Rooms**: `GET /api/v1/rooms/active` $\to$ Returns active room counts and participant numbers.
- **Public Certificate Lookup**: `GET /verify/{cert_hash}` $\to$ Renders public proof-of-completion verification card.

---

## 🔒 Security Hardening Recommendations
1. Ensure `SECRET_KEY` is at least 32 cryptographically secure random bytes in production.
2. Protect admin endpoints (`/api/v1/admin/*`) behind role-based access control (RBAC).
3. Enable rate limiting on authentication and certificate verification endpoints.
