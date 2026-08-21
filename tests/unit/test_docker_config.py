"""Unit tests validating Docker, Docker Compose, and Nginx configurations."""

import os
from pathlib import Path
import yaml
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_docker_files_exist():
    """Verify that all essential Dockerization files are present."""
    required_files = [
        "Dockerfile",
        "docker-compose.yml",
        "docker-compose.dev.yml",
        ".dockerignore",
        ".env.example",
        "docker/entrypoint.sh",
        "docker/nginx/nginx.conf",
        "docker/nginx/conf.d/default.conf",
    ]

    for rel_path in required_files:
        target = REPO_ROOT / rel_path
        assert target.exists(), f"Missing required container file: {rel_path}"
        assert target.stat().st_size > 0, f"File {rel_path} is empty"


def test_docker_compose_production_syntax():
    """Verify valid YAML structure and services in docker-compose.yml."""
    compose_path = REPO_ROOT / "docker-compose.yml"
    with open(compose_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "services" in data
    services = data["services"]
    assert "backend" in services
    assert "db" in services
    assert "redis" in services
    assert "nginx" in services

    # Check backend dependencies and networks
    backend = services["backend"]
    assert "depends_on" in backend
    assert "db" in backend["depends_on"]
    assert "redis" in backend["depends_on"]
    assert backend["depends_on"]["db"]["condition"] == "service_healthy"

    # Check database configuration
    db = services["db"]
    assert "image" in db and "postgres" in db["image"]
    assert "healthcheck" in db

    # Check redis configuration
    redis = services["redis"]
    assert "image" in redis and "redis" in redis["image"]
    assert "healthcheck" in redis

    # Check nginx ports
    nginx = services["nginx"]
    assert "80:80" in nginx.get("ports", [])


def test_docker_compose_dev_syntax():
    """Verify valid YAML structure and services in docker-compose.dev.yml."""
    compose_path = REPO_ROOT / "docker-compose.dev.yml"
    with open(compose_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "services" in data
    services = data["services"]
    assert "backend" in services
    assert "db" in services
    assert "redis" in services
    assert "8000:8000" in services["backend"].get("ports", [])


def test_dockerfile_multi_stage_structure():
    """Verify multi-stage build, non-root user, and healthcheck in Dockerfile."""
    dockerfile_path = REPO_ROOT / "Dockerfile"
    content = dockerfile_path.read_text(encoding="utf-8")

    assert "FROM python:3.11-slim as builder" in content
    assert "FROM python:3.11-slim as runner" in content
    assert "useradd" in content and "appuser" in content
    assert "USER appuser" in content
    assert "HEALTHCHECK" in content
    assert "ENTRYPOINT" in content


def test_nginx_configuration():
    """Verify key proxy headers and routing in Nginx configs."""
    nginx_conf = (REPO_ROOT / "docker/nginx/nginx.conf").read_text(encoding="utf-8")
    default_conf = (REPO_ROOT / "docker/nginx/conf.d/default.conf").read_text(encoding="utf-8")

    assert "gzip on;" in nginx_conf
    assert "client_max_body_size 50M;" in nginx_conf

    assert "upstream backend_upstream" in default_conf
    assert "proxy_pass http://backend_upstream;" in default_conf
    assert "Upgrade $http_upgrade;" in default_conf
    assert "location /static/" in default_conf
    assert "location /visual_cards/" in default_conf


def test_env_example_contains_required_keys():
    """Verify that .env.example contains necessary variables."""
    env_content = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")

    expected_keys = [
        "ENVIRONMENT",
        "SECRET_KEY",
        "DATABASE_URL",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DB",
        "REDIS_URL",
        "ALLOWED_ORIGINS",
    ]

    for key in expected_keys:
        assert f"{key}=" in env_content, f"Missing {key} in .env.example"


def test_entrypoint_script_contents():
    """Verify entrypoint script performs DB probe and starts Uvicorn."""
    entrypoint_content = (REPO_ROOT / "docker/entrypoint.sh").read_text(encoding="utf-8")

    assert "#!/bin/bash" in entrypoint_content
    assert "init_db" in entrypoint_content
    assert "uvicorn backend.main:app" in entrypoint_content
