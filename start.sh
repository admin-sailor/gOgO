#!/usr/bin/env bash
set -e
cd backend
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
PORT="${PORT:-8080}"
exec gunicorn app:app --bind "0.0.0.0:${PORT}" --workers 1 --threads 2 --timeout 120
