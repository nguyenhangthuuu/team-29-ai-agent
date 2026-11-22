#!/usr/bin/env bash
# Belfius ReBel Agentic Trader — Quickstart Script
# Team 29 | QTEM Data Challenge (QDC) Fall 2026

set -e

echo "=========================================================="
echo "   BELFIUS ReBel — Multi-Agent Institutional Terminal    "
echo "=========================================================="

# 1. Virtual Environment Setup
if [ ! -d ".venv" ]; then
    echo "[1/3] Creating Python virtual environment (.venv)..."
    python3 -m venv .venv
fi

echo "[2/3] Activating virtual environment & verifying packages..."
source .venv/bin/activate
pip install -r requirements.txt -q

# 2. Environment file setup
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "Creating local .env configuration from .env.example..."
    cp .env.example .env
fi

# 3. Launch Server
echo "[3/3] Starting Neo-Terminal backend server at http://localhost:8000..."
echo "Open http://localhost:8000 in your browser to interact with the dashboard."
echo "Press CTRL+C to stop."
echo "=========================================================="
exec uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
