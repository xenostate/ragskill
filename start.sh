#!/bin/bash
# Start the web-rag RAG API server
# Usage: ./start.sh [--port 8090]

SKILL_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SKILL_DIR/.venv/bin"
PORT="${1:-8090}"

echo "🕸️  web-rag server"
echo "   Skill dir: $SKILL_DIR"
echo "   Port:      $PORT"
echo ""

# Check venv
if [ ! -f "$VENV/uvicorn" ]; then
    echo "❌ venv not found. Run:"
    echo "   python3 -m venv $SKILL_DIR/.venv"
    echo "   $VENV/pip install -r $SKILL_DIR/scripts/requirements.txt"
    exit 1
fi

# Check .env
if [ ! -f "$SKILL_DIR/.env" ]; then
    echo "❌ .env not found at $SKILL_DIR/.env"
    exit 1
fi

cd "$SKILL_DIR"
exec "$VENV/uvicorn" scripts.server:app --host 0.0.0.0 --port "$PORT"
