#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "==> Building and starting containers..."
docker compose up -d --build

echo "==> Waiting for app to be healthy..."
timeout=120
elapsed=0
until docker compose exec app python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" 2>/dev/null; do
    if [ $elapsed -ge $timeout ]; then
        echo "ERROR: App failed to start within ${timeout}s"
        docker compose logs app
        exit 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    echo "    Waiting... (${elapsed}s)"
done

echo "==> App is running at http://localhost:${APP_PORT:-8000}"
docker compose ps
