#!/usr/bin/env bash
# startup.sh — Start (or stop) all services for the doc-classification demo.
#
# Services managed:
#   app            FastAPI demo app          Docker Compose  → http://localhost:{APP_PORT}
#   evidently      Evidently UI dashboard    Local poetry    → http://localhost:8080
#
# Usage:
#   scripts/startup.sh              # start both services
#   scripts/startup.sh --reset      # rebuild Docker image before starting
#   scripts/startup.sh --logs       # tail app logs after startup
#   scripts/startup.sh --monitoring # also run the Evidently batch job on startup
#   scripts/startup.sh --stop       # stop both services cleanly
#   scripts/startup.sh -h           # show this help
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Config ────────────────────────────────────────────────────────────────────
APP_SERVICE="app"
EVIDENTLY_PORT=8080
EVIDENTLY_WORKSPACE="$PROJECT_ROOT/monitoring/evidently_workspace"
EVIDENTLY_PIDFILE="/tmp/evidently-doc-classification.pid"
REFERENCE_PARQUET="$PROJECT_ROOT/monitoring/reference/reference_dataset.parquet"

MODELS=(
    "exp14b_finetuned_resnet50_cnn.pt"
    "exp16_fusion_featknn_graphsage.pt"
    "exp23_gat_fusion.pt"
    "exp25_boc_sage.pt"
    "exp26_gated_boc.pt"
    "exp27_attn_pool.pt"
)

# ── Flags ─────────────────────────────────────────────────────────────────────
RESET=0
FOLLOW_LOGS=0
RUN_MONITORING=0
STOP=0

# ── Help ──────────────────────────────────────────────────────────────────────
usage() {
    cat <<'EOF'
Usage: scripts/startup.sh [OPTIONS]

Options:
  --reset       Stop app, rebuild Docker image with --no-cache, recreate container.
  --logs        Follow app logs after startup (Ctrl+C to detach).
  --monitoring  Run the Evidently batch job (last 24h) right after startup.
  --stop        Stop all services (app container + Evidently UI server).
  -h, --help    Show this help text.

Services started:
  app            FastAPI demo  (Docker Compose)  → http://localhost:<APP_PORT>
  evidently ui   Drift dashboard (poetry local)  → http://localhost:8080

Notes:
  - Model checkpoints must exist in models/ before the app can start.
  - The Evidently UI server reads monitoring/evidently_workspace directly.
  - To push new monitoring snapshots manually:
      poetry run python scripts/monitoring/run_evidently.py --window-hours 24
EOF
}

# ── Argument parsing ──────────────────────────────────────────────────────────
while [ "$#" -gt 0 ]; do
    case "$1" in
        --reset)      RESET=1 ;;
        --logs)       FOLLOW_LOGS=1 ;;
        --monitoring) RUN_MONITORING=1 ;;
        --stop)       STOP=1 ;;
        -h|--help)    usage; exit 0 ;;
        *)
            echo "ERROR: Unknown argument: $1"
            echo
            usage
            exit 1
            ;;
    esac
    shift
done

# ── Helpers ───────────────────────────────────────────────────────────────────
require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: Required command not found: $1"
        exit 1
    fi
}

ensure_env_file() {
    if [ -f "$PROJECT_ROOT/.env.local" ]; then
        return
    fi
    if [ ! -f "$PROJECT_ROOT/.env.example" ]; then
        echo "ERROR: .env.local missing and .env.example not available to bootstrap it."
        exit 1
    fi
    echo "==> .env.local not found. Creating from .env.example"
    cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env.local"
}

load_env_file() {
    set -a
    # shellcheck source=/dev/null
    source "$PROJECT_ROOT/.env.local"
    set +a
}

validate_models() {
    local missing=0
    for model in "${MODELS[@]}"; do
        if [ ! -f "$PROJECT_ROOT/models/$model" ]; then
            echo "WARNING: Missing model checkpoint: models/$model"
            missing=$((missing + 1))
        fi
    done
    if [ "$missing" -gt 0 ]; then
        echo "ERROR: $missing model checkpoint(s) missing. Place them in models/ before startup."
        exit 1
    fi
}

ensure_monitoring_dirs() {
    mkdir -p "$PROJECT_ROOT/monitoring/data"
    mkdir -p "$PROJECT_ROOT/monitoring/output"
    mkdir -p "$PROJECT_ROOT/monitoring/reference"
    mkdir -p "$EVIDENTLY_WORKSPACE"

    if [ ! -f "$REFERENCE_PARQUET" ]; then
        echo "WARNING: Reference dataset not found."
        echo "         Generate it with:"
        echo "         poetry run python scripts/monitoring/bootstrap_reference.py --synthetic"
    fi
}

wait_for_app() {
    local timeout=120
    local elapsed=0
    local app_port="${APP_PORT:-9000}"

    echo "==> Waiting for app to be ready on port ${app_port}"
    until docker compose exec -T "$APP_SERVICE" \
        python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"APP_PORT\",\"9000\")}/')" \
        >/dev/null 2>&1; do
        if [ "$elapsed" -ge "$timeout" ]; then
            echo "ERROR: App failed to become ready within ${timeout}s"
            docker compose logs "$APP_SERVICE"
            exit 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        echo "    Waiting... (${elapsed}s)"
    done
}

# ── Evidently UI server ───────────────────────────────────────────────────────
evidently_is_running() {
    if [ -f "$EVIDENTLY_PIDFILE" ]; then
        local pid
        pid=$(cat "$EVIDENTLY_PIDFILE")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

start_evidently() {
    if evidently_is_running; then
        echo "==> Evidently UI already running (PID $(cat "$EVIDENTLY_PIDFILE"))"
        return
    fi

    echo "==> Starting Evidently UI server on port ${EVIDENTLY_PORT}"
    cd "$PROJECT_ROOT"
    poetry run evidently ui \
        --workspace "$EVIDENTLY_WORKSPACE" \
        --port "$EVIDENTLY_PORT" \
        --host 0.0.0.0 \
        > /tmp/evidently-doc-classification.log 2>&1 &

    echo $! > "$EVIDENTLY_PIDFILE"
    echo "    PID $(cat "$EVIDENTLY_PIDFILE") — logs: tail /tmp/evidently-doc-classification.log"

    # Wait for it to bind the port
    local attempts=0
    until curl -sf "http://localhost:${EVIDENTLY_PORT}" >/dev/null 2>&1; do
        sleep 1
        attempts=$((attempts + 1))
        if [ "$attempts" -ge 15 ]; then
            echo "WARNING: Evidently UI did not respond after 15s — check logs:"
            echo "         tail /tmp/evidently-doc-classification.log"
            return
        fi
    done
    echo "    Evidently UI ready"
}

stop_evidently() {
    if evidently_is_running; then
        local pid
        pid=$(cat "$EVIDENTLY_PIDFILE")
        echo "==> Stopping Evidently UI server (PID ${pid})"
        kill "$pid" 2>/dev/null || true
        rm -f "$EVIDENTLY_PIDFILE"
    else
        echo "==> Evidently UI not running"
    fi
}

run_monitoring_job() {
    echo "==> Running Evidently batch job (last 24h)"
    cd "$PROJECT_ROOT"
    poetry run python scripts/monitoring/run_evidently.py --window-hours 24
}

# ── Stop mode ─────────────────────────────────────────────────────────────────
if [ "$STOP" -eq 1 ]; then
    cd "$PROJECT_ROOT"
    echo "==> Stopping app container"
    docker compose down --remove-orphans
    stop_evidently
    echo ""
    echo "All services stopped."
    exit 0
fi

# ── Pre-flight checks ─────────────────────────────────────────────────────────
require_command docker
require_command poetry

if ! docker compose version >/dev/null 2>&1; then
    echo "ERROR: docker compose is required but not available."
    exit 1
fi

cd "$PROJECT_ROOT"
ensure_env_file
load_env_file
validate_models
ensure_monitoring_dirs

# ── Start app (Docker) ────────────────────────────────────────────────────────
if [ "$RESET" -eq 1 ]; then
    echo "==> Resetting app: stopping, removing image, rebuilding"
    docker compose down --remove-orphans --rmi local
    docker compose build --no-cache "$APP_SERVICE"
fi

echo "==> Starting app container"
docker compose up -d --force-recreate "$APP_SERVICE"
wait_for_app

# ── Start Evidently UI (local poetry) ─────────────────────────────────────────
start_evidently

# ── Optional: run monitoring batch job ───────────────────────────────────────
if [ "$RUN_MONITORING" -eq 1 ]; then
    run_monitoring_job
fi

# ── Summary ───────────────────────────────────────────────────────────────────
APP_PORT="${APP_PORT:-9000}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Services running"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Demo app       http://localhost:${APP_PORT}"
echo "  Drift Monitor  http://localhost:${EVIDENTLY_PORT}"
echo ""
echo "  Stop all:       scripts/startup.sh --stop"
echo "  Push snapshots: poetry run python scripts/monitoring/run_evidently.py --window-hours 24"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

docker compose ps

if [ "$FOLLOW_LOGS" -eq 1 ]; then
    echo ""
    echo "==> Tailing app logs (Ctrl+C to stop)"
    docker compose logs -f "$APP_SERVICE"
fi
