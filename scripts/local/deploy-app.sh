#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Validate model checkpoints exist
MODELS=("exp14b_finetuned_resnet50_cnn.pt" "exp16_fusion_featknn_graphsage.pt" "exp23_gat_fusion.pt" "exp25_boc_sage.pt" "exp26_gated_boc.pt" "exp27_attn_pool.pt")
MISSING=0
for m in "${MODELS[@]}"; do
    if [ ! -f "models/$m" ]; then
        echo "WARNING: Missing model checkpoint: models/$m"
        MISSING=$((MISSING + 1))
    fi
done
if [ $MISSING -gt 0 ]; then
    echo "ERROR: $MISSING model checkpoint(s) missing. Place them in models/ before running."
    exit 1
fi

# Validate .env.local exists
if [ ! -f ".env.local" ]; then
    echo "==> .env.local not found, creating from .env.example..."
    cp .env.example .env.local
fi

echo "==> Rebuilding and restarting app container..."
docker compose up -d --build app

echo "==> Tailing logs (Ctrl+C to stop)..."
docker compose logs -f app
