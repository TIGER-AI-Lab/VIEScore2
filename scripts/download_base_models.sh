#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Download Qwen3-VL base model into checkpoints/.
#
# Usage:
#   bash scripts/download_base_models.sh
#
# Downloads:
#   checkpoints/qwen3_8B_base  <- Qwen/Qwen3-VL-8B-Instruct
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

MODELS=(
    "Qwen/Qwen3-VL-8B-Instruct|qwen3_8B_base"
)

CHECKPOINT_BASE="${CHECKPOINT_BASE:-$DATA_ROOT/checkpoints}"
mkdir -p "${CHECKPOINT_BASE}"

for entry in "${MODELS[@]}"; do
    IFS='|' read -r hf_name local_name <<< "$entry"
    local_dir="${CHECKPOINT_BASE}/${local_name}"

    echo "============================================"
    echo "Downloading $hf_name"
    echo "  -> $local_dir"
    echo "============================================"

    if [[ -d "$local_dir" ]] && [[ -f "$local_dir/config.json" ]]; then
        echo "Already exists, skipping."
        echo ""
        continue
    fi

    hf download "$hf_name" --local-dir "$local_dir"
    echo "[done] $hf_name -> $local_dir"
    echo ""
done

echo "============================================"
echo "All base models downloaded."
echo "============================================"
