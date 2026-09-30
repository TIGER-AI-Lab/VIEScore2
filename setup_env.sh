#!/usr/bin/env bash
# VIEScore2 environment setup — source this at the start of each session
# Usage: source setup_env.sh

DATA_ROOT=$DATA_ROOT

export HF_HOME="${DATA_ROOT}/hf_cache"
export DATALAB_CACHE_DIR="${DATA_ROOT}/datalab_cache"
export MODEL_CACHE_DIR="${DATA_ROOT}/datalab_cache/models"
export PIP_CACHE_DIR="${DATA_ROOT}/pip-cache"
export TMPDIR="${DATA_ROOT}/tmp"
export TRITON_CACHE_DIR="${DATA_ROOT}/.triton"
export CHECKPOINT_BASE="${DATA_ROOT}/checkpoints"
export CLAUDE_CONFIG_DIR="${DATA_ROOT}/VIEScore2/.claude"

mkdir -p "${DATA_ROOT}/tmp"
mkdir -p "${DATA_ROOT}/pip-cache"
mkdir -p "${DATA_ROOT}/checkpoints"

# Activate venv
VENV_DIR="./.venv"
export PATH="${VENV_DIR}/bin:${PATH}"
export VIRTUAL_ENV="${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
hash -r

echo "VIEScore2 environment ready (Python $(python --version 2>&1 | cut -d' ' -f2))"
