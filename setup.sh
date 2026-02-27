#!/bin/bash
# Setup script for Differentiable MPC experiments
# Usage: source setup.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create and activate a virtual environment if not already in one
if [ -z "$VIRTUAL_ENV" ]; then
    if [ ! -d "${SCRIPT_DIR}/.venv" ]; then
        echo "Creating virtual environment in .venv/ ..."
        python3 -m venv "${SCRIPT_DIR}/.venv"
    fi
    source "${SCRIPT_DIR}/.venv/bin/activate"
fi

# Install dependencies only if not already installed
if ! python3 -c "import setproctitle" 2>/dev/null; then
    echo "Installing dependencies ..."
    pip install -q -r "${SCRIPT_DIR}/requirements.txt"
fi

export PYTHONPATH="${SCRIPT_DIR}/mpc_pytorch_lib:${PYTHONPATH}"
export PYTHONWARNINGS="ignore::UserWarning"
