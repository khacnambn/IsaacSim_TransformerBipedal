#!/bin/bash
# filepath: transformer_nam/isaaclab.sh
set -e

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate conda env
echo "[INFO] Activating conda env: isaaclab"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate isaaclab

# ✅ SOURCE ISAAC SIM ENVIRONMENT
ISAAC_SIM_PATH="${HOME}/IsaacLab/isaac-sim"

if [ ! -d "${ISAAC_SIM_PATH}" ]; then
    echo "[ERROR] Isaac Sim not found at: ${ISAAC_SIM_PATH}"
    exit 1
fi

echo "[INFO] Setting up Isaac Sim environment from: ${ISAAC_SIM_PATH}"

# Source Isaac Sim setup script
source "${ISAAC_SIM_PATH}/setup_python_env.sh"

# Add Isaac Sim Python paths
export PYTHONPATH="${ISAAC_SIM_PATH}/kit/python/lib/python3.11/site-packages:${PYTHONPATH}"
export PYTHONPATH="${ISAAC_SIM_PATH}/exts/omni.isaac.kit/pip_prebundle:${PYTHONPATH}"
export PYTHONPATH="${ISAAC_SIM_PATH}/python_packages:${PYTHONPATH}"

# Add extension source to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}/source"

echo "[INFO] Running: python $@"
python "$@"