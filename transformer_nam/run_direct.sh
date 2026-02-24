#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Isaac Sim python
ISAAC_PYTHON="${HOME}/IsaacLab/isaac-sim/python.sh"

if [ ! -f "$ISAAC_PYTHON" ]; then
    echo "❌ Isaac Sim python not found at: $ISAAC_PYTHON"
    exit 1
fi

# Add extension and IsaacLab to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}/source"
export PYTHONPATH="${PYTHONPATH}:${HOME}/IsaacLab/source"

echo "✅ Using Isaac Sim Python: $ISAAC_PYTHON"

# Run
"$ISAAC_PYTHON" "$@"
