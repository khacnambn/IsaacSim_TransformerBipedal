#!/bin/bash
set -e

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add extension source to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}/source"

# Run IsaacLab
~/IsaacLab/isaaclab.sh "$@"
