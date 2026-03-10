#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_HUMANOID_XML="${REPO_ROOT}/closd/data/assets/mjcf/smpl_0_humanoid.xml"
HUMANOID_XML="${1:-${DEFAULT_HUMANOID_XML}}"

export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"

if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "Error: python interpreter not found." >&2
    exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/view_mjcf.py" "${HUMANOID_XML}"
