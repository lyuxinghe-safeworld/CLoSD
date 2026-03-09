#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/view_xml.sh [XML_PATH] [-- <extra python args>]

Examples:
  bash scripts/view_xml.sh
  bash scripts/view_xml.sh closd/data/robot_cache/smpl_humanoid_0.xml
  bash scripts/view_xml.sh /abs/path/model.xml -- --dry-run --base-mode fixed
  bash scripts/view_xml.sh /home/lyuxinghe/code/CLoSD/closd/data/robot_cache/smpl_humanoid_0.xml -- --base-mode fixed
EOF
}

fail() {
  echo "Error: $*" >&2
  exit 1
}

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || fail "Required command not found: $cmd"
}

display_to_socket() {
  local disp="$1"
  local dnum="${disp#:}"
  dnum="${dnum%%.*}"
  echo "/tmp/.X11-unix/X${dnum}"
}

is_display_ready() {
  local disp="$1"
  local socket_path
  socket_path="$(display_to_socket "$disp")"
  [ -S "$socket_path" ] || return 1
  DISPLAY="$disp" xdpyinfo >/dev/null 2>&1
}

wait_for_display() {
  local disp="$1"
  local timeout_sec="$2"
  local i
  for ((i = 0; i < timeout_sec; i++)); do
    if is_display_ready "$disp"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

resolve_xml_path() {
  local raw="$1"
  if [[ "$raw" = /* ]]; then
    printf '%s\n' "$raw"
  else
    printf '%s/%s\n' "$REPO_DIR" "$raw"
  fi
}

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_XML_REL="closd/data/robot_cache/smpl_humanoid_0.xml"

CONDA_ENV="${CONDA_ENV:-closd}"
VNC_DISPLAY="${VNC_DISPLAY:-:1}"
ISAAC_DISPLAY="${ISAAC_DISPLAY:-:0}"
VNC_GEOMETRY="${VNC_GEOMETRY:-1600x900}"
VNC_DEPTH="${VNC_DEPTH:-24}"

XML_INPUT="${DEFAULT_XML_REL}"
PY_ARGS=()

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      PY_ARGS=("$@")
      ;;
    *)
      XML_INPUT="$1"
      shift
      if [[ $# -gt 0 ]]; then
        [[ "$1" == "--" ]] || fail "Unexpected argument '$1'. Use: bash scripts/view_xml.sh [XML_PATH] [-- <extra python args>]"
        shift
        PY_ARGS=("$@")
      fi
      ;;
  esac
fi

XML_PATH="$(resolve_xml_path "$XML_INPUT")"
[ -f "$XML_PATH" ] || fail "XML file not found: $XML_PATH"

CONDA_SH="$HOME/miniforge3/etc/profile.d/conda.sh"
[ -f "$CONDA_SH" ] || fail "Conda activation script not found: $CONDA_SH"
# shellcheck disable=SC1090
source "$CONDA_SH" || fail "Failed to source conda script: $CONDA_SH"
conda activate "$CONDA_ENV" || fail "Failed to activate conda env: $CONDA_ENV"

PYTHON_BIN="$(command -v python || true)"
[ -n "$PYTHON_BIN" ] || fail "Python interpreter not found after activating conda env '$CONDA_ENV'"

VNC_SERVER_BIN="/opt/TurboVNC/bin/vncserver"
[ -x "$VNC_SERVER_BIN" ] || fail "TurboVNC server not executable: $VNC_SERVER_BIN"
require_cmd vglrun
require_cmd xdpyinfo

if ! is_display_ready "$VNC_DISPLAY"; then
  echo "[view_xml] TurboVNC display $VNC_DISPLAY not ready; starting it..." >&2
  "$VNC_SERVER_BIN" "$VNC_DISPLAY" -geometry "$VNC_GEOMETRY" -depth "$VNC_DEPTH" >/dev/null
  wait_for_display "$VNC_DISPLAY" 20 || fail "TurboVNC display $VNC_DISPLAY did not become ready within timeout"
fi

export DISPLAY="$VNC_DISPLAY"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"

if [ -n "${CONDA_PREFIX:-}" ] && [ -d "$CONDA_PREFIX/lib" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

if [ -z "${VK_ICD_FILENAMES:-}" ]; then
  if [ -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
    export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
  elif [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
  fi
fi

cd "$REPO_DIR"
exec vglrun -d "$ISAAC_DISPLAY" "$PYTHON_BIN" "$REPO_DIR/scripts/view_xml.py" --xml "$XML_PATH" "${PY_ARGS[@]}"
