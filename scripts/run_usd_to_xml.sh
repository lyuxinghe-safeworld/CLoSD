#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_usd_to_xml.sh <input.usd|input.xml> [output.xml] [converter args] [--no-view] [-- <view_xml.py args>]

Examples:
  bash scripts/run_usd_to_xml.sh Characters/Biped_Setup.usd output/biped_smpl24.xml
  bash scripts/run_usd_to_xml.sh Characters/Biped_Setup.usd output/biped_smpl24_mesh.xml --mesh-mode visual --hide-template-geoms --no-view
  bash scripts/run_usd_to_xml.sh Characters/Biped_Setup.usd output/biped_smpl24.xml -- --dry-run
  bash scripts/run_usd_to_xml.sh closd/data/robot_cache/smpl_humanoid_0.xml output/smpl_copy.xml --no-view
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

resolve_path() {
  local raw="$1"
  if [[ "$raw" = /* ]]; then
    printf '%s\n' "$raw"
  else
    printf '%s/%s\n' "$REPO_DIR" "$raw"
  fi
}

default_output_for_input() {
  local input_path="$1"
  local base
  base="$(basename "$input_path")"
  local stem="${base%.*}"
  printf '%s/output/%s_smpl24.xml\n' "$REPO_DIR" "$stem"
}

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-closd}"
VNC_DISPLAY="${VNC_DISPLAY:-:1}"
ISAAC_DISPLAY="${ISAAC_DISPLAY:-:0}"
VNC_GEOMETRY="${VNC_GEOMETRY:-1600x900}"
VNC_DEPTH="${VNC_DEPTH:-24}"
NO_VIEW=0
CONVERTER_ARGS=()

PHYSICS_SOURCE_REL="closd/data/robot_cache/smpl_humanoid_0.xml"
MAPPING_REL="closd/data/robot_cache/usd_smpl24_map.json"

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

INPUT_RAW="$1"
shift

OUTPUT_RAW=""
if [[ $# -gt 0 ]]; then
  case "$1" in
    --no-view|--)
      ;;
    *)
      OUTPUT_RAW="$1"
      shift
      ;;
  esac
fi

VIEW_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-view)
      NO_VIEW=1
      shift
      ;;
    --mesh-mode|--mesh-min-triangles|--usd-skel-prim|--usd-mesh-prim|--mesh-overrides|--axis-remap|--root-policy|--mapping|--physics-source)
      [[ $# -ge 2 ]] || fail "Flag '$1' requires a value"
      CONVERTER_ARGS+=("$1" "$2")
      shift 2
      ;;
    --hide-template-geoms|--show-template-geoms|--validate)
      CONVERTER_ARGS+=("$1")
      shift
      ;;
    --)
      shift
      VIEW_ARGS=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unexpected argument '$1'. Use: bash scripts/run_usd_to_xml.sh <input> [output] [converter args] [--no-view] [-- <view args>]"
      ;;
  esac
done

INPUT_PATH="$(resolve_path "$INPUT_RAW")"
[ -f "$INPUT_PATH" ] || fail "Input file not found: $INPUT_PATH"

if [[ -n "$OUTPUT_RAW" ]]; then
  OUTPUT_PATH="$(resolve_path "$OUTPUT_RAW")"
else
  OUTPUT_PATH="$(default_output_for_input "$INPUT_PATH")"
fi

CONDA_SH="$HOME/miniforge3/etc/profile.d/conda.sh"
[ -f "$CONDA_SH" ] || fail "Conda activation script not found: $CONDA_SH"
# shellcheck disable=SC1090
source "$CONDA_SH" || fail "Failed to source conda script: $CONDA_SH"
conda activate "$CONDA_ENV" || fail "Failed to activate conda env: $CONDA_ENV"

PYTHON_BIN="$(command -v python || true)"
[ -n "$PYTHON_BIN" ] || fail "Python interpreter not found after activating conda env '$CONDA_ENV'"

require_cmd xdpyinfo
VNC_SERVER_BIN="/opt/TurboVNC/bin/vncserver"
[ -x "$VNC_SERVER_BIN" ] || fail "TurboVNC server not executable: $VNC_SERVER_BIN"

if ! is_display_ready "$VNC_DISPLAY"; then
  echo "[run_usd_to_xml] TurboVNC display $VNC_DISPLAY not ready; starting it..." >&2
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

mkdir -p "$(dirname "$OUTPUT_PATH")"

cd "$REPO_DIR"
"$PYTHON_BIN" "$REPO_DIR/scripts/convert_usd_to_mjcf.py" \
  --input "$INPUT_PATH" \
  --output "$OUTPUT_PATH" \
  --mode smpl24 \
  --physics-source "$REPO_DIR/$PHYSICS_SOURCE_REL" \
  --mapping "$REPO_DIR/$MAPPING_REL" \
  "${CONVERTER_ARGS[@]}" \
  --validate

echo "[run_usd_to_xml] Output XML: $OUTPUT_PATH"

if [[ "$NO_VIEW" -eq 1 ]]; then
  exit 0
fi

require_cmd vglrun
exec vglrun -d "${ISAAC_DISPLAY:-:0}" "$PYTHON_BIN" "$REPO_DIR/scripts/view_xml.py" --xml "$OUTPUT_PATH" "${VIEW_ARGS[@]}"
