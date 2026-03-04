#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_t2m_long_horizon.sh [options] [-- hydra_overrides...]

Options:
  --prompt TEXT         Long prompt text
  --prompt-file PATH    Read long prompt text from file
  --exp-name NAME       Experiment name prefix (default: t2m_long_horizon)
  --planning-horizon N  Planning horizon in 20fps units (default: 40)
  --episode-length N    Optional override for env.episode_length
  --record-frames N|auto
                        Recording cap in rendered frames. "auto" uses
                        episode_length * controlFrequencyInv (default: auto)
  --isaac-display D     VirtualGL display for Isaac Gym (default: :0)
  --plan-only           Generate and print schedule, do not run simulation
  --help                Show this help
EOF
}

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-closd}"
EXP_NAME="${EXP_NAME:-t2m_long_horizon}"
OUTPUT_PATH="${OUTPUT_PATH:-output/CLoSD/CLoSD_no_finetune}"
NUM_ENVS="${NUM_ENVS:-1}"
PLANNING_HORIZON="${PLANNING_HORIZON:-40}"
EPISODE_LENGTH_OVERRIDE=""
PROMPT_TEXT="${PROMPT:-}"
PROMPT_FILE="${PROMPT_FILE:-}"
RECORD_FRAMES="${RECORD_FRAMES:-auto}"
ISAAC_DISPLAY="${ISAAC_DISPLAY:-:0}"
VNC_DISPLAY="${VNC_DISPLAY:-:1}"
PLAN_ONLY=0
CONTROL_FREQ_INV="${CONTROL_FREQ_INV:-2}"

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt)
      PROMPT_TEXT="${2:?Missing value for --prompt}"
      shift 2
      ;;
    --prompt=*)
      PROMPT_TEXT="${1#*=}"
      shift
      ;;
    --prompt-file)
      PROMPT_FILE="${2:?Missing value for --prompt-file}"
      shift 2
      ;;
    --prompt-file=*)
      PROMPT_FILE="${1#*=}"
      shift
      ;;
    --exp-name)
      EXP_NAME="${2:?Missing value for --exp-name}"
      shift 2
      ;;
    --exp-name=*)
      EXP_NAME="${1#*=}"
      shift
      ;;
    --planning-horizon)
      PLANNING_HORIZON="${2:?Missing value for --planning-horizon}"
      shift 2
      ;;
    --planning-horizon=*)
      PLANNING_HORIZON="${1#*=}"
      shift
      ;;
    --episode-length)
      EPISODE_LENGTH_OVERRIDE="${2:?Missing value for --episode-length}"
      shift 2
      ;;
    --episode-length=*)
      EPISODE_LENGTH_OVERRIDE="${1#*=}"
      shift
      ;;
    --record-frames)
      RECORD_FRAMES="${2:?Missing value for --record-frames}"
      shift 2
      ;;
    --record-frames=*)
      RECORD_FRAMES="${1#*=}"
      shift
      ;;
    --isaac-display)
      ISAAC_DISPLAY="${2:?Missing value for --isaac-display}"
      shift 2
      ;;
    --isaac-display=*)
      ISAAC_DISPLAY="${1#*=}"
      shift
      ;;
    --plan-only)
      PLAN_ONLY=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

RESOLVED_CONTROL_FREQ_INV="$CONTROL_FREQ_INV"
for extra_arg in "${EXTRA_ARGS[@]}"; do
  case "$extra_arg" in
    env.controlFrequencyInv=*)
      RESOLVED_CONTROL_FREQ_INV="${extra_arg#*=}"
      ;;
  esac
done

if ! [[ "$RESOLVED_CONTROL_FREQ_INV" =~ ^[0-9]+$ ]] || [ "$RESOLVED_CONTROL_FREQ_INV" -le 0 ]; then
  echo "Error: controlFrequencyInv must be a positive integer. Resolve via CONTROL_FREQ_INV or env.controlFrequencyInv override." >&2
  exit 2
fi

if ! [[ "$PLANNING_HORIZON" =~ ^[0-9]+$ ]] || [ "$PLANNING_HORIZON" -le 0 ]; then
  echo "Error: --planning-horizon must be a positive integer." >&2
  exit 2
fi

if [ -n "$EPISODE_LENGTH_OVERRIDE" ]; then
  if ! [[ "$EPISODE_LENGTH_OVERRIDE" =~ ^[0-9]+$ ]] || [ "$EPISODE_LENGTH_OVERRIDE" -le 0 ]; then
    echo "Error: --episode-length must be a positive integer." >&2
    exit 2
  fi
fi

if [ "$RECORD_FRAMES" != "auto" ]; then
  if ! [[ "$RECORD_FRAMES" =~ ^[0-9]+$ ]] || [ "$RECORD_FRAMES" -le 0 ]; then
    echo "Error: --record-frames must be a positive integer or 'auto'." >&2
    exit 2
  fi
fi

if [ -n "$PROMPT_TEXT" ] && [ -n "$PROMPT_FILE" ]; then
  echo "Error: --prompt and --prompt-file are mutually exclusive." >&2
  exit 2
fi
if [ -z "$PROMPT_TEXT" ] && [ -z "$PROMPT_FILE" ]; then
  echo "Error: one of --prompt or --prompt-file is required." >&2
  exit 2
fi

if [ -n "$PROMPT_FILE" ]; then
  if [ ! -f "$PROMPT_FILE" ]; then
    echo "Error: --prompt-file not found: $PROMPT_FILE" >&2
    exit 2
  fi
  PROMPT_TEXT="$(<"$PROMPT_FILE")"
fi

trimmed_prompt="${PROMPT_TEXT#"${PROMPT_TEXT%%[![:space:]]*}"}"
trimmed_prompt="${trimmed_prompt%"${trimmed_prompt##*[![:space:]]}"}"
PROMPT_TEXT="$trimmed_prompt"
if [ -z "$PROMPT_TEXT" ]; then
  echo "Error: prompt is empty after trimming whitespace." >&2
  exit 2
fi

if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
else
  echo "Warning: conda.sh not found. Proceeding without explicit conda activation." >&2
fi

export DISPLAY="${DISPLAY:-$VNC_DISPLAY}"
cd "$REPO_DIR"

planner_summary_json="$(
  python scripts/plan_t2m_long_horizon.py \
    --prompt "$PROMPT_TEXT" \
    --planning-horizon "$PLANNING_HORIZON"
)"

mapfile -t planner_fields < <(
  python - "$planner_summary_json" <<'PY'
import json
import sys

summary = json.loads(sys.argv[1])
print(summary["schedule_path"])
print(int(summary["episode_length"]))
print(int(summary["total_horizons"]))
PY
)

SCHEDULE_PATH="${planner_fields[0]}"
PLANNED_EPISODE_LENGTH="${planner_fields[1]}"
TOTAL_HORIZONS="${planner_fields[2]}"

FINAL_EPISODE_LENGTH="$PLANNED_EPISODE_LENGTH"
if [ -n "$EPISODE_LENGTH_OVERRIDE" ]; then
  FINAL_EPISODE_LENGTH="$EPISODE_LENGTH_OVERRIDE"
fi

FINAL_RECORD_FRAMES="$RECORD_FRAMES"
if [ "$FINAL_RECORD_FRAMES" = "auto" ]; then
  FINAL_RECORD_FRAMES="$((FINAL_EPISODE_LENGTH * RESOLVED_CONTROL_FREQ_INV))"
fi

export CLOSD_AUTO_RECORD=1
export CLOSD_AUTO_RECORD_FRAMES="$FINAL_RECORD_FRAMES"

echo "Prompt: $PROMPT_TEXT"
echo "Planning horizon (20fps): $PLANNING_HORIZON"
echo "Total horizons: $TOTAL_HORIZONS"
echo "Planned episode length: $PLANNED_EPISODE_LENGTH"
echo "Final episode length: $FINAL_EPISODE_LENGTH"
echo "controlFrequencyInv: $RESOLVED_CONTROL_FREQ_INV"
echo "Record frames: $FINAL_RECORD_FRAMES"
echo "Schedule JSON: $SCHEDULE_PATH"

if [ "$PLAN_ONLY" -eq 1 ]; then
  exit 0
fi

escape_hydra_string() {
  local escaped="$1"
  escaped="${escaped//\\/\\\\}"
  escaped="${escaped//\"/\\\"}"
  printf '"%s"' "$escaped"
}

hydra_schedule_path="$(escape_hydra_string "$SCHEDULE_PATH")"

cmd=(
  python closd/run.py
  learning=im_big
  robot=smpl_humanoid
  test=True
  no_log=True
  epoch=-1
  headless=False
  no_virtual_display=True
  env=closd_t2m
  env.num_envs="$NUM_ENVS"
  env.episode_length="$FINAL_EPISODE_LENGTH"
  learning.params.config.player.games_num=1
  output_path="$OUTPUT_PATH"
  exp_name="$EXP_NAME"
  env.use_dataset_prefix=False
  env.dip.planning_horizon="$PLANNING_HORIZON"
  'env.custom_prompt=""'
  "env.segment_schedule_path=$hydra_schedule_path"
)

set +e
if command -v vglrun >/dev/null 2>&1; then
  vglrun -d "$ISAAC_DISPLAY" "${cmd[@]}" "${EXTRA_ARGS[@]}"
  run_status=$?
else
  echo "Warning: vglrun not found. Running without VirtualGL." >&2
  "${cmd[@]}" "${EXTRA_ARGS[@]}"
  run_status=$?
fi
set -e

if [ "$run_status" -ne 0 ]; then
  echo "CLoSD run failed with exit code $run_status." >&2
  exit "$run_status"
fi

latest_mp4="$(ls -1t output/renderings/"$EXP_NAME"-*.mp4 2>/dev/null | head -n1 || true)"
if [ -z "$latest_mp4" ]; then
  echo "Run completed but no MP4 found under output/renderings for exp_name=$EXP_NAME." >&2
  echo "Check recording setup (DISPLAY, VirtualGL, CLOSD_AUTO_RECORD settings)." >&2
  exit 1
fi

base_name="$(basename "$latest_mp4" .mp4)"
state_pkl="output/states/${base_name}.pkl"
if [ -f "$state_pkl" ]; then
  rm -f "$state_pkl"
fi

echo "Final MP4: $latest_mp4"
