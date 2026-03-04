#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_t2m_condition.sh [options]

Options:
  --episode-length N   Episode length (default: 300)
  --exp-name NAME      Experiment name prefix for output files (default: t2m_one_condition)
  --index N            Non-interactive caption index
  --prompt TEXT        Use a direct custom prompt (overrides --index)
  --prompt-file PATH   Load prompt text from file (mutually exclusive with --prompt)
  --use-dataset-prefix Enable HumanML dataset prefix lead-in motion
  --no-dataset-prefix  Disable HumanML dataset prefix lead-in motion (default)
  --record-frames N    Auto-stop recording after N frames
  --record-until-exit  Keep recording until sim/process exit (default)
  --isaac-display D    VirtualGL display for Isaac Gym (default: :0)
  --list-only          Print indexed condition list and exit
                       Note: --list-only ignores --prompt/--prompt-file and does not run sim
  --help               Show this help
EOF
}

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-closd}"
EXP_NAME="${EXP_NAME:-t2m_one_condition}"
OUTPUT_PATH="${OUTPUT_PATH:-output/CLoSD/CLoSD_no_finetune}"
NUM_ENVS="${NUM_ENVS:-1}"
EPISODE_LENGTH="${EPISODE_LENGTH:-300}"
VNC_DISPLAY="${VNC_DISPLAY:-:1}"
ISAAC_DISPLAY="${ISAAC_DISPLAY:-:0}"
AUTO_RECORD="${AUTO_RECORD:-1}"
AUTO_RECORD_FRAMES="${AUTO_RECORD_FRAMES:-0}"
LIST_ONLY=0
SELECTED_INDEX="${INDEX:-}"
CUSTOM_PROMPT="${PROMPT:-}"
CUSTOM_PROMPT_FILE="${PROMPT_FILE:-}"
PROMPT_MODE=0
USE_DATASET_PREFIX="${USE_DATASET_PREFIX:-0}"
RECORD_MODE=""

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --episode-length)
      EPISODE_LENGTH="${2:?Missing value for --episode-length}"
      shift 2
      ;;
    --episode-length=*)
      EPISODE_LENGTH="${1#*=}"
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
    --index)
      SELECTED_INDEX="${2:?Missing value for --index}"
      shift 2
      ;;
    --index=*)
      SELECTED_INDEX="${1#*=}"
      shift
      ;;
    --prompt)
      CUSTOM_PROMPT="${2:?Missing value for --prompt}"
      shift 2
      ;;
    --prompt=*)
      CUSTOM_PROMPT="${1#*=}"
      shift
      ;;
    --prompt-file)
      CUSTOM_PROMPT_FILE="${2:?Missing value for --prompt-file}"
      shift 2
      ;;
    --prompt-file=*)
      CUSTOM_PROMPT_FILE="${1#*=}"
      shift
      ;;
    --use-dataset-prefix)
      USE_DATASET_PREFIX=1
      shift
      ;;
    --no-dataset-prefix)
      USE_DATASET_PREFIX=0
      shift
      ;;
    --record-frames)
      AUTO_RECORD_FRAMES="${2:?Missing value for --record-frames}"
      RECORD_MODE="frames"
      shift 2
      ;;
    --record-frames=*)
      AUTO_RECORD_FRAMES="${1#*=}"
      RECORD_MODE="frames"
      shift
      ;;
    --record-until-exit)
      AUTO_RECORD_FRAMES=0
      RECORD_MODE="until_exit"
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
    --list-only)
      LIST_ONLY=1
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

if ! [[ "$EPISODE_LENGTH" =~ ^[0-9]+$ ]] || [ "$EPISODE_LENGTH" -le 0 ]; then
  echo "Error: --episode-length must be a positive integer." >&2
  exit 2
fi

if ! [[ "$AUTO_RECORD_FRAMES" =~ ^[0-9]+$ ]]; then
  echo "Error: AUTO_RECORD_FRAMES/--record-frames must be a non-negative integer." >&2
  exit 2
fi
if [ "$AUTO_RECORD_FRAMES" -lt 0 ]; then
  echo "Error: AUTO_RECORD_FRAMES/--record-frames must be >= 0." >&2
  exit 2
fi
if [ -z "$RECORD_MODE" ]; then
  if [ "$AUTO_RECORD_FRAMES" -gt 0 ]; then
    RECORD_MODE="frames"
  else
    RECORD_MODE="until_exit"
  fi
fi
if [ "$RECORD_MODE" = "frames" ] && [ "$AUTO_RECORD_FRAMES" -le 0 ]; then
  echo "Error: --record-frames must be a positive integer." >&2
  exit 2
fi

if ! [[ "$USE_DATASET_PREFIX" =~ ^[01]$ ]]; then
  echo "Error: USE_DATASET_PREFIX must be 0 or 1." >&2
  exit 2
fi

if [ "$LIST_ONLY" -ne 1 ]; then
  if [ -n "$CUSTOM_PROMPT" ] && [ -n "$CUSTOM_PROMPT_FILE" ]; then
    echo "Error: --prompt and --prompt-file are mutually exclusive." >&2
    exit 2
  fi

  if [ -n "$CUSTOM_PROMPT_FILE" ]; then
    if [ ! -f "$CUSTOM_PROMPT_FILE" ]; then
      echo "Error: --prompt-file not found: $CUSTOM_PROMPT_FILE" >&2
      exit 2
    fi

    CUSTOM_PROMPT="$(<"$CUSTOM_PROMPT_FILE")"
    trimmed_prompt="${CUSTOM_PROMPT#"${CUSTOM_PROMPT%%[![:space:]]*}"}"
    trimmed_prompt="${trimmed_prompt%"${trimmed_prompt##*[![:space:]]}"}"
    CUSTOM_PROMPT="$trimmed_prompt"

    if [ -z "$CUSTOM_PROMPT" ]; then
      echo "Error: --prompt-file content is empty after trimming whitespace." >&2
      exit 2
    fi
  fi

  if [ -n "$CUSTOM_PROMPT" ]; then
    if [ -z "${CUSTOM_PROMPT//[[:space:]]/}" ]; then
      echo "Error: --prompt must contain non-whitespace text." >&2
      exit 2
    fi
    PROMPT_MODE=1
  fi
fi

if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
else
  echo "Warning: conda.sh not found. Proceeding without explicit conda activation." >&2
fi

export DISPLAY="${DISPLAY:-$VNC_DISPLAY}"
export CLOSD_AUTO_RECORD="$AUTO_RECORD"
export CLOSD_AUTO_RECORD_FRAMES="$AUTO_RECORD_FRAMES"

cd "$REPO_DIR"

captions_json=""
trap 'if [ -n "${captions_json:-}" ] && [ -f "$captions_json" ]; then rm -f "$captions_json"; fi' EXIT

selected_prompt=""
if [ "$LIST_ONLY" -eq 1 ] || [ "$PROMPT_MODE" -eq 0 ]; then
  captions_json="$(mktemp)"

  deps_path_output="$(
    python - <<'PY'
import os
import sys

repo_id = "guytevet/CLoSD"
env_path = os.environ.get("DEPENDENCIES_PATH", "").strip()

def has_cache(path: str) -> bool:
    if not path:
        return False
    train = os.path.join(path, "data", "humanml3d", "t2m_train.npy")
    test = os.path.join(path, "data", "humanml3d", "t2m_test.npy")
    return os.path.isfile(train) and os.path.isfile(test)

if has_cache(env_path):
    print(env_path)
    sys.exit(0)

try:
    from huggingface_hub import snapshot_download
    cached_path = snapshot_download(repo_id=repo_id, local_files_only=True)
    if has_cache(cached_path):
        print(cached_path)
        sys.exit(0)
except Exception:
    pass

try:
    from closd.utils import hf_handler
    downloaded_path = hf_handler.get_dependencies()
    if has_cache(downloaded_path):
        print(downloaded_path)
        sys.exit(0)
except Exception as exc:
    print(f"Failed to fetch CLoSD dependencies: {exc}", file=sys.stderr)
    sys.exit(1)

print("HumanML cache files were not found after dependency fetch.", file=sys.stderr)
sys.exit(1)
PY
  )"

  deps_path="$(printf '%s\n' "$deps_path_output" | tail -n1)"
  train_npy="$deps_path/data/humanml3d/t2m_train.npy"
  test_npy="$deps_path/data/humanml3d/t2m_test.npy"

  if [ ! -f "$train_npy" ] || [ ! -f "$test_npy" ]; then
    echo "Error: Missing HumanML cache files." >&2
    echo "Expected:" >&2
    echo "  $train_npy" >&2
    echo "  $test_npy" >&2
    echo "Try running a normal CLoSD command once to populate dependencies." >&2
    exit 1
  fi

  caption_count="$(
    python - "$train_npy" "$test_npy" "$captions_json" <<'PY'
import json
import numpy as np
import sys

train_path, test_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
captions = set()

for path in (train_path, test_path):
    data = np.load(path, allow_pickle=True).item()
    data_dict = data.get("data_dict", {})
    for sample in data_dict.values():
        for text_info in sample.get("text", []):
            caption = text_info.get("caption", "")
            if isinstance(caption, str):
                caption = caption.strip()
                if caption:
                    captions.add(caption)

sorted_captions = sorted(captions)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(sorted_captions, f, ensure_ascii=False)

print(len(sorted_captions))
PY
  )"

  if ! [[ "$caption_count" =~ ^[0-9]+$ ]] || [ "$caption_count" -eq 0 ]; then
    echo "Error: No text conditions were found in HumanML cache." >&2
    exit 1
  fi

  python - "$captions_json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    captions = json.load(f)

for idx, caption in enumerate(captions):
    print(f"[{idx}] {caption}")
PY

  if [ "$LIST_ONLY" -eq 1 ]; then
    echo "Listed $caption_count unique conditions."
    exit 0
  fi

  if [ -z "$SELECTED_INDEX" ]; then
    while true; do
      read -r -p "Select caption index [0-$((caption_count - 1))]: " SELECTED_INDEX
      if [[ "$SELECTED_INDEX" =~ ^[0-9]+$ ]] && [ "$SELECTED_INDEX" -ge 0 ] && [ "$SELECTED_INDEX" -lt "$caption_count" ]; then
        break
      fi
      echo "Invalid index. Enter an integer in [0, $((caption_count - 1))]." >&2
    done
  else
    if ! [[ "$SELECTED_INDEX" =~ ^[0-9]+$ ]] || [ "$SELECTED_INDEX" -lt 0 ] || [ "$SELECTED_INDEX" -ge "$caption_count" ]; then
      echo "Error: --index out of range. Expected [0, $((caption_count - 1))]." >&2
      exit 2
    fi
  fi

  selected_prompt="$(
    python - "$captions_json" "$SELECTED_INDEX" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    captions = json.load(f)

idx = int(sys.argv[2])
print(captions[idx], end="")
PY
  )"

  echo "Selected prompt [$SELECTED_INDEX]: $selected_prompt"
else
  if [ -n "$SELECTED_INDEX" ]; then
    echo "Note: --prompt/--prompt-file provided; ignoring --index." >&2
  fi
  selected_prompt="$CUSTOM_PROMPT"
  echo "Selected prompt (custom): $selected_prompt"
fi

escape_hydra_string() {
  local escaped="$1"
  escaped="${escaped//\\/\\\\}"
  escaped="${escaped//\"/\\\"}"
  printf '"%s"' "$escaped"
}

hydra_prompt="$(escape_hydra_string "$selected_prompt")"
hydra_use_dataset_prefix=False
dataset_prefix_label="disabled"
if [ "$USE_DATASET_PREFIX" -eq 1 ]; then
  hydra_use_dataset_prefix=True
  dataset_prefix_label="enabled"
fi

record_mode_label="until process exit"
if [ "$AUTO_RECORD_FRAMES" -gt 0 ]; then
  record_mode_label="${AUTO_RECORD_FRAMES} frames"
fi

echo "Dataset prefix: $dataset_prefix_label"
echo "Recording mode: $record_mode_label"
echo "Episode length: $EPISODE_LENGTH"

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
  env.episode_length="$EPISODE_LENGTH"
  learning.params.config.player.games_num=1
  output_path="$OUTPUT_PATH"
  exp_name="$EXP_NAME"
  env.use_dataset_prefix="$hydra_use_dataset_prefix"
  "env.custom_prompt=$hydra_prompt"
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
