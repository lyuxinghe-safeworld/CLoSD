#!/usr/bin/env bash
set -euo pipefail

# One-sample CLoSD run for VM/VNC setups.
# Recording starts automatically and is finalized on process exit.
# In the viewer, "L" is an optional manual toggle.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-closd}"
EXP_NAME="${EXP_NAME:-one_sample_seq}"
OUTPUT_PATH="${OUTPUT_PATH:-output/CLoSD/CLoSD_no_finetune}"
NUM_ENVS="${NUM_ENVS:-1}"
EPISODE_LENGTH="${EPISODE_LENGTH:-480}"
VNC_DISPLAY="${VNC_DISPLAY:-:1}"
ISAAC_DISPLAY="${ISAAC_DISPLAY:-:0}"
AUTO_RECORD="${AUTO_RECORD:-1}"
AUTO_RECORD_FRAMES="${AUTO_RECORD_FRAMES:-0}"
CUSTOM_PROMPT="${CUSTOM_PROMPT:-}"

if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

export DISPLAY="${DISPLAY:-$VNC_DISPLAY}"
export CLOSD_AUTO_RECORD="$AUTO_RECORD"
export CLOSD_AUTO_RECORD_FRAMES="$AUTO_RECORD_FRAMES"

cd "$REPO_DIR"

cmd=(
  python closd/run.py
  learning=im_big
  robot=smpl_humanoid
  test=True
  no_log=True
  epoch=-1
  headless=False
  no_virtual_display=True
  env=closd_sequence
  env.num_envs="$NUM_ENVS"
  env.episode_length="$EPISODE_LENGTH"
  learning.params.config.player.games_num=1
  output_path="$OUTPUT_PATH"
  exp_name="$EXP_NAME"
)

if [ -n "$CUSTOM_PROMPT" ]; then
  cmd+=(env.custom_prompt="$CUSTOM_PROMPT")
fi

if command -v vglrun >/dev/null 2>&1; then
  exec vglrun -d "$ISAAC_DISPLAY" "${cmd[@]}" "$@"
else
  echo "Warning: vglrun not found. Running without VirtualGL." >&2
  exec "${cmd[@]}" "$@"
fi
