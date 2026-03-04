# `run_t2m_condition.sh`

Interactive one-condition runner for CLoSD text-to-motion (`env=closd_t2m`).

## What it does

1. Finds HumanML cached files:
   - `data/humanml3d/t2m_train.npy`
   - `data/humanml3d/t2m_test.npy`
2. Builds a unique caption list from both splits (`strip` + dedupe + sort).
3. Lets you select a caption index (or use `--index`), or bypass the list with `--prompt` / `--prompt-file`.
4. Runs CLoSD with:
   - `env=closd_t2m`
   - `env.num_envs=1`
   - `learning.params.config.player.games_num=1`
   - `env.custom_prompt="<selected caption>"`
   - `env.use_dataset_prefix=<True|False>`
5. Auto-records MP4 and removes the matching state `.pkl` file.

## Usage

```bash
bash scripts/run_t2m_condition.sh
```

### Useful modes

```bash
# Only print indexed captions
bash scripts/run_t2m_condition.sh --list-only

# Non-interactive selection
bash scripts/run_t2m_condition.sh --index 42

# Direct custom prompt
bash scripts/run_t2m_condition.sh --prompt "A person is moonwalking."

# Read custom prompt from file
bash scripts/run_t2m_condition.sh --prompt-file /tmp/my_prompt.txt

# Enable dataset prefix lead-in from HumanML for first planning step
bash scripts/run_t2m_condition.sh --index 42 --use-dataset-prefix

# Record fixed duration (otherwise records until process exit)
bash scripts/run_t2m_condition.sh --record-frames 300 --index 42
```

## Options

- `--episode-length N` (default: `300`)
- `--exp-name NAME` (default: `t2m_one_condition`)
- `--index N` (optional, skip interactive prompt)
- `--prompt TEXT` (direct custom prompt; overrides `--index`)
- `--prompt-file PATH` (read prompt from file; mutually exclusive with `--prompt`)
- `--use-dataset-prefix` (enable HumanML prefix lead-in)
- `--no-dataset-prefix` (disable HumanML prefix lead-in; default)
- `--record-frames N` (cap recording to `N` frames)
- `--record-until-exit` (default; no frame cap)
- `--isaac-display DISPLAY` (default: `:0`)
- `--list-only`

Precedence:

- `--list-only` always lists dataset captions and exits.
- `--prompt` / `--prompt-file` skip caption loading + index selection and run directly.
- If both `--prompt` and `--prompt-file` are provided, the script exits with an error.

Extra Hydra overrides can be passed after `--`:

```bash
bash scripts/run_t2m_condition.sh --index 12 -- env.dip.debug_hml=True
```

## Environment defaults

- `CONDA_ENV=closd`
- `NUM_ENVS=1`
- `EPISODE_LENGTH=300`
- `VNC_DISPLAY=:1` (`DISPLAY` defaults to this if unset)
- `ISAAC_DISPLAY=:0`
- `AUTO_RECORD=1`
- `AUTO_RECORD_FRAMES=0` by default (record until process exit)
- `USE_DATASET_PREFIX=0` by default (no dataset lead-in)
- `EXP_NAME=t2m_one_condition`

## Outputs

- MP4: `output/renderings/<exp_name>-<timestamp>.mp4`
- State file: `output/states/<exp_name>-<timestamp>.pkl` (removed automatically)

At the end, the script prints:

- The selected caption
- Whether dataset prefix is enabled
- Recording mode (`until process exit` or `N frames`)
- Episode length
- The final MP4 path

## Failure notes

- If cache files are missing, the script tries to fetch CLoSD dependencies.
- If `vglrun` is not available, it falls back to direct `python` execution.
- If run fails, it exits non-zero.
- If MP4 is missing after run, check display/recording setup (`DISPLAY`, VirtualGL, auto-record settings).
