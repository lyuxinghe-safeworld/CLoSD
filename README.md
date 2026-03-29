# CLoSD: Closing the Loop between Simulation and Diffusion for multi-task character control.
### ICLR 2025 Spotlight
[Project Page](https://guytevet.github.io/CLoSD-page/) | [Arxiv](https://arxiv.org/abs/2410.03441) | [Video](https://www.youtube.com/watch?feature=shared&v=O1tzbiDMW8U)

![teaser](https://github.com/GuyTevet/CLoSD-page/blob/main/static/figures/demo1.gif?raw=true)


## Bibtex

If you find this code useful in your research, please cite:

```
@inproceedings{
  tevet2025closd,
  title={{CL}o{SD}: Closing the Loop between Simulation and Diffusion for multi-task character control},
  author={Guy Tevet and Sigal Raab and Setareh Cohan and Daniele Reda and Zhengyi Luo and Xue Bin Peng and Amit Haim Bermano and Michiel van de Panne},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=pZISppZSTv}
}
```


## Getting Started


- The code was tested on `Ubuntu 20.04.5` with `Python 3.8.19`.
- Running CLoSD requires a single GPU with `~4GB RAM` and a monitor.
- Training and evaluation require a single GPU with `~50GB RAM` (monitor is not required).
- You only need to setup the Python environment. All the dependencies (data, checkpoints, etc.) will be cached automatically on the first run!

<details>
  <summary><b>Setup env</b></summary>

  - Create a Conda env and setup the requirements:

```
conda create -n closd python=3.8
conda activate closd
pip install -r requirement.txt
python -m spacy download en_core_web_sm
```

  - Optional: initialize conda for all new bash terminals:

```
~/miniforge3/bin/conda init bash
exec bash -l
```

  - Install Isaac Gym Preview 4 (required by CLoSD).  
    Option A: reuse an existing Isaac Gym checkout:

```
conda activate closd
cd <ISAAC_GYM_DIR>/python
pip install -e .
```

  - Option B: compact install from scratch (tested on Ubuntu 22.04):

```
conda activate closd

# Download + extract Isaac Gym Preview 4
mkdir -p ~/.cache/isaacgym ~/.local/src
curl -fL https://developer.nvidia.com/isaac-gym-preview-4 \
  -o ~/.cache/isaacgym/IsaacGym_Preview_4_Package.tar.gz
tar -xzf ~/.cache/isaacgym/IsaacGym_Preview_4_Package.tar.gz -C ~/.local/src

# Optional stable symlink used by commands below
ln -sfn ~/.local/src/isaacgym ~/isaacgym

# Install python package
pip install -e ~/isaacgym/python
```

  - Runtime env vars (recommended for VM/VNC setups):

```
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$HOME/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH}"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

  - Persist the runtime env vars on every `conda activate closd` (recommended):

```
conda activate closd
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/isaacgym-runtime.sh" <<'EOF'
export ISAACGYM_PATH="${ISAACGYM_PATH:-$HOME/isaacgym}"
if [ -z "${ISAACGYM_OLD_LD_LIBRARY_PATH+x}" ]; then
  export ISAACGYM_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
fi
if [ -z "${ISAACGYM_OLD_VK_ICD_FILENAMES+x}" ]; then
  export ISAACGYM_OLD_VK_ICD_FILENAMES="${VK_ICD_FILENAMES-}"
fi
new_ld="$CONDA_PREFIX/lib:$ISAACGYM_PATH/python/isaacgym/_bindings/linux-x86_64"
if [ -n "${LD_LIBRARY_PATH-}" ]; then
  export LD_LIBRARY_PATH="$new_ld:$LD_LIBRARY_PATH"
else
  export LD_LIBRARY_PATH="$new_ld"
fi
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi
EOF

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/isaacgym-runtime.sh" <<'EOF'
if [ -n "${ISAACGYM_OLD_LD_LIBRARY_PATH+x}" ]; then
  export LD_LIBRARY_PATH="$ISAACGYM_OLD_LD_LIBRARY_PATH"
  unset ISAACGYM_OLD_LD_LIBRARY_PATH
fi
if [ -n "${ISAACGYM_OLD_VK_ICD_FILENAMES+x}" ]; then
  if [ -n "$ISAACGYM_OLD_VK_ICD_FILENAMES" ]; then
    export VK_ICD_FILENAMES="$ISAACGYM_OLD_VK_ICD_FILENAMES"
  else
    unset VK_ICD_FILENAMES
  fi
  unset ISAACGYM_OLD_VK_ICD_FILENAMES
fi
EOF
```

  - Quick verification:

```
nvidia-smi
python -c "from isaacgym import gymapi; print('isaacgym ok')"
```

  - Note: when writing custom scripts, import order must be `isaacgym` before `torch`.

</details>

<details>
  <summary><b>Working with VNC</b></summary>
  To render within VM, run the following command in the local machine, then connect to the VM monitor through the localhost port
```
gcloud compute ssh human-motion-exps \
  --zone us-central1-a \
  -- -N -L 5901:localhost:5901
```
</details>

<details>
  <summary><b>Copyright notes</b></summary>
  
The code will automatically download cached versions of the following datasets and models. You must adhere to their terms of use!

- SMPL license is according to https://smpl-x.is.tue.mpg.de/
- AMASS license is according to  https://amass.is.tue.mpg.de/
- HumanML3D dataset license is according to https://github.com/EricGuo5513/HumanML3D

</details>

## Run CLoSD

<details>
  <summary><b>Multi-task</b></summary>

```
python closd/run.py\
  learning=im_big robot=smpl_humanoid\
  epoch=-1 test=True no_virtual_display=True\
  headless=False env.num_envs=9\
  env=closd_multitask exp_name=CLoSD_multitask_finetune
```

</details>

<details>
  <summary><b>Sequence of tasks</b></summary>

```
python closd/run.py\
  learning=im_big robot=smpl_humanoid\
  epoch=-1 test=True no_virtual_display=True\
  headless=False env.num_envs=9\
  env=closd_sequence exp_name=CLoSD_multitask_finetune
```

</details>

<details id="run-closd-t2m">
  <summary><b>Text-to-motion</b></summary>

```
python closd/run.py\
  learning=im_big robot=smpl_humanoid\
  epoch=-1 test=True no_virtual_display=True\
  headless=False env.num_envs=9\
  env=closd_t2m exp_name=CLoSD_t2m_finetune
```

</details>

- To run the model without fine-tuning, use `exp_name=CLoSD_no_finetune`
- To run without a monitor, use `headless=True`

<details>
  <summary><b>Run one sample + save one video (Ubuntu 22.04 VM/VNC)</b></summary>

- In VNC sessions, first check if there are VNC session running:

```
ss -lntp | grep 5901
```

If not, start one:
```
/opt/TurboVNC/bin/vncserver :1
```

Set display and use VirtualGL:

```
export DISPLAY=:1
```

- Run a single-episode evaluation (uses pretrained `CLoSD_no_finetune` checkpoint):

```
vglrun -d :0 python closd/run.py \
  learning=im_big robot=smpl_humanoid \
  test=True no_log=True epoch=-1 \
  headless=False no_virtual_display=True \
  env=closd_sequence env.num_envs=1 env.episode_length=480 \
  learning.params.config.player.games_num=1 \
  output_path=output/CLoSD/CLoSD_no_finetune \
  exp_name=one_sample_seq
```

- Or use the helper script:

```
bash scripts/run_one_sample.sh
```

- Optional overrides:

```
EXP_NAME=my_sample EPISODE_LENGTH=240 NUM_ENVS=1 bash scripts/run_one_sample.sh
```

- Custom language prompt override (all sequence states use this text):

```
bash scripts/run_one_sample.sh env.custom_prompt="A person is moonwalking."
```

- Equivalent env-var form:

```
CUSTOM_PROMPT="A person is moonwalking." bash scripts/run_one_sample.sh
```

- `scripts/run_one_sample.sh` auto-records MP4 by default.
- For manual command mode, in Isaac Gym viewer, press `L` to start recording, then `L` again to stop/write.
- Video output: `output/renderings/<exp_name>-<timestamp>.mp4`
- State output: `output/states/<exp_name>-<timestamp>.pkl`

- If MP4 writing fails due missing backend:

```
pip install imageio imageio-ffmpeg
```

- If interrupted while compiling `gymtorch`, clear stale lock and rerun:

```
rm -f ~/.cache/torch_extensions/py38_cu121/gymtorch/lock
```

</details>

<details>
  <summary><b>Run one Text-to-Motion condition (interactive picker + MP4-only output)</b></summary>

- This helper script:
  - Loads HumanML cached captions from both `t2m_train.npy` and `t2m_test.npy`
  - Deduplicates + sorts captions, then lets you pick one by index
  - Also supports direct custom prompts via `--prompt` or `--prompt-file`
  - Runs `env=closd_t2m` with `env.custom_prompt="<selected caption>"`
  - Supports toggling HumanML dataset prefix lead-in via `--use-dataset-prefix/--no-dataset-prefix`
  - Auto-records a video and removes matching state `.pkl` file (MP4-only policy)

- Interactive mode:

```
bash scripts/run_t2m_condition.sh
```

- List captions only:

```
bash scripts/run_t2m_condition.sh --list-only
```

- Non-interactive mode (pick an index directly):

```
bash scripts/run_t2m_condition.sh --index 42
```

- Direct custom prompt (skips caption indexing/selection):

```
bash scripts/run_t2m_condition.sh --prompt "A person is moonwalking."
```

- Common options:
  - `--episode-length <N>` (default `300`)
  - `--exp-name <name>` (default `t2m_one_condition`)
  - `--prompt "<text>"` (overrides `--index`)
  - `--prompt-file <path>` (mutually exclusive with `--prompt`)
  - `--use-dataset-prefix` / `--no-dataset-prefix` (default disabled)
  - `--record-frames <N>` (fixed recording cap)
  - `--record-until-exit` (default; unlimited recording until process exit)
  - `--isaac-display <display>` (default `:0`)

- Example with overrides:

```
bash scripts/run_t2m_condition.sh --episode-length 360 --exp-name t2m_jump_demo --index 12
```

- Output:
  - MP4: `output/renderings/<exp_name>-<timestamp>.mp4`
  - Matching state file `output/states/<exp_name>-<timestamp>.pkl` is deleted automatically

- You can pass extra Hydra overrides after `--`:

```
bash scripts/run_t2m_condition.sh --index 12 -- env.dip.debug_hml=True
```

- Script details: `scripts/README_t2m_condition.md`

</details>

<details>
  <summary><b>Run long-horizon one-shot Text-to-Motion (LLM segments + prompt scheduling)</b></summary>

- This flow runs one single CLoSD episode and switches the text prompt at planning-horizon boundaries.
- It does not stitch multiple runs; output is one continuous MP4 trajectory.

- Set planner env vars (OpenAI Responses API):

```
export OPENAI_API_KEY=<your_key>
# optional
export OPENAI_MODEL=gpt-4o-mini
export OPENAI_BASE_URL=https://api.openai.com/v1
```

- One-shot run:

```
bash scripts/run_t2m_long_horizon.sh \
  --prompt "A person walks forward, then waves, then crouches, then stands and turns."
```

- Plan only (create schedule JSON, do not run simulator):

```
bash scripts/run_t2m_long_horizon.sh \
  --prompt "A person walks, jumps, and spins." \
  --plan-only
```

- Prompt file + common overrides:

```
bash scripts/run_t2m_long_horizon.sh \
  --prompt "A person walks, jumps, and spins." \
  --planning-horizon 40 \
  --episode-length 420 \
  --record-frames auto \
  --exp-name t2m_long_demo
```

- Runtime behavior:
  - Script first calls `scripts/plan_t2m_long_horizon.py` to build a schedule JSON.
  - Then launches one `closd/run.py` with `env=closd_t2m` and `env.segment_schedule_path=<schedule.json>`.
  - `--record-frames auto` records `episode_length * env.controlFrequencyInv` rendered frames (60fps by default).
  - Segment prompts are normalized to `a person is ...` (including `a person is ... while ...` when applicable).
  - During generation, terminal prints current/switching prompt:
    - `[CLoSDT2M][USE] ... prompt=...`
    - `[CLoSDT2M][SWITCH] ... prompt=...`
    - `[CLoSDT2M][EXHAUSTED] ...` (if schedule ends and `env.segment_hold_last=False`)

- If OpenAI planning fails (or key is missing), planner falls back to a heuristic segmentation so the run can still proceed.

</details>


## Evaluate

<details>
  <summary><b>Multi-task success rate</b></summary>

- To reproduce Table 1 in the paper.

```
python closd/run.py\
 learning=im_big env=closd_multitask robot=smpl_humanoid\
 exp_name=CLoSD_multitask_finetune\
 epoch=-1\
 env.episode_length=500\
 env.dip.cfg_param=7.5\
 env.num_envs=4096\
 test=True\
 no_virtual_display=True\
 headless=True\
 closd_eval=True
```

</details>

<details>
  <summary><b>Text-to-motion</b></summary>

```
python -m closd.diffusion_planner.eval.eval_humanml --external_results_file closd/diffusion_planner/saved_motions/closd/CloSD.pkl --do_unique
```
- To log resutls in WandB, add:
  ```
  --train_platform_type WandBPlatform --eval_name <wandb_exp_name>
  ```
- The evaluation process runs on pre-recorded data and reproduces Table 3 in the paper.
- The raw results are at `https://huggingface.co/guytevet/CLoSD/blob/main/evaluation/closd/eval.log`, this code should reproduce it.
- In case you want to re-record the data yourself (reproduce the `external_results_file` .pkl file), run:
  ```
  python closd/run.py\
    learning=im_big robot=smpl_humanoid\
    epoch=-1 test=True no_virtual_display=True\
    headless=True env.num_envs=4096\
    env=closd_t2m exp_name=CLoSD_t2m_finetune \
    env.episode_length=300 \
    env.save_motion.save_hml_episodes=True \
    env.save_motion.save_hml_episodes_dir=<target_folder_name>
  ```

</details>

## Visualizations

- To record motions with IsaacGym, while simulation is running (on IsaacGym GUI), press `L` to start/stop recording.
- The recorded file will be saved to `output/states/`


<details>
  <summary><b>Blender vizualization</b></summary>

- This script runs with Blender interpreter and visualizes IsaacGym recordings.
- The code is based on https://github.com/xizaoqu/blender_for_UniHSI and was tested on Blender 4.2


First, setup the Blender interpreter with:

```
blender -b -P closd/blender/setup_blender.py
```

Then visualize with:

```
blender -b -P closd/blender/record2anim.py -- --record_path output/states/YOUR_RECORD_NAME.pkl
```

</details>

<details>
  <summary><b>Extract SMPL parameters</b></summary>

To extract the SMPL parameters of the humanoid, first download [SMPL](https://smpl.is.tue.mpg.de/) and place it in `closd/data/smpl`.

Then run:

```
python closd/utils/extract_smpl.py --record_path output/states/YOUR_RECORD_NAME.pkl
```

The script will save the SMPL parameters that can be visualize with standard SMPL tools, for example those of [MDM](https://github.com/GuyTevet/motion-diffusion-model) or [PHC](https://github.com/ZhengyiLuo/PHC).

</details>


## Train your own CLoSD

<details>
  <summary><b>Tracking controller (PHC based)</b></summary>

```
python closd/run.py\
 learning=im_big env=im_single_prim robot=smpl_humanoid\
 env.cycle_motion=True epoch=-1\
 exp_name=my_CLoSD_no_finetune
```

- Train for 62K epochs


</details>

<details>
  <summary><b>Fine-tune for Multi-task</b></summary>

```
python closd/run.py\
 learning=im_big env=closd_multitask robot=smpl_humanoid\
 learning.params.load_checkpoint=True\
 learning.params.load_path=output/CLoSD/my_CLoSD_no_finetune/Humanoid.pth\
 env.dip.cfg_param=2.5 env.num_envs=3072\
 has_eval=False epoch=-1\
 exp_name=my_CLoSD_multitask_finetune
```

- Train for 4K epochs

</details>


<details>
  <summary><b>Fine-tune for Text-to-motion</b></summary>

```
python closd/run.py\
 learning=im_big env=closd_t2m robot=smpl_humanoid\
 learning.params.load_checkpoint=True\
 learning.params.load_path=output/CLoSD/my_CLoSD_no_finetune/Humanoid.pth\
 env.dip.cfg_param=2.5 env.num_envs=3072\
 has_eval=False epoch=-1\
 exp_name=my_CLoSD_t2m_finetune
```

- Train for 1K epochs

</details>

- For debug run, use `learning=im_toy` and add `no_log=True env.num_envs=4`

## DiP

- Diffusion Planner (DiP) is a real-time autoregressive diffusion model that serves as the planner for the CLoSD agent.
- Instead of running it as part of CLoSD, you can also run DiP in a stand-alone mode, fed by its own generated motions.
- The following details how to sample/evaluate/train DiP in the **stand-alone** mode.

### 

<details>
  <summary><b>Generate Motion with the Stand-alone DiP</b></summary>

Full autoregressive generation (without target):

```
python -m closd.diffusion_planner.sample.generate\
 --model_path closd/diffusion_planner/save/DiP_no-target_10steps_context20_predict40/model000200000.pt\
 --num_repetitions 1 --autoregressive
```

Prefix completion with target trajectory:

```
python -m closd.diffusion_planner.sample.generate\
 --model_path closd/diffusion_planner/save/DiP_multi-target_10steps_context20_predict40/model000300000.pt\
 --num_repetitions 1 --sampling_mode goal\
 --target_joint_names "traj,heading" --target_joint_source data
```

- To sample with random joint target (instead of sampling it from the data, which is more challenging), use `--target_joint_source random`
- Other 'legal' joint conditions are:

```
--target_joint_names 
[traj,heading|
pelvis,heading|
right_wrist,heading|
left_wrist,heading|
right_foot,heading|
left_foot,heading]
```

</details>

<details>
  <summary><b>Stand-alone Evaluation</b></summary>

- Evaluate DiP fed by its own predictions (without the CLoSD framework):
- To reproduce Tables 2 and 3 (the DiP entry) in the paper.

```
python -m closd.diffusion_planner.eval.eval_humanml\
 --guidance_param 7.5\
 --model_path closd/diffusion_planner/save/DiP_no-target_10steps_context20_predict40/model000600343.pt\
 --autoregressive
```

</details>


<details>
  <summary><b>Train your own DiP</b></summary>

The following will reproduce the DiP used in the paper:

```
python -m closd.diffusion_planner.train.train_mdm\
 --save_dir closd/diffusion_planner/save/my_DiP\
 --dataset humanml --arch trans_dec --text_encoder_type bert\
 --diffusion_steps 10 --context_len 20 --pred_len 40\
 --mask_frames --eval_during_training --gen_during_training --overwrite --use_ema --autoregressive --train_platform_type WandBPlatform
```

To train DiP without target conditioning, add `--lambda_target_loc 0`

</details>

## Acknowledgments

This code is standing on the shoulders of giants. We want to thank the following contributors
that our code is based on:

[MDM](https://github.com/GuyTevet/motion-diffusion-model), [PHC](https://github.com/ZhengyiLuo/PHC), [MotionCLIP](https://github.com/GuyTevet/MotionCLIP), [text-to-motion](https://github.com/EricGuo5513/text-to-motion), [actor](https://github.com/Mathux/ACTOR), [joints2smpl](https://github.com/wangsen1312/joints2smpl), [MoDi](https://github.com/sigal-raab/MoDi).

## License
This code is distributed under an [MIT LICENSE](LICENSE).
