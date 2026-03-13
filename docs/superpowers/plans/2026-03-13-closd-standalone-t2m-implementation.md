# CLoSD Standalone T2M Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone sibling repo, `CLoSD_t2m_standalone`, that runs DiP text-to-motion inference from a prompt using a synthetic standing-start prefix and writes an offline MP4 without any simulator dependency.

**Architecture:** Keep the extracted repo narrow: vendor only the inference-related DiP/HumanML code that is actually needed, then wrap it with a small `standalone_t2m` package that owns checkpoint loading, standing-prefix synthesis, autoregressive rollout, artifact writing, and offline rendering. The bootstrap prefix is created in HumanML joint space, not from an existing motion clip and not from a zero tensor, then converted into the model-ready feature representation before the first sampling step.

**Tech Stack:** Python 3.8+, PyTorch, NumPy, SciPy, `transformers`, Matplotlib, pytest.

---

## File Structure

### Repo bootstrap

- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/.gitignore`
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/pyproject.toml`
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/pytest.ini`
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/README.md`

### Vendored inference subset

- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/vendor/diffusion_planner/`
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/assets/t2m_mean.npy`
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/assets/t2m_std.npy`

### Core package

- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/__init__.py`
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/config.py`
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/checkpoint.py`
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/prefix/standing.py`
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/generation.py`
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/decode.py`
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/artifacts.py`
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/render.py`
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/cli.py`

### Tests

- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/tests/test_checkpoint.py`
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/tests/test_standing_prefix.py`
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/tests/test_generation.py`
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/tests/test_decode_render.py`
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/tests/test_no_sim_imports.py`
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/tests/integration/test_generate_cli.py`

## Chunk 1: Bootstrap And Vendored Inference Core

### Task 1: Create The Standalone Repo Skeleton

**Files:**
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/.gitignore`
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/pyproject.toml`
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/pytest.ini`
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/__init__.py`

- [ ] **Step 1: Create the new sibling repo and initialize git**

Run:
```bash
mkdir -p /home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m
cd /home/lyuxinghe/code/CLoSD_t2m_standalone
git init
```
Expected: the sibling repo exists and has an empty git history.

- [ ] **Step 2: Add packaging and test discovery files**

```toml
[project]
name = "closd-t2m-standalone"
version = "0.1.0"
requires-python = ">=3.8"
dependencies = [
  "numpy",
  "scipy",
  "torch",
  "transformers",
  "matplotlib",
]

[project.scripts]
closd-t2m = "standalone_t2m.cli:main"
```

```ini
[pytest]
testpaths = tests
pythonpath = .
```

- [ ] **Step 3: Add a minimal `.gitignore`**

```gitignore
__pycache__/
.pytest_cache/
dist/
build/
outputs/
*.pyc
```

- [ ] **Step 4: Verify import and pytest collection work**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_t2m_standalone
python -m pytest --collect-only
```
Expected: pytest starts cleanly and collects zero or more tests without import errors.

- [ ] **Step 5: Commit**

```bash
cd /home/lyuxinghe/code/CLoSD_t2m_standalone
git add .gitignore pyproject.toml pytest.ini standalone_t2m/__init__.py
git commit -m "chore: bootstrap standalone t2m repo"
```

### Task 2: Vendor The Inference-Only Upstream Modules

**Files:**
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/vendor/diffusion_planner/`
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/assets/t2m_mean.npy`
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/assets/t2m_std.npy`
- Test: `/home/lyuxinghe/code/CLoSD_t2m_standalone/tests/test_checkpoint.py`

- [ ] **Step 1: Write the failing checkpoint smoke test**

```python
from standalone_t2m.checkpoint import CheckpointBundle, resolve_checkpoint_bundle


def test_resolve_checkpoint_bundle_requires_args_json(tmp_path):
    model_path = tmp_path / "model000200000.pt"
    model_path.write_bytes(b"fake")
    try:
        resolve_checkpoint_bundle(model_path)
    except FileNotFoundError as exc:
        assert "args.json" in str(exc)
    else:
        raise AssertionError("resolve_checkpoint_bundle should fail when args.json is missing")
```

- [ ] **Step 2: Run the checkpoint test to verify it fails**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_t2m_standalone
python -m pytest tests/test_checkpoint.py::test_resolve_checkpoint_bundle_requires_args_json -v
```
Expected: FAIL because `standalone_t2m.checkpoint` does not exist yet.

- [ ] **Step 3: Copy only the inference subset from CLoSD**

Run:
```bash
mkdir -p /home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/vendor/diffusion_planner
mkdir -p /home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/assets
rsync -a /home/lyuxinghe/code/CLoSD/closd/diffusion_planner/model /home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/vendor/diffusion_planner/
rsync -a /home/lyuxinghe/code/CLoSD/closd/diffusion_planner/diffusion /home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/vendor/diffusion_planner/
rsync -a /home/lyuxinghe/code/CLoSD/closd/diffusion_planner/utils /home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/vendor/diffusion_planner/
rsync -a /home/lyuxinghe/code/CLoSD/closd/diffusion_planner/data_loaders /home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/vendor/diffusion_planner/
cp /home/lyuxinghe/code/CLoSD/closd/diffusion_planner/dataset/t2m_mean.npy /home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/assets/t2m_mean.npy
cp /home/lyuxinghe/code/CLoSD/closd/diffusion_planner/dataset/t2m_std.npy /home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/assets/t2m_std.npy
```
Expected: the standalone repo now contains vendored DiP/HumanML inference code and normalization assets, but no train/eval entrypoints are copied into the public package.

- [ ] **Step 4: Implement the checkpoint bundle resolver**

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckpointBundle:
    model_path: Path
    args_path: Path
    mean_path: Path
    std_path: Path


def resolve_checkpoint_bundle(model_path: Path) -> CheckpointBundle:
    model_path = Path(model_path).resolve()
    args_path = model_path.parent / "args.json"
    if not args_path.is_file():
        raise FileNotFoundError(f"Missing args.json next to checkpoint: {args_path}")
    return CheckpointBundle(
        model_path=model_path,
        args_path=args_path,
        mean_path=Path(__file__).resolve().parent / "assets" / "t2m_mean.npy",
        std_path=Path(__file__).resolve().parent / "assets" / "t2m_std.npy",
    )
```

- [ ] **Step 5: Re-run the checkpoint test**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_t2m_standalone
python -m pytest tests/test_checkpoint.py -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/lyuxinghe/code/CLoSD_t2m_standalone
git add standalone_t2m/assets standalone_t2m/vendor standalone_t2m/checkpoint.py tests/test_checkpoint.py
git commit -m "feat: vendor dip inference subset and checkpoint resolver"
```

### Task 3: Load The Vendored DiP Model Without Simulator Imports

**Files:**
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/config.py`
- Modify: `/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/checkpoint.py`
- Test: `/home/lyuxinghe/code/CLoSD_t2m_standalone/tests/test_checkpoint.py`

- [ ] **Step 1: Add a failing model-config test**

```python
import json

from standalone_t2m.config import load_model_args


def test_load_model_args_keeps_context_and_pred_lengths(tmp_path):
    args_path = tmp_path / "args.json"
    args_path.write_text(json.dumps({"context_len": 20, "pred_len": 40, "guidance_param": 2.5}), encoding="utf-8")
    args = load_model_args(args_path)
    assert args.context_len == 20
    assert args.pred_len == 40
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_t2m_standalone
python -m pytest tests/test_checkpoint.py::test_load_model_args_keeps_context_and_pred_lengths -v
```
Expected: FAIL because `standalone_t2m.config` does not exist yet.

- [ ] **Step 3: Implement a small `args.json` loader and model builder wrapper**

```python
import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from standalone_t2m.vendor.diffusion_planner.utils.model_util import create_model_and_diffusion, load_model_wo_clip
from standalone_t2m.vendor.diffusion_planner.utils.sampler_util import ClassifierFreeSampleModel


def load_model_args(args_path: Path) -> Namespace:
    with open(args_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Namespace(**data)


@dataclass
class LoadedModelBundle:
    model: torch.nn.Module
    diffusion: object
    sample_fn: object
    args: Namespace
    mean: torch.Tensor
    std: torch.Tensor
    context_len: int
    pred_len: int
    device: torch.device
```

```python
def build_model_and_diffusion(bundle):
    args = load_model_args(bundle.args_path)
    dummy_data = type("DummyData", (), {"dataset": type("DummyDataset", (), {"t2m_dataset": type("T2M", (), {})()})()})()
    model, diffusion = create_model_and_diffusion(args, dummy_data)
    state_dict = torch.load(bundle.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)
    if getattr(args, "guidance_param", 2.5) != 1:
        model = ClassifierFreeSampleModel(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return LoadedModelBundle(
        model=model,
        diffusion=diffusion,
        sample_fn=diffusion.p_sample_loop,
        args=args,
        mean=torch.from_numpy(np.load(bundle.mean_path)).to(device),
        std=torch.from_numpy(np.load(bundle.std_path)).to(device),
        context_len=args.context_len,
        pred_len=args.pred_len,
        device=device,
    )
```

- [ ] **Step 4: Add and run a no-simulator import test**

```python
from pathlib import Path


def test_repo_contains_no_simulator_imports():
    root = Path("standalone_t2m")
    banned = ("isaacgym", "omni.isaac", "isaacsim")
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in banned), f"Unexpected simulator import in {path}"
```

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_t2m_standalone
python -m pytest tests/test_checkpoint.py tests/test_no_sim_imports.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/lyuxinghe/code/CLoSD_t2m_standalone
git add standalone_t2m/config.py standalone_t2m/checkpoint.py tests/test_no_sim_imports.py tests/test_checkpoint.py
git commit -m "feat: add standalone model config loader"
```

## Chunk 2: Standing Prefix And Autoregressive Generation

### Task 4: Implement The Synthetic Standing Prefix Builder

**Files:**
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/prefix/standing.py`
- Test: `/home/lyuxinghe/code/CLoSD_t2m_standalone/tests/test_standing_prefix.py`

- [ ] **Step 1: Write the failing standing-prefix tests**

```python
import torch

from standalone_t2m.prefix.standing import build_standing_prefix, neutral_standing_joints


def test_neutral_standing_joints_has_humanml_joint_count():
    joints = neutral_standing_joints()
    assert joints.shape == (22, 3)


def test_build_standing_prefix_returns_model_ready_shape():
    prefix = build_standing_prefix(context_len=20)
    assert prefix.shape == (1, 263, 1, 20)
    assert torch.isfinite(prefix).all()
```

- [ ] **Step 2: Run the prefix tests to verify they fail**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_t2m_standalone
python -m pytest tests/test_standing_prefix.py -v
```
Expected: FAIL because the prefix module does not exist yet.

- [ ] **Step 3: Implement the neutral standing joints and feature conversion**

```python
import numpy as np
import torch

from standalone_t2m.vendor.diffusion_planner.data_loaders.humanml.scripts.motion_process_torch import extract_features_t2m


def neutral_standing_joints() -> torch.Tensor:
    return torch.tensor(np.array([
        [0.0, 1.00, 0.0], [0.1, 0.92, 0.0], [0.1, 0.52, 0.02], [0.1, 0.08, 0.04],
        [0.1, 0.00, 0.12], [-0.1, 0.92, 0.0], [-0.1, 0.52, 0.02], [-0.1, 0.08, 0.04],
        [-0.1, 0.00, 0.12], [0.0, 1.18, 0.0], [0.0, 1.34, 0.0], [0.0, 1.52, 0.0],
        [0.0, 1.66, 0.0], [0.0, 1.82, 0.02], [0.16, 1.46, 0.0], [0.32, 1.42, 0.0],
        [0.48, 1.32, 0.0], [0.60, 1.20, 0.0], [-0.16, 1.46, 0.0], [-0.32, 1.42, 0.0],
        [-0.48, 1.32, 0.0], [-0.60, 1.20, 0.0],
    ]), dtype=torch.float32)


def build_standing_prefix(context_len: int = 20) -> torch.Tensor:
    seq = neutral_standing_joints().unsqueeze(0).unsqueeze(0).repeat(1, context_len + 1, 1, 1)
    features, _ = extract_features_t2m(seq, fix_ik_bug=True)
    assert features.shape[1] == context_len
    return features.permute(0, 2, 1).unsqueeze(2)
```

- [ ] **Step 4: Normalize the prefix with shipped mean/std and assert exact context length**

```python
def normalize_prefix(features, mean, std):
    return ((features - mean) / std).float()
```

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_t2m_standalone
python -m pytest tests/test_standing_prefix.py -v
```
Expected: PASS with shape `(1, 263, 1, 20)`.

- [ ] **Step 5: Commit**

```bash
cd /home/lyuxinghe/code/CLoSD_t2m_standalone
git add standalone_t2m/prefix/standing.py tests/test_standing_prefix.py
git commit -m "feat: add standing-start prefix builder"
```

### Task 5: Implement Autoregressive Chunked Generation

**Files:**
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/generation.py`
- Test: `/home/lyuxinghe/code/CLoSD_t2m_standalone/tests/test_generation.py`

- [ ] **Step 1: Write the failing generation tests**

```python
import torch

from standalone_t2m.generation import next_prefix, stitch_prediction_chunks


def test_next_prefix_takes_last_context_frames():
    sample = torch.arange(1 * 263 * 1 * 60, dtype=torch.float32).reshape(1, 263, 1, 60)
    prefix = next_prefix(sample, context_len=20)
    assert prefix.shape[-1] == 20
    assert torch.equal(prefix, sample[..., -20:])


def test_stitch_prediction_chunks_truncates_to_requested_frames():
    chunk = torch.zeros(1, 263, 1, 40)
    full = stitch_prediction_chunks([chunk, chunk, chunk], target_frames=90)
    assert full.shape[-1] == 90
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_t2m_standalone
python -m pytest tests/test_generation.py -v
```
Expected: FAIL because the generation helpers do not exist yet.

- [ ] **Step 3: Implement pure tensor helpers first**

```python
import torch


def next_prefix(sample: torch.Tensor, context_len: int) -> torch.Tensor:
    return sample[..., -context_len:].clone()


def stitch_prediction_chunks(chunks, target_frames: int) -> torch.Tensor:
    return torch.cat(chunks, dim=-1)[..., :target_frames]
```

- [ ] **Step 4: Implement the real generator wrapper around the vendored sampler**

```python
import torch


def sample_once(model_bundle, prompt: str, prefix: torch.Tensor, guidance: float) -> torch.Tensor:
    model_kwargs = {
        "y": {
            "text": [prompt],
            "prefix": prefix.to(model_bundle.device),
            "mask": torch.ones(1, 1, 1, model_bundle.pred_len, device=model_bundle.device),
        }
    }
    if guidance != 1:
        model_kwargs["y"]["scale"] = torch.tensor([guidance], device=model_bundle.device)
    return model_bundle.sample_fn(
        model_bundle.model,
        (1, model_bundle.model.njoints, model_bundle.model.nfeats, model_bundle.pred_len),
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,
        init_image=None,
        progress=False,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )


def generate_motion(model_bundle, prompt: str, target_frames: int, guidance: float, prefix: torch.Tensor):
    chunks = []
    while sum(chunk.shape[-1] for chunk in chunks) < target_frames:
        sample = sample_once(model_bundle, prompt=prompt, prefix=prefix, guidance=guidance)
        pred = sample[..., -model_bundle.pred_len:]
        chunks.append(pred)
        prefix = next_prefix(sample, context_len=model_bundle.context_len)
    return stitch_prediction_chunks(chunks, target_frames)
```

- [ ] **Step 5: Re-run the tests**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_t2m_standalone
python -m pytest tests/test_generation.py -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/lyuxinghe/code/CLoSD_t2m_standalone
git add standalone_t2m/generation.py tests/test_generation.py
git commit -m "feat: add autoregressive generation loop"
```

## Chunk 3: Decode, Render, CLI, And End-To-End Verification

### Task 6: Decode Generated Motion And Render Offline MP4

**Files:**
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/decode.py`
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/render.py`
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/artifacts.py`
- Test: `/home/lyuxinghe/code/CLoSD_t2m_standalone/tests/test_decode_render.py`

- [ ] **Step 1: Write the failing decode/render tests**

```python
import torch

from standalone_t2m.decode import decode_to_xyz
from standalone_t2m.render import output_mp4_path


def test_decode_to_xyz_returns_22_joint_sequence():
    sample = torch.zeros(1, 263, 1, 40)
    xyz = decode_to_xyz(sample, mean=torch.zeros(263), std=torch.ones(263))
    assert xyz.shape[0] == 1
    assert xyz.shape[-2] == 22


def test_output_mp4_path_uses_prompt_slug(tmp_path):
    path = output_mp4_path(tmp_path, "A person is moonwalking.")
    assert path.name.endswith(".mp4")
    assert "moonwalking" in path.name
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_t2m_standalone
python -m pytest tests/test_decode_render.py -v
```
Expected: FAIL because decode/render modules do not exist yet.

- [ ] **Step 3: Implement decode and artifact helpers**

```python
from pathlib import Path

import torch

from standalone_t2m.vendor.diffusion_planner.data_loaders.humanml.scripts.motion_process import recover_from_ric


def decode_to_xyz(sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    motion = sample.squeeze(2).permute(0, 2, 1)
    motion = (motion * std) + mean
    return recover_from_ric(motion, 22)


def output_mp4_path(output_dir: Path, prompt: str) -> Path:
    slug = prompt.lower().replace(" ", "_").replace(".", "")
    return output_dir / f"{slug}.mp4"
```

- [ ] **Step 4: Implement the renderer and artifact writer**

```python
def write_artifacts(output_dir, prompt, generated, xyz, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(generated.cpu(), output_dir / "motion.pt")
    torch.save(xyz.cpu(), output_dir / "xyz.pt")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
```

```python
def render_xyz_motion(xyz, prompt, mp4_path):
    motion = xyz[0].detach().cpu().numpy()
    plot_3d_motion(str(mp4_path), paramUtil.t2m_kinematic_chain, motion, dataset="humanml", title=prompt, fps=20)
```

- [ ] **Step 5: Re-run the decode/render tests**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_t2m_standalone
python -m pytest tests/test_decode_render.py -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/lyuxinghe/code/CLoSD_t2m_standalone
git add standalone_t2m/decode.py standalone_t2m/render.py standalone_t2m/artifacts.py tests/test_decode_render.py
git commit -m "feat: add decode and offline render pipeline"
```

### Task 7: Add The User-Facing CLI And Integration Smoke Test

**Files:**
- Create: `/home/lyuxinghe/code/CLoSD_t2m_standalone/standalone_t2m/cli.py`
- Modify: `/home/lyuxinghe/code/CLoSD_t2m_standalone/README.md`
- Test: `/home/lyuxinghe/code/CLoSD_t2m_standalone/tests/integration/test_generate_cli.py`

- [ ] **Step 1: Write the failing CLI parser test**

```python
from standalone_t2m.cli import build_parser


def test_parser_defaults_to_standing_start():
    parser = build_parser()
    args = parser.parse_args(["--prompt", "a person is moonwalking."])
    assert args.prefix_mode == "standing_start"
```

- [ ] **Step 2: Run the parser test to verify it fails**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_t2m_standalone
python -m pytest tests/integration/test_generate_cli.py::test_parser_defaults_to_standing_start -v
```
Expected: FAIL because the CLI module does not exist yet.

- [ ] **Step 3: Implement the narrow CLI**

```python
import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-path", default="checkpoints/DiP_no-target_10steps_context20_predict40/model000200000.pt")
    parser.add_argument("--num-seconds", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--prefix-mode", default="standing_start", choices=["standing_start"])
    parser.add_argument("--output-dir", default="outputs")
    return parser
```

- [ ] **Step 4: Add an integration smoke test with a real checkpoint path gate**

```python
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
def test_generate_cli_writes_outputs(tmp_path):
    model_path = os.environ.get("CLOSD_T2M_MODEL_PATH")
    if not model_path:
        pytest.skip("CLOSD_T2M_MODEL_PATH is not set")
    subprocess.run(
        [
            "python",
            "-m",
            "standalone_t2m.cli",
            "--prompt",
            "a person is moonwalking.",
            "--model-path",
            model_path,
            "--num-seconds",
            "2",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
    )
    assert (tmp_path / "motion.pt").is_file()
    assert any(path.suffix == ".mp4" for path in tmp_path.iterdir())
```

- [ ] **Step 5: Update the README with one exact command**

```markdown
python -m standalone_t2m.cli \
  --prompt "a person is moonwalking." \
  --model-path /path/to/model000200000.pt \
  --output-dir outputs/moonwalk
```

- [ ] **Step 6: Run the parser test and the simulator-import regression test**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_t2m_standalone
python -m pytest tests/integration/test_generate_cli.py::test_parser_defaults_to_standing_start tests/test_no_sim_imports.py -v
```
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
cd /home/lyuxinghe/code/CLoSD_t2m_standalone
git add standalone_t2m/cli.py README.md tests/integration/test_generate_cli.py
git commit -m "feat: add standalone t2m cli"
```

### Task 8: Run The Final End-To-End Verification

**Files:**
- Verify: `/home/lyuxinghe/code/CLoSD_t2m_standalone/tests/`
- Verify: `/home/lyuxinghe/code/CLoSD_t2m_standalone/README.md`

- [ ] **Step 1: Run the full unit test suite**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_t2m_standalone
python -m pytest tests -m "not integration" -v
```
Expected: all non-integration tests PASS.

- [ ] **Step 2: Run the gated integration smoke test if a checkpoint path is available**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_t2m_standalone
CLOSD_T2M_MODEL_PATH=/path/to/model000200000.pt python -m pytest tests/integration/test_generate_cli.py -m integration -v
```
Expected: PASS and the temp output directory contains `motion.pt`, `xyz.pt`, `metadata.json`, and an `.mp4`.

- [ ] **Step 3: Manually smoke-test the CLI**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_t2m_standalone
python -m standalone_t2m.cli \
  --prompt "a person is moonwalking." \
  --model-path /path/to/model000200000.pt \
  --num-seconds 4 \
  --output-dir outputs/moonwalk
```
Expected: the command exits zero and writes the four expected artifacts.

- [ ] **Step 4: Verify the repo contains no simulator dependencies**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_t2m_standalone
rg -n "isaacgym|omni\\.isaac|isaacsim" standalone_t2m
```
Expected: no matches.

- [ ] **Step 5: Commit**

```bash
cd /home/lyuxinghe/code/CLoSD_t2m_standalone
git add README.md tests
git commit -m "test: verify standalone t2m inference flow"
```
