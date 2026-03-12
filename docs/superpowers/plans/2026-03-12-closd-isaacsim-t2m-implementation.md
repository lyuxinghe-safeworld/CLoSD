# CLoSD Isaac Sim T2M Migration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained `CLoSD_isaacsim` fork that runs `scripts/run_t2m_condition.sh` in Isaac Sim with the copied SMPL USD humanoid and the approved three-checkpoint validation flow.

**Architecture:** Keep the existing `closd` package mostly intact and add a new `closd_isaacsim` package that owns runtime checks, copied assets, metadata extraction, mapping, Isaac Sim stepping, and the new `run_t2m_isaacsim.py` entrypoint. Preserve the current shell-script prompt UX, but replace the Isaac Gym task stack with a narrow single-env Isaac Sim path executed only through `/isaac-sim/python.sh`.

**Tech Stack:** Bash, Python, Isaac Sim 5.x, Omniverse/Isaac core APIs, PyTorch, Hydra config reuse, pytest.

---

## File Structure

### Repo bootstrap

- Create: `CLoSD_isaacsim/.gitignore`
- Create: `CLoSD_isaacsim/pytest.ini`
- Create: `CLoSD_isaacsim/scripts/setup_isaacsim_env.sh`
- Modify: `CLoSD_isaacsim/scripts/run_t2m_condition.sh`

### New Isaac Sim package

- Create: `CLoSD_isaacsim/closd_isaacsim/__init__.py`
- Create: `CLoSD_isaacsim/closd_isaacsim/runtime.py`
- Create: `CLoSD_isaacsim/closd_isaacsim/run_t2m_isaacsim.py`

### Copied assets and committed metadata

- Create: `CLoSD_isaacsim/closd_isaacsim/assets/usd/smpl_humanoid.usda`
- Create: `CLoSD_isaacsim/closd_isaacsim/assets/reference/smpl_humanoid_closd.xml`
- Create: `CLoSD_isaacsim/closd_isaacsim/assets/reference/smpl_humanoid_protomotions.xml`
- Create: `CLoSD_isaacsim/closd_isaacsim/assets/metadata/smpl_closd.json`
- Create: `CLoSD_isaacsim/closd_isaacsim/assets/metadata/smpl_protomotions.json`
- Create: `CLoSD_isaacsim/closd_isaacsim/assets/metadata/smpl_usd_articulation.json`
- Create: `CLoSD_isaacsim/closd_isaacsim/assets/metadata/closd_to_usd_dof_map.json`

### Metadata and mapping helpers

- Create: `CLoSD_isaacsim/closd_isaacsim/metadata/extract_xml_metadata.py`
- Create: `CLoSD_isaacsim/closd_isaacsim/metadata/extract_usd_metadata.py`
- Create: `CLoSD_isaacsim/closd_isaacsim/metadata/build_mapping.py`
- Create: `CLoSD_isaacsim/closd_isaacsim/sim/pd_utils.py`
- Create: `CLoSD_isaacsim/closd_isaacsim/sim/mapping.py`

### Isaac Sim runners and checkpoint tools

- Create: `CLoSD_isaacsim/closd_isaacsim/sim/runner.py`
- Create: `CLoSD_isaacsim/closd_isaacsim/tools/view_usd_humanoid.py`
- Create: `CLoSD_isaacsim/closd_isaacsim/tools/preview_diffusion_motion.py`
- Create: `CLoSD_isaacsim/closd_isaacsim/tools/check_action_mapping.py`

### T2M inference isolation

- Create: `CLoSD_isaacsim/closd_isaacsim/t2m/adapter.py`
- Create: `CLoSD_isaacsim/closd_isaacsim/t2m/config.py`

### Tests

- Create: `CLoSD_isaacsim/tests/isaacsim/test_runtime.py`
- Create: `CLoSD_isaacsim/tests/isaacsim/test_xml_metadata.py`
- Create: `CLoSD_isaacsim/tests/isaacsim/test_mapping.py`
- Create: `CLoSD_isaacsim/tests/isaacsim/test_t2m_adapter.py`

## Chunk 1: Bootstrap And Runtime Contract

### Task 1: Create The Self-Contained Fork

**Files:**
- Create: `CLoSD_isaacsim/`
- Create: `CLoSD_isaacsim/.gitignore`
- Create: `CLoSD_isaacsim/pytest.ini`

- [ ] **Step 1: Create the sibling copy from the current working tree**

Run:
```bash
rsync -a --exclude '.git' --exclude '__pycache__' /home/lyuxinghe/code/CLoSD/ /home/lyuxinghe/code/CLoSD_isaacsim/
git init /home/lyuxinghe/code/CLoSD_isaacsim
```
Expected: `CLoSD_isaacsim` exists and contains the current local Isaac Sim probe scripts if they are present in the source tree.

- [ ] **Step 2: Add a minimal Python test harness**

```ini
[pytest]
testpaths = tests
pythonpath = .
```

- [ ] **Step 3: Add a fork-specific `.gitignore`**

```gitignore
__pycache__/
.pytest_cache/
output/
*.pyc
```

- [ ] **Step 4: Verify the copied repo can run pytest discovery**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_isaacsim
/isaac-sim/python.sh -m pytest --collect-only
```
Expected: pytest starts without import-order errors and collects zero or more tests.

- [ ] **Step 5: Commit**

```bash
cd /home/lyuxinghe/code/CLoSD_isaacsim
git add .gitignore pytest.ini
git commit -m "chore: bootstrap CLoSD Isaac Sim fork"
```

### Task 2: Enforce The Isaac Sim Runtime Contract

**Files:**
- Create: `CLoSD_isaacsim/closd_isaacsim/__init__.py`
- Create: `CLoSD_isaacsim/closd_isaacsim/runtime.py`
- Create: `CLoSD_isaacsim/scripts/setup_isaacsim_env.sh`
- Test: `CLoSD_isaacsim/tests/isaacsim/test_runtime.py`

- [ ] **Step 1: Write the failing runtime test**

```python
from closd_isaacsim.runtime import RuntimeConfig, validate_runtime


def test_validate_runtime_requires_python_sh(tmp_path):
    cfg = RuntimeConfig(
        isaacsim_python=tmp_path / "missing-python.sh",
        usd_asset=tmp_path / "smpl_humanoid.usda",
        display=":1",
    )
    try:
        validate_runtime(cfg)
    except FileNotFoundError as exc:
        assert "python.sh" in str(exc)
    else:
        raise AssertionError("validate_runtime should fail when python.sh is missing")
```

- [ ] **Step 2: Run the runtime test to verify it fails**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_isaacsim
/isaac-sim/python.sh -m pytest tests/isaacsim/test_runtime.py -v
```
Expected: FAIL because `closd_isaacsim.runtime` does not exist yet.

- [ ] **Step 3: Implement the runtime module and setup script**

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RuntimeConfig:
    isaacsim_python: Path
    usd_asset: Path
    display: str


def validate_runtime(cfg: RuntimeConfig) -> None:
    if not cfg.isaacsim_python.is_file():
        raise FileNotFoundError(f"Missing Isaac Sim launcher: {cfg.isaacsim_python}")
    if not cfg.usd_asset.is_file():
        raise FileNotFoundError(f"Missing USD asset: {cfg.usd_asset}")
```

- [ ] **Step 4: Re-run the runtime test**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_isaacsim
/isaac-sim/python.sh -m pytest tests/isaacsim/test_runtime.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/lyuxinghe/code/CLoSD_isaacsim
git add closd_isaacsim/__init__.py closd_isaacsim/runtime.py scripts/setup_isaacsim_env.sh tests/isaacsim/test_runtime.py
git commit -m "feat: add Isaac Sim runtime contract checks"
```

## Chunk 2: Assets, Metadata, And Spawn Checkpoint

### Task 3: Copy The USD And Reference XML Assets

**Files:**
- Create: `CLoSD_isaacsim/closd_isaacsim/assets/usd/smpl_humanoid.usda`
- Create: `CLoSD_isaacsim/closd_isaacsim/assets/reference/smpl_humanoid_closd.xml`
- Create: `CLoSD_isaacsim/closd_isaacsim/assets/reference/smpl_humanoid_protomotions.xml`
- Create: `CLoSD_isaacsim/closd_isaacsim/metadata/extract_xml_metadata.py`
- Create: `CLoSD_isaacsim/closd_isaacsim/assets/metadata/smpl_closd.json`
- Create: `CLoSD_isaacsim/closd_isaacsim/assets/metadata/smpl_protomotions.json`
- Test: `CLoSD_isaacsim/tests/isaacsim/test_xml_metadata.py`

- [ ] **Step 1: Write the failing XML metadata test**

```python
from closd_isaacsim.metadata.extract_xml_metadata import extract_xml_metadata


def test_extract_xml_metadata_reads_body_and_dof_names(tmp_path):
    xml_path = tmp_path / "mini.xml"
    xml_path.write_text(
        "<mujoco><worldbody><body name='pelvis'><joint name='pelvis_x' axis='1 0 0'/></body></worldbody></mujoco>",
        encoding="utf-8",
    )
    meta = extract_xml_metadata(xml_path)
    assert meta["body_names"] == ["pelvis"]
    assert meta["dof_names"] == ["pelvis_x"]
```

- [ ] **Step 2: Run the XML metadata test to verify it fails**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_isaacsim
/isaac-sim/python.sh -m pytest tests/isaacsim/test_xml_metadata.py -v
```
Expected: FAIL because the extractor does not exist yet.

- [ ] **Step 3: Copy the three source assets and implement XML metadata extraction**

Run:
```bash
cp /home/lyuxinghe/code/ProtoMotions/protomotions/data/assets/usd/smpl_humanoid.usda /home/lyuxinghe/code/CLoSD_isaacsim/closd_isaacsim/assets/usd/smpl_humanoid.usda
cp /home/lyuxinghe/code/CLoSD/closd/data/robot_cache/smpl_humanoid_0.xml /home/lyuxinghe/code/CLoSD_isaacsim/closd_isaacsim/assets/reference/smpl_humanoid_closd.xml
cp /home/lyuxinghe/code/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml /home/lyuxinghe/code/CLoSD_isaacsim/closd_isaacsim/assets/reference/smpl_humanoid_protomotions.xml
```

```python
import xml.etree.ElementTree as ET


def extract_xml_metadata(xml_path):
    root = ET.parse(xml_path).getroot()
    body_names = [body.attrib["name"] for body in root.findall(".//body") if "name" in body.attrib]
    dof_names = [joint.attrib["name"] for joint in root.findall(".//joint") if "name" in joint.attrib]
    return {"body_names": body_names, "dof_names": dof_names}
```

- [ ] **Step 4: Generate and commit the two JSON metadata files**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_isaacsim
/isaac-sim/python.sh -m closd_isaacsim.metadata.extract_xml_metadata \
  --xml closd_isaacsim/assets/reference/smpl_humanoid_closd.xml \
  --out closd_isaacsim/assets/metadata/smpl_closd.json
/isaac-sim/python.sh -m closd_isaacsim.metadata.extract_xml_metadata \
  --xml closd_isaacsim/assets/reference/smpl_humanoid_protomotions.xml \
  --out closd_isaacsim/assets/metadata/smpl_protomotions.json
/isaac-sim/python.sh -m pytest tests/isaacsim/test_xml_metadata.py -v
```
Expected: PASS and both JSON files are created.

- [ ] **Step 5: Commit**

```bash
cd /home/lyuxinghe/code/CLoSD_isaacsim
git add closd_isaacsim/assets closd_isaacsim/metadata/extract_xml_metadata.py tests/isaacsim/test_xml_metadata.py
git commit -m "feat: add SMPL USD asset and XML metadata extraction"
```

### Task 4: Implement The USD Spawn Checkpoint

**Files:**
- Create: `CLoSD_isaacsim/closd_isaacsim/metadata/extract_usd_metadata.py`
- Create: `CLoSD_isaacsim/closd_isaacsim/tools/view_usd_humanoid.py`
- Create: `CLoSD_isaacsim/closd_isaacsim/assets/metadata/smpl_usd_articulation.json`

- [ ] **Step 1: Write the failing USD metadata unit test around a dumped fixture**

```python
from closd_isaacsim.metadata.extract_usd_metadata import normalize_articulation_dump


def test_normalize_articulation_dump_keeps_joint_names():
    dump = {"joint_names": ["Pelvis", "L_Hip"], "body_names": ["pelvis", "left_hip"]}
    norm = normalize_articulation_dump(dump)
    assert norm["joint_names"][0] == "Pelvis"
    assert "body_names" in norm
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_isaacsim
/isaac-sim/python.sh -m pytest tests/isaacsim/test_mapping.py -k articulation_dump -v
```
Expected: FAIL because the extractor is missing.

- [ ] **Step 3: Implement the headless viewer and articulation dumper**

```python
def normalize_articulation_dump(dump):
    return {
        "joint_names": list(dump["joint_names"]),
        "body_names": list(dump["body_names"]),
        "num_joints": len(dump["joint_names"]),
        "num_bodies": len(dump["body_names"]),
    }
```

- [ ] **Step 4: Run the spawn checkpoint headlessly and save the metadata JSON**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_isaacsim
/isaac-sim/python.sh closd_isaacsim/tools/view_usd_humanoid.py \
  --usd-path closd_isaacsim/assets/usd/smpl_humanoid.usda \
  --dump-json closd_isaacsim/assets/metadata/smpl_usd_articulation.json \
  --headless --max-frames 2
```
Expected: the script prints body/joint/DOF counts and writes `smpl_usd_articulation.json`.

- [ ] **Step 5: Commit**

```bash
cd /home/lyuxinghe/code/CLoSD_isaacsim
git add closd_isaacsim/metadata/extract_usd_metadata.py closd_isaacsim/tools/view_usd_humanoid.py closd_isaacsim/assets/metadata/smpl_usd_articulation.json
git commit -m "feat: add USD spawn checkpoint and articulation dump"
```

## Chunk 3: Mapping And Diffusion Preview

### Task 5: Build The Explicit CLoSD-To-USD Mapping Layer

**Files:**
- Create: `CLoSD_isaacsim/closd_isaacsim/metadata/build_mapping.py`
- Create: `CLoSD_isaacsim/closd_isaacsim/sim/pd_utils.py`
- Create: `CLoSD_isaacsim/closd_isaacsim/sim/mapping.py`
- Create: `CLoSD_isaacsim/closd_isaacsim/assets/metadata/closd_to_usd_dof_map.json`
- Test: `CLoSD_isaacsim/tests/isaacsim/test_mapping.py`

- [ ] **Step 1: Write the failing mapping test**

```python
from closd_isaacsim.sim.mapping import build_index_map


def test_build_index_map_aligns_named_entries():
    mapping = build_index_map(["Pelvis", "L_Hip", "R_Hip"], ["R_Hip", "Pelvis", "L_Hip"])
    assert mapping == [1, 2, 0]
```

- [ ] **Step 2: Run the mapping test to verify it fails**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_isaacsim
/isaac-sim/python.sh -m pytest tests/isaacsim/test_mapping.py -v
```
Expected: FAIL because the mapping helpers do not exist yet.

- [ ] **Step 3: Implement the mapping helpers and PD scaling utility**

```python
def build_index_map(source_names, target_names):
    target_lookup = {name: idx for idx, name in enumerate(target_names)}
    return [target_lookup[name] for name in source_names]
```

```python
def build_pd_action_offset_scale(lower, upper):
    return (upper + lower) * 0.5, (upper - lower) * 0.5
```

- [ ] **Step 4: Generate and review the committed mapping JSON**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_isaacsim
/isaac-sim/python.sh -m closd_isaacsim.metadata.build_mapping \
  --closd-meta closd_isaacsim/assets/metadata/smpl_closd.json \
  --usd-meta closd_isaacsim/assets/metadata/smpl_usd_articulation.json \
  --out closd_isaacsim/assets/metadata/closd_to_usd_dof_map.json
/isaac-sim/python.sh -m pytest tests/isaacsim/test_mapping.py -v
```
Expected: PASS and `closd_to_usd_dof_map.json` is produced with explicit name-based mappings.

- [ ] **Step 5: Commit**

```bash
cd /home/lyuxinghe/code/CLoSD_isaacsim
git add closd_isaacsim/metadata/build_mapping.py closd_isaacsim/sim/pd_utils.py closd_isaacsim/sim/mapping.py closd_isaacsim/assets/metadata/closd_to_usd_dof_map.json tests/isaacsim/test_mapping.py
git commit -m "feat: add explicit CLoSD to USD mapping metadata"
```

### Task 6: Isolate The T2M Inference Adapter And Diffusion Preview Checkpoint

**Files:**
- Create: `CLoSD_isaacsim/closd_isaacsim/t2m/config.py`
- Create: `CLoSD_isaacsim/closd_isaacsim/t2m/adapter.py`
- Create: `CLoSD_isaacsim/closd_isaacsim/tools/preview_diffusion_motion.py`
- Test: `CLoSD_isaacsim/tests/isaacsim/test_t2m_adapter.py`

- [ ] **Step 1: Write the failing T2M adapter test**

```python
from closd_isaacsim.t2m.adapter import normalize_prompt


def test_normalize_prompt_adds_person_prefix():
    assert normalize_prompt("moonwalking") == "a person is moonwalking."
```

- [ ] **Step 2: Run the adapter test to verify it fails**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_isaacsim
/isaac-sim/python.sh -m pytest tests/isaacsim/test_t2m_adapter.py -v
```
Expected: FAIL because the adapter does not exist yet.

- [ ] **Step 3: Implement a minimal simulator-independent T2M adapter**

```python
def normalize_prompt(text: str) -> str:
    text = " ".join(text.strip().split())
    if not text.endswith("."):
        text = f"{text}."
    if not text.lower().startswith("a person is "):
        text = f"a person is {text[0].lower() + text[1:] if text else 'moving.'}"
    return text
```

The real implementation should then wrap the existing CLoSD diffusion loader and return tensors/shapes that the Isaac Sim preview tool can consume.

- [ ] **Step 4: Implement and run the diffusion preview checkpoint**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_isaacsim
/isaac-sim/python.sh closd_isaacsim/tools/preview_diffusion_motion.py \
  --prompt "A person is moonwalking." \
  --headless --max-frames 90
```
Expected: the tool prints generated motion tensor shapes and visibly replays the generated motion when run without `--headless`.

- [ ] **Step 5: Commit**

```bash
cd /home/lyuxinghe/code/CLoSD_isaacsim
git add closd_isaacsim/t2m/config.py closd_isaacsim/t2m/adapter.py closd_isaacsim/tools/preview_diffusion_motion.py tests/isaacsim/test_t2m_adapter.py
git commit -m "feat: add simulator-independent T2M adapter and preview checkpoint"
```

## Chunk 4: Action Mapping Checkpoint And Final Runner

### Task 7: Implement The Isaac Sim Runner And Action Mapping Checkpoint

**Files:**
- Create: `CLoSD_isaacsim/closd_isaacsim/sim/runner.py`
- Create: `CLoSD_isaacsim/closd_isaacsim/tools/check_action_mapping.py`

- [ ] **Step 1: Add a failing integration smoke test for the runner constructor**

```python
from closd_isaacsim.sim.runner import IsaacSimHumanoidRunner


def test_runner_exposes_required_handles():
    assert hasattr(IsaacSimHumanoidRunner, "step")
    assert hasattr(IsaacSimHumanoidRunner, "get_articulation_metadata")
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_isaacsim
/isaac-sim/python.sh -m pytest tests/isaacsim/test_mapping.py -k runner -v
```
Expected: FAIL because the runner class does not exist yet.

- [ ] **Step 3: Implement the narrow Isaac Sim single-env runner**

```python
class IsaacSimHumanoidRunner:
    def __init__(self, runtime_cfg, mapping_cfg):
        self.runtime_cfg = runtime_cfg
        self.mapping_cfg = mapping_cfg

    def step(self, actions):
        raise NotImplementedError

    def get_articulation_metadata(self):
        raise NotImplementedError
```

The real implementation should own world creation, articulation spawn, action application, state reads, and recording hooks.

- [ ] **Step 4: Implement and run the action mapping checkpoint**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_isaacsim
/isaac-sim/python.sh closd_isaacsim/tools/check_action_mapping.py \
  --headless --max-frames 120
```
Expected: the tool prints mapped joint names, root orientation, and a short successful stepping loop without immediate instability.

- [ ] **Step 5: Commit**

```bash
cd /home/lyuxinghe/code/CLoSD_isaacsim
git add closd_isaacsim/sim/runner.py closd_isaacsim/tools/check_action_mapping.py
git commit -m "feat: add Isaac Sim runner and action mapping checkpoint"
```

### Task 8: Wire The Final `run_t2m_condition.sh` Path

**Files:**
- Create: `CLoSD_isaacsim/closd_isaacsim/run_t2m_isaacsim.py`
- Modify: `CLoSD_isaacsim/scripts/run_t2m_condition.sh`

- [ ] **Step 1: Write the failing launcher test**

```python
from closd_isaacsim.runtime import build_runner_command


def test_build_runner_command_uses_python_sh():
    cmd = build_runner_command("/isaac-sim/python.sh", "closd_isaacsim/run_t2m_isaacsim.py")
    assert cmd[0] == "/isaac-sim/python.sh"
```

- [ ] **Step 2: Run the launcher test to verify it fails**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_isaacsim
/isaac-sim/python.sh -m pytest tests/isaacsim/test_runtime.py -k build_runner_command -v
```
Expected: FAIL because `build_runner_command` does not exist yet.

- [ ] **Step 3: Implement the final entrypoint and repoint the shell script**

The shell script should keep:
- caption listing
- prompt override behavior
- dataset prefix behavior
- VNC/display defaults

It should replace:
- `python closd/run.py ...`

With:
- `/isaac-sim/python.sh closd_isaacsim/run_t2m_isaacsim.py ...`

- [ ] **Step 4: Run the end-to-end single-prompt smoke test**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_isaacsim
bash scripts/run_t2m_condition.sh --prompt "A person is moonwalking." --record-frames 120
```
Expected: the run starts through `/isaac-sim/python.sh`, uses the copied USD asset, completes one visible rollout in VNC mode, and produces a video or equivalent artifact.

- [ ] **Step 5: Commit**

```bash
cd /home/lyuxinghe/code/CLoSD_isaacsim
git add closd_isaacsim/run_t2m_isaacsim.py scripts/run_t2m_condition.sh tests/isaacsim/test_runtime.py
git commit -m "feat: run T2M through Isaac Sim"
```

### Task 9: Final Verification And Closeout

**Files:**
- Modify: `CLoSD_isaacsim/README.md`
- Modify: `CLoSD_isaacsim/scripts/README_t2m_condition.md`

- [ ] **Step 1: Document the Isaac Sim path and the three checkpoints**

Add concise instructions for:
- `scripts/setup_isaacsim_env.sh`
- `closd_isaacsim/tools/view_usd_humanoid.py`
- `closd_isaacsim/tools/preview_diffusion_motion.py`
- `closd_isaacsim/tools/check_action_mapping.py`
- the final `bash scripts/run_t2m_condition.sh` flow

- [ ] **Step 2: Run the full verification sequence**

Run:
```bash
cd /home/lyuxinghe/code/CLoSD_isaacsim
/isaac-sim/python.sh -m pytest tests/isaacsim -v
/isaac-sim/python.sh closd_isaacsim/tools/view_usd_humanoid.py --headless --max-frames 2
/isaac-sim/python.sh closd_isaacsim/tools/preview_diffusion_motion.py --prompt "A person is walking." --headless --max-frames 90
/isaac-sim/python.sh closd_isaacsim/tools/check_action_mapping.py --headless --max-frames 120
bash scripts/run_t2m_condition.sh --prompt "A person is walking." --record-frames 120
```
Expected: all tests pass, all three checkpoints succeed, and the final runner produces an inspectable artifact.

- [ ] **Step 3: Commit**

```bash
cd /home/lyuxinghe/code/CLoSD_isaacsim
git add README.md scripts/README_t2m_condition.md
git commit -m "docs: document Isaac Sim T2M workflow"
```
