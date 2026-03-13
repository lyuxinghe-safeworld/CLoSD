# CLoSD Standalone T2M Inference Design

**Date:** 2026-03-13

## Goal

Create a standalone sibling repo, `CLoSD_t2m_standalone`, that extracts the text-to-motion inference pipeline from CLoSD into a simulator-free workflow:

- prompt in
- synthetic standing prefix bootstrap
- autoregressive DiP generation
- offline motion visualization

The user-facing experience should stay close to `scripts/run_t2m_condition.sh`, but without Isaac Gym, Isaac Sim, or any closed-loop controller.

## Scope

### In scope

- Extract the minimum inference-only subset of `closd/diffusion_planner` and supporting utilities needed to run the DiP text-conditioned motion generator.
- Build a standalone CLI that accepts a text prompt and produces generated human motion.
- Replace simulator-provided prefix context with a synthetic `standing_start` prefix.
- Run autoregressive generation using the existing DiP checkpoint contract of `context_len=20` and `pred_len=40`.
- Save raw generation artifacts and decoded motion artifacts.
- Produce an offline MP4 visualizer output for the generated motion.

### Out of scope

- Training or evaluation pipelines.
- Isaac Gym integration.
- Isaac Sim integration.
- Closed-loop character control.
- Multi-task or sequence task support.
- Dataset-backed retrieval prefixing.
- Alternate prefix modes or hidden fallbacks.
- True SMPL mesh fitting/rendering in v1.

## Current State Summary

CLoSD currently couples two separate concerns:

1. DiP, an autoregressive diffusion planner that predicts future motion from text and prefix context.
2. A simulator/controller loop that provides the live motion prefix and uses the generated plan for closed-loop control.

Relevant current behavior:

- The text-to-motion flow is launched through `scripts/run_t2m_condition.sh`.
- The DiP checkpoint used by `closd_t2m` is a prefix-completion model with `context_len=20`, `pred_len=40`, and autoregressive sampling.
- In normal CLoSD operation, the prefix comes from live simulator motion converted into HumanML space.
- In the text-to-motion task, there is also an optional first-step dataset-prefix override from HumanML.

For the standalone repo, the model-side contract remains valid, but the simulator-side source of prefix context is removed.

## Considered Approaches

### 1. Retrieval-backed prefix bootstrap

Use a HumanML caption match, take the first 20 frames of a retrieved sample, then switch to self-prefix autoregression.

Pros:

- Closest to the training distribution.
- Likely strong first-window quality.

Cons:

- Uses existing motion clips, which is explicitly not desired.
- Adds dataset lookup and retrieval complexity.

### 2. Synthetic standing-start bootstrap

Construct a static standing pose sequence for the first 20 frames, convert it into HumanML features, then switch to self-prefix autoregression.

Pros:

- No dependency on existing motion clips.
- Very small, deterministic bootstrap surface.
- Keeps the model contract intact.

Cons:

- Dynamic prompts may start more slowly than with a semantically matched bootstrap.

### 3. Zero-prefix bootstrap

Feed zeros or normalized means directly as the first prefix tensor.

Pros:

- Minimal code.

Cons:

- High risk of unstable or off-manifold behavior.
- Does not reflect how the checkpoint is used in upstream CLoSD.

## Recommended Approach

Use the synthetic standing-start bootstrap.

This keeps the extracted repo aligned with the actual DiP inference contract while avoiding any dependency on retrieved motion clips or simulator state. It also avoids designing around zero-prefix behavior, which is easy to implement but low confidence.

## Repository Boundary

Create a new sibling repo named `CLoSD_t2m_standalone` under the same workspace root.

The repo should support exactly one runtime path:

`text prompt -> standing_start prefix -> DiP autoregressive generation -> decoded motion artifacts -> offline MP4`

The repo is inference-only and should not carry over unrelated CLoSD subsystems.

## Architecture

The standalone repo should be organized around four narrow responsibilities:

### 1. Inference loader

Responsibilities:

- load the DiP checkpoint
- load `args.json`
- load normalization stats and text-conditioning dependencies
- expose a generator-ready model and sampler configuration

This unit should not know how prefixes are created or how output is rendered.

### 2. Prefix builder

Responsibilities:

- create the synthetic `standing_start` bootstrap sequence
- enforce the checkpoint’s expected context length
- convert the bootstrap from pose space into normalized HumanML prefix features

This unit replaces both simulator-derived context and dataset-prefix overrides.

### 3. Autoregressive generator

Responsibilities:

- run the first DiP sampling window using the synthetic prefix
- feed the model’s own last 20 generated frames back as the next prefix
- continue until the requested motion duration is reached

This unit should depend only on model inputs and outputs, not on visualization.

### 4. Decoding and rendering

Responsibilities:

- decode generated motion into joint-space outputs
- save raw and decoded artifacts
- render an offline MP4 visualization

This unit should not know how prompts are parsed or how the prefix was created.

## Prefix Policy

The standalone repo uses one supported bootstrap mode: `standing_start`.

`standing_start` is defined as:

- a neutral standing human pose
- repeated exactly for the first 20 frames
- no retrieval from HumanML
- no simulator-derived prefix
- no explicit sway or anticipation motion
- no alternate fallback behavior

Implementation requirements:

- build the prefix in physical pose space first
- convert that pose sequence into the normalized HumanML representation expected by the checkpoint
- pass the result as the model prefix tensor

After the first prediction window:

- always use the model’s own last 20 generated frames as the next prefix
- continue standard autoregressive generation

If the synthetic standing pose cannot be converted into a valid prefix tensor, the run should fail explicitly rather than silently switching prefix behavior.

## Visualization And Artifacts

v1 visualization should be offline-only.

### Required outputs

- generated motion tensor in model/native representation
- decoded joint-space motion suitable for later reuse
- metadata JSON containing prompt, seed, checkpoint path, frame counts, and prefix mode
- offline MP4 visualizing the generated motion

### Rendering contract

The required v1 renderer is joint/skeleton based, using the existing HumanML motion visualization path that already exists in the CLoSD codebase.

This is intentionally narrower than true SMPL mesh rendering. The current repo already supports:

- converting generated HumanML-style motion into decoded joint/pose sequences
- rendering those sequences to offline videos

The current repo does not already provide a direct, standalone path from generated DiP output to fitted SMPL mesh video. Therefore:

- v1 guarantees offline MP4 visualization of the generated motion
- v1 saves decoded artifacts in a form that can support future SMPL-focused visualization work
- true SMPL mesh rendering is a future extension, not a v1 requirement

## Execution Flow

The runtime flow should be:

1. load checkpoint, config, and normalization assets
2. create the synthetic `standing_start` prefix as a 20-frame pose sequence
3. convert the prefix into normalized HumanML features
4. run one DiP prediction window using text plus prefix
5. reuse the model’s own last 20 generated frames as the next prefix
6. continue autoregressively until the requested duration is reached
7. decode the generated motion into joint-space artifacts
8. render the offline MP4 and save metadata

## Public Interfaces

The standalone repo should expose one Python API surface and one CLI surface.

### Python API

- `generate_motion(prompt, num_frames, seed, guidance, checkpoint, prefix_mode="standing_start")`
- `build_standing_prefix(context_len=20)`
- `decode_motion(generated_motion)`
- `render_motion(decoded_motion, output_path)`

### CLI

The CLI should mirror the role of `scripts/run_t2m_condition.sh`, but without simulator arguments.

Example:

```bash
python -m standalone_t2m.generate \
  --prompt "a person is moonwalking." \
  --num-seconds 8 \
  --output-dir outputs/moonwalk
```

The CLI should remain narrow:

- one prompt
- one checkpoint
- one bootstrap mode
- one offline render path

## Dependency Strategy

The new repo should vendor or copy only the minimum code required for:

- DiP model loading
- text conditioning
- sampling
- HumanML normalization and representation conversion
- offline rendering

The inference path must not import:

- Isaac Gym
- Isaac Sim
- task/environment code for closed-loop control

## Error Handling

The repo should fail early and specifically for:

- missing checkpoint files
- missing `args.json`
- missing normalization stats or required representation assets
- inability to build the `standing_start` prefix
- prefix tensor shape mismatch with the checkpoint contract
- NaNs or invalid frame counts returned by generation

Failure behavior should preserve useful outputs:

- if generation succeeds but rendering fails, keep raw motion artifacts, decoded artifacts, and metadata
- if decoding succeeds but MP4 writing fails, keep decoded motion and metadata so rendering can be rerun independently
- if prefix construction fails, stop immediately and do not attempt alternate prefix strategies

## Verification Strategy

Implementation should verify at least the following:

### Unit-level

- `standing_start` produces a valid 20-frame prefix tensor
- one generation window accepts that prefix and returns the expected 40-frame chunk

### End-to-end

- a fixed prompt produces raw motion artifacts
- the same run produces decoded motion artifacts
- the same run produces an offline MP4

### Boundary checks

- the standalone inference path contains no simulator imports
- the CLI does not depend on simulator-only runtime flags or assets

## Success Criteria

The extraction is successful when:

- `CLoSD_t2m_standalone` exists as a standalone repo
- the repo accepts a text prompt from a single CLI command
- the repo does not depend on simulator state to bootstrap the first prefix
- the bootstrap prefix is the synthetic `standing_start` sequence
- generation runs autoregressively after the first window
- the repo saves raw and decoded motion artifacts
- the repo writes an offline MP4 visualization for a known prompt

## Non-Goals For v1

The following are intentionally deferred:

- multiple bootstrap policies
- retrieval-based prefixing
- zero-prefix robustness work
- mesh-based SMPL fitting/rendering
- training and evaluation commands
- generalized simulator reintegration
