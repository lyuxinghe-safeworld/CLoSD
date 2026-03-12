# CLoSD Isaac Sim T2M Migration Design

**Date:** 2026-03-12

## Goal

Create a self-contained fork named `CLoSD_isaacsim` that ports only the single-environment text-to-motion path behind `scripts/run_t2m_condition.sh` from Isaac Gym to Isaac Sim, using the SMPL USD humanoid asset and keeping the diffusion/text-conditioning side as close to upstream CLoSD as possible.

## Scope

### In scope

- Copy `CLoSD` into a new repo or sibling directory named `CLoSD_isaacsim`.
- Support only the single-environment `run_t2m_condition.sh` flow.
- Use `/isaac-sim/python.sh` as the primary runtime for the new path.
- Use the SMPL USD humanoid asset currently located at `ProtoMotions/protomotions/data/assets/usd/smpl_humanoid.usda`, copied into the fork under a stable internal path.
- Vendor the minimum Isaac Sim / Isaac Lab / SMPL support code needed to keep the fork self-contained.
- Add validation checkpoints for:
  - USD asset spawn and articulation inspection
  - diffusion motion visualization before closed-loop control
  - action-to-humanoid mapping sanity checks

### Out of scope

- Preserving Isaac Gym compatibility for the new path
- Supporting `closd/run.py` as a generic simulator-agnostic entrypoint
- Supporting `multitask`, `sequence`, or multi-environment execution
- Supporting dynamic XML generation, shape-varying SMPL assets, or arbitrary humanoid assets
- Generalizing the simulator integration beyond the `run_t2m_condition.sh` use case

## Current State Summary

### CLoSD

- `scripts/run_t2m_condition.sh` currently launches `python closd/run.py` with `env=closd_t2m`.
- The simulator-facing stack is tightly coupled to Isaac Gym:
  - `closd/run.py` imports `isaacgym` directly and constructs Gym `SimParams`.
  - `closd/env/tasks/humanoid.py` owns asset loading, state refresh, PD target application, force control, viewer integration, and Isaac Gym tensor access.
- The T2M runner already separates prompt selection and dataset caption loading in the shell script, which can be reused.
- The repo already contains early Isaac Sim probing scripts such as `scripts/view_human_isaacsim.py`, which confirms this migration direction is already being explored.

### ProtoMotions

- ProtoMotions already supports SMPL in both MJCF and USD forms.
- Its Isaac Sim path is implemented through Isaac Lab, with a common simulator abstraction and shared PD/action conversion logic.
- Relevant reusable pieces already exist for:
  - SMPL robot metadata
  - USD asset configuration
  - body and DOF ordering conversion
  - PD action offset/scale generation
  - single-robot Isaac Lab simulation stepping

## Recommended Approach

Build a new single-purpose Isaac Sim runner inside `CLoSD_isaacsim` rather than trying to retrofit the existing Isaac Gym task stack.

This is the recommended approach because it minimizes scope, preserves the existing diffusion/text path, avoids trying to emulate Isaac Gym APIs on top of Isaac Sim, and aligns directly with the actual success criterion: making `run_t2m_condition.sh` work with the SMPL USD humanoid in Isaac Sim.

## Architecture

`CLoSD_isaacsim` should fork the execution path early.

- Keep the prompt-selection and caption-loading behavior from `scripts/run_t2m_condition.sh`.
- Replace the Isaac Gym launch target with a new Isaac Sim entrypoint executed via `/isaac-sim/python.sh`.
- Reuse the existing CLoSD text-to-motion inference and conditioning logic as much as possible.
- Introduce a new single-environment Isaac Sim runner dedicated to the SMPL USD humanoid.
- Vendor only the minimal ProtoMotions-derived modules needed for SMPL USD loading, simulator configuration, articulation stepping, and PD/action mapping.

The result is a split between stable logic and unstable logic:

- Stable side: prompt selection, HumanML caption lookup, diffusion inference, motion preprocessing
- Unstable side: simulator integration, articulation naming/order conversion, PD/action mapping, viewer and recording behavior

## Component Design

### 1. Isaac Sim launcher path

The forked `scripts/run_t2m_condition.sh` remains the user-facing entrypoint, but it launches a new Isaac Sim Python module instead of `closd/run.py`.

Responsibilities:

- keep prompt and caption selection UX unchanged
- keep HumanML cache lookup behavior
- validate runtime prerequisites
- launch the new entrypoint with explicit asset and display settings

### 2. T2M inference adapter

A new adapter layer should isolate the minimum CLoSD code needed to turn a prompt into motion/control targets without depending on Isaac Gym.

Responsibilities:

- load prompt-conditioned diffusion or planner artifacts
- produce motion targets in a simulator-independent intermediate form
- reuse existing CLoSD coordinate transforms only where still valid

This layer must not import Isaac Gym or Isaac Sim.

### 3. Single-env Isaac Sim humanoid runner

The runner owns the simulator lifecycle.

Responsibilities:

- create the stage and world
- spawn the copied SMPL USD asset
- expose articulation metadata
- step the physics loop
- apply actions or PD targets
- manage camera/viewer/recording behavior for VNC use

This runner is intentionally narrow and should only support:

- one humanoid
- one environment
- the SMPL USD asset copied into the fork
- the T2M evaluation flow

### 4. Compatibility and mapping layer

This layer bridges CLoSD’s expectations to the USD articulation.

Responsibilities:

- map body names and DOF names/orderings
- convert action conventions into Isaac Sim PD targets
- verify root pose and axis conventions
- expose explicit checks for permutation or frame mistakes

This boundary is the highest-risk part of the port and must be separately testable.

## Runtime and Dependency Strategy

The new path should support one Python runtime only: `/isaac-sim/python.sh`.

Rationale:

- Isaac Sim, Omniverse, and PyTorch import-order constraints make mixed Conda plus Isaac Sim setups fragile.
- Using Isaac Sim’s Python as the primary process avoids ABI mismatches and hidden import-order failures.
- The existing `closd` conda environment should not be treated as the default runtime for the new path.

### Dependency layout

- Base runtime:
  - Isaac Sim’s bundled Python and Omniverse packages
- Added packages:
  - only the subset of CLoSD Python dependencies needed for T2M inference and prompt/data handling
- Vendored local code:
  - minimal ProtoMotions-derived modules copied into `CLoSD_isaacsim`
- Assets:
  - copied SMPL USD asset and any directly required config metadata stored inside the fork

### Launcher contract

Before attempting to run T2M, the launcher should fail fast if any of the following are missing:

- `/isaac-sim/python.sh`
- required importable Python packages inside the Isaac Sim runtime
- copied USD asset inside the fork
- required mapping/config files
- a usable VNC/display configuration when non-headless visualization is requested

## Validation Plan

Validation should happen in three checkpoints before the final runner is considered complete.

### Checkpoint 1: USD spawn checkpoint

Create a standalone viewer that:

- loads the copied `smpl_humanoid.usda`
- opens visibly in Isaac Sim over VNC
- prints articulation body names, joint names, and DOF counts

Purpose:

- verify asset pathing
- verify Isaac Sim runtime and display setup
- verify the articulation structure matches mapping assumptions

### Checkpoint 2: Diffusion motion checkpoint

Create a motion-visualization checkpoint that:

- runs only the CLoSD T2M inference path
- visualizes generated motion in Isaac Sim before closed-loop control

Purpose:

- sanity check the generated motion in the new coordinate conventions
- separate motion-generation issues from physics-control issues

### Checkpoint 3: Action mapping checkpoint

Create a control sanity checkpoint that:

- applies mapped actions or PD targets to the humanoid
- verifies that major joints move along the expected axes
- checks root orientation and body correspondence
- runs for a short episode without immediate catastrophic instability

Purpose:

- verify the action-to-articulation mapping before running the full loop

## Success Criteria

The first-pass migration is successful when:

- `CLoSD_isaacsim` exists as a self-contained copy/fork
- `scripts/run_t2m_condition.sh` launches through `/isaac-sim/python.sh`
- prompt selection still works
- the copied SMPL USD humanoid is used from within the fork
- a single visible T2M rollout can be observed over VNC
- the run reaches episode completion
- the run produces a recording or equivalent artifact for inspection

## Error Handling Expectations

The new path should fail early and explicitly for:

- wrong Python runtime
- missing USD asset or copied config metadata
- unresolved body or DOF name mismatches
- unsupported non-T2M or multi-env usage
- display or recording misconfiguration in VNC mode

Errors should report the exact missing prerequisite or mapping mismatch instead of failing later inside Isaac Sim.

## Key Risks

### Isaac Gym vs Isaac Sim API mismatch

CLoSD currently assumes Isaac Gym ownership of state tensors, asset loading, DOF properties, and viewer behaviors. This cannot be ported mechanically.

Mitigation:

- avoid reusing the Gym task stack for the new path
- isolate simulator integration in a dedicated runner

### Action and frame mismatch

The largest technical risk is incorrect body ordering, DOF ordering, root frame, or PD scale assumptions.

Mitigation:

- explicit mapping layer
- dedicated diffusion visualization checkpoint
- dedicated action mapping checkpoint

### Runtime instability from mixed environments

Mixing Conda Python and Isaac Sim Python is likely to produce unstable imports and binary conflicts.

Mitigation:

- support only `/isaac-sim/python.sh` for the new path
- add a dedicated setup script for required Python packages

## Implementation Direction

The implementation plan should follow this order:

1. create the self-contained fork skeleton and copy the required USD/config assets
2. stand up the Isaac Sim runtime and USD spawn checkpoint
3. isolate the minimal CLoSD T2M inference path
4. vendor and adapt the minimum ProtoMotions simulator/mapping pieces
5. implement the diffusion motion checkpoint
6. implement the action mapping checkpoint
7. wire the final `run_t2m_condition.sh` path
8. verify VNC-visible execution and output artifacts
