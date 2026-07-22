# SOMA Retargeter
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![SOMA Retargeter Banner](assets/docs/banner.gif)

Convert [SOMA](https://github.com/NVlabs/SOMA-X) human motion captures into humanoid robot joint animation. Takes BVH motion files as input and produces robot-playable CSV joint data as output using GPU-optimized inverse kinematics via [Newton](https://github.com/newton-physics/newton) and high-performance computation with [NVIDIA Warp](https://github.com/NVIDIA/warp).

The retargeting pipeline handles proportional human-to-robot scaling, multi-objective IK solving with joint limits, feet stabilization to maintain ground contact, and per-DOF joint limit clamping. It uses SOMA as the input skeleton and ships with Unitree G1 (29 DOF) and Humanoid Ultra (27 DOF) as output robots. Adding a new robot target is data-driven and is documented under [Adding a New Robot Target](#adding-a-new-robot-target).

SOMA Retargeter is part of the [SOMA body model](https://github.com/NVlabs/SOMA-X) ecosystem for humanoid motion data.

> **Note:** This project is in active development. The API may change between releases as the design is refined.

## Requirements

- **Python:** 3.12
- **Git LFS:** Installed and initialized for asset downloads
- **OS:** Windows (x86-64) and Linux (x86-64, aarch64)
- **GPU:** NVIDIA GPU (Maxwell or newer), driver 545+ (CUDA 12). No local CUDA Toolkit installation required.

## Installation

<details>

<summary>Setup instructions</summary>

### Method 1 (conda + pip)

#### 1. Create and Activate Conda Environment

```bash
conda create -n soma-retargeter python=3.12 -y
conda activate soma-retargeter
```

#### 2. Download LFS Assets

```bash
git lfs pull
```

#### 3. Install the Library

```bash
pip install .
```

### Method 2 (uv)

#### 1. Install uv

Follow the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/) if `uv` is not yet installed.

#### 2. Download LFS Assets

```bash
git lfs pull
```

#### 3. Sync the Project

`uv sync` creates an isolated `.venv` virtual environment inside the project directory, installs the correct Python version and resolves all dependencies.

```bash
uv sync
```

### Platform-specific notes

**Note (Linux):** For the GUI viewer to work, install `tkinter`

```bash
sudo apt-get install python3.12-tk
```

**Note (Windows):** If `imgui-bundle` fails to install, the Microsoft Visual C++ Redistributables may be missing. Download from the [official Microsoft documentation](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

</details>

## Motion Data

This repo includes 10 sample BVH/CSV pairs in `assets/motions/` for immediate testing.

For large-scale motion data, see the [SEED dataset](https://huggingface.co/datasets/bones-studio/seed) (Skeletal Everyday Embodiment Dataset) published by [Bones Studio](https://huggingface.co/bones-studio). SEED provides a large-scale collection of human motions on the SOMA uniform-proportion skeleton, which is the expected input format for this tool. The G1 robot motion data included in SEED was retargeted using SOMA Retargeter.

### Convert YM/FZ BVH to SOMA

Use `tools/convert_ym_bvh_to_soma.py` to convert a YM/FZ 57-joint BVH file to the SOMA 78-joint layout. Run the following command from the repository root:

```bash
python3 tools/convert_ym_bvh_to_soma.py \
  --file "mocap/Data_2026-07-10_10-30-50.bvh" \
  --output "mocap_soma/july_heading_normalized.bvh" \
  --soma-geometry \
  --no-orientation-offsets \
  --source-reference-seconds 1.0 \
  --normalize-root-heading \
  --global-bind-offset \
  --overwrite
```

The options used above are:

| Option | Purpose |
|--------|---------|
| `--file` | Selects one source BVH file. Quote the path when it contains spaces. |
| `--output` | Sets the converted SOMA BVH path. Parent directories are created automatically. |
| `--soma-geometry` | Uses the SOMA template bone geometry instead of the source bone offsets. |
| `--no-orientation-offsets` | Disables the fixed adjustments in `tools/soma_effector_orientation_offsets.json`. |
| `--source-reference-seconds 1.0` | Averages the first second of the source motion as its reference pose. |
| `--normalize-root-heading` | Rotates the reference root heading to world `+Z` and applies the same correction to the root X/Z trajectory. This keeps facing direction and travel direction consistent across capture sessions. |
| `--global-bind-offset` | Aligns source and SOMA joint coordinate frames using fixed global bind-pose offsets. |
| `--overwrite` | Replaces the output file if it already exists. Omit this option to protect an existing result. |

The first second should contain a stable pose with the performer facing the intended forward direction. If the motion starts immediately, provide a compatible static YM/FZ reference BVH instead:

```bash
python3 tools/convert_ym_bvh_to_soma.py \
  --file "mocap/input.bvh" \
  --output "mocap_soma/output.bvh" \
  --soma-geometry \
  --source-reference "mocap/static_reference.bvh" \
  --normalize-root-heading \
  --global-bind-offset
```

Use `--dry-run` first to validate the skeleton, frame count, reference settings, and output path without writing a file.

## Quick Start

> When using **uv** (Method 2), replace `python` with `uv run` in the commands below.

### Interactive viewer (OpenGL)

```bash
python ./app/bvh_to_csv_converter.py --config ./assets/default_bvh_to_csv_converter_config.json --viewer gl
```

![Interactive viewer interface](assets/docs/interactive-viewer-screenshot.png)

The viewer displays the source SOMA motion alongside the retargeted robot in a 3D viewport. Use the right panel to load BVH files, run retargeting, and save CSV output. Playback controls at the bottom allow scrubbing, speed adjustment, and looping. Toggle visibility of the skinned mesh, skeleton, joint axes, and positioning gizmos.

### Batch conversion (headless)

Process a folder of BVH files without a display. Set `import_folder` and `export_folder` in the config file, then run:

```bash
python ./app/bvh_to_csv_converter.py --config ./assets/default_bvh_to_csv_converter_config.json --viewer null
```

Batch mode recursively finds all `.bvh` files in the import folder, processes them in configurable batch sizes, and writes CSV files to the export folder mirroring the input directory structure.

### Selecting the target robot

The target robot is chosen by the `retarget_target` field in the config file. Both the viewer and batch modes read it:

```json
{
    "retarget_source": "soma",
    "retarget_target": "humanoid_ultra",
    "retarget_source_facing_direction": "Mujoco"
}
```

A ready-to-run config for Humanoid Ultra is provided at `assets/humanoid_ultra_bvh_to_csv_converter_config.json`.

## Adding a New Robot Target

The pipeline is data-driven: the IK solver, scaling, feet stabilization, and joint-limit clamping are all robot-agnostic. Adding a robot means registering it in a few places and providing three JSON config files. Humanoid Ultra (`soma_retargeter/configs/humanoid_ultra/`) is a complete worked example built from a URDF — mirror it for your own robot.

### 1. Register the robot in code

**`soma_retargeter/pipelines/utils.py`**

- Add a member to the `TargetType` enum.
- Add its string name to `_TARGET_TYPE_TO_STR`.
- Add a `(SourceType.SOMA, TargetType.YOUR_ROBOT)` entry to `_RETARGETER_CONFIG_FILES` pointing at your retargeter config.
- Add a branch to `create_robot_model_builder()` that loads your robot model into the `newton.ModelBuilder`. Use `builder.add_mjcf(...)` for an MJCF, or `builder.add_urdf(path, floating=True)` for a URDF.

**`soma_retargeter/assets/csv.py`**

- Subclass `_EulerRootCSVConfigBase` and define `csv_header`. The CSV layout is `[Frame, root translate xyz (cm), root rotate xyz (euler deg), joint dofs (deg)]`. **The joint columns must be listed in the robot's Newton DOF order** — for a URDF this is the depth-first joint order, which you can print with `[newton_utils.get_name_from_label(l) for l in builder.body_label]` (skip the root/base body).
- Register the class in `_ROBOT_CSV_CONFIGS` under the robot's string name.

### 2. Create the config directory

Create `soma_retargeter/configs/<your_robot>/` with three files (see Humanoid Ultra for the exact schema):

| File | Purpose |
|------|---------|
| `soma_to_<robot>_retargeter_config.json` | IK settings + `ik_map`: maps each SOMA joint (`Hips`, `LeftHand`, …) to a robot link and per-effector position/rotation weights. |
| `soma_to_<robot>_scaler_config.json` | Per-joint human→robot `joint_scales`, the SOMA `joint_parents` topology, and per-joint `joint_offsets` (translation + quaternion) that align the SOMA zero pose onto the robot. |
| `<robot>_feet_stabilizer_config.json` | Two-bone leg IK effectors and hints used to keep feet planted on the ground. |

For a URDF-based robot, also place the robot description in this directory. Newton cannot resolve `package://` mesh URIs, so rewrite the mesh paths to absolute (or relative-to-URDF) paths in your copy — otherwise the kinematics still load but the viewer shows no meshes.

### 3. Generating the scaler offsets

`joint_scales` and `joint_offsets` are the only genuinely robot-specific numbers, and they are derived rather than hand-authored. The alignment method is:

- Compute the SOMA **zero pose** global joint transforms (load `configs/soma/soma_zero_frame0.bvh` with `Mujoco` facing) and the robot **rest pose** body transforms (`newton.eval_fk` on the freshly built model, lifted so the soles touch the ground).
- The pipeline expects each effector target at the zero pose to equal the robot rest pose pre-rotated by `W = Rz(-90°)`. So for each mapped joint:
  - `offset_q = conj(q_soma_zero) · W · q_robot_rest`
  - `offset_t = rotate(conj(W · q_robot_rest), W(p_robot_rest) − p_scaled_zero)`
  - `joint_scale` = ratio of the robot rest limb length to the SOMA zero limb length (radial from the root).
- Note the scaler code copies `LeftToe`/`RightToe` offsets onto `LeftToeBase`/`RightToeBase`, so those keys must be present in `joint_offsets`.

This can be scripted; validate it by running the same procedure against the shipped G1 config and checking the computed offsets match to within a few degrees before trusting it on a new robot.

### 4. Run and tune

Point a converter config at the new `retarget_target` and run. Verify the output CSV keeps every joint within its URDF limits, that per-frame deltas stay small (smooth), and that a forward-kinematics reconstruction of the hands/feet tracks the SOMA source. Fine-tune the `t_weight`/`r_weight` values in the retargeter config and the `joint_scales` in the scaler config against the OpenGL viewer for the best-looking motion.

## Code Overview

### `app/`

| File | Description |
|------|-------------|
| `bvh_to_csv_converter.py` | Main entry point. Drives both interactive and headless batch retargeting modes. |

### `soma_retargeter/`

| Module | Description |
|--------|-------------|
| `animation/` | Core data structures for skeletons, animation buffers, IK, and skinned meshes. |
| `assets/` | File I/O for BVH, CSV, and USD formats. |
| `pipelines/` | Retargeting pipeline: IK solving, feet stabilization, and joint limit clamping. |
| `robotics/` | Human-to-robot scaling and robot output formatting. |
| `renderers/` | Visualization for the interactive viewer. |
| `utils/` | Math, pose, coordinate conversion, Newton and Warp helpers. |
| `configs/` | JSON configuration for retargeting, scaling, and feet stabilization parameters. |

## Related Work

SOMA Retargeter is a support tool within the SOMA ecosystem for humanoid motion data:

* [SOMA Body Model](https://github.com/NVlabs/SOMA-X) - Parametric human body model with standardized skeleton, mesh, and shape parameters
* [GEM-X](https://github.com/NVlabs/GEM-X) - Human motion estimation from video
* [Kimodo](https://github.com/nv-tlabs/kimodo) - Kinematic motion diffusion model for text and constraint-driven 3D human and robot motion generation
* [ProtoMotions](https://github.com/NVlabs/ProtoMotions) - GPU-accelerated simulation and learning framework for training physically simulated digital humans and humanoid robots
* [SONIC](https://nvlabs.github.io/GEAR-SONIC/) - Whole-body control for humanoid robots, training locomotion and interaction policies

## Acknowledgments

This project draws inspiration and builds upon excellent open-source work, including:
* [GMR](https://github.com/YanjieZe/GMR) - General Motion Retargeting
* [PyRoki](https://pyroki-toolkit.github.io/) - A Modular Toolkit for Robot Kinematic Optimization

## License

This codebase is licensed under [Apache-2.0](LICENSE).

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.
