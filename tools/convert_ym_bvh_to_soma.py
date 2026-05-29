#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert YM/FZ 57-joint BVH files to the SOMA 78-joint BVH layout.

The converter keeps the source motion timing and remaps each frame into the
joint/channel order used by SOMA Retargeter sample BVH files. Target joints
that are not present in the YM/FZ skeleton are written as static zero channels.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TEMPLATE = Path("assets/motions/bvh/Neutral_walk_forward_002__A057.bvh")
DEFAULT_REFERENCE = Path("soma_retargeter/configs/soma/soma_zero_frame0.bvh")
DEFAULT_ORIENTATION_OFFSETS = Path("tools/soma_effector_orientation_offsets.json")
ROTATION_CHANNELS = ("Zrotation", "Yrotation", "Xrotation")


FZMOTION_IK_OFFSETS_XYZW = {
    "Hips": (0.5, -0.5, -0.5, -0.5),
    "Chest": (0.5, -0.5, -0.5, -0.5),
    "LeftLeg": (0.5, -0.5, -0.5, -0.5),
    "LeftShin": (0.5, -0.5, -0.5, -0.5),
    "LeftFoot": (0.5, -0.5, -0.5, -0.5),
    "RightLeg": (0.5, -0.5, -0.5, -0.5),
    "RightShin": (0.5, -0.5, -0.5, -0.5),
    "RightFoot": (0.5, -0.5, -0.5, -0.5),
    "LeftArm": (0.7071, 0.0, -0.7071, 0.0),
    "LeftForeArm": (1.0, 0.0, 0.0, 0.0),
    "LeftHand": (1.0, 0.0, 0.0, 0.0),
    "RightArm": (0.0, 0.7071, 0.0, 0.7071),
    "RightForeArm": (0.0, 0.0, 0.0, 1.0),
    "RightHand": (0.0, 0.0, 0.0, 1.0),
}


SOMA_IK_OFFSETS_XYZW = {
    "Hips": (0.5, 0.5, 0.5, 0.5),
    "Chest": (0.478, 0.521, 0.521, 0.478),
    "LeftLeg": (0.459, -0.538, -0.538, 0.459),
    "LeftShin": (0.5, -0.5, -0.5, 0.5),
    "LeftFoot": (0.695, -0.128, -0.128, 0.695),
    "RightLeg": (0.538, 0.459, 0.459, 0.538),
    "RightShin": (0.5, 0.5, 0.5, 0.5),
    "RightFoot": (0.128, 0.695, 0.695, 0.128),
    "LeftArm": (-0.5756, -0.5053, 0.4530, 0.4561),
    "LeftForeArm": (-0.707, 0.0, 0.0, 0.707),
    "LeftHand": (-0.7071, 0.0, 0.0, 0.7071),
    "RightArm": (-0.4561, 0.45307, -0.5756, 0.5053),
    "RightForeArm": (0.0, 0.707, -0.707, 0.0),
    "RightHand": (0.0, 0.707, -0.707, 0.0),
}


FZMOTION_CORRECTION_SOURCE = {
    # The fzmotion IK table drives the robot torso from Spine2, while SOMA's
    # stock retargeter uses Chest for the same robot link.
    "Chest": "Spine2",
}


ZERO_LENGTH_INSERTED_JOINTS = {
    "Neck2",
    "LeftHandThumbEnd",
    "LeftHandIndex4",
    "LeftHandIndexEnd",
    "LeftHandMiddle4",
    "LeftHandMiddleEnd",
    "LeftHandRing4",
    "LeftHandRingEnd",
    "LeftHandPinky4",
    "LeftHandPinkyEnd",
    "RightHandThumbEnd",
    "RightHandIndex4",
    "RightHandIndexEnd",
    "RightHandMiddle4",
    "RightHandMiddleEnd",
    "RightHandRing4",
    "RightHandRingEnd",
    "RightHandPinky4",
    "RightHandPinkyEnd",
}


END_EFFECTOR_IK_CHAINS = (
    (("LeftArm", "LeftForeArm", "LeftHand"), ("LeftArm", "LeftForeArm", "LeftHand")),
    (("RightArm", "RightForeArm", "RightHand"), ("RightArm", "RightForeArm", "RightHand")),
    (("LeftLeg", "LeftShin", "LeftFoot"), ("LeftUpLeg", "LeftLeg", "LeftFoot")),
    (("RightLeg", "RightShin", "RightFoot"), ("RightUpLeg", "RightLeg", "RightFoot")),
)


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str | None
    offset: tuple[float, float, float]
    channels: tuple[str, ...]
    channel_start: int

    @property
    def channel_count(self) -> int:
        return len(self.channels)


@dataclass
class BVHData:
    path: Path
    header_lines: list[str]
    joints: list[Joint]
    frames: int
    frame_time: str
    motion_rows: list[list[float]]

    @property
    def channel_count(self) -> int:
        return sum(j.channel_count for j in self.joints)

    @property
    def joint_by_name(self) -> dict[str, Joint]:
        return {j.name: j for j in self.joints}


def _parse_numeric_row(line: str, line_number: int, path: Path) -> list[float]:
    try:
        return [float(value) for value in line.split()]
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number}: invalid numeric motion row") from exc


def parse_bvh(path: Path) -> BVHData:
    lines = path.read_text(encoding="utf-8").splitlines()

    motion_index = None
    raw_joints: list[dict[str, object]] = []
    stack: list[str] = []
    pending_joint: str | None = None
    pending_end_site = False
    end_site_depth = 0
    channel_start = 0

    for index, line in enumerate(lines):
        token = line.split()
        if not token:
            continue

        if token[0] == "MOTION":
            motion_index = index
            break

        if end_site_depth == 0 and token[0] in ("ROOT", "JOINT"):
            if len(token) < 2:
                raise ValueError(f"{path}:{index + 1}: missing joint name")
            name = token[1]
            parent = stack[-1] if stack else None
            raw_joints.append(
                {
                    "name": name,
                    "parent": parent,
                    "offset": (0.0, 0.0, 0.0),
                    "channels": (),
                    "channel_start": None,
                }
            )
            pending_joint = name
        elif token[0] == "{":
            if pending_joint is not None:
                stack.append(pending_joint)
                pending_joint = None
            elif pending_end_site:
                end_site_depth = 1
                pending_end_site = False
            elif end_site_depth > 0:
                end_site_depth += 1
        elif token[0] == "}":
            if end_site_depth > 0:
                end_site_depth -= 1
            elif stack:
                stack.pop()
        elif end_site_depth == 0 and token[:2] == ["End", "Site"]:
            pending_end_site = True
        elif end_site_depth == 0 and token[0] == "OFFSET":
            if not raw_joints:
                raise ValueError(f"{path}:{index + 1}: OFFSET before any joint")
            if len(token) < 4:
                raise ValueError(f"{path}:{index + 1}: malformed OFFSET line")
            raw_joints[-1]["offset"] = (
                float(token[1]),
                float(token[2]),
                float(token[3]),
            )
        elif end_site_depth == 0 and token[0] == "CHANNELS":
            if not raw_joints:
                raise ValueError(f"{path}:{index + 1}: CHANNELS before any joint")
            if len(token) < 2:
                raise ValueError(f"{path}:{index + 1}: malformed CHANNELS line")
            count = int(token[1])
            channels = tuple(token[2:])
            if len(channels) != count:
                raise ValueError(
                    f"{path}:{index + 1}: CHANNELS declares {count} values "
                    f"but has {len(channels)}"
                )
            raw_joints[-1]["channels"] = channels
            raw_joints[-1]["channel_start"] = channel_start
            channel_start += count

    if motion_index is None:
        raise ValueError(f"{path}: missing MOTION section")

    joints: list[Joint] = []
    for raw in raw_joints:
        if raw["channel_start"] is None:
            raise ValueError(f"{path}: joint {raw['name']} has no CHANNELS line")
        joints.append(
            Joint(
                name=str(raw["name"]),
                parent=raw["parent"] if raw["parent"] is None else str(raw["parent"]),
                offset=raw["offset"],  # type: ignore[arg-type]
                channels=tuple(raw["channels"]),  # type: ignore[arg-type]
                channel_start=int(raw["channel_start"]),
            )
        )

    frames: int | None = None
    frame_time: str | None = None
    motion_rows: list[list[float]] = []

    for index in range(motion_index + 1, len(lines)):
        line = lines[index].strip()
        if not line:
            continue

        token = line.split()
        if token[0] == "Frames:":
            frames = int(token[1])
        elif token[:2] == ["Frame", "Time:"]:
            frame_time = token[2]
        else:
            row = _parse_numeric_row(line, index + 1, path)
            if len(row) != channel_start:
                raise ValueError(
                    f"{path}:{index + 1}: expected {channel_start} motion values, "
                    f"got {len(row)}"
                )
            motion_rows.append(row)

    if frames is None:
        raise ValueError(f"{path}: missing Frames line")
    if frame_time is None:
        raise ValueError(f"{path}: missing Frame Time line")
    if len(motion_rows) != frames:
        raise ValueError(
            f"{path}: Frames says {frames}, but parsed {len(motion_rows)} rows"
        )

    return BVHData(
        path=path,
        header_lines=lines[:motion_index],
        joints=joints,
        frames=frames,
        frame_time=frame_time,
        motion_rows=motion_rows,
    )


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [
            a[row][0] * b[0][col]
            + a[row][1] * b[1][col]
            + a[row][2] * b[2][col]
            for col in range(3)
        ]
        for row in range(3)
    ]


def mat_identity() -> list[list[float]]:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def mat_transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[col][row] for col in range(3)] for row in range(3)]


def vec_add(a: list[float], b: list[float]) -> list[float]:
    return [a[index] + b[index] for index in range(3)]


def vec_sub(a: list[float], b: list[float]) -> list[float]:
    return [a[index] - b[index] for index in range(3)]


def vec_mul(vector: list[float], scalar: float) -> list[float]:
    return [value * scalar for value in vector]


def vec_dot(a: list[float], b: list[float]) -> float:
    return sum(a[index] * b[index] for index in range(3))


def vec_cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def vec_length(vector: tuple[float, float, float] | list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def vec_normalize(vector: list[float]) -> list[float]:
    length = vec_length(vector)
    if length < 1e-9:
        return [0.0, 0.0, 0.0]
    return [value / length for value in vector]


def mat_vec_mul(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [
        sum(matrix[row][col] * vector[col] for col in range(3))
        for row in range(3)
    ]


def axis_angle_matrix(axis: list[float], angle: float) -> list[list[float]]:
    x, y, z = vec_normalize(axis)
    c = math.cos(angle)
    s = math.sin(angle)
    one_minus_c = 1.0 - c
    return [
        [
            c + x * x * one_minus_c,
            x * y * one_minus_c - z * s,
            x * z * one_minus_c + y * s,
        ],
        [
            y * x * one_minus_c + z * s,
            c + y * y * one_minus_c,
            y * z * one_minus_c - x * s,
        ],
        [
            z * x * one_minus_c - y * s,
            z * y * one_minus_c + x * s,
            c + z * z * one_minus_c,
        ],
    ]


def rotation_between_vectors(source: list[float], target: list[float]) -> list[list[float]]:
    source_unit = vec_normalize(source)
    target_unit = vec_normalize(target)
    if vec_length(source_unit) < 1e-9 or vec_length(target_unit) < 1e-9:
        return mat_identity()

    cosine = max(-1.0, min(1.0, vec_dot(source_unit, target_unit)))
    if cosine > 1.0 - 1e-8:
        return mat_identity()
    if cosine < -1.0 + 1e-8:
        axis = vec_cross(source_unit, [1.0, 0.0, 0.0])
        if vec_length(axis) < 1e-6:
            axis = vec_cross(source_unit, [0.0, 1.0, 0.0])
        return axis_angle_matrix(axis, math.pi)

    return axis_angle_matrix(vec_cross(source_unit, target_unit), math.acos(cosine))


def normalize_quat_xyzw(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, z, w = quat
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / norm, y / norm, z / norm, w / norm)


def quat_to_matrix_xyzw(quat: tuple[float, float, float, float]) -> list[list[float]]:
    x, y, z, w = normalize_quat_xyzw(quat)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def axis_rotation_matrix(axis: str, degrees: float) -> list[list[float]]:
    angle = math.radians(degrees)
    c = math.cos(angle)
    s = math.sin(angle)
    axis = axis.upper()

    if axis == "X":
        return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
    if axis == "Y":
        return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    if axis == "Z":
        return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]

    raise ValueError(f"unsupported rotation axis: {axis}")


def euler_to_matrix(values: list[float], channels: tuple[str, ...]) -> list[list[float]]:
    matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    rotation_channels = tuple(channel for channel in channels if "rotation" in channel)
    for channel, value in zip(rotation_channels, values):
        matrix = matmul(matrix, axis_rotation_matrix(channel[0], value))
    return matrix


def matrix_to_zyx_euler(matrix: list[list[float]]) -> list[float]:
    # Decompose R = Rz(z) * Ry(y) * Rx(x), matching the SOMA BVH channel order.
    sy = max(-1.0, min(1.0, -matrix[2][0]))
    y = math.asin(sy)
    cy = math.cos(y)

    if abs(cy) > 1e-8:
        x = math.atan2(matrix[2][1], matrix[2][2])
        z = math.atan2(matrix[1][0], matrix[0][0])
    else:
        x = 0.0
        z = math.atan2(-matrix[0][1], matrix[1][1])

    return [math.degrees(z), math.degrees(y), math.degrees(x)]


def rotation_values(row: list[float], joint: Joint) -> list[float]:
    values = []
    for offset, channel in enumerate(joint.channels):
        if "rotation" in channel:
            values.append(row[joint.channel_start + offset])
    return values


def position_values(row: list[float], joint: Joint) -> dict[str, float]:
    values: dict[str, float] = {}
    for offset, channel in enumerate(joint.channels):
        if "position" in channel:
            values[channel] = row[joint.channel_start + offset]
    return values


def local_position_values(row: list[float], joint: Joint) -> list[float]:
    values = [joint.offset[0], joint.offset[1], joint.offset[2]]
    for offset, channel in enumerate(joint.channels):
        if channel == "Xposition":
            values[0] = row[joint.channel_start + offset]
        elif channel == "Yposition":
            values[1] = row[joint.channel_start + offset]
        elif channel == "Zposition":
            values[2] = row[joint.channel_start + offset]
    return values


def local_rotation_matrix(row: list[float], joint: Joint) -> list[list[float]]:
    return euler_to_matrix(rotation_values(row, joint), joint.channels)


def compute_global_rotations(data: BVHData, row: list[float]) -> dict[str, list[list[float]]]:
    rotations: dict[str, list[list[float]]] = {}
    for joint in data.joints:
        local_matrix = local_rotation_matrix(row, joint)
        if joint.parent is None:
            rotations[joint.name] = local_matrix
        else:
            rotations[joint.name] = matmul(rotations[joint.parent], local_matrix)
    return rotations


def compute_global_positions_and_rotations(
    data: BVHData,
    row: list[float],
) -> tuple[dict[str, list[float]], dict[str, list[list[float]]]]:
    positions: dict[str, list[float]] = {}
    rotations: dict[str, list[list[float]]] = {}
    for joint in data.joints:
        local_position = local_position_values(row, joint)
        local_rotation = local_rotation_matrix(row, joint)
        if joint.parent is None:
            positions[joint.name] = local_position
            rotations[joint.name] = local_rotation
        else:
            parent_rotation = rotations[joint.parent]
            positions[joint.name] = vec_add(
                positions[joint.parent],
                mat_vec_mul(parent_rotation, local_position),
            )
            rotations[joint.name] = matmul(parent_rotation, local_rotation)
    return positions, rotations


def compose_rotations(
    first_values: list[float],
    first_channels: tuple[str, ...],
    second_values: list[float],
    second_channels: tuple[str, ...],
) -> list[float]:
    first_matrix = euler_to_matrix(first_values, first_channels)
    second_matrix = euler_to_matrix(second_values, second_channels)
    return matrix_to_zyx_euler(matmul(first_matrix, second_matrix))


def build_joint_mapping(source: BVHData, target: BVHData) -> dict[str, str | None]:
    explicit: dict[str, str | None] = {
        "Root": None,
        "Hips": "Hips",
        "Neck1": "Neck",
        "Neck2": None,
        "Jaw": None,
        "LeftEye": None,
        "RightEye": None,
        "LeftLeg": "LeftUpLeg",
        "LeftShin": "LeftLeg",
        "LeftFoot": "LeftFoot",
        "LeftToeBase": "LeftToe",
        "LeftToeEnd": "LToeEnd",
        "RightLeg": "RightUpLeg",
        "RightShin": "RightLeg",
        "RightFoot": "RightFoot",
        "RightToeBase": "RightToe",
        "RightToeEnd": "RToeEnd",
    }

    for side in ("Left", "Right"):
        prefix = f"{side}Hand"
        fz_prefix = f"FZ{side}"
        for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
            for index in (1, 2, 3):
                explicit[f"{prefix}{finger}{index}"] = f"{fz_prefix}{finger}{index}"
            explicit[f"{prefix}{finger}4"] = None
            explicit[f"{prefix}{finger}End"] = None

    source_names = source.joint_by_name
    mapping: dict[str, str | None] = {}
    for target_joint in target.joints:
        if target_joint.name in explicit:
            mapping[target_joint.name] = explicit[target_joint.name]
        elif target_joint.name in source_names:
            mapping[target_joint.name] = target_joint.name
        else:
            mapping[target_joint.name] = None

    missing_sources = sorted(
        {
            source_name
            for source_name in mapping.values()
            if source_name is not None and source_name not in source_names
        }
    )
    if missing_sources:
        raise ValueError(
            f"{source.path}: missing required source joints: {', '.join(missing_sources)}"
        )

    return mapping


def joint_rotation_channels(joint: Joint) -> tuple[str, ...]:
    return tuple(channel for channel in joint.channels if "rotation" in channel)


def rotation_summary(data: BVHData) -> str:
    orders = sorted(
        {"".join(channel[0] for channel in joint_rotation_channels(joint)) for joint in data.joints}
    )
    return ",".join(orders)


def validate_zyx_rotations(data: BVHData) -> None:
    bad_joints = [
        joint.name
        for joint in data.joints
        if joint_rotation_channels(joint) != ROTATION_CHANNELS
    ]
    if bad_joints:
        raise ValueError(
            f"{data.path}: expected every joint to use "
            f"{' '.join(ROTATION_CHANNELS)}; bad joints: {', '.join(bad_joints)}"
        )


def validate_source_bvh(source: BVHData) -> None:
    validate_zyx_rotations(source)
    if len(source.joints) != 57 or source.channel_count != 174:
        raise ValueError(
            f"{source.path}: expected YM/FZ source layout with 57 joints and "
            f"174 channels, got {len(source.joints)} joints and {source.channel_count} channels"
        )


def validate_source_reference_compatible(source: BVHData, reference: BVHData) -> None:
    source_signature = [(joint.name, joint.channels) for joint in source.joints]
    reference_signature = [(joint.name, joint.channels) for joint in reference.joints]
    if source_signature != reference_signature:
        raise ValueError(
            f"{reference.path}: --source-reference must use the same joint order "
            f"and channels as {source.path}"
        )


def validate_template_bvh(target: BVHData) -> None:
    validate_zyx_rotations(target)
    if len(target.joints) != 78 or target.channel_count != 240:
        raise ValueError(
            f"{target.path}: expected SOMA template layout with 78 joints and "
            f"240 channels, got {len(target.joints)} joints and {target.channel_count} channels"
        )


def validate_reference_bvh(reference: BVHData, target: BVHData) -> None:
    validate_zyx_rotations(reference)
    missing = [joint.name for joint in target.joints if joint.name not in reference.joint_by_name]
    if missing:
        raise ValueError(
            f"{reference.path}: reference pose is missing target joints: {', '.join(missing)}"
        )


def _parse_orientation_offset_block(
    path: Path,
    data: object,
    channels: tuple[str, ...],
    block_name: str,
) -> dict[str, list[list[float]]]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: {block_name} must be an object")

    offsets: dict[str, list[list[float]]] = {}
    for joint_name, values in data.items():
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"{path}: {block_name}.{joint_name} must be a list of 3 degree values")
        offsets[str(joint_name)] = euler_to_matrix(
            [float(value) for value in values],
            channels,
        )
    return offsets


def load_orientation_offsets(
    path: Path | None,
) -> tuple[dict[str, list[list[float]]], dict[str, list[list[float]]], str]:
    if path is None or not path.exists():
        return {}, {}, "local"

    data = json.loads(path.read_text(encoding="utf-8"))
    order = str(data.get("order", "ZYX")).upper()
    if sorted(order) != ["X", "Y", "Z"]:
        raise ValueError(f"{path}: order must contain X, Y, and Z exactly once")

    post_ik_space = str(data.get("post_ik_space", data.get("space", "local"))).lower()
    if post_ik_space not in ("local", "world"):
        raise ValueError(f"{path}: post_ik_space must be either 'local' or 'world'")

    channels = tuple(f"{axis}rotation" for axis in order)
    pre_ik_offsets = _parse_orientation_offset_block(
        path,
        data.get("pre_ik_offsets_degrees", {}),
        channels,
        "pre_ik_offsets_degrees",
    )
    post_ik_offsets = _parse_orientation_offset_block(
        path,
        data.get("post_ik_offsets_degrees", data.get("offsets_degrees", {})),
        channels,
        "post_ik_offsets_degrees",
    )
    return pre_ik_offsets, post_ik_offsets, post_ik_space


def zero_rotation_reference_row(data: BVHData) -> list[float]:
    """Build a synthetic rest-pose row from the BVH hierarchy.

    BVH rotations are authored relative to the hierarchy rest pose. When we do
    not have a dedicated static FZMotion T-pose clip, an all-zero rotation row
    is a safer source reference than using the first frame of an arbitrary
    motion.
    """
    row = [0.0] * data.channel_count
    for joint in data.joints:
        for offset, channel in enumerate(joint.channels):
            if channel == "Xposition":
                row[joint.channel_start + offset] = joint.offset[0]
            elif channel == "Yposition":
                row[joint.channel_start + offset] = joint.offset[1]
            elif channel == "Zposition":
                row[joint.channel_start + offset] = joint.offset[2]
    return row


def first_finger_palm_joint(target_joint_name: str) -> str | None:
    for side in ("Left", "Right"):
        prefix = f"{side}Hand"
        for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
            if target_joint_name == f"{prefix}{finger}1":
                return f"{side}HandPalm"
    return None


def target_offset(
    target_joint: Joint,
    source: BVHData,
    mapping: dict[str, str | None],
    use_source_geometry: bool,
) -> tuple[float, float, float]:
    if not use_source_geometry:
        return target_joint.offset
    if target_joint.name == "Root":
        return (0.0, 0.0, 0.0)
    if target_joint.name in ZERO_LENGTH_INSERTED_JOINTS:
        return (0.0, 0.0, 0.0)

    source_name = mapping.get(target_joint.name)
    if source_name is not None and source_name in source.joint_by_name:
        return source.joint_by_name[source_name].offset
    return target_joint.offset


def build_output_header(
    target: BVHData,
    source: BVHData,
    mapping: dict[str, str | None],
    use_source_geometry: bool,
) -> list[str]:
    children: dict[str | None, list[Joint]] = {}
    for joint in target.joints:
        children.setdefault(joint.parent, []).append(joint)

    roots = children.get(None, [])
    if len(roots) != 1:
        raise ValueError(f"{target.path}: expected exactly one root joint")

    lines = ["HIERARCHY"]

    def emit(joint: Joint, depth: int) -> None:
        indent = "  " * depth
        keyword = "ROOT" if joint.parent is None else "JOINT"
        offset = target_offset(joint, source, mapping, use_source_geometry)
        lines.append(f"{indent}{keyword} {joint.name}")
        lines.append(f"{indent}{{")
        lines.append(
            f"{indent}  OFFSET {format_float(offset[0])} {format_float(offset[1])} {format_float(offset[2])}"
        )
        lines.append(
            f"{indent}  CHANNELS {joint.channel_count} {' '.join(joint.channels)}"
        )
        for child in children.get(joint.name, []):
            emit(child, depth + 1)
        lines.append(f"{indent}}}")

    emit(roots[0], 0)
    return lines


def source_rotation_matrix_for_target(
    row: list[float],
    target_joint_name: str,
    source: BVHData,
    mapping: dict[str, str | None],
    palm_fold: bool,
) -> list[list[float]] | None:
    source_by_name = source.joint_by_name
    source_name = mapping[target_joint_name]

    if source_name is None:
        return None

    source_joint = source_by_name[source_name]
    palm_name = first_finger_palm_joint(target_joint_name) if palm_fold else None

    if palm_name is not None and palm_name in source_by_name:
        palm_joint = source_by_name[palm_name]
        folded_rotation = compose_rotations(
            rotation_values(row, palm_joint),
            palm_joint.channels,
            rotation_values(row, source_joint),
            source_joint.channels,
        )
        return euler_to_matrix(folded_rotation, ROTATION_CHANNELS)

    return euler_to_matrix(rotation_values(row, source_joint), source_joint.channels)


def reference_rotation_matrix(
    reference: BVHData,
    target_joint_name: str,
) -> list[list[float]]:
    reference_joint = reference.joint_by_name[target_joint_name]
    return euler_to_matrix(rotation_values(reference.motion_rows[0], reference_joint), reference_joint.channels)


def corrected_rotation_values(
    row: list[float],
    source_reference_row: list[float],
    target_joint: Joint,
    source: BVHData,
    target_reference: BVHData,
    mapping: dict[str, str | None],
    palm_fold: bool,
) -> list[float]:
    target_reference_matrix = reference_rotation_matrix(target_reference, target_joint.name)
    source_matrix = source_rotation_matrix_for_target(
        row, target_joint.name, source, mapping, palm_fold
    )
    source_reference_matrix = source_rotation_matrix_for_target(
        source_reference_row, target_joint.name, source, mapping, palm_fold
    )

    if source_matrix is None or source_reference_matrix is None:
        corrected_matrix = target_reference_matrix
    else:
        delta_matrix = matmul(mat_transpose(source_reference_matrix), source_matrix)
        corrected_matrix = matmul(target_reference_matrix, delta_matrix)

    return matrix_to_zyx_euler(corrected_matrix)


def fzmotion_to_soma_correction_matrix(target_joint_name: str) -> list[list[float]] | None:
    fzmotion_offset = FZMOTION_IK_OFFSETS_XYZW.get(target_joint_name)
    soma_offset = SOMA_IK_OFFSETS_XYZW.get(target_joint_name)
    if fzmotion_offset is None or soma_offset is None:
        return None
    return matmul(
        quat_to_matrix_xyzw(fzmotion_offset),
        mat_transpose(quat_to_matrix_xyzw(soma_offset)),
    )


def corrected_global_rotation(
    target_joint: Joint,
    source: BVHData,
    source_global_rotations: dict[str, list[list[float]]],
    mapping: dict[str, str | None],
) -> list[list[float]] | None:
    correction = fzmotion_to_soma_correction_matrix(target_joint.name)
    if correction is None:
        return None

    source_name = FZMOTION_CORRECTION_SOURCE.get(target_joint.name, mapping[target_joint.name])
    if source_name is None or source_name not in source.joint_by_name:
        return None
    return matmul(source_global_rotations[source_name], correction)


def mapped_joint_values(
    row: list[float],
    source_reference_row: list[float],
    target_joint: Joint,
    source: BVHData,
    target_reference: BVHData,
    mapping: dict[str, str | None],
    palm_fold: bool,
) -> list[float]:
    source_by_name = source.joint_by_name
    source_name = mapping[target_joint.name]
    source_joint = source_by_name[source_name] if source_name is not None else None
    corrected_rotation = corrected_rotation_values(
        row,
        source_reference_row,
        target_joint,
        source,
        target_reference,
        mapping,
        palm_fold,
    )
    reference_joint = target_reference.joint_by_name[target_joint.name]
    reference_positions = position_values(target_reference.motion_rows[0], reference_joint)

    output: list[float] = []
    rotation_index = 0
    for target_channel in target_joint.channels:
        if "rotation" in target_channel:
            output.append(corrected_rotation[rotation_index])
            rotation_index += 1
            continue

        if source_joint is not None and target_channel in source_joint.channels:
            source_offset = source_joint.channels.index(target_channel)
            output.append(row[source_joint.channel_start + source_offset])
        else:
            output.append(reference_positions.get(target_channel, 0.0))

    return output


def output_position_values_for_target(
    row: list[float],
    target_joint: Joint,
    source: BVHData,
    target_reference: BVHData,
    mapping: dict[str, str | None],
) -> list[float]:
    source_name = mapping[target_joint.name]
    source_joint = source.joint_by_name[source_name] if source_name else None
    reference_positions = position_values(
        target_reference.motion_rows[0],
        target_reference.joint_by_name[target_joint.name],
    )

    values = [target_joint.offset[0], target_joint.offset[1], target_joint.offset[2]]
    for offset, channel in enumerate(target_joint.channels):
        if "position" not in channel:
            continue
        if source_joint is not None and channel in source_joint.channels:
            value = row[source_joint.channel_start + source_joint.channels.index(channel)]
        else:
            value = reference_positions.get(channel, 0.0)

        if channel == "Xposition":
            values[0] = value
        elif channel == "Yposition":
            values[1] = value
        elif channel == "Zposition":
            values[2] = value
    return values


def compute_target_globals_from_local(
    target: BVHData,
    local_positions: dict[str, list[float]],
    local_rotations: dict[str, list[list[float]]],
) -> tuple[dict[str, list[float]], dict[str, list[list[float]]]]:
    positions: dict[str, list[float]] = {}
    rotations: dict[str, list[list[float]]] = {}
    for joint in target.joints:
        local_position = local_positions[joint.name]
        local_rotation = local_rotations[joint.name]
        if joint.parent is None:
            positions[joint.name] = local_position
            rotations[joint.name] = local_rotation
        else:
            parent_rotation = rotations[joint.parent]
            positions[joint.name] = vec_add(
                positions[joint.parent],
                mat_vec_mul(parent_rotation, local_position),
            )
            rotations[joint.name] = matmul(parent_rotation, local_rotation)
    return positions, rotations


def apply_two_bone_ik(
    source_positions: dict[str, list[float]],
    source_rotations: dict[str, list[list[float]]],
    target: BVHData,
    target_chain: tuple[str, str, str],
    source_chain: tuple[str, str, str],
    local_positions: dict[str, list[float]],
    local_rotations: dict[str, list[list[float]]],
) -> None:
    start_name, mid_name, end_name = target_chain
    source_start, source_mid, source_end = source_chain

    target_positions, target_rotations = compute_target_globals_from_local(
        target, local_positions, local_rotations
    )
    start_position = target_positions[start_name]
    target_position = source_positions[source_end]

    mid_offset = target.joint_by_name[mid_name].offset
    end_offset = target.joint_by_name[end_name].offset
    first_length = vec_length(mid_offset)
    second_length = vec_length(end_offset)
    if first_length < 1e-6 or second_length < 1e-6:
        return

    start_to_target = vec_sub(target_position, start_position)
    distance = vec_length(start_to_target)
    if distance < 1e-6:
        return

    clamped_distance = min(distance, first_length + second_length - 1e-5)
    direction = vec_normalize(start_to_target)
    reachable_target = vec_add(start_position, vec_mul(direction, clamped_distance))

    along = (
        first_length * first_length
        + clamped_distance * clamped_distance
        - second_length * second_length
    ) / (2.0 * clamped_distance)
    height = math.sqrt(max(0.0, first_length * first_length - along * along))

    source_hint = vec_sub(source_positions[source_mid], source_positions[source_start])
    bend_hint = vec_sub(source_hint, vec_mul(direction, vec_dot(source_hint, direction)))
    if vec_length(bend_hint) < 1e-6:
        current_mid = target_positions[mid_name]
        bend_hint = vec_sub(
            current_mid,
            vec_add(
                start_position,
                vec_mul(direction, vec_dot(vec_sub(current_mid, start_position), direction)),
            ),
        )
    if vec_length(bend_hint) < 1e-6:
        bend_hint = vec_cross(direction, [0.0, 1.0, 0.0])
    if vec_length(bend_hint) < 1e-6:
        bend_hint = vec_cross(direction, [1.0, 0.0, 0.0])

    desired_mid = vec_add(
        vec_add(start_position, vec_mul(direction, along)),
        vec_mul(vec_normalize(bend_hint), height),
    )

    start_joint = target.joint_by_name[start_name]
    parent_rotation = (
        target_rotations[start_joint.parent]
        if start_joint.parent is not None
        else mat_identity()
    )
    start_align = rotation_between_vectors(
        vec_sub(target_positions[mid_name], start_position),
        vec_sub(desired_mid, start_position),
    )
    desired_start_global = matmul(start_align, target_rotations[start_name])
    local_rotations[start_name] = matmul(mat_transpose(parent_rotation), desired_start_global)

    target_positions, target_rotations = compute_target_globals_from_local(
        target, local_positions, local_rotations
    )
    mid_align = rotation_between_vectors(
        vec_sub(target_positions[end_name], target_positions[mid_name]),
        vec_sub(reachable_target, target_positions[mid_name]),
    )
    desired_mid_global = matmul(mid_align, target_rotations[mid_name])
    local_rotations[mid_name] = matmul(
        mat_transpose(target_rotations[start_name]),
        desired_mid_global,
    )


def soma_geometry_row(
    row: list[float],
    source: BVHData,
    target: BVHData,
    target_reference: BVHData,
    source_reference_row: list[float],
    mapping: dict[str, str | None],
    palm_fold: bool,
    end_effector_ik: bool,
    pre_ik_orientation_offsets: dict[str, list[list[float]]],
    post_ik_orientation_offsets: dict[str, list[list[float]]],
    post_ik_orientation_offset_space: str,
) -> list[float]:
    local_rotations = {
        target_joint.name: euler_to_matrix(
            corrected_rotation_values(
                row,
                source_reference_row,
                target_joint,
                source,
                target_reference,
                mapping,
                palm_fold,
            ),
            ROTATION_CHANNELS,
        )
        for target_joint in target.joints
    }
    for joint_name, orientation_offset in pre_ik_orientation_offsets.items():
        if joint_name in local_rotations:
            local_rotations[joint_name] = matmul(
                local_rotations[joint_name],
                orientation_offset,
            )

    local_positions = {
        target_joint.name: output_position_values_for_target(
            row, target_joint, source, target_reference, mapping
        )
        for target_joint in target.joints
    }

    if end_effector_ik:
        source_positions, source_rotations = compute_global_positions_and_rotations(source, row)
        for target_chain, source_chain in END_EFFECTOR_IK_CHAINS:
            apply_two_bone_ik(
                source_positions,
                source_rotations,
                target,
                target_chain,
                source_chain,
                local_positions,
                local_rotations,
            )

        for target_chain, _ in END_EFFECTOR_IK_CHAINS:
            end_name = target_chain[2]
            if end_name not in post_ik_orientation_offsets:
                continue

            orientation_offset = post_ik_orientation_offsets[end_name]
            if post_ik_orientation_offset_space == "local":
                local_rotations[end_name] = matmul(
                    local_rotations[end_name],
                    orientation_offset,
                )
            else:
                _, target_rotations = compute_target_globals_from_local(
                    target, local_positions, local_rotations
                )
                end_parent = target.joint_by_name[end_name].parent
                parent_rotation = (
                    target_rotations[end_parent]
                    if end_parent is not None
                    else mat_identity()
                )
                local_rotations[end_name] = matmul(
                    mat_transpose(parent_rotation),
                    matmul(orientation_offset, target_rotations[end_name]),
                )

    output_row: list[float] = []
    for target_joint in target.joints:
        local_rotation = matrix_to_zyx_euler(local_rotations[target_joint.name])
        local_position = local_positions[target_joint.name]
        rotation_index = 0
        for target_channel in target_joint.channels:
            if "rotation" in target_channel:
                output_row.append(local_rotation[rotation_index])
                rotation_index += 1
            elif target_channel == "Xposition":
                output_row.append(local_position[0])
            elif target_channel == "Yposition":
                output_row.append(local_position[1])
            elif target_channel == "Zposition":
                output_row.append(local_position[2])
    return output_row


def convert_rows(
    source: BVHData,
    target: BVHData,
    target_reference: BVHData,
    source_reference_row: list[float],
    mapping: dict[str, str | None],
    palm_fold: bool,
    use_source_geometry: bool,
    apply_ik_offset_correction: bool,
    end_effector_ik: bool,
    pre_ik_orientation_offsets: dict[str, list[list[float]]],
    post_ik_orientation_offsets: dict[str, list[list[float]]],
    post_ik_orientation_offset_space: str,
) -> list[list[float]]:
    converted: list[list[float]] = []
    for row in source.motion_rows:
        if not use_source_geometry:
            converted.append(
                soma_geometry_row(
                    row,
                    source,
                    target,
                    target_reference,
                    source_reference_row,
                    mapping,
                    palm_fold,
                    end_effector_ik,
                    pre_ik_orientation_offsets,
                    post_ik_orientation_offsets,
                    post_ik_orientation_offset_space,
                )
            )
            continue

        source_global_rotations = compute_global_rotations(source, row)
        target_global_rotations: dict[str, list[list[float]]] = {}
        output_row: list[float] = []
        for target_joint in target.joints:
            raw_local_matrix = source_rotation_matrix_for_target(
                row, target_joint.name, source, mapping, palm_fold
            )
            if raw_local_matrix is None:
                raw_local_matrix = (
                    mat_identity()
                    if target_joint.name in ZERO_LENGTH_INSERTED_JOINTS
                    else reference_rotation_matrix(target_reference, target_joint.name)
                )

            parent_global = (
                target_global_rotations[target_joint.parent]
                if target_joint.parent is not None
                else mat_identity()
            )

            desired_global_matrix = (
                corrected_global_rotation(target_joint, source, source_global_rotations, mapping)
                if apply_ik_offset_correction
                else None
            )
            if desired_global_matrix is None:
                local_matrix = raw_local_matrix
                global_matrix = matmul(parent_global, local_matrix)
            else:
                global_matrix = desired_global_matrix
                local_matrix = matmul(mat_transpose(parent_global), global_matrix)

            target_global_rotations[target_joint.name] = global_matrix
            local_rotation = matrix_to_zyx_euler(local_matrix)

            target_values: list[float] = []
            rotation_index = 0
            source_name = mapping[target_joint.name]
            source_joint = source.joint_by_name[source_name] if source_name else None
            reference_positions = position_values(
                target_reference.motion_rows[0],
                target_reference.joint_by_name[target_joint.name],
            )

            for target_channel in target_joint.channels:
                if "rotation" in target_channel:
                    target_values.append(local_rotation[rotation_index])
                    rotation_index += 1
                elif source_joint is not None and target_channel in source_joint.channels:
                    source_offset = source_joint.channels.index(target_channel)
                    target_values.append(row[source_joint.channel_start + source_offset])
                else:
                    target_values.append(reference_positions.get(target_channel, 0.0))

            output_row.extend(target_values)
        converted.append(output_row)

    expected_count = target.channel_count
    for frame_index, row in enumerate(converted):
        if len(row) != expected_count:
            raise ValueError(
                f"{source.path}: converted frame {frame_index} has {len(row)} "
                f"values, expected {expected_count}"
            )
    return converted


def format_float(value: float) -> str:
    if abs(value) < 0.0000005:
        value = 0.0
    return f"{value:.6f}"


def write_bvh(path: Path, header_lines: list[str], frame_time: str, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for line in header_lines:
            file.write(f"{line}\n")
        file.write("MOTION\n")
        file.write(f"Frames: {len(rows)}\n")
        file.write(f"Frame Time: {frame_time}\n")
        for row in rows:
            file.write(" ".join(format_float(value) for value in row))
            file.write("\n")


def summarize_conversion(
    source: BVHData,
    target: BVHData,
    mapping: dict[str, str | None],
    output_path: Path,
    dry_run: bool,
    palm_fold: bool,
    use_source_geometry: bool,
    source_reference: BVHData | None,
) -> None:
    used_sources = {name for name in mapping.values() if name is not None}
    if palm_fold:
        for palm_name in ("LeftHandPalm", "RightHandPalm"):
            if palm_name in source.joint_by_name:
                used_sources.add(palm_name)

    zero_targets = [joint.name for joint in target.joints if mapping[joint.name] is None]
    ignored_sources = [joint.name for joint in source.joints if joint.name not in used_sources]
    fps = 1.0 / float(source.frame_time)
    action = "DRY-RUN" if dry_run else "WROTE"

    print(f"[{action}] {source.path}")
    print(
        f"  input:  joints={len(source.joints)} channels={source.channel_count} "
        f"frames={source.frames} fps={fps:.3f} rotation={rotation_summary(source)}"
    )
    print(
        f"  output: joints={len(target.joints)} channels={target.channel_count} "
        f"rotation={rotation_summary(target)} path={output_path}"
    )
    print(
        "  geometry: "
        + (
            "FZMotion source offsets"
            if use_source_geometry
            else "SOMA template offsets + rest-pose delta rotations"
        )
    )
    if not use_source_geometry:
        print(
            "  source reference: "
            + (
                str(source_reference.path)
                if source_reference is not None
                else "synthetic zero-rotation rest pose"
            )
        )
    print(f"  reference-filled target joints ({len(zero_targets)}): {', '.join(zero_targets)}")
    print(f"  ignored source joints ({len(ignored_sources)}): {', '.join(ignored_sources)}")


def output_path_for(source_path: Path, input_root: Path, output: Path, single_file: bool) -> Path:
    if single_file and output.suffix.lower() == ".bvh":
        return output

    try:
        relative = source_path.relative_to(input_root)
    except ValueError:
        relative = Path(source_path.name)
    return output / relative


def collect_sources(input_path: Path, file_path: Path | None) -> tuple[list[Path], Path, bool]:
    if file_path is not None:
        return [file_path], file_path.parent, True

    if not input_path.is_dir():
        raise ValueError(f"--input must be a directory when --file is not used: {input_path}")

    sources = sorted(input_path.glob("*.bvh"))
    if not sources:
        raise ValueError(f"no .bvh files found in {input_path}")
    return sources, input_path, False


def convert_file(
    source_path: Path,
    input_root: Path,
    output_root: Path,
    target_template: BVHData,
    target_reference: BVHData,
    source_reference: BVHData | None,
    overwrite: bool,
    dry_run: bool,
    palm_fold: bool,
    use_source_geometry: bool,
    apply_ik_offset_correction: bool,
    end_effector_ik: bool,
    pre_ik_orientation_offsets: dict[str, list[list[float]]],
    post_ik_orientation_offsets: dict[str, list[list[float]]],
    post_ik_orientation_offset_space: str,
    single_file: bool,
) -> None:
    source = parse_bvh(source_path)
    validate_source_bvh(source)
    if source_reference is not None:
        validate_source_reference_compatible(source, source_reference)
    mapping = build_joint_mapping(source, target_template)
    source_reference_row = (
        source_reference.motion_rows[0]
        if source_reference is not None
        else zero_rotation_reference_row(source)
    )
    out_path = output_path_for(source_path, input_root, output_root, single_file)

    if out_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"{out_path} already exists; pass --overwrite to replace it")

    summarize_conversion(
        source,
        target_template,
        mapping,
        out_path,
        dry_run,
        palm_fold,
        use_source_geometry,
        source_reference,
    )

    if dry_run:
        return

    header_lines = build_output_header(
        target_template, source, mapping, use_source_geometry
    )
    rows = convert_rows(
        source,
        target_template,
        target_reference,
        source_reference_row,
        mapping,
        palm_fold,
        use_source_geometry,
        apply_ik_offset_correction,
        end_effector_ik,
        pre_ik_orientation_offsets,
        post_ik_orientation_offsets,
        post_ik_orientation_offset_space,
    )
    write_bvh(out_path, header_lines, source.frame_time, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert YM/FZ 57-joint BVH files into SOMA 78-joint BVH files."
    )
    parser.add_argument("--input", type=Path, default=Path("mocap"))
    parser.add_argument("--output", type=Path, default=Path("mocap_soma"))
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="SOMA pose used as the rotational baseline for converted motion.",
    )
    parser.add_argument(
        "--source-reference",
        type=Path,
        default=None,
        help=(
            "Optional YM/FZ static rest-pose BVH. Used by --soma-geometry to "
            "compute source rotation deltas. If omitted, an all-zero rotation "
            "row from each source hierarchy is used."
        ),
    )
    parser.add_argument("--file", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--soma-geometry",
        action="store_true",
        help=(
            "Use the SOMA template offsets and rest-pose delta rotations so "
            "the result can drive the SOMA mesh. By default the output keeps "
            "the FZMotion source bone offsets under SOMA-compatible joint names."
        ),
    )
    parser.add_argument(
        "--apply-ik-offset-correction",
        action="store_true",
        help=(
            "Bake fzmotion-vs-SOMA IK orientation offsets into the BVH. This is "
            "usually not desired for visual BVH conversion; keep the offsets in "
            "the retarget/scaler config instead."
        ),
    )
    parser.add_argument(
        "--no-palm-fold",
        action="store_true",
        help="Do not fold LeftHandPalm/RightHandPalm rotations into finger roots.",
    )
    parser.add_argument(
        "--no-end-effector-ik",
        action="store_true",
        help=(
            "Disable the hand/foot two-bone IK pass used by --soma-geometry "
            "to keep SOMA mesh limbs aligned with source end-effectors."
        ),
    )
    parser.add_argument(
        "--orientation-offsets",
        type=Path,
        default=DEFAULT_ORIENTATION_OFFSETS,
        help=(
            "JSON file with editable fixed Z/Y/X degree offsets for body joints "
            "and LeftHand/RightHand/LeftFoot/RightFoot. Used by --soma-geometry."
        ),
    )
    parser.add_argument(
        "--no-orientation-offsets",
        action="store_true",
        help="Ignore any hand/foot fixed orientation offset JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        if args.soma_geometry and args.apply_ik_offset_correction:
            raise ValueError(
                "--apply-ik-offset-correction is only supported with the default "
                "FZMotion-geometry output; omit it when using --soma-geometry"
            )

        target_template = parse_bvh(args.template)
        validate_template_bvh(target_template)
        target_reference = parse_bvh(args.reference)
        validate_reference_bvh(target_reference, target_template)
        source_reference = parse_bvh(args.source_reference) if args.source_reference else None
        if source_reference is not None:
            validate_source_bvh(source_reference)
        (
            pre_ik_orientation_offsets,
            post_ik_orientation_offsets,
            post_ik_orientation_offset_space,
        ) = load_orientation_offsets(
            None if args.no_orientation_offsets else args.orientation_offsets
        )
        sources, input_root, single_file = collect_sources(args.input, args.file)

        for source_path in sources:
            convert_file(
                source_path=source_path,
                input_root=input_root,
                output_root=args.output,
                target_template=target_template,
                target_reference=target_reference,
                source_reference=source_reference,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                palm_fold=not args.no_palm_fold,
                use_source_geometry=not args.soma_geometry,
                apply_ik_offset_correction=args.apply_ik_offset_correction,
                end_effector_ik=not args.no_end_effector_ik,
                pre_ik_orientation_offsets=pre_ik_orientation_offsets,
                post_ik_orientation_offsets=post_ik_orientation_offsets,
                post_ik_orientation_offset_space=post_ik_orientation_offset_space,
                single_file=single_file,
            )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
