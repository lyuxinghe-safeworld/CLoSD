#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from usd_mesh_to_mjcf_visual import (
    MeshConversionError,
    export_partitioned_visual_meshes,
    inject_visual_meshes_into_mjcf,
)


Vec3 = Tuple[float, float, float]
USD_SUFFIXES = {".usd", ".usda", ".usdc"}
XML_SUFFIXES = {".xml"}
EXPECTED_HINGE_COUNT = 69
EXPECTED_MOTOR_COUNT = 69

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_PHYSICS_SOURCE = REPO_ROOT / "closd/data/robot_cache/smpl_humanoid_0.xml"
DEFAULT_MAPPING_PATH = REPO_ROOT / "closd/data/robot_cache/usd_smpl24_map.json"
DEFAULT_MESH_OVERRIDES_PATH = REPO_ROOT / "closd/data/robot_cache/usd_smpl24_mesh_overrides.json"


class ConversionError(RuntimeError):
    pass


def resolve_path(path_str: str, *, must_exist: bool) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise ConversionError(f"Path does not exist: {path}")
    return path


def format_vec3(vec: Vec3) -> str:
    return f"{vec[0]:.6f} {vec[1]:.6f} {vec[2]:.6f}"


def vec_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_mean(values: Sequence[Vec3]) -> Vec3:
    if not values:
        raise ConversionError("Cannot compute mean of an empty vector list.")
    inv = 1.0 / len(values)
    return (
        sum(v[0] for v in values) * inv,
        sum(v[1] for v in values) * inv,
        sum(v[2] for v in values) * inv,
    )


def vec_norm(vec: Vec3) -> float:
    return math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])


def clamp_small_offset(vec: Vec3, eps: float) -> Vec3:
    norm = vec_norm(vec)
    if norm >= eps:
        return vec
    if norm > 0.0:
        scale = eps / norm
        return (vec[0] * scale, vec[1] * scale, vec[2] * scale)
    return (eps, 0.0, 0.0)


def load_mapping(mapping_path: Path, mode: str) -> Mapping[str, object]:
    with mapping_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    if mapping.get("mode") != mode:
        raise ConversionError(
            f"Mapping mode mismatch. Expected '{mode}', got '{mapping.get('mode')}'."
        )
    required_top = ("target_order", "target_parents", "target_to_source")
    for key in required_top:
        if key not in mapping:
            raise ConversionError(f"Mapping is missing required key '{key}'.")
    return mapping


def _find_skeleton(stage, usdskel_module, usd_skel_prim: Optional[str]):
    if usd_skel_prim:
        prim = stage.GetPrimAtPath(usd_skel_prim)
        if not prim or not prim.IsValid():
            raise ConversionError(f"--usd-skel-prim not found: {usd_skel_prim}")
        if not prim.IsA(usdskel_module.Skeleton):
            raise ConversionError(f"Prim is not UsdSkel.Skeleton: {usd_skel_prim}")
        return usdskel_module.Skeleton(prim)

    matches = []
    for prim in stage.Traverse():
        if prim.IsA(usdskel_module.Skeleton):
            matches.append(prim)
    if not matches:
        return None
    if len(matches) > 1:
        joined = ", ".join(str(p.GetPath()) for p in matches)
        raise ConversionError(
            "Multiple UsdSkel.Skeleton prims found. Provide --usd-skel-prim to disambiguate: "
            + joined
        )
    return usdskel_module.Skeleton(matches[0])


def _build_joint_resolver(joint_names: Sequence[str]):
    exact: Dict[str, int] = {}
    leaf_to_indices: Dict[str, List[int]] = {}
    for idx, raw in enumerate(joint_names):
        name = str(raw).lstrip("/")
        exact[name] = idx
        leaf = name.split("/")[-1]
        leaf_to_indices.setdefault(leaf, []).append(idx)
    return exact, leaf_to_indices


def _resolve_source_joint_index(
    source_name: str,
    exact: Mapping[str, int],
    leaf_to_indices: Mapping[str, List[int]],
) -> int:
    normalized = source_name.lstrip("/")
    if normalized in exact:
        return exact[normalized]
    candidates = leaf_to_indices.get(normalized)
    if not candidates:
        raise ConversionError(f"Required source joint '{source_name}' not found in USD skeleton.")
    if len(candidates) > 1:
        raise ConversionError(
            f"Source joint '{source_name}' is ambiguous ({len(candidates)} matches by leaf name)."
        )
    return candidates[0]


def extract_usd_global_positions(
    usd_path: Path, usd_skel_prim: Optional[str]
) -> Tuple[List[str], List[int], List[Vec3]]:
    try:
        from pxr import Usd, UsdSkel  # type: ignore
    except ModuleNotFoundError as exc:
        raise ConversionError(
            "USD conversion requires pxr (USD Python bindings). "
            "Please use an environment with pxr installed."
        ) from exc

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ConversionError(f"Failed to open USD stage: {usd_path}")

    skeleton = _find_skeleton(stage, UsdSkel, usd_skel_prim)
    if skeleton is None:
        raise ConversionError(f"No UsdSkel.Skeleton found in stage: {usd_path}")

    joints_attr = skeleton.GetJointsAttr().Get()
    if not joints_attr:
        raise ConversionError("Skeleton has no joints.")
    joint_names = [str(j) for j in joints_attr]

    topology = UsdSkel.Topology(joints_attr)
    parent_indices = [int(i) for i in topology.GetParentIndices()]
    if len(parent_indices) != len(joint_names):
        raise ConversionError("USD topology parent index count does not match joint count.")

    rest_transforms = skeleton.GetRestTransformsAttr().Get()
    if not rest_transforms:
        raise ConversionError("Skeleton has no restTransforms; cannot build static rest skeleton.")
    if len(rest_transforms) != len(joint_names):
        raise ConversionError("restTransforms count does not match joint count.")

    global_mats: List[object] = [None] * len(joint_names)  # type: ignore[assignment]

    def compute_global(idx: int):
        cached = global_mats[idx]
        if cached is not None:
            return cached
        local = rest_transforms[idx]
        parent_idx = parent_indices[idx]
        if parent_idx < 0:
            global_mats[idx] = local
        else:
            # Some USD assets store restTransforms in a space that requires
            # local-first multiplication for correct world rest positions.
            global_mats[idx] = local * compute_global(parent_idx)
        return global_mats[idx]

    global_positions: List[Vec3] = []
    for i in range(len(joint_names)):
        mat = compute_global(i)
        t = mat.ExtractTranslation()
        global_positions.append((float(t[0]), float(t[1]), float(t[2])))

    return joint_names, parent_indices, global_positions


def remap_offset(vec: Vec3, axis_remap: str) -> Vec3:
    if axis_remap == "none":
        return vec
    if axis_remap == "yup_to_zup":
        return (vec[2], vec[0], vec[1])
    raise ConversionError(f"Unsupported axis remap policy: {axis_remap}")


def run_offset_sanity_checks(
    offsets: Mapping[str, Vec3],
    target_parents: Mapping[str, Optional[str]],
    *,
    root_name: str,
) -> None:
    pelvis_children = [name for name, parent in target_parents.items() if parent == root_name]
    for child in pelvis_children:
        pos = offsets.get(child)
        if pos is None:
            continue
        if vec_norm(pos) > 0.5:
            raise ConversionError(
                f"Sanity check failed: root child '{child}' has offset norm {vec_norm(pos):.4f}m (> 0.5m)."
            )

    for name, parent in target_parents.items():
        if parent is None:
            continue
        pos = offsets.get(name)
        if pos is None:
            continue
        if vec_norm(pos) > 1.0:
            raise ConversionError(
                f"Sanity check failed: body '{name}' has offset norm {vec_norm(pos):.4f}m (> 1.0m)."
            )


def build_target_globals_from_usd(
    usd_path: Path,
    mapping: Mapping[str, object],
    usd_skel_prim: Optional[str],
) -> Dict[str, Vec3]:
    target_order = list(mapping["target_order"])  # type: ignore[index]
    target_to_source: Mapping[str, Mapping[str, object]] = mapping["target_to_source"]  # type: ignore[assignment]

    joint_names, _, global_positions = extract_usd_global_positions(usd_path, usd_skel_prim)
    exact, leaf_to_indices = _build_joint_resolver(joint_names)

    target_globals: Dict[str, Vec3] = {}
    for target in target_order:
        rule = target_to_source.get(target)
        if rule is None:
            raise ConversionError(f"Mapping is missing target rule for '{target}'.")
        if "source" in rule:
            source = str(rule["source"])
            idx = _resolve_source_joint_index(source, exact, leaf_to_indices)
            target_globals[target] = global_positions[idx]
            continue
        if "synthetic_mean" in rule:
            source_list = rule["synthetic_mean"]
            if not isinstance(source_list, list) or not source_list:
                raise ConversionError(
                    f"Invalid synthetic_mean list for target '{target}'."
                )
            points: List[Vec3] = []
            for source in source_list:
                idx = _resolve_source_joint_index(str(source), exact, leaf_to_indices)
                points.append(global_positions[idx])
            target_globals[target] = vec_mean(points)
            continue
        raise ConversionError(
            f"Mapping rule for target '{target}' must define 'source' or 'synthetic_mean'."
        )
    return target_globals


def synthesize_offsets_from_target_globals(
    mapping: Mapping[str, object],
    target_globals: Mapping[str, Vec3],
    eps: float,
    axis_remap: str,
    root_policy: str,
) -> Dict[str, Vec3]:
    target_order = list(mapping["target_order"])  # type: ignore[index]
    target_parents: Mapping[str, Optional[str]] = mapping["target_parents"]  # type: ignore[assignment]

    offsets: Dict[str, Vec3] = {}
    for target in target_order:
        parent = target_parents.get(target)
        if parent is None:
            if root_policy == "usd":
                offsets[target] = target_globals[target]
            elif root_policy == "template":
                continue
            else:
                raise ConversionError(f"Unsupported root policy: {root_policy}")
        else:
            local = vec_sub(target_globals[target], target_globals[parent])
            remapped = remap_offset(local, axis_remap)
            offsets[target] = clamp_small_offset(remapped, eps)
    run_offset_sanity_checks(offsets, target_parents, root_name=target_order[0])
    return offsets


def _find_body_root(xml_root: ET.Element) -> ET.Element:
    worldbody = xml_root.find("worldbody")
    if worldbody is None:
        raise ConversionError("MJCF is missing <worldbody>.")
    body_root = worldbody.find("body")
    if body_root is None:
        raise ConversionError("MJCF <worldbody> has no root <body>.")
    return body_root


def _collect_body_order(body_root: ET.Element) -> List[str]:
    order: List[str] = []

    def walk(body: ET.Element) -> None:
        name = body.get("name")
        if not name:
            raise ConversionError("Found <body> without a name attribute.")
        order.append(name)
        for child in body.findall("body"):
            walk(child)

    walk(body_root)
    return order


def _collect_body_map(body_root: ET.Element) -> Dict[str, ET.Element]:
    body_map: Dict[str, ET.Element] = {}

    def walk(body: ET.Element) -> None:
        name = body.get("name")
        if not name:
            raise ConversionError("Found <body> without a name attribute.")
        body_map[name] = body
        for child in body.findall("body"):
            walk(child)

    walk(body_root)
    return body_map


def write_converted_mjcf(
    physics_source: Path,
    output_path: Path,
    offsets: Mapping[str, Vec3],
) -> None:
    tree = ET.parse(physics_source)
    xml_root = tree.getroot()
    body_root = _find_body_root(xml_root)
    body_map = _collect_body_map(body_root)

    for body_name, pos in offsets.items():
        body = body_map.get(body_name)
        if body is None:
            raise ConversionError(
                f"Physics template is missing expected body '{body_name}'."
            )
        body.set("pos", format_vec3(pos))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8")


def validate_mjcf(xml_path: Path, expected_body_order: Sequence[str]) -> None:
    tree = ET.parse(xml_path)
    xml_root = tree.getroot()
    body_root = _find_body_root(xml_root)
    body_order = _collect_body_order(body_root)
    if body_order != list(expected_body_order):
        raise ConversionError(
            "Body order mismatch.\n"
            f"Expected: {list(expected_body_order)}\n"
            f"Actual:   {body_order}"
        )

    worldbody = xml_root.find("worldbody")
    assert worldbody is not None
    hinge_count = sum(
        1
        for joint in worldbody.findall(".//joint")
        if joint.get("type", "hinge") == "hinge"
    )
    if hinge_count != EXPECTED_HINGE_COUNT:
        raise ConversionError(
            f"Expected {EXPECTED_HINGE_COUNT} hinge joints, found {hinge_count}."
        )

    actuator = xml_root.find("actuator")
    motor_count = 0 if actuator is None else len(actuator.findall("motor"))
    if motor_count != EXPECTED_MOTOR_COUNT:
        raise ConversionError(
            f"Expected {EXPECTED_MOTOR_COUNT} motors, found {motor_count}."
        )

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from closd.utils.poselib.poselib.skeleton.skeleton3d import SkeletonTree
    except Exception as exc:
        raise ConversionError(
            "Validation requires CLoSD runtime deps (including numpy/torch). "
            "Please run in the project conda environment."
        ) from exc

    try:
        SkeletonTree.from_mjcf(str(xml_path))
    except Exception as exc:
        raise ConversionError(f"SkeletonTree.from_mjcf failed for {xml_path}: {exc}") from exc


def run(args: argparse.Namespace) -> int:
    input_path = resolve_path(args.input, must_exist=True)
    output_path = resolve_path(args.output, must_exist=False) if args.output else None
    mapping_path = resolve_path(args.mapping, must_exist=True)
    physics_source = resolve_path(args.physics_source, must_exist=True)

    mapping = load_mapping(mapping_path, args.mode)
    target_order = list(mapping["target_order"])  # type: ignore[index]

    suffix = input_path.suffix.lower()
    if suffix in XML_SUFFIXES:
        xml_target = input_path
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path != input_path:
                shutil.copy2(input_path, output_path)
            xml_target = output_path
        if args.validate:
            validate_mjcf(xml_target, target_order)
        print(f"XML passthrough complete: {xml_target}")
        return 0

    if suffix in USD_SUFFIXES:
        if output_path is None:
            raise ConversionError("--output is required for USD input.")
        target_globals = build_target_globals_from_usd(
            input_path,
            mapping,
            args.usd_skel_prim,
        )
        offsets = synthesize_offsets_from_target_globals(
            mapping,
            target_globals,
            args.eps,
            args.axis_remap,
            args.root_policy,
        )
        write_converted_mjcf(physics_source, output_path, offsets)

        if args.mesh_mode == "visual":
            hide_template_geoms = (
                args.hide_template_geoms
                if args.hide_template_geoms is not None
                else True
            )
            if args.mesh_overrides:
                mesh_overrides_path = resolve_path(args.mesh_overrides, must_exist=True)
            elif DEFAULT_MESH_OVERRIDES_PATH.exists():
                mesh_overrides_path = DEFAULT_MESH_OVERRIDES_PATH
            else:
                mesh_overrides_path = None

            target_parents: Mapping[str, Optional[str]] = mapping["target_parents"]  # type: ignore[assignment]
            target_to_source: Mapping[str, Mapping[str, object]] = mapping["target_to_source"]  # type: ignore[assignment]
            try:
                mesh_export = export_partitioned_visual_meshes(
                    usd_path=input_path,
                    output_xml_path=output_path,
                    target_order=target_order,
                    target_parents=target_parents,
                    target_to_source=target_to_source,
                    target_globals=target_globals,
                    axis_remap=args.axis_remap,
                    mesh_min_triangles=args.mesh_min_triangles,
                    usd_skel_prim=args.usd_skel_prim,
                    usd_mesh_prim=args.usd_mesh_prim,
                    mesh_overrides_path=mesh_overrides_path,
                )
                inject_visual_meshes_into_mjcf(
                    xml_path=output_path,
                    body_mesh_files=mesh_export["body_mesh_files"],  # type: ignore[index]
                    hide_template_geoms=hide_template_geoms,
                )
                print(f"Injected USD visual mesh assets: {mesh_export['manifest_path']}")  # type: ignore[index]
            except MeshConversionError as exc:
                raise ConversionError(f"Mesh visual conversion failed: {exc}") from exc

        if args.validate:
            validate_mjcf(output_path, target_order)
        print(f"Converted USD -> MJCF: {output_path}")
        return 0

    raise ConversionError(
        f"Unsupported input extension '{input_path.suffix}'. "
        "Expected .xml or one of .usd/.usda/.usdc."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert USD (UsdSkel) to SMPL-24-compatible CLoSD MJCF by replacing "
            "body offsets in a stable physics template."
        )
    )
    parser.add_argument("--input", required=True, help="Input .usd/.usda/.usdc or .xml path.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output .xml path. Required for USD input; optional for XML passthrough.",
    )
    parser.add_argument(
        "--mode",
        default="smpl24",
        choices=["smpl24"],
        help="Target articulation mode.",
    )
    parser.add_argument(
        "--physics-source",
        default=str(DEFAULT_PHYSICS_SOURCE),
        help="Template MJCF to copy physics defaults from.",
    )
    parser.add_argument(
        "--mapping",
        default=str(DEFAULT_MAPPING_PATH),
        help="JSON mapping config for USD -> target joints.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-4,
        help="Minimum local offset norm for non-root joints.",
    )
    parser.add_argument(
        "--axis-remap",
        default="yup_to_zup",
        choices=["yup_to_zup", "none"],
        help="Axis remapping policy applied to non-root local offsets.",
    )
    parser.add_argument(
        "--root-policy",
        default="template",
        choices=["template", "usd"],
        help="Whether to preserve template root position or overwrite from USD.",
    )
    parser.add_argument(
        "--mesh-mode",
        default="visual",
        choices=["visual", "none"],
        help="USD mesh handling mode. 'visual' exports per-body STL meshes and injects visual geoms.",
    )
    parser.add_argument(
        "--hide-template-geoms",
        dest="hide_template_geoms",
        action="store_true",
        help="Hide template primitive body geoms (rgba=0) when injecting USD visual meshes.",
    )
    parser.add_argument(
        "--show-template-geoms",
        dest="hide_template_geoms",
        action="store_false",
        help="Keep template primitive body geoms visible with USD visual meshes.",
    )
    parser.set_defaults(hide_template_geoms=None)
    parser.add_argument(
        "--mesh-min-triangles",
        type=int,
        default=12,
        help="Drop tiny mesh body fragments below this triangle count.",
    )
    parser.add_argument(
        "--usd-skel-prim",
        default=None,
        help="Optional UsdSkel.Skeleton prim path to disambiguate skeleton selection.",
    )
    parser.add_argument(
        "--usd-mesh-prim",
        default=None,
        help="Optional UsdGeom.Mesh prim path to disambiguate skinned mesh selection.",
    )
    parser.add_argument(
        "--mesh-overrides",
        default=None,
        help=(
            "Optional JSON with source-joint -> SMPL24 body overrides. "
            "If omitted and closd/data/robot_cache/usd_smpl24_mesh_overrides.json exists, it is used."
        ),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run post-conversion validation checks (XML parse, bodies, counts, SkeletonTree).",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except ConversionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
