#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


Vec3 = Tuple[float, float, float]


class MeshConversionError(RuntimeError):
    pass


def _warn(message: str) -> None:
    print(f"[usd_mesh_to_mjcf_visual] Warning: {message}", file=sys.stderr)


def remap_vec(vec: Vec3, axis_remap: str) -> Vec3:
    if axis_remap == "none":
        return vec
    if axis_remap == "yup_to_zup":
        return (vec[2], vec[0], vec[1])
    raise MeshConversionError(f"Unsupported axis remap policy: {axis_remap}")


def _vec_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _vec_norm(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _normalize(v: Vec3) -> Vec3:
    n = _vec_norm(v)
    if n <= 1e-12:
        return (0.0, 0.0, 0.0)
    inv = 1.0 / n
    return (v[0] * inv, v[1] * inv, v[2] * inv)


def _build_joint_resolver(joint_names: Sequence[str]) -> Tuple[Dict[str, int], Dict[str, List[int]]]:
    exact: Dict[str, int] = {}
    leaf_to_indices: Dict[str, List[int]] = {}
    for idx, raw in enumerate(joint_names):
        normalized = str(raw).lstrip("/")
        exact[normalized] = idx
        leaf = normalized.split("/")[-1]
        leaf_to_indices.setdefault(leaf, []).append(idx)
    return exact, leaf_to_indices


def _resolve_joint_index(
    source_name: str,
    exact: Mapping[str, int],
    leaf_to_indices: Mapping[str, List[int]],
) -> int:
    normalized = source_name.lstrip("/")
    if normalized in exact:
        return exact[normalized]
    candidates = leaf_to_indices.get(normalized)
    if not candidates:
        raise MeshConversionError(f"Required source joint '{source_name}' not found.")
    if len(candidates) > 1:
        raise MeshConversionError(
            f"Source joint '{source_name}' is ambiguous ({len(candidates)} leaf-name matches)."
        )
    return candidates[0]


def _find_skeleton(stage, usdskel_module, usd_skel_prim: Optional[str]):
    if usd_skel_prim:
        prim = stage.GetPrimAtPath(usd_skel_prim)
        if not prim or not prim.IsValid():
            raise MeshConversionError(f"--usd-skel-prim not found: {usd_skel_prim}")
        if not prim.IsA(usdskel_module.Skeleton):
            raise MeshConversionError(f"Prim is not UsdSkel.Skeleton: {usd_skel_prim}")
        return usdskel_module.Skeleton(prim)

    matches = []
    for prim in stage.Traverse():
        if prim.IsA(usdskel_module.Skeleton):
            matches.append(prim)
    if not matches:
        raise MeshConversionError("No UsdSkel.Skeleton found in USD stage.")
    if len(matches) > 1:
        joined = ", ".join(str(p.GetPath()) for p in matches)
        raise MeshConversionError(
            "Multiple skeletons found. Provide --usd-skel-prim to disambiguate: "
            + joined
        )
    return usdskel_module.Skeleton(matches[0])


def _mesh_has_skinning(mesh_prim, usdskel_module) -> bool:
    binding = usdskel_module.BindingAPI(mesh_prim)
    joint_indices = binding.GetJointIndicesPrimvar()
    joint_weights = binding.GetJointWeightsPrimvar()
    if not joint_indices or not joint_weights:
        return False
    ji_attr = joint_indices.GetAttr()
    jw_attr = joint_weights.GetAttr()
    return bool(ji_attr and ji_attr.HasAuthoredValue()) and bool(jw_attr and jw_attr.HasAuthoredValue())


def _find_mesh(stage, usdgeom_module, usdskel_module, skeleton_prim, usd_mesh_prim: Optional[str]):
    if usd_mesh_prim:
        prim = stage.GetPrimAtPath(usd_mesh_prim)
        if not prim or not prim.IsValid():
            raise MeshConversionError(f"--usd-mesh-prim not found: {usd_mesh_prim}")
        if not prim.IsA(usdgeom_module.Mesh):
            raise MeshConversionError(f"Prim is not UsdGeom.Mesh: {usd_mesh_prim}")
        if not _mesh_has_skinning(prim, usdskel_module):
            raise MeshConversionError(f"Mesh has no skinning joint primvars: {usd_mesh_prim}")
        return usdgeom_module.Mesh(prim)

    skeleton_path = skeleton_prim.GetPath()
    matches = []
    for prim in stage.Traverse():
        if not prim.IsA(usdgeom_module.Mesh):
            continue
        if not _mesh_has_skinning(prim, usdskel_module):
            continue
        binding = usdskel_module.BindingAPI(prim)
        targets = list(binding.GetSkeletonRel().GetTargets() or [])
        if targets and skeleton_path not in targets:
            continue
        matches.append(prim)

    if not matches:
        raise MeshConversionError("No skinned UsdGeom.Mesh bound to the skeleton was found.")
    if len(matches) > 1:
        joined = ", ".join(str(p.GetPath()) for p in matches)
        raise MeshConversionError(
            "Multiple skinned meshes found. Provide --usd-mesh-prim to disambiguate: "
            + joined
        )
    return usdgeom_module.Mesh(matches[0])


def _triangulate_faces(face_vertex_counts: Sequence[int], face_vertex_indices: Sequence[int]) -> List[Tuple[int, int, int]]:
    triangles: List[Tuple[int, int, int]] = []
    cursor = 0
    for count in face_vertex_counts:
        c = int(count)
        if c < 3:
            cursor += c
            continue
        base = int(face_vertex_indices[cursor])
        for i in range(1, c - 1):
            b = int(face_vertex_indices[cursor + i])
            d = int(face_vertex_indices[cursor + i + 1])
            triangles.append((base, b, d))
        cursor += c
    return triangles


def _expand_vertex_influences(
    values: Sequence[float],
    *,
    element_size: int,
    vertex_count: int,
    label: str,
) -> List[List[float]]:
    if element_size <= 0:
        raise MeshConversionError(f"Invalid {label} elementSize={element_size}; expected positive.")
    flat = [float(v) for v in values]
    expected = vertex_count * element_size
    if len(flat) != expected:
        raise MeshConversionError(
            f"{label} value count mismatch: got {len(flat)}, expected {expected} "
            f"({vertex_count} vertices x {element_size} influences)."
        )

    rows: List[List[float]] = []
    for i in range(vertex_count):
        start = i * element_size
        rows.append(flat[start : start + element_size])
    return rows


def _load_overrides(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        raise MeshConversionError(f"Mesh overrides JSON does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "source_to_target" in payload:
        mapping = payload["source_to_target"]
    else:
        mapping = payload
    if not isinstance(mapping, dict):
        raise MeshConversionError(
            "Mesh overrides must be an object or have a top-level 'source_to_target' object."
        )

    result: Dict[str, str] = {}
    for source_name, target_body in mapping.items():
        result[str(source_name)] = str(target_body)
    return result


def _build_source_joint_to_target_body(
    skeleton_joint_names: Sequence[str],
    parent_indices: Sequence[int],
    target_order: Sequence[str],
    target_to_source: Mapping[str, Mapping[str, object]],
    mesh_overrides: Mapping[str, str],
) -> Dict[int, str]:
    target_set = set(target_order)
    exact, leaf_to_indices = _build_joint_resolver(skeleton_joint_names)

    anchor: Dict[int, str] = {}
    for target_body in target_order:
        rule = target_to_source.get(target_body)
        if rule is None:
            continue

        if "source" in rule:
            idx = _resolve_joint_index(str(rule["source"]), exact, leaf_to_indices)
            anchor[idx] = target_body

        if "synthetic_mean" in rule:
            values = rule["synthetic_mean"]
            if isinstance(values, list):
                for source_name in values:
                    idx = _resolve_joint_index(str(source_name), exact, leaf_to_indices)
                    anchor[idx] = target_body

    if not anchor:
        raise MeshConversionError("No source->target anchor joints found in mapping config.")

    root_target = target_order[0]
    source_to_target: Dict[int, str] = {}
    for idx in range(len(skeleton_joint_names)):
        cur = idx
        mapped: Optional[str] = None
        while cur >= 0:
            mapped = anchor.get(cur)
            if mapped is not None:
                break
            cur = int(parent_indices[cur])
        source_to_target[idx] = mapped if mapped is not None else root_target

    for source_name, target_body in mesh_overrides.items():
        if target_body not in target_set:
            raise MeshConversionError(
                f"Mesh override target '{target_body}' is not in SMPL24 target list."
            )
        idx = _resolve_joint_index(source_name, exact, leaf_to_indices)
        source_to_target[idx] = target_body

    return source_to_target


def _get_binding_joint_to_skeleton_index(
    binding_joint_names: Sequence[str],
    skeleton_exact: Mapping[str, int],
    skeleton_leaf_to_indices: Mapping[str, List[int]],
) -> Dict[int, int]:
    result: Dict[int, int] = {}
    for binding_idx, name in enumerate(binding_joint_names):
        source_idx = _resolve_joint_index(str(name), skeleton_exact, skeleton_leaf_to_indices)
        result[binding_idx] = source_idx
    return result


def _label_triangles(
    triangles: Sequence[Tuple[int, int, int]],
    vertex_label: Sequence[str],
    vertex_target_weights: Sequence[Mapping[str, float]],
    target_order: Sequence[str],
) -> Dict[str, List[Tuple[int, int, int]]]:
    order_index = {name: i for i, name in enumerate(target_order)}
    out: Dict[str, List[Tuple[int, int, int]]] = {name: [] for name in target_order}

    for tri in triangles:
        a, b, c = tri
        labels = [vertex_label[a], vertex_label[b], vertex_label[c]]

        counts: Dict[str, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        max_count = max(counts.values())
        candidates = [name for name, count in counts.items() if count == max_count]

        if len(candidates) == 1:
            selected = candidates[0]
        else:
            best = None
            best_score = -1.0
            for name in candidates:
                score = (
                    vertex_target_weights[a].get(name, 0.0)
                    + vertex_target_weights[b].get(name, 0.0)
                    + vertex_target_weights[c].get(name, 0.0)
                )
                candidate_key = (score, -order_index.get(name, 10**9))
                if best is None or candidate_key > best:
                    best = candidate_key
                    best_score = score
                    selected = name
            if best_score < 0.0:
                selected = candidates[0]

        out[selected].append(tri)

    return out


def _write_ascii_stl(path: Path, solid_name: str, vertices: Sequence[Vec3], triangles: Sequence[Tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"solid {solid_name}\n")
        for i0, i1, i2 in triangles:
            v0 = vertices[i0]
            v1 = vertices[i1]
            v2 = vertices[i2]
            e1 = _vec_sub(v1, v0)
            e2 = _vec_sub(v2, v0)
            n = _normalize(_vec_cross(e1, e2))
            f.write(f"  facet normal {n[0]:.9g} {n[1]:.9g} {n[2]:.9g}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]:.9g} {v0[1]:.9g} {v0[2]:.9g}\n")
            f.write(f"      vertex {v1[0]:.9g} {v1[1]:.9g} {v1[2]:.9g}\n")
            f.write(f"      vertex {v2[0]:.9g} {v2[1]:.9g} {v2[2]:.9g}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {solid_name}\n")


def export_partitioned_visual_meshes(
    *,
    usd_path: Path,
    output_xml_path: Path,
    target_order: Sequence[str],
    target_parents: Mapping[str, Optional[str]],  # kept for parity and future checks
    target_to_source: Mapping[str, Mapping[str, object]],
    target_globals: Mapping[str, Vec3],
    axis_remap: str,
    mesh_min_triangles: int,
    usd_skel_prim: Optional[str],
    usd_mesh_prim: Optional[str],
    mesh_overrides_path: Optional[Path],
) -> Mapping[str, object]:
    del target_parents
    try:
        from pxr import Usd, UsdGeom, UsdSkel  # type: ignore
    except ModuleNotFoundError as exc:
        raise MeshConversionError(
            "USD mesh conversion requires pxr (USD Python bindings)."
        ) from exc

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise MeshConversionError(f"Failed to open USD stage: {usd_path}")

    skeleton = _find_skeleton(stage, UsdSkel, usd_skel_prim)
    skeleton_prim = skeleton.GetPrim()
    mesh = _find_mesh(stage, UsdGeom, UsdSkel, skeleton_prim, usd_mesh_prim)
    mesh_prim = mesh.GetPrim()

    skeleton_joint_tokens = list(skeleton.GetJointsAttr().Get() or [])
    if not skeleton_joint_tokens:
        raise MeshConversionError("Skeleton has no joints.")
    skeleton_joint_names = [str(v) for v in skeleton_joint_tokens]
    topology = UsdSkel.Topology(skeleton_joint_tokens)
    parent_indices = [int(v) for v in topology.GetParentIndices()]
    if len(parent_indices) != len(skeleton_joint_names):
        raise MeshConversionError("Skeleton parent index count does not match skeleton joint count.")

    mesh_overrides = _load_overrides(mesh_overrides_path)
    source_to_target = _build_source_joint_to_target_body(
        skeleton_joint_names,
        parent_indices,
        target_order,
        target_to_source,
        mesh_overrides,
    )

    binding = UsdSkel.BindingAPI(mesh_prim)
    binding_joint_names = [str(v) for v in (binding.GetJointsAttr().Get() or [])]
    if not binding_joint_names:
        raise MeshConversionError("Skinned mesh has no binding joints list.")

    skeleton_exact, skeleton_leaf_to_indices = _build_joint_resolver(skeleton_joint_names)
    binding_to_skeleton = _get_binding_joint_to_skeleton_index(
        binding_joint_names,
        skeleton_exact,
        skeleton_leaf_to_indices,
    )

    points_attr = mesh.GetPointsAttr().Get()
    if not points_attr:
        raise MeshConversionError("Mesh has no points.")
    points = [(float(p[0]), float(p[1]), float(p[2])) for p in points_attr]
    vertex_count = len(points)

    xform_cache = UsdGeom.XformCache()
    mesh_world = xform_cache.GetLocalToWorldTransform(mesh_prim)
    world_points: List[Vec3] = []
    for p in points:
        transformed = mesh_world.Transform((p[0], p[1], p[2]))
        world_points.append((float(transformed[0]), float(transformed[1]), float(transformed[2])))

    counts = mesh.GetFaceVertexCountsAttr().Get() or []
    indices = mesh.GetFaceVertexIndicesAttr().Get() or []
    if not counts or not indices:
        raise MeshConversionError("Mesh has no face topology.")
    triangles = _triangulate_faces(counts, indices)
    if not triangles:
        raise MeshConversionError("Mesh triangulation produced zero triangles.")

    joint_indices_pv = binding.GetJointIndicesPrimvar()
    joint_weights_pv = binding.GetJointWeightsPrimvar()
    if not joint_indices_pv or not joint_weights_pv:
        raise MeshConversionError("Skinned mesh is missing joint indices/weights primvars.")

    if joint_indices_pv.GetInterpolation() != "vertex" or joint_weights_pv.GetInterpolation() != "vertex":
        raise MeshConversionError(
            "Only vertex-interpolated joint indices/weights are supported for mesh partitioning."
        )

    joint_indices_rows = _expand_vertex_influences(
        joint_indices_pv.Get() or [],
        element_size=joint_indices_pv.GetElementSize(),
        vertex_count=vertex_count,
        label="jointIndices",
    )
    joint_weights_rows = _expand_vertex_influences(
        joint_weights_pv.Get() or [],
        element_size=joint_weights_pv.GetElementSize(),
        vertex_count=vertex_count,
        label="jointWeights",
    )

    if len(joint_indices_rows) != len(joint_weights_rows):
        raise MeshConversionError("jointIndices/jointWeights row count mismatch.")

    root_target = target_order[0]
    vertex_label: List[str] = []
    vertex_target_weights: List[Dict[str, float]] = []

    for vid in range(vertex_count):
        target_weights: Dict[str, float] = {}
        joint_indices = joint_indices_rows[vid]
        joint_weights = joint_weights_rows[vid]
        for ji, jw in zip(joint_indices, joint_weights):
            weight = float(jw)
            if weight <= 0.0:
                continue
            binding_joint_idx = int(ji)
            skeleton_joint_idx = binding_to_skeleton.get(binding_joint_idx)
            if skeleton_joint_idx is None:
                continue
            target = source_to_target[skeleton_joint_idx]
            target_weights[target] = target_weights.get(target, 0.0) + weight

        if not target_weights:
            vertex_label.append(root_target)
            vertex_target_weights.append({root_target: 1.0})
            continue

        selected = max(target_weights.items(), key=lambda item: item[1])[0]
        vertex_label.append(selected)
        vertex_target_weights.append(target_weights)

    tri_by_body = _label_triangles(triangles, vertex_label, vertex_target_weights, target_order)

    output_stem = output_xml_path.stem
    mesh_root = output_xml_path.parent / f"{output_stem}_usdmesh"
    geom_dir = mesh_root / "geom"
    geom_dir.mkdir(parents=True, exist_ok=True)

    body_mesh_files: Dict[str, str] = {}
    body_stats: Dict[str, Dict[str, int]] = {}

    for body in target_order:
        body_tris = tri_by_body.get(body, [])
        if len(body_tris) < mesh_min_triangles:
            if body_tris:
                _warn(
                    f"Dropping tiny mesh fragment for {body}: {len(body_tris)} triangles "
                    f"(< {mesh_min_triangles})."
                )
            continue

        if body not in target_globals:
            _warn(f"No target global position found for {body}; skipping mesh export for this body.")
            continue

        joint_world = target_globals[body]
        local_vertices: List[Vec3] = []
        remap_index: Dict[int, int] = {}
        local_triangles: List[Tuple[int, int, int]] = []

        for a, b, c in body_tris:
            tri_local_idx: List[int] = []
            for src_vi in (a, b, c):
                dst_vi = remap_index.get(src_vi)
                if dst_vi is None:
                    world_v = world_points[src_vi]
                    local_v = remap_vec(_vec_sub(world_v, joint_world), axis_remap)
                    dst_vi = len(local_vertices)
                    remap_index[src_vi] = dst_vi
                    local_vertices.append(local_v)
                tri_local_idx.append(dst_vi)
            local_triangles.append((tri_local_idx[0], tri_local_idx[1], tri_local_idx[2]))

        stl_name = f"{body}.stl"
        stl_path = geom_dir / stl_name
        _write_ascii_stl(stl_path, f"{body}_usd_mesh", local_vertices, local_triangles)

        rel_path = f"{output_stem}_usdmesh/geom/{stl_name}"
        body_mesh_files[body] = rel_path
        body_stats[body] = {
            "vertices": len(local_vertices),
            "triangles": len(local_triangles),
        }

    if not body_mesh_files:
        raise MeshConversionError(
            "No per-body mesh pieces were exported. Try lowering --mesh-min-triangles."
        )

    manifest = {
        "usd": str(usd_path),
        "skeleton_prim": str(skeleton_prim.GetPath()),
        "mesh_prim": str(mesh_prim.GetPath()),
        "mesh_min_triangles": int(mesh_min_triangles),
        "axis_remap": axis_remap,
        "bodies": body_stats,
    }
    manifest_path = mesh_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "body_mesh_files": body_mesh_files,
        "manifest_path": manifest_path,
        "mesh_root": mesh_root,
    }


def _find_body_root(xml_root: ET.Element) -> ET.Element:
    worldbody = xml_root.find("worldbody")
    if worldbody is None:
        raise MeshConversionError("MJCF is missing <worldbody>.")
    body_root = worldbody.find("body")
    if body_root is None:
        raise MeshConversionError("MJCF <worldbody> has no root <body>.")
    return body_root


def _collect_body_map(body_root: ET.Element) -> Dict[str, ET.Element]:
    out: Dict[str, ET.Element] = {}

    def walk(node: ET.Element) -> None:
        name = node.get("name")
        if not name:
            raise MeshConversionError("Found <body> without a name in MJCF.")
        out[name] = node
        for child in node.findall("body"):
            walk(child)

    walk(body_root)
    return out


def inject_visual_meshes_into_mjcf(
    *,
    xml_path: Path,
    body_mesh_files: Mapping[str, str],
    hide_template_geoms: bool,
) -> None:
    tree = ET.parse(xml_path)
    xml_root = tree.getroot()

    body_root = _find_body_root(xml_root)
    body_map = _collect_body_map(body_root)

    asset = xml_root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        xml_root.insert(0, asset)

    mesh_names = {body: f"{body}_usd_mesh" for body in body_mesh_files}
    mesh_geom_names = {body: f"{body}_usd_mesh_geom" for body in body_mesh_files}

    for mesh_elem in list(asset.findall("mesh")):
        name = mesh_elem.get("name", "")
        if name in mesh_names.values():
            asset.remove(mesh_elem)

    for body_name, body_elem in body_map.items():
        expected_geom_name = mesh_geom_names.get(body_name)
        if expected_geom_name is None:
            continue
        for geom in list(body_elem.findall("geom")):
            if geom.get("name") == expected_geom_name:
                body_elem.remove(geom)

    if hide_template_geoms:
        for body_name, body_elem in body_map.items():
            # Preserve template geoms for bodies with no exported mesh fragment.
            if body_name not in body_mesh_files:
                continue
            for geom in body_elem.findall("geom"):
                geom.set("rgba", "0 0 0 0")

    for body, rel_file in body_mesh_files.items():
        mesh_name = mesh_names[body]
        ET.SubElement(asset, "mesh", {"name": mesh_name, "file": rel_file})

    for body, rel_file in body_mesh_files.items():
        del rel_file
        body_elem = body_map.get(body)
        if body_elem is None:
            _warn(f"MJCF body '{body}' not found; skipping visual mesh attachment.")
            continue
        ET.SubElement(
            body_elem,
            "geom",
            {
                "name": mesh_geom_names[body],
                "type": "mesh",
                "mesh": mesh_names[body],
                "contype": "0",
                "conaffinity": "0",
                "density": "0",
                "group": "1",
                "rgba": "0.8 0.8 0.8 1",
            },
        )

    tree.write(xml_path, encoding="utf-8")
