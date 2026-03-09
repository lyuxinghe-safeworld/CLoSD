#!/usr/bin/env python3
from isaacgym import gymapi

import argparse
import sys
from pathlib import Path


DEFAULT_XML_REL = "closd/data/robot_cache/smpl_humanoid_0.xml"


def parse_vec3(text: str, name: str) -> gymapi.Vec3:
    tokens = text.replace(",", " ").split()
    if len(tokens) != 3:
        raise ValueError(f"{name} must have exactly 3 numbers, got: {text!r}")
    try:
        x, y, z = (float(v) for v in tokens)
    except ValueError as exc:
        raise ValueError(f"{name} must contain valid floats, got: {text!r}") from exc
    return gymapi.Vec3(x, y, z)


def resolve_xml_path(raw_path: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"XML file not found: {path}")
    return path


def build_sim_params(args: argparse.Namespace) -> gymapi.SimParams:
    sim_params = gymapi.SimParams()
    sim_params.dt = args.dt
    sim_params.substeps = 2

    if args.up_axis == "z":
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    else:
        sim_params.up_axis = gymapi.UP_AXIS_Y
        sim_params.gravity = gymapi.Vec3(0.0, -9.81, 0.0)

    # Match project defaults used for PhysX.
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = 4
    sim_params.physx.num_subscenes = 0
    sim_params.physx.max_gpu_contact_pairs = 16 * 1024 * 1024
    sim_params.physx.use_gpu = args.compute_device_id >= 0
    sim_params.use_gpu_pipeline = args.compute_device_id >= 0
    return sim_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View a single MJCF XML in Isaac Gym."
    )
    parser.add_argument(
        "--xml",
        type=str,
        default=DEFAULT_XML_REL,
        help=(
            "Path to MJCF XML. Relative paths are resolved from repo root "
            f"(default: {DEFAULT_XML_REL})."
        ),
    )
    parser.add_argument("--compute-device-id", type=int, default=0)
    parser.add_argument("--graphics-device-id", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.0166667)
    parser.add_argument("--up-axis", choices=["z", "y"], default="z")
    parser.add_argument("--spawn-height", type=float, default=1.0)
    parser.add_argument(
        "--camera-pos",
        type=str,
        default="3.0 3.0 2.0",
        help='Camera position as "x y z".',
    )
    parser.add_argument(
        "--camera-target",
        type=str,
        default="0.0 0.0 1.0",
        help='Camera target as "x y z".',
    )
    parser.add_argument(
        "--base-mode",
        choices=["free", "fixed"],
        default="free",
        help="Set base behavior: free dynamics or fixed base.",
    )
    parser.add_argument("--no-ground", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        xml_path = resolve_xml_path(args.xml)
        camera_pos = parse_vec3(args.camera_pos, "--camera-pos")
        camera_target = parse_vec3(args.camera_target, "--camera-target")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    gym = gymapi.acquire_gym()
    sim = None
    viewer = None

    try:
        sim_params = build_sim_params(args)
        sim = gym.create_sim(
            args.compute_device_id,
            args.graphics_device_id,
            gymapi.SIM_PHYSX,
            sim_params,
        )
        if sim is None:
            print("Error: failed to create Isaac Gym sim.", file=sys.stderr)
            return 1

        if not args.no_ground:
            plane_params = gymapi.PlaneParams()
            if args.up_axis == "z":
                plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
            else:
                plane_params.normal = gymapi.Vec3(0.0, 1.0, 0.0)
            gym.add_ground(sim, plane_params)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
        asset_options.fix_base_link = args.base_mode == "fixed"

        asset = gym.load_asset(sim, str(xml_path.parent), xml_path.name, asset_options)
        if asset is None:
            print(f"Error: failed to load asset from {xml_path}", file=sys.stderr)
            return 1

        rb_count = gym.get_asset_rigid_body_count(asset)
        dof_count = gym.get_asset_dof_count(asset)
        joint_count = gym.get_asset_joint_count(asset)

        print(f"Loaded XML: {xml_path}")
        print(f"Asset root: {xml_path.parent}")
        print(f"Asset file: {xml_path.name}")
        print(f"Base mode: {args.base_mode}")
        print(f"Rigid bodies: {rb_count}")
        print(f"DOFs: {dof_count}")
        print(f"Joints: {joint_count}")

        env_lower = gymapi.Vec3(-2.0, -2.0, -2.0)
        env_upper = gymapi.Vec3(2.0, 2.0, 2.0)
        env = gym.create_env(sim, env_lower, env_upper, 1)

        pose = gymapi.Transform()
        if args.up_axis == "z":
            pose.p = gymapi.Vec3(0.0, 0.0, args.spawn_height)
        else:
            pose.p = gymapi.Vec3(0.0, args.spawn_height, 0.0)

        actor_handle = gym.create_actor(env, asset, pose, "xml_actor", 0, 0)
        if actor_handle < 0:
            print("Error: failed to create actor.", file=sys.stderr)
            return 1

        # Required for stable GPU pipeline stepping after actors are created.
        gym.prepare_sim(sim)

        if args.dry_run:
            print("Dry run complete. Exiting before opening viewer loop.")
            return 0

        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            print("Error: failed to create viewer.", file=sys.stderr)
            return 1

        gym.viewer_camera_look_at(viewer, env, camera_pos, camera_target)

        while not gym.query_viewer_has_closed(viewer):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

        return 0
    finally:
        if viewer is not None:
            gym.destroy_viewer(viewer)
        if sim is not None:
            gym.destroy_sim(sim)


if __name__ == "__main__":
    raise SystemExit(main())
