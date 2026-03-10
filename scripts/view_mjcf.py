import argparse
from pathlib import Path


DEFAULT_HUMANOID_XML = (
    Path(__file__).resolve().parents[1]
    / "closd"
    / "data"
    / "assets"
    / "mjcf"
    / "smpl_0_humanoid.xml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View a MuJoCo humanoid MJCF file.")
    parser.add_argument(
        "humanoid_xml",
        nargs="?",
        default=str(DEFAULT_HUMANOID_XML),
        help="Path to humanoid MJCF XML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(args.humanoid_xml)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)


if __name__ == "__main__":
    main()
