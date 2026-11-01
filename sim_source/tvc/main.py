import argparse
from pathlib import Path

from .allocator import TorqueAllocator
from .parameters import build_default_config, load_config
from .rendering import setup_plot


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Thrust vector torque allocator demo.")
    parser.add_argument("--config", type=str, help="Path to JSON config file.")
    parser.add_argument("--n", type=int, default=3, help="Number of thrust mounts.")
    parser.add_argument(
        "--radius", type=float, default=0.098, help="Mount ring radius (meters)."
    )
    parser.add_argument("--thrust", type=float, default=10.0, help="Thrust per mount.")
    parser.add_argument(
        "--max-angle-deg",
        type=float,
        default=None,
        help="Symmetric gimbal angle limit for both axes (ring default only).",
    )
    parser.add_argument("--force-weight", type=float, default=None)
    parser.add_argument("--torque-limit", type=float, default=None)
    parser.add_argument("--force-scale", type=float, default=None)
    parser.add_argument(
        "--torque-damping",
        type=float,
        default=None,
        help="Ridge scale for the linear a2 torque map (solver.damping).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    default_config_path = Path("config.example.json")
    if args.config:
        config = load_config(args.config)
        config_source = f"config file: {args.config}"
    elif default_config_path.exists():
        config = load_config(default_config_path)
        config_source = f"config file: {default_config_path}"
    else:
        config = build_default_config(
            count=args.n,
            radius=args.radius,
            thrust=args.thrust,
            max_angle_deg=args.max_angle_deg,
        )
        config_source = "built-in defaults"

    if args.force_weight is not None:
        config.render.force_weight = args.force_weight
    if args.torque_limit is not None:
        config.render.torque_limit = args.torque_limit
    if args.force_scale is not None:
        config.render.force_scale = args.force_scale
    if args.torque_damping is not None:
        config.solver.damping = args.torque_damping
    print(
        "Loaded "
        f"{config_source} | "
        f"force_weight={config.render.force_weight:.3f} | "
        f"mixer=collective+differential | "
        f"torque_damping={config.solver.damping:.3f}"
    )

    allocator = TorqueAllocator(
        config.mounts,
        config.solver,
        force_axis=config.render.force_axis,
        center_of_mass=config.center_of_mass,
    )
    setup_plot(config.mounts, allocator, config.render, catch_arms=config.catch_arms)


if __name__ == "__main__":
    main()
