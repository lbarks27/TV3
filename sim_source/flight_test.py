from __future__ import annotations

import argparse
from pathlib import Path

from tvc.flight_log import FlightLog
from tvc.flight_sim import FlightScenario, FlightSimulator, plot_flight_log, summarize_log
from tvc.parameters import (
    FlightTestConfig,
    build_default_config,
    load_config,
    load_vehicle_profile,
)
from tvc.replay_view import replay_3d


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run closed-loop 6-DOF flight tests and replay logs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a flight test.")
    run_parser.add_argument(
        "--config",
        type=str,
        help="Combined JSON config (legacy). Prefer --vehicle/--profile.",
    )
    run_parser.add_argument(
        "--vehicle",
        type=str,
        help="Vehicle config under config/vehicles/ (or absolute path).",
    )
    run_parser.add_argument(
        "--profile",
        type=str,
        help="Flight profile under config/profiles/ (or absolute path).",
    )
    run_parser.add_argument(
        "--motion",
        choices=("attitude", "lateral", "position", "hover"),
        default=None,
        help="Motion mode override.",
    )
    run_parser.add_argument(
        "--scenario",
        choices=("hold", "step", "doublet"),
        default=None,
        help="Attitude reference profile when --motion=attitude.",
    )
    run_parser.add_argument("--duration", type=float, default=None)
    run_parser.add_argument("--dt", type=float, default=None)
    run_parser.add_argument("--roll-deg", type=float, default=None)
    run_parser.add_argument("--pitch-deg", type=float, default=None)
    run_parser.add_argument("--yaw-deg", type=float, default=None)
    run_parser.add_argument("--step-time", type=float, default=1.0)
    run_parser.add_argument("--doublet-axis", choices=("roll", "pitch", "yaw"), default="roll")
    run_parser.add_argument("--doublet-deg", type=float, default=15.0)
    run_parser.add_argument("--doublet-width", type=float, default=0.5)
    run_parser.add_argument("--x", type=float, default=None, help="Target X position (m).")
    run_parser.add_argument("--y", type=float, default=None, help="Target Y position (m).")
    run_parser.add_argument("--z", type=float, default=None, help="Target Z / hover altitude (m).")
    run_parser.add_argument("--force-target", type=float, default=None)
    run_parser.add_argument("--force-weight", type=float, default=None)
    run_parser.add_argument(
        "--log",
        type=str,
        default="flight_logs/latest.csv",
        help="Output path for the recorded flight log (.csv).",
    )
    run_parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot 2D position/attitude/rate/torque traces after the run.",
    )
    run_parser.add_argument(
        "--view-3d",
        action="store_true",
        help="Open an animated 3D replay after the run.",
    )

    replay_parser = subparsers.add_parser("replay", help="Replay a saved flight log.")
    replay_parser.add_argument("--log", type=str, required=True, help="Path to flight log CSV.")
    replay_parser.add_argument("--config", type=str, help="Mount geometry config for 3D replay.")
    replay_parser.add_argument("--vehicle", type=str, help="Vehicle config for 3D replay geometry.")
    replay_parser.add_argument("--profile", type=str, help="Profile used to resolve vehicle geometry.")
    replay_parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot 2D position, attitude, rates, and torque traces.",
    )
    replay_parser.add_argument(
        "--view-3d",
        action="store_true",
        help="Open an animated 3D replay of the vehicle trajectory.",
    )
    return parser


def _load_system_config(args: argparse.Namespace):
    if getattr(args, "vehicle", None) or getattr(args, "profile", None):
        return load_vehicle_profile(
            vehicle_path=getattr(args, "vehicle", None),
            profile_path=getattr(args, "profile", None),
        )
    if getattr(args, "config", None):
        return load_config(args.config)
    default_profile = Path("config/profiles/hover_10m.json")
    if default_profile.exists():
        return load_vehicle_profile(profile_path=default_profile)
    default_config_path = Path("config.example.json")
    if default_config_path.exists():
        return load_config(default_config_path)
    return build_default_config(count=3, radius=0.098, thrust=10.0)


def _build_flight_config(system_config, args: argparse.Namespace) -> FlightTestConfig:
    flight = system_config.flight_test or FlightTestConfig()
    if args.duration is not None:
        flight.duration = args.duration
    if args.dt is not None:
        flight.dt = args.dt
    if args.force_target is not None:
        flight.force_target = args.force_target
    if args.force_weight is not None:
        flight.force_weight = args.force_weight
    if flight.thrust_per_mount is None and system_config.flight_test is not None:
        flight.thrust_per_mount = system_config.flight_test.thrust_per_mount
    flight.vehicle = system_config.vehicle
    flight.controller = system_config.controller
    if system_config.flight_test is not None:
        flight.position_controller = system_config.flight_test.position_controller
        flight.altitude_controller = system_config.flight_test.altitude_controller
    elif system_config.profile is not None:
        flight.altitude_controller = system_config.profile.altitude_controller
    return flight


def _build_scenario(args: argparse.Namespace, system_config) -> FlightScenario:
    profile = system_config.profile
    motion = args.motion
    if motion is None:
        motion = profile.motion if profile is not None else "attitude"

    target = (
        profile.target_position_m
        if profile is not None
        else (0.0, 0.0, 0.0)
    )
    x = args.x if args.x is not None else target[0]
    y = args.y if args.y is not None else target[1]
    z = args.z if args.z is not None else (target[2] if target[2] != 0.0 else 10.0)

    roll = args.roll_deg if args.roll_deg is not None else 0.0
    pitch = args.pitch_deg if args.pitch_deg is not None else 0.0
    yaw = args.yaw_deg if args.yaw_deg is not None else 0.0

    if motion == "hover":
        if profile is not None and profile.real_landing is not None:
            return FlightScenario.real_landing(
                profile.real_landing,
                euler_deg=(roll, pitch, yaw),
                name=profile.name or "real_landing",
            )
        profile_waypoints = profile.waypoints if profile is not None else []
        cli_overrides_xy = args.x is not None or args.y is not None or args.z is not None
        if profile_waypoints and not cli_overrides_xy:
            advance = "timed"
            capture_radius_m = 2.0
            if profile is not None:
                advance = profile.waypoint_advance
                capture_radius_m = profile.waypoint_capture_radius_m
            return FlightScenario.hover_waypoints(
                [(wp.t, wp.position_m) for wp in profile_waypoints],
                euler_deg=(roll, pitch, yaw),
                advance=advance,
                capture_radius_m=capture_radius_m,
            )
        return FlightScenario.hover(z, xy_m=(x, y), euler_deg=(roll, pitch, yaw))
    if motion == "lateral":
        lat_pitch = args.pitch_deg if args.pitch_deg is not None else 8.0
        return FlightScenario.lateral((0.0, lat_pitch, 0.0))
    if motion == "position":
        return FlightScenario.position_step((x, y, z), step_time=args.step_time)

    scenario = args.scenario
    if scenario is None:
        scenario = profile.scenario if profile is not None else "hold"

    # Legacy defaults for attitude demos when not driven by a profile.
    if args.roll_deg is None and profile is None:
        roll = 20.0
    if args.pitch_deg is None and profile is None and scenario != "hold":
        pitch = 8.0

    if scenario == "hold":
        return FlightScenario.hold((roll, pitch, yaw))
    if scenario == "step":
        return FlightScenario.step((roll, pitch, yaw), step_time=args.step_time)
    return FlightScenario.doublet(
        args.doublet_axis,
        args.doublet_deg,
        pulse_width=args.doublet_width,
        start_time=args.step_time,
    )


def run_command(args: argparse.Namespace) -> None:
    system_config = _load_system_config(args)
    flight_config = _build_flight_config(system_config, args)

    # Real-landing: fixed launch offset from the pinned pad (default 10 m east).
    if system_config.profile is not None and system_config.profile.real_landing is not None:
        mission = system_config.profile.real_landing
        launch = mission.launch_position_m()
        flight_config.initial_position_m = launch
        site = mission.landing_site_m
        first_wp = mission.first_waypoint_m()
        path = str(getattr(mission, "path_shape", "capture") or "capture")
        note = (
            f"real_landing path={path} "
            f"launch=({launch[0]:.2f},{launch[1]:.2f},{launch[2]:.2f}) "
            f"site=({site[0]:.2f},{site[1]:.2f},{site[2]:.2f}) "
            f"first_wp=({first_wp[0]:.2f},{first_wp[1]:.2f},{first_wp[2]:.2f}) "
            f"track=launch->first_wp->site descent={mission.descent_duration_s:.1f}s"
        )
        base_notes = (flight_config.notes or system_config.profile.notes or "").strip()
        flight_config.notes = f"{base_notes} | {note}" if base_notes else note
        print(
            f"Real landing: launch ({launch[0]:.2f}, {launch[1]:.2f}, {launch[2]:.2f}) m "
            f"| pad {site} | first WP {first_wp}"
        )
        variance = system_config.profile.engine_variance
        if variance is not None and variance.draws:
            bits = []
            for index, draw in enumerate(variance.draws):
                thrust_scale, time_scale = float(draw[0]), float(draw[1])
                ign = float(draw[2]) if len(draw) > 2 else 0.0
                bit = f"M{index} thrust={thrust_scale:.3f}× burn={time_scale:.3f}×"
                if ign > 1e-9:
                    bit += f" ign={ign:.3f}s"
                bits.append(bit)
            print("Engine variance: " + " | ".join(bits))
            scatter = "engine_variance " + " ".join(bits)
            flight_config.notes = (
                f"{flight_config.notes} | {scatter}"
                if flight_config.notes
                else scatter
            )

    scenario = _build_scenario(args, system_config)
    simulator = FlightSimulator(system_config, flight_config)
    log = simulator.run(scenario)

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.save(log_path)
    print(f"Saved flight log to {log_path}")
    print(summarize_log(log))
    if args.plot:
        plot_flight_log(log)
    if args.view_3d:
        replay_3d(log, config=system_config, title=f"3D replay: {log_path}")


def replay_command(args: argparse.Namespace) -> None:
    log = FlightLog.load(args.log)
    print(summarize_log(log))
    if args.plot:
        plot_flight_log(log, title=f"Replay: {args.log}")
    if args.view_3d or not args.plot:
        system_config = _load_system_config(args)
        replay_3d(log, config=system_config, title=f"3D replay: {args.log}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.command == "run":
        run_command(args)
    elif args.command == "replay":
        replay_command(args)


if __name__ == "__main__":
    main()
