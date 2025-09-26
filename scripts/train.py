#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INFER = HERE / "gr00t_inference.py"

def run(cmd, env=None, dry_run=False):
    print("\n==> CMD:", " ".join(shlex.quote(c) for c in cmd))
    if env:
        print("==> ENV:", {k: env[k] for k in sorted(env) if k.startswith(("ROS_", "RMW_", "CYCLONEDDS_"))})
    if dry_run:
        return 0
    return subprocess.call(cmd, env=env)

def build_common(args):
    cmd = [sys.executable, str(INFER)]
    if args.model:
        cmd += ["--model", args.model]
    if args.device:
        cmd += ["--device", args.device]
    if args.extra:
        cmd += args.extra
    return cmd

def scenario_image(args):
    cmd = build_common(args)
    cmd += ["--mode", "image", "--input", args.input, "--output", args.output]
    if args.prompt:
        cmd += ["--prompt", args.prompt]
    return run(cmd, dry_run=args.dry_run)

def scenario_folder(args):
    cmd = build_common(args)
    cmd += ["--mode", "folder", "--input_dir", args.input, "--output_dir", args.output]
    if args.prompt:
        cmd += ["--prompt", args.prompt]
    return run(cmd, dry_run=args.dry_run)

def scenario_video(args):
    cmd = build_common(args)
    cmd += ["--mode", "video", "--input", args.input, "--output", args.output]
    if args.prompt:
        cmd += ["--prompt", args.prompt]
    return run(cmd, dry_run=args.dry_run)

def scenario_webcam(args):
    cmd = build_common(args)
    cmd += ["--mode", "webcam", "--camera", str(args.camera_index), "--output", args.output]
    if args.prompt:
        cmd += ["--prompt", args.prompt]
    return run(cmd, dry_run=args.dry_run)

def scenario_ros2(args):
    env = os.environ.copy()
    if args.ros_domain_id is not None:
        env["ROS_DOMAIN_ID"] = str(args.ros_domain_id)
    if args.rmw_impl:
        env["RMW_IMPLEMENTATION"] = args.rmw_impl
    if args.cyclonedds_uri:
        env["CYCLONEDDS_URI"] = args.cyclonedds_uri

    cmd = build_common(args)
    cmd += ["--mode", "ros2", "--ros-topic", args.ros_topic]
    if args.prompt:
        cmd += ["--prompt", args.prompt]
    if args.output:
        cmd += ["--output", args.output]
    return run(cmd, env=env, dry_run=args.dry_run)

def scenario_replay(args):
    cmd = build_common(args)
    cmd += ["--mode", "replay", "--dataset", args.input, "--output", args.output]
    if args.prompt:
        cmd += ["--prompt", args.prompt]
    return run(cmd, dry_run=args.dry_run)

SCENARIOS = {
    "image": scenario_image,
    "folder": scenario_folder,
    "video": scenario_video,
    "webcam": scenario_webcam,
    "ros2": scenario_ros2,
    "replay": scenario_replay,
}

def parse_args():
    p = argparse.ArgumentParser(
        description="Run gr00t_inference.py with preset argument sets."
    )
    p.add_argument(
        "--scenario",
        choices=SCENARIOS.keys(),
        required=True,
        help="Which preset to run."
    )
    p.add_argument(
        "--model", default="nvidia/GR00T-N1.5-3B",
        help="Model ID or path passed to gr00t_inference.py."
    )
    p.add_argument(
        "--device", default="cuda",
        help="Device passed to gr00t_inference.py (e.g., cuda, cpu)."
    )
    p.add_argument(
        "--input", help="Input file/folder/topic/dataset depending on the scenario."
    )
    p.add_argument(
        "--output", default="./outputs",
        help="Output file/folder depending on the scenario."
    )
    p.add_argument(
        "--prompt", default=None, help="Optional text prompt."
    )
    p.add_argument(
        "--camera-index", type=int, default=0,
        help="Webcam index (webcam scenario)."
    )
    # ROS 2 specific
    p.add_argument(
        "--ros-topic", default="/camera/color/image_raw",
        help="ROS 2 topic (ros2 scenario)."
    )
    p.add_argument(
        "--ros-domain-id", type=int, default=None,
        help="Set ROS_DOMAIN_ID in the environment (ros2 scenario)."
    )
    p.add_argument(
        "--rmw-impl", default=None,
        help="Set RMW_IMPLEMENTATION (e.g., rmw_cyclonedds_cpp) (ros2 scenario)."
    )
    p.add_argument(
        "--cyclonedds-uri", default=None,
        help="Set CYCLONEDDS_URI (ros2 scenario)."
    )
    # Extra passthrough args to gr00t_inference.py
    p.add_argument(
        "--extra", nargs=argparse.REMAINDER, default=[],
        help="Anything after this flag is passed verbatim to gr00t_inference.py"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing."
    )
    return p.parse_args()

def main():
    args = parse_args()

    # Basic path checks
    if not INFER.exists():
        print(f"ERROR: {INFER} not found next to train.py", file=sys.stderr)
        sys.exit(1)

    # Scenario-specific sanity checks
    if args.scenario in {"image", "video"} and not args.input:
        print("ERROR: --input is required for image/video scenarios.", file=sys.stderr)
        sys.exit(2)
    if args.scenario == "folder" and not args.input:
        print("ERROR: --input (folder path) is required for folder scenario.", file=sys.stderr)
        sys.exit(2)
    if args.scenario == "ros2" and not args.ros_topic:
        print("ERROR: --ros-topic is required for ros2 scenario.", file=sys.stderr)
        sys.exit(2)
    if args.scenario == "replay" and not args.input:
        print("ERROR: --input (dataset path) is required for replay scenario.", file=sys.stderr)
        sys.exit(2)

    rc = SCENARIOS[args.scenario](args)
    sys.exit(rc)

if __name__ == "__main__":
    main()
