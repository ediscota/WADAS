#!/usr/bin/env python3
# Description: Launch N simulated cameras against the WADAS pipeline, once per scheduler use
# case, and collect event latencies and node/device placements into CSV files.

"""Run a camera load test. Run it on the Ray head node, with the cluster and exporters up.

  --cameras 17 --video-probability 0       17 cameras sending photos only
  --cameras 10 --video-probability 0.2     10 cameras, 1 event in 5 is a video
  --use-cases MONITORING LOCAL             schedulers to compare; LOCAL runs everything on this
                                           host without Ray (how a single node works today)

Each camera is a process with its own pipeline, so the scheduler decides the placement of each
camera once, when the camera starts. Cameras start --camera-stagger seconds apart so that every
decision sees the load of the previous ones. After all cameras are warm, they start together
and run for --duration seconds. The same seed gives every use case the same arrival times.
"""

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from importlib import metadata
from pathlib import Path

from loadgen import plan_runs, summarize_events

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMERA_SCRIPT = Path(__file__).resolve().with_name("run_camera.py")
MODES = ("MONITORING", "ACTUATOR", "CRITICAL", "LOCAL")

RUN_KEYS = ["run_id", "run_index", "repeat", "mode"]
CAMERA_FIELDS = RUN_KEYS + [
    "camera_id",
    "status",
    "error",
    "returncode",
    "ready",
    "det_node",
    "det_device",
    "cls_node",
    "cls_device",
    "t_init_s",
    "t_warmup_s",
    "warmup_inferences",
    "n_events_planned",
    "n_events_done",
]
EVENT_FIELDS = RUN_KEYS + [
    "camera_id",
    "event_id",
    "kind",
    "media",
    "scheduled_at",
    "started_at",
    "finished_at",
    "wait_s",
    "service_s",
    "latency_s",
    "t_detection_s",
    "t_classification_s",
    "n_animals",
    "n_frames_with_animals",
    "status",
    "error",
]
RUN_FIELDS = RUN_KEYS + [
    "cameras",
    "cameras_ready",
    "duration_s",
    "events",
    "ok",
    "errors",
    "queued_share",
    "photo_count",
    "photo_p50_s",
    "photo_p95_s",
    "photo_max_s",
    "video_count",
    "video_p50_s",
    "video_p95_s",
    "video_max_s",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--cameras", type=int, default=17)
    parser.add_argument("--duration", type=float, default=300.0, help="Seconds of load per run")
    parser.add_argument("--photo-interval", type=float, default=30.0, help="Mean s between events")
    parser.add_argument("--video-probability", type=float, default=0.0)
    parser.add_argument(
        "--use-cases", nargs="+", choices=MODES, default=list(MODES[:3]), help="Runs to execute"
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--camera-stagger", type=float, default=2.0, help="Seconds between starts")
    parser.add_argument("--init-timeout", type=float, default=900.0, help="Seconds to warm up")
    parser.add_argument("--grace", type=float, default=300.0, help="Extra seconds after the load")
    parser.add_argument("--cooldown", type=float, default=30.0, help="Seconds between runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=1, help="Video frames sampled per second")
    parser.add_argument("--local-device", default="GPU", help="Device used by the LOCAL run")
    parser.add_argument("--photos-dir", type=Path, default=Path.home() / "PHOTO_TEST")
    parser.add_argument("--videos-dir", type=Path, default=Path.home() / "VIDEO_TEST")
    parser.add_argument("--results-dir", type=Path, default=Path.home() / "wadas_results")
    parser.add_argument("--run-id", default=datetime.now().strftime("load-%Y%m%d-%H%M%S"))
    return parser.parse_args(argv)


def git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def write_metadata(args, run_dir):
    photos = [p.name for p in sorted(Path(args.photos_dir).expanduser().glob("*.jp*g"))]
    videos = [p.name for p in sorted(Path(args.videos_dir).expanduser().glob("*.mp4"))]
    info = {
        "args": {key: str(value) for key, value in vars(args).items()},
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "ray": package_version("ray"),
        "openvino": package_version("openvino"),
        "host": platform.node(),
        "photos": photos,
        "videos": videos,
    }
    (run_dir / "meta.json").write_text(json.dumps(info, indent=2))


def launch_camera(args, mode, camera_id, run_dir, start_file):
    camera_dir = run_dir / f"camera_{camera_id:02d}"
    camera_dir.mkdir(parents=True)
    cmd = [
        sys.executable,
        str(CAMERA_SCRIPT),
        "--camera-id",
        str(camera_id),
        "--mode",
        mode,
        "--out-dir",
        str(camera_dir),
        "--start-file",
        str(start_file),
        "--photos-dir",
        str(args.photos_dir),
        "--videos-dir",
        str(args.videos_dir),
        "--duration",
        str(args.duration),
        "--photo-interval",
        str(args.photo_interval),
        "--video-probability",
        str(args.video_probability),
        "--seed",
        str(args.seed),
        "--fps",
        str(args.fps),
        "--local-device",
        args.local_device,
        "--start-timeout",
        str(args.init_timeout + 120),
    ]
    log = open(camera_dir / "log.txt", "w")
    process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    return {"id": camera_id, "dir": camera_dir, "process": process, "log": log}


def is_ready(camera):
    return (camera["dir"] / "ready").exists()


def wait_until_ready(cameras, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(is_ready(c) or c["process"].poll() is not None for c in cameras):
            return
        time.sleep(1)


def wait_for_cameras(cameras, deadline):
    for camera in cameras:
        try:
            camera["process"].wait(timeout=max(0.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            camera["process"].kill()
            camera["process"].wait()


def collect(run_base, cameras):
    camera_rows, event_rows = [], []
    for camera in cameras:
        row = dict(
            run_base,
            camera_id=camera["id"],
            ready=camera["ready"],
            returncode=camera["process"].returncode,
            status="crashed",
        )
        camera_file = camera["dir"] / "camera.json"
        if camera_file.exists():
            row.update(json.loads(camera_file.read_text()))
        camera_rows.append(row)

        events_file = camera["dir"] / "events.jsonl"
        if events_file.exists():
            for line in events_file.read_text().splitlines():
                try:
                    event_rows.append(dict(run_base, **json.loads(line)))
                except ValueError:  # last line of a camera that was killed
                    continue
    return camera_rows, event_rows


def run_once(args, index, repeat, mode, run_dir):
    run_base = {"run_id": args.run_id, "run_index": index, "repeat": repeat, "mode": mode}
    run_dir.mkdir(parents=True)
    start_file = run_dir / "start_at"

    cameras = []
    for camera_id in range(args.cameras):
        cameras.append(launch_camera(args, mode, camera_id, run_dir, start_file))
        if camera_id < args.cameras - 1:
            time.sleep(args.camera_stagger)

    wait_until_ready(cameras, args.init_timeout)
    for camera in cameras:
        camera["ready"] = is_ready(camera)
        if not camera["ready"] and camera["process"].poll() is None:
            camera["process"].kill()
    ready = [c for c in cameras if c["ready"] and c["process"].poll() is None]

    if ready:
        start_at = time.time() + 3
        pending = run_dir / "start_at.tmp"
        pending.write_text(repr(start_at))
        pending.replace(start_file)
        wait_for_cameras(ready, start_at + args.duration + args.grace)
    for camera in cameras:
        camera["process"].wait()
        camera["log"].close()

    camera_rows, event_rows = collect(run_base, cameras)
    run_row = dict(
        run_base,
        cameras=args.cameras,
        cameras_ready=len(ready),
        duration_s=args.duration,
        **summarize_events(event_rows),
    )
    return camera_rows, event_rows, run_row


def fmt(value, scale=1):
    return "-" if value is None else f"{value * scale:.1f}"


def print_summary(run_rows, camera_rows):
    for run in run_rows:
        print(
            f"\n[{run['mode']} rep{run['repeat']}] cameras {run['cameras_ready']}/{run['cameras']}"
            f" | events {run['ok']}/{run['events']} ok"
            f" | queued {fmt(run['queued_share'], 100)}%"
        )
        print(
            f"  photo latency p50 {fmt(run['photo_p50_s'])}s p95 {fmt(run['photo_p95_s'])}s"
            f" | video latency p50 {fmt(run['video_p50_s'])}s p95 {fmt(run['video_p95_s'])}s"
        )
        cams = [c for c in camera_rows if c["run_index"] == run["run_index"] and c.get("det_node")]
        for label, node, device in (
            ("detection", "det_node", "det_device"),
            ("classification", "cls_node", "cls_device"),
        ):
            placement = Counter((c[node], c[device]) for c in cams)
            counts = ", ".join(f"{n}/{d} x{k}" for (n, d), k in placement.most_common())
            print(f"  {label} placement: {counts or '-'}")


def main(argv=None):
    args = parse_args(argv)
    args.photos_dir = args.photos_dir.expanduser()
    args.videos_dir = args.videos_dir.expanduser()
    if not any(args.photos_dir.glob("*.jp*g")) and args.video_probability < 1:
        sys.exit(f"No photos in {args.photos_dir}: run extract_photos.py first")
    if not any(args.videos_dir.glob("*.mp4")) and args.video_probability > 0:
        sys.exit(f"No .mp4 files in {args.videos_dir}")

    base_dir = args.results_dir.expanduser() / args.run_id
    base_dir.mkdir(parents=True)
    write_metadata(args, base_dir)

    camera_rows, run_rows = [], []
    plan = plan_runs(args.use_cases, args.repeats, args.seed)
    with (
        open(base_dir / "cameras.csv", "w", newline="") as cameras_file,
        open(base_dir / "events.csv", "w", newline="") as events_file,
        open(base_dir / "runs.csv", "w", newline="") as runs_file,
    ):
        writers = {
            "cameras": csv.DictWriter(cameras_file, CAMERA_FIELDS, extrasaction="ignore"),
            "events": csv.DictWriter(events_file, EVENT_FIELDS, extrasaction="ignore"),
            "runs": csv.DictWriter(runs_file, RUN_FIELDS, extrasaction="ignore"),
        }
        for writer in writers.values():
            writer.writeheader()

        for index, (repeat, mode) in enumerate(plan):
            print(
                f"[{index + 1}/{len(plan)}] {mode} rep{repeat}: {args.cameras} camera(s), "
                f"{args.duration:g}s of load",
                flush=True,
            )
            run_dir = base_dir / f"{index:02d}_{mode}_rep{repeat}"
            cameras, events, run = run_once(args, index, repeat, mode, run_dir)
            writers["cameras"].writerows(cameras)
            writers["events"].writerows(events)
            writers["runs"].writerow(run)
            for handle in (cameras_file, events_file, runs_file):
                handle.flush()
            camera_rows += cameras
            run_rows.append(run)
            print(
                f"    done: {run['ok']}/{run['events']} events ok,"
                f" {run['cameras_ready']}/{run['cameras']} cameras ready",
                flush=True,
            )
            if index < len(plan) - 1:
                time.sleep(args.cooldown)

    print_summary(run_rows, camera_rows)
    print(f"\nResults: {base_dir}")


if __name__ == "__main__":
    main()
