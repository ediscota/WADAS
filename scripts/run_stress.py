# Description: Stress test of the Ray cluster with continuous camera streams: a ramp of N
# simultaneous streams at a fixed frame rate, to find how many frames per second a scheduler
# configuration sustains, with which latency and energy.

"""Ramp of continuous streams (run_stream_task.py) against one scheduler configuration.

Run it on the Ray head node, with the cluster and the per-node exporters up (the power exporter
too, for --measure-energy). For each step N of --steps:

  1. N streams start --stagger seconds apart, each placing its own pipeline once;
  2. after the last start, --warmup seconds are left out (model loading, first frames);
  3. the next --window seconds are measured: frames, latency, energy;
  4. the streams stop sending, finish their queue for at most --drain seconds, and exit.

A step is sustainable when the frames completed in the window are at least --min-throughput
of those offered (N x fps x window), the 95th percentile of the latency (end of processing
minus scheduled arrival) is at most --latency-limit, and no stream failed. The ramp stops
--extra-steps after the first step that is not sustainable. The capacity is the largest N
sustainable before that step: "N x fps frames per second".

  python scripts/run_stress.py --use-case MONITORING --measure-energy --run-id st_mon
  python scripts/run_stress.py --use-case ACTUATOR --scheduler-params '{"pending_penalty": 0.05}'
"""

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import median

from energy import energy_between, measure_idle_watts, read_energy
from run_scenario import alive_node_names, git_commit, package_version, prepare_pending_registry

STREAM_SCRIPT = Path(__file__).resolve().with_name("run_stream_task.py")
USE_CASES = ("MONITORING", "ACTUATOR", "CRITICAL", "PINNED")
DEFAULT_STEPS = "1,2,4,6,8,12,16,20,24,32"

STEP_FIELDS = [
    "run_id",
    "use_case",
    "scheduler_params",
    "n_streams",
    "offered_fps",
    "throughput_fps",
    "throughput_ratio",
    "latency_p50_s",
    "latency_p95_s",
    "latency_max_s",
    "late_share",
    "unprocessed",
    "failed_streams",
    "sustainable",
    "reasons",
    "idle_w",
    "energy_j",
    "energy_active_j",
    "power_active_w",
    "j_per_frame",
    "energy_by_node_j",
]
STREAM_FIELDS = [
    "run_id",
    "n_streams",
    "stream_id",
    "video",
    "status",
    "error",
    "returncode",
    "det_node",
    "det_device",
    "cls_node",
    "cls_device",
    "t_init_s",
    "first_arrival",
    "n_frames",
    "unprocessed",
]
FRAME_FIELDS = ["n_streams", "stream_id", "frame", "arrival", "start", "end", "latency_s",
                "late", "det_s", "cls_s", "n_animals"]  # fmt: skip


def parse_steps(text):
    steps = [int(part) for part in text.split(",") if part.strip()]
    if not steps or any(n < 1 for n in steps) or steps != sorted(set(steps)):
        raise ValueError(f"steps must be increasing positive integers, got {text!r}")
    return steps


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--use-case", required=True, choices=USE_CASES)
    parser.add_argument("--scheduler-params", default="{}", help="JSON with scheduler options")
    parser.add_argument("--detection-model", help="Detection model version")
    parser.add_argument("--classification-model", help="Classification model version")
    parser.add_argument("--videos-dir", type=Path, default=Path.home() / "VIDEO_TEST")
    parser.add_argument("--steps", default=DEFAULT_STEPS, help="Numbers of streams, increasing")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second per stream")
    parser.add_argument("--stagger", type=float, default=1.0, help="Seconds between stream starts")
    parser.add_argument("--warmup", type=float, default=30.0, help="Seconds left out after starts")
    parser.add_argument("--window", type=float, default=120.0, help="Seconds measured per step")
    parser.add_argument("--drain", type=float, default=10.0, help="Seconds to finish the queue")
    parser.add_argument("--cooldown", type=float, default=20.0, help="Seconds between steps")
    parser.add_argument("--latency-limit", type=float, default=2.0, help="Max p95 latency (s)")
    parser.add_argument(
        "--min-throughput", type=float, default=0.95, help="Min completed / offered frames"
    )
    parser.add_argument(
        "--extra-steps", type=int, default=1, help="Steps run after the first unsustainable one"
    )
    parser.add_argument("--measure-energy", action="store_true")
    parser.add_argument(
        "--idle-seconds", type=float, default=30.0, help="Idle power measured before the first step"
    )
    parser.add_argument(
        "--step-idle",
        type=float,
        default=15.0,
        help="Idle power measured again at the end of every cooldown, against drift (seconds)",
    )
    parser.add_argument(
        "--expected-nodes", type=int, default=5, help="Abort if fewer Ray nodes are alive"
    )
    parser.add_argument("--results-dir", type=Path, default=Path.home() / "wadas_results")
    parser.add_argument("--run-id", default=datetime.now().strftime("stress-%Y%m%d-%H%M%S"))
    args = parser.parse_args(argv)
    try:
        json.loads(args.scheduler_params)
        args.steps = parse_steps(args.steps)
    except ValueError as exc:
        parser.error(str(exc))
    if args.step_idle > args.cooldown:
        parser.error("--step-idle must fit in --cooldown")
    return args


def percentile(values, q):
    """Nearest-rank percentile (q in 0..100) of a non-empty list."""
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def summarize_step(streams, n_streams, measure_start, measure_end, fps, latency_limit,
                   min_throughput):  # fmt: skip
    """Summary of one step from its streams: [{"status", "frames": [records], "unprocessed"}].

    Frames count for the throughput when they finish inside the window, and for the latency
    when they arrive inside it. Streams that never started still count in the offered load.
    """
    window = measure_end - measure_start
    offered = n_streams * fps * window
    frames = [f for s in streams for f in s.get("frames", [])]
    completed = [f for f in frames if measure_start <= f["end"] <= measure_end]
    arrived = [f for f in frames if measure_start <= f["arrival"] < measure_end]
    latencies = [f["latency_s"] for f in arrived]
    failed = sum(1 for s in streams if s.get("status") != "ok")
    unprocessed = sum(int(s.get("unprocessed") or 0) for s in streams)

    summary = {
        "n_streams": n_streams,
        "offered_fps": n_streams * fps,
        "throughput_fps": len(completed) / window,
        "throughput_ratio": len(completed) / offered if offered else 0.0,
        "latency_p50_s": median(latencies) if latencies else None,
        "latency_p95_s": percentile(latencies, 95) if latencies else None,
        "latency_max_s": max(latencies) if latencies else None,
        "late_share": sum(1 for f in arrived if f["late"]) / len(arrived) if arrived else None,
        "unprocessed": unprocessed,
        "failed_streams": failed,
        "n_completed": len(completed),
    }
    reasons = []
    if summary["throughput_ratio"] < min_throughput:
        reasons.append(f"throughput {summary['throughput_ratio']:.0%} < {min_throughput:.0%}")
    if summary["latency_p95_s"] is None or summary["latency_p95_s"] > latency_limit:
        reasons.append(f"p95 latency above {latency_limit:g} s")
    if failed:
        reasons.append(f"{failed} stream(s) failed")
    summary["sustainable"] = not reasons
    summary["reasons"] = "; ".join(reasons)
    return summary


def ramp_should_continue(sustainable_so_far, extra_steps):
    """Whether to run another step: until extra_steps steps after the first unsustainable."""
    if False not in sustainable_so_far:
        return True
    first_bad = sustainable_so_far.index(False)
    return len(sustainable_so_far) - 1 - first_bad < extra_steps


def capacity(summaries):
    """Largest n_streams sustainable before the first unsustainable step (0 if none)."""
    best = 0
    for summary in summaries:
        if not summary["sustainable"]:
            break
        best = summary["n_streams"]
    return best


def rotated(videos, start):
    """Videos starting at position start, so that streams do not all show the same frames."""
    start %= len(videos)
    return videos[start:] + videos[:start]


def read_stream(stream_dir):
    """result.json plus the frames of one stream, with numbers parsed."""
    result_file = stream_dir / "result.json"
    stream = json.loads(result_file.read_text()) if result_file.exists() else {"status": "crashed"}
    frames = []
    frames_file = stream_dir / "frames.csv"
    if frames_file.exists():
        with open(frames_file, newline="") as f:
            for row in csv.DictReader(f):
                frames.append(
                    {
                        **{k: float(row[k]) for k in ("arrival", "start", "end", "latency_s")},
                        **{k: float(row[k] or 0) for k in ("det_s", "cls_s")},
                        "frame": int(row["frame"]),
                        "n_animals": int(row["n_animals"]),
                        "late": row["late"] == "True",
                    }
                )
    stream["frames"] = frames
    return stream


def launch_stream(args, index, videos, stream_dir, stop_at):
    stream_dir.mkdir(parents=True)
    cmd = [
        sys.executable,
        str(STREAM_SCRIPT),
        "--videos",
        *[str(v) for v in rotated(videos, index)],
        "--use-case",
        args.use_case,
        "--scheduler-params",
        args.scheduler_params,
        *(["--detection-model", args.detection_model] if args.detection_model else []),
        *(
            ["--classification-model", args.classification_model]
            if args.classification_model
            else []
        ),
        "--fps",
        str(args.fps),
        "--stop-at",
        str(stop_at),
        "--drain",
        str(args.drain),
        "--stream-id",
        str(index),
        "--out-dir",
        str(stream_dir),
    ]
    log = open(stream_dir / "log.txt", "w")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log


def sleep_until(epoch):
    delay = epoch - time.time()
    if delay > 0:
        time.sleep(delay)


def run_step(args, n_streams, videos, run_dir, node_ips, idle_w):
    step_dir = run_dir / f"n{n_streams:03d}"
    launch_start = time.time() + 2.0
    measure_start = launch_start + (n_streams - 1) * args.stagger + args.warmup
    measure_end = measure_start + args.window

    launched = []
    for index in range(n_streams):
        sleep_until(launch_start + index * args.stagger)
        launched.append(
            launch_stream(args, index, videos, step_dir / f"s{index:03d}", measure_end)
        )

    sleep_until(measure_start)
    before = read_energy(node_ips) if args.measure_energy else None
    sleep_until(measure_end)
    after = read_energy(node_ips) if args.measure_energy else None

    deadline = measure_end + args.drain + 120.0
    returncodes = []
    for process, log in launched:
        try:
            process.wait(timeout=max(0.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        finally:
            log.close()
        returncodes.append(process.returncode)

    streams = []
    for index, returncode in enumerate(returncodes):
        stream = read_stream(step_dir / f"s{index:03d}")
        stream["returncode"] = returncode
        streams.append(stream)

    summary = summarize_step(
        streams,
        n_streams,
        measure_start,
        measure_end,
        args.fps,
        args.latency_limit,
        args.min_throughput,
    )
    if args.measure_energy:
        per_node, joules = energy_between(before, after)
        if joules is not None:
            active = joules - idle_w * args.window
            summary.update(
                idle_w=idle_w,
                energy_j=joules,
                energy_active_j=active,
                power_active_w=active / args.window,
                j_per_frame=active / summary["n_completed"] if summary["n_completed"] else None,
                energy_by_node_j=json.dumps({ip: round(j, 1) for ip, j in per_node.items()}),
            )
    return summary, streams


def cooldown(args, node_ips, idle_w):
    """Wait --cooldown seconds; with energy, measure the idle power again in the last
    --step-idle of them, so that every step subtracts a baseline taken just before it."""
    if not args.measure_energy:
        time.sleep(args.cooldown)
        return idle_w
    time.sleep(args.cooldown - args.step_idle)
    idle_by_ip = measure_idle_watts(node_ips, args.step_idle)
    return sum(idle_by_ip.values()) if idle_by_ip else idle_w


def write_metadata(args, run_dir, videos):
    info = {
        "args": {key: str(value) for key, value in vars(args).items()},
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "ray": package_version("ray"),
        "openvino": package_version("openvino"),
        "host": platform.node(),
        "videos": [video.name for video in videos],
    }
    (run_dir / "meta.json").write_text(json.dumps(info, indent=2))


def main(argv=None):
    args = parse_args(argv)
    videos = sorted(args.videos_dir.expanduser().glob("*.mp4"))
    if not videos:
        sys.exit(f"No .mp4 files found in {args.videos_dir}")
    params = json.loads(args.scheduler_params)
    if params.get("pending_penalty", 0) > 0 or params.get("energy_profile"):
        prepare_pending_registry()

    run_dir = args.results_dir.expanduser() / args.run_id
    run_dir.mkdir(parents=True)
    write_metadata(args, run_dir, videos)

    node_names = alive_node_names()
    node_ips, idle_w = list(node_names), None
    if args.measure_energy:
        print(f"Measuring the idle power of {len(node_ips)} node(s) for {args.idle_seconds:g}s...")
        idle_by_ip = measure_idle_watts(node_ips, args.idle_seconds)
        if idle_by_ip is None:
            sys.exit("Cannot read the energy counters of every node: is the power exporter up?")
        idle_w = sum(idle_by_ip.values())
        print(f"Idle power of the cluster: {idle_w:.1f} W", flush=True)

    base = {"run_id": args.run_id, "use_case": args.use_case,
            "scheduler_params": args.scheduler_params}  # fmt: skip
    summaries = []
    with (
        open(run_dir / "steps.csv", "w", newline="") as steps_file,
        open(run_dir / "streams.csv", "w", newline="") as streams_file,
        open(run_dir / "frames.csv", "w", newline="") as frames_file,
    ):
        steps_csv = csv.DictWriter(steps_file, STEP_FIELDS, extrasaction="ignore")
        streams_csv = csv.DictWriter(streams_file, STREAM_FIELDS, extrasaction="ignore")
        frames_csv = csv.DictWriter(frames_file, FRAME_FIELDS, extrasaction="ignore")
        for writer in (steps_csv, streams_csv, frames_csv):
            writer.writeheader()

        for n_streams in args.steps:
            alive = len(alive_node_names())
            if alive < args.expected_nodes:
                print(f"ABORT: {alive} Ray nodes alive instead of {args.expected_nodes}")
                break
            print(f"=== {datetime.now():%H:%M:%S} {n_streams} stream(s)", flush=True)
            summary, streams = run_step(args, n_streams, videos, run_dir, node_ips, idle_w)
            names = {ip: node_names.get(ip, ip) for ip in node_ips}
            if summary.get("energy_by_node_j"):
                summary["energy_by_node_j"] = json.dumps(
                    {names[ip]: j for ip, j in json.loads(summary["energy_by_node_j"]).items()}
                )
            summaries.append(summary)
            steps_csv.writerow(dict(base, **summary))
            for stream in streams:
                streams_csv.writerow(dict(stream, run_id=args.run_id, n_streams=n_streams))
                for frame in stream["frames"]:
                    frames_csv.writerow(
                        dict(frame, n_streams=n_streams, stream_id=stream.get("stream_id"))
                    )
            for f in (steps_file, streams_file, frames_file):
                f.flush()

            p95 = summary["latency_p95_s"]
            print(
                f"    {summary['throughput_fps']:.1f}/{summary['offered_fps']:g} frames/s, "
                f"p95 latency {'n/a' if p95 is None else f'{p95:.2f} s'}, "
                f"{summary['failed_streams']} failed stream(s): "
                + ("sustainable" if summary["sustainable"] else summary["reasons"]),
                flush=True,
            )
            if not ramp_should_continue([s["sustainable"] for s in summaries], args.extra_steps):
                break
            idle_w = cooldown(args, node_ips, idle_w)

    print(f"\nCapacity: {capacity(summaries)} stream(s) at {args.fps:g} frame/s each")
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
