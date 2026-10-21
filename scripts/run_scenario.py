# Description: Launch batches of video and photo tasks against the Ray cluster, one batch
# per scheduler use case, and collect the outcome into CSV files.

"""Launch batches of run_video_task.py (videos and photos) and collect the results as CSV.

Run it on the Ray head node, with the cluster and the per-node exporters already up.

  --n-tasks 1                              one video per use case (baseline)
  --n-tasks 5 --use-cases MONITORING       concurrent load on one scheduler
  --n-tasks 5 --repeats 3                  same load on every scheduler, in random order
  --n-videos 10 --n-photos 10 --stagger 0 --repeats 10
                                           10 videos and 10 photos arriving together

Task i of each kind always uses media i (cycling through the sorted files), so every use
case is compared on the same workload. Photos come from --photos-dir (see extract_photos.py).
"""

import argparse
import csv
import json
import platform
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from importlib import metadata
from pathlib import Path
from statistics import mean, median, stdev

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_SCRIPT = Path(__file__).resolve().with_name("run_video_task.py")
USE_CASES = ("MONITORING", "ACTUATOR", "CRITICAL")

TASK_FIELDS = [
    "run_id",
    "scenario",
    "repeat",
    "use_case",
    "n_tasks",
    "stagger_s",
    "task_id",
    "kind",
    "video",
    "status",
    "error",
    "returncode",
    "driver_host",
    "started_at",
    "det_node",
    "det_device",
    "cls_node",
    "cls_device",
    "n_frames",
    "n_frames_with_animals",
    "t_init_s",
    "t_detection_phase_s",
    "t_classification_phase_s",
    "t_total_s",
    "det_first_call_s",
    "det_median_s",
    "cls_median_s",
]
SCENARIO_FIELDS = [
    "run_id",
    "scenario",
    "repeat",
    "use_case",
    "n_tasks",
    "stagger_s",
    "makespan_s",
    "n_ok",
    "n_failed",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--videos-dir", type=Path, default=Path.home() / "VIDEO_TEST")
    parser.add_argument("--use-cases", nargs="+", choices=USE_CASES, default=list(USE_CASES))
    parser.add_argument("--photos-dir", type=Path, default=Path.home() / "PHOTO_TEST")
    parser.add_argument(
        "--n-tasks", type=int, default=None, help="Video tasks, if --n-videos is unset"
    )
    parser.add_argument("--n-videos", type=int, default=None, help="Video tasks per scenario")
    parser.add_argument("--n-photos", type=int, default=0, help="Photo tasks per scenario")
    parser.add_argument(
        "--stagger", type=float, default=3.0, help="Seconds between launches (0 = all together)"
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cooldown", type=float, default=20.0, help="Seconds between scenarios")
    parser.add_argument("--task-timeout", type=float, default=1800.0, help="Seconds per task")
    parser.add_argument("--fps", type=int, default=1, help="Video frames sampled per second")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the scenario order")
    parser.add_argument("--results-dir", type=Path, default=Path.home() / "wadas_results")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args(argv)
    if args.n_videos is None:
        # --n-tasks keeps meaning "videos"; asking only for photos must not add a video.
        args.n_videos = args.n_tasks if args.n_tasks is not None else (0 if args.n_photos else 1)
    if args.n_videos < 0 or args.n_photos < 0 or args.n_videos + args.n_photos == 0:
        parser.error("need at least one video or photo task")
    return args


def build_tasks(videos, photos, n_videos, n_photos):
    """Return [(kind, file)]: n_videos videos then n_photos photos, cycling through the files."""
    tasks = [("video", videos[i % len(videos)]) for i in range(n_videos)]
    tasks += [("photo", photos[i % len(photos)]) for i in range(n_photos)]
    return tasks


def plan_scenarios(use_cases, repeats, seed):
    """Return [(repeat, use_case)], shuffling the use case order within each repeat."""
    plan = []
    for repeat in range(repeats):
        order = list(use_cases)
        random.Random(seed + repeat).shuffle(order)
        plan.extend((repeat, use_case) for use_case in order)
    return plan


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


def write_metadata(args, run_dir, videos, photos):
    info = {
        "args": {key: str(value) for key, value in vars(args).items()},
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "ray": package_version("ray"),
        "openvino": package_version("openvino"),
        "host": platform.node(),
        "videos": [video.name for video in videos],
        "photos": [photo.name for photo in photos],
    }
    (run_dir / "meta.json").write_text(json.dumps(info, indent=2))


def launch_task(args, kind, media, use_case, index, task_dir):
    task_dir.mkdir(parents=True)
    cmd = [
        sys.executable,
        str(TASK_SCRIPT),
        "--video",
        str(media),
        "--kind",
        kind,
        "--use-case",
        use_case,
        "--task-id",
        str(index),
        "--out-dir",
        str(task_dir),
        "--fps",
        str(args.fps),
    ]
    log = open(task_dir / "log.txt", "w")
    process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    return process, log, time.monotonic()


def collect_task(process, log, launched_at, task_dir, timeout, base_row):
    timed_out = False
    try:
        process.wait(timeout=max(0.0, launched_at + timeout - time.monotonic()))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        timed_out = True
    finally:
        log.close()

    row = dict(base_row, status="timeout" if timed_out else "crashed")
    result_file = task_dir / "result.json"
    if result_file.exists():
        row.update(json.loads(result_file.read_text()))
    if timed_out:
        row["status"] = "timeout"
    row["returncode"] = process.returncode
    return row


def run_scenario(args, tasks, index, repeat, use_case, run_dir):
    scenario_dir = run_dir / f"{index:02d}_{use_case}_rep{repeat}"
    base = {
        "run_id": args.run_id,
        "scenario": index,
        "repeat": repeat,
        "use_case": use_case,
        "n_tasks": len(tasks),
        "stagger_s": args.stagger,
    }

    started = time.monotonic()
    launched = []
    for i, (kind, media) in enumerate(tasks):
        task_dir = scenario_dir / f"task_{i:02d}"
        process, log, launched_at = launch_task(args, kind, media, use_case, i, task_dir)
        launched.append((i, kind, media, task_dir, process, log, launched_at))
        if i < len(tasks) - 1 and args.stagger > 0:
            time.sleep(args.stagger)

    rows = []
    for i, kind, media, task_dir, process, log, launched_at in launched:
        base_row = dict(base, task_id=i, kind=kind, video=media.name)
        rows.append(collect_task(process, log, launched_at, task_dir, args.task_timeout, base_row))

    n_ok = sum(1 for row in rows if row["status"] == "ok")
    scenario_row = dict(
        base, makespan_s=time.monotonic() - started, n_ok=n_ok, n_failed=len(rows) - n_ok
    )
    return rows, scenario_row


def print_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["use_case"], row["kind"])].append(row)

    for (use_case, kind), group in sorted(groups.items()):
        ok = [row for row in group if row["status"] == "ok"]
        print(f"\n{use_case} {kind}: {len(ok)}/{len(group)} tasks ok")
        if not ok:
            continue
        times = [row["t_total_s"] for row in ok]
        spread = f", std {stdev(times):.1f}" if len(times) > 1 else ""
        print(f"  t_total_s: mean {mean(times):.1f}, median {median(times):.1f}{spread}")
        for label, node_key, device_key in (
            ("detection", "det_node", "det_device"),
            ("classification", "cls_node", "cls_device"),
        ):
            placement = Counter((row[node_key], row[device_key]) for row in ok)
            counts = ", ".join(f"{n}/{d} x{c}" for (n, d), c in placement.most_common())
            print(f"  {label} placement: {counts}")


def main(argv=None):
    args = parse_args(argv)
    videos = sorted(args.videos_dir.expanduser().glob("*.mp4"))
    if args.n_videos and not videos:
        sys.exit(f"No .mp4 files found in {args.videos_dir}")
    photos = sorted(
        path
        for pattern in ("*.jpg", "*.jpeg", "*.png")
        for path in args.photos_dir.expanduser().glob(pattern)
    )
    if args.n_photos and not photos:
        sys.exit(f"No photos found in {args.photos_dir}: run scripts/extract_photos.py first")
    tasks = build_tasks(videos, photos, args.n_videos, args.n_photos)

    run_dir = args.results_dir.expanduser() / args.run_id
    run_dir.mkdir(parents=True)
    write_metadata(args, run_dir, videos, photos)

    plan = plan_scenarios(args.use_cases, args.repeats, args.seed)
    all_rows = []
    with (
        open(run_dir / "tasks.csv", "w", newline="") as tasks_file,
        open(run_dir / "scenarios.csv", "w", newline="") as scenarios_file,
    ):
        tasks_csv = csv.DictWriter(tasks_file, TASK_FIELDS, extrasaction="ignore")
        scenarios_csv = csv.DictWriter(scenarios_file, SCENARIO_FIELDS, extrasaction="ignore")
        tasks_csv.writeheader()
        scenarios_csv.writeheader()

        for index, (repeat, use_case) in enumerate(plan):
            print(
                f"[{index + 1}/{len(plan)}] {use_case} rep{repeat}: launching "
                f"{args.n_videos} video(s) + {args.n_photos} photo(s), {args.stagger:g}s apart",
                flush=True,
            )
            rows, scenario_row = run_scenario(args, tasks, index, repeat, use_case, run_dir)
            tasks_csv.writerows(rows)
            scenarios_csv.writerow(scenario_row)
            tasks_file.flush()
            scenarios_file.flush()
            all_rows.extend(rows)
            print(
                f"    done in {scenario_row['makespan_s']:.1f}s: "
                f"{scenario_row['n_ok']} ok, {scenario_row['n_failed']} failed",
                flush=True,
            )
            if index < len(plan) - 1:
                time.sleep(args.cooldown)

    print_summary(all_rows)
    print(f"\nResults: {run_dir}")


if __name__ == "__main__":
    main()
