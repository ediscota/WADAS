# Description: Launch batches of staggered video tasks against the Ray cluster, one batch
# per scheduler use case, and collect the outcome into CSV files.

"""Launch staggered batches of run_video_task.py and collect the results as CSV.

Run it on the Ray head node, with the cluster and the per-node exporters already up.

  --n-tasks 1                              one task per use case (baseline)
  --n-tasks 5 --use-cases MONITORING       concurrent load on one scheduler
  --n-tasks 5 --repeats 3                  same load on every scheduler, in random order

Task i always uses video i (cycling through the sorted videos), so every use case is
compared on the same workload.
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
from statistics import median

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
    parser.add_argument("--n-tasks", type=int, default=1, help="Concurrent tasks per scenario")
    parser.add_argument("--stagger", type=float, default=3.0, help="Seconds between task launches")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cooldown", type=float, default=20.0, help="Seconds between scenarios")
    parser.add_argument("--task-timeout", type=float, default=1800.0, help="Seconds per task")
    parser.add_argument("--fps", type=int, default=1, help="Video frames sampled per second")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the scenario order")
    parser.add_argument("--results-dir", type=Path, default=Path.home() / "wadas_results")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    return parser.parse_args(argv)


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


def launch_task(args, video, use_case, index, task_dir):
    task_dir.mkdir(parents=True)
    cmd = [
        sys.executable,
        str(TASK_SCRIPT),
        "--video",
        str(video),
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


def run_scenario(args, videos, index, repeat, use_case, run_dir):
    scenario_dir = run_dir / f"{index:02d}_{use_case}_rep{repeat}"
    base = {
        "run_id": args.run_id,
        "scenario": index,
        "repeat": repeat,
        "use_case": use_case,
        "n_tasks": args.n_tasks,
        "stagger_s": args.stagger,
    }

    started = time.monotonic()
    launched = []
    for i in range(args.n_tasks):
        video = videos[i % len(videos)]
        task_dir = scenario_dir / f"task_{i:02d}"
        launched.append((i, video, task_dir, *launch_task(args, video, use_case, i, task_dir)))
        if i < args.n_tasks - 1:
            time.sleep(args.stagger)

    rows = []
    for i, video, task_dir, process, log, launched_at in launched:
        base_row = dict(base, task_id=i, video=video.name)
        rows.append(collect_task(process, log, launched_at, task_dir, args.task_timeout, base_row))

    n_ok = sum(1 for row in rows if row["status"] == "ok")
    scenario_row = dict(
        base, makespan_s=time.monotonic() - started, n_ok=n_ok, n_failed=len(rows) - n_ok
    )
    return rows, scenario_row


def print_summary(rows):
    by_use_case = defaultdict(list)
    for row in rows:
        by_use_case[row["use_case"]].append(row)

    for use_case, group in by_use_case.items():
        ok = [row for row in group if row["status"] == "ok"]
        print(f"\n{use_case}: {len(ok)}/{len(group)} tasks ok")
        if not ok:
            continue
        print(f"  median t_total_s: {median(row['t_total_s'] for row in ok):.1f}")
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
    if not videos:
        sys.exit(f"No .mp4 files found in {args.videos_dir}")

    run_dir = args.results_dir.expanduser() / args.run_id
    run_dir.mkdir(parents=True)
    write_metadata(args, run_dir, videos)

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
                f"[{index + 1}/{len(plan)}] {use_case} rep{repeat}: launching {args.n_tasks} "
                f"task(s), {args.stagger:g}s apart",
                flush=True,
            )
            rows, scenario_row = run_scenario(args, videos, index, repeat, use_case, run_dir)
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
