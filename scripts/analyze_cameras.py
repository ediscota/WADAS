# Description: Summarise and plot the results of run_camera_test.py: for every scheduler
# configuration, where WADAS placed its pipeline, up to how many cameras it keeps up, the latency
# of photos and clips and the energy used.

"""Summarise run_camera_test.py runs and draw the camera stress-test charts.

  python scripts/analyze_cameras.py ~/wadas_results/cam_*_A ~/wadas_results/cam_*_B --out charts

Charts (PNG, in --out):
  test_a_latency.png     photo p95 latency against the number of photo cameras (log scale)
  test_a_throughput.png  photos per second processed against photos per second sent
  test_b_latency.png     photo and clip p95 latency of every configuration, 5 + 5 cameras
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

NUMERIC = (
    "photo_cameras",
    "video_cameras",
    "photos_offered",
    "photos_done",
    "photo_p95_s",
    "clips_offered",
    "clips_done",
    "clip_p95_s",
    "backlog",
    "delay_growth_s_per_min",
    "energy_active_j",
    "power_active_w",
)


def number(value):
    return float(value) if value not in (None, "") else None


def read_run(run_dir):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    with open(run_dir / "steps.csv", newline="") as f:
        steps = list(csv.DictReader(f))
    for step in steps:
        for key in NUMERIC:
            step[key] = number(step.get(key))
        step["keeps_up"] = step["keeps_up"] == "True"
    window = float(meta["args"]["window"])
    items_file = run_dir / "items.csv"
    if items_file.exists():
        with open(items_file, newline="") as f:
            add_throughput(steps, list(csv.DictReader(f)), window, float(meta["args"]["warmup"]))
    return {"name": run_dir.name, "placement": meta.get("placement", {}), "steps": steps,
            "window": window}  # fmt: skip


def add_throughput(steps, items, window, warmup):
    """Photos/s and clips/min processed by the pipeline in each measured window.

    These count every item that *finished* inside the window, whenever it arrived. The
    "done" counts of steps.csv only include items that also *arrived* in the window: under
    overload those are reached late or never, so they shrink although the pipeline works at
    the same speed. The throughput is the processing rate; "done" is what the cameras got back.
    """
    for step in steps:
        rows = [r for r in items if r["step"] == step["step"]]
        step["photos_per_s"] = step["clips_per_min"] = None
        if not rows:
            continue
        measure_start = min(float(r["arrival"]) for r in rows) + warmup
        measure_end = measure_start + window

        def finished(kind, rows=rows, start=measure_start, end=measure_end):
            return sum(1 for r in rows if r["kind"] == kind and r["status"] == "ok"
                       and start <= float(r["end"]) <= end)  # fmt: skip

        step["photos_per_s"] = finished("photo") / window
        step["clips_per_min"] = finished("video") / window * 60


def max_cameras_kept_up(steps):
    """Largest number of photo cameras of a step that kept up, before the first that did not."""
    best = 0
    for step in steps:
        if not step["keeps_up"]:
            break
        best = int(step["photo_cameras"])
    return best


def by_cameras(steps, key):
    """{photo cameras: mean of key over the steps with that many cameras}."""
    groups = defaultdict(list)
    for step in steps:
        if step.get(key) is not None:
            groups[int(step["photo_cameras"])].append(step[key])
    return {n: mean(values) for n, values in sorted(groups.items())}


def placement_text(placement):
    if not placement:
        return "-"
    return (f"det {placement['det_node']}:{placement['det_device']}, "
            f"cls {placement['cls_node']}:{placement['cls_device']}")  # fmt: skip


def print_table(runs):
    for run in runs:
        print(f"\n{run['name']}  ({placement_text(run['placement'])})")
        print(f"  {'cams':>6s} {'photos/s':>8s} {'clips/min':>9s} {'photos':>11s} {'p95 photo':>9s} "
              f"{'clips':>7s} {'p95 clip':>8s} {'backlog':>7s} {'delay/min':>9s} "
              f"{'W above idle':>12s}  verdict")  # fmt: skip
        for s in run["steps"]:
            cams = f"{int(s['photo_cameras'])}+{int(s['video_cameras'])}"
            photos = (f"{int(s['photos_done'])}/{int(s['photos_offered'])}"
                      if s["photos_offered"] else "-")  # fmt: skip
            clips = f"{int(s['clips_done'])}/{int(s['clips_offered'])}" if s["clips_offered"] else "-"
            p95p = f"{s['photo_p95_s']:.1f} s" if s["photo_p95_s"] is not None else "-"
            p95c = f"{s['clip_p95_s']:.1f} s" if s["clip_p95_s"] is not None else "-"
            watts = f"{s['power_active_w']:.1f}" if s["power_active_w"] is not None else "-"
            verdict = "keeps up" if s["keeps_up"] else s["reasons"]
            rate = f"{s['photos_per_s']:.2f}" if s.get("photos_per_s") is not None else "-"
            clip_rate = f"{s['clips_per_min']:.1f}" if s.get("clips_per_min") else "-"
            print(f"  {cams:>6s} {rate:>8s} {clip_rate:>9s} {photos:>11s} {p95p:>9s} {clips:>7s} "
                  f"{p95c:>8s} "
                  f"{int(s['backlog']):>7d} {s['delay_growth_s_per_min']:>+9.1f} {watts:>12s}  "
                  f"{verdict}")  # fmt: skip
        if not any(s["video_cameras"] for s in run["steps"]):
            print(f"  keeps up with at most {max_cameras_kept_up(run['steps'])} photo camera(s)")


def plot(runs, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    runs_a = [r for r in runs if not any(s["video_cameras"] for s in r["steps"])]
    runs_b = [r for r in runs if any(s["video_cameras"] for s in r["steps"])]

    if runs_a:
        fig, ax = plt.subplots(figsize=(7, 5))
        for run in runs_a:
            points = by_cameras(run["steps"], "photo_p95_s")
            ax.plot(list(points), list(points.values()), marker="o", label=run["name"])
        ax.axhline(2.0, color="0.6", linestyle="--", label="2 s")
        ax.set(xlabel="photo cameras at 1 photo/s", ylabel="photo p95 latency (s)", yscale="log")
        ax.legend()
        fig.savefig(out_dir / "test_a_latency.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        top = max(int(s["photo_cameras"]) for r in runs_a for s in r["steps"])
        ax.plot([0, top], [0, top], color="0.7", linestyle="--", label="sent = processed")
        for run in runs_a:
            done = by_cameras(run["steps"], "photos_per_s")
            ax.plot(list(done), list(done.values()), marker="o", label=run["name"])
        ax.set(xlabel="photos/s sent", ylabel="photos/s processed")
        ax.legend()
        fig.savefig(out_dir / "test_a_throughput.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if runs_b:
        fig, ax = plt.subplots(figsize=(8, 5))
        names = [r["name"] for r in runs_b]
        photo = [mean([s["photo_p95_s"] for s in r["steps"] if s["photo_p95_s"] is not None] or [0])
                 for r in runs_b]  # fmt: skip
        clip = [mean([s["clip_p95_s"] for s in r["steps"] if s["clip_p95_s"] is not None] or [0])
                for r in runs_b]  # fmt: skip
        x = range(len(names))
        ax.bar([i - 0.2 for i in x], photo, width=0.4, label="photo p95")
        ax.bar([i + 0.2 for i in x], clip, width=0.4, label="clip p95")
        ax.set_xticks(list(x), names, rotation=30, ha="right")
        ax.set(ylabel="latency (s)", title="5 photo + 5 video cameras")
        ax.legend()
        fig.savefig(out_dir / "test_b_latency.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("runs", nargs="+", type=Path, help="run_camera_test.py result directories")
    parser.add_argument("--out", type=Path, default=Path("camera_charts"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    runs = [read_run(run) for run in args.runs]
    print_table(runs)
    if not args.no_plots:
        plot(runs, args.out)
        print(f"\nCharts in {args.out}")


if __name__ == "__main__":
    main()
