# Description: Summarise and plot the results of run_stress.py: capacity of every scheduler
# configuration, throughput and latency against the load, energy per frame, and where the
# streams were placed at a light and at a heavy load.

"""Summarise one or more run_stress.py runs and draw the stress-test charts.

  python scripts/analyze_stress.py ~/wadas_results/st_monorig_1 ~/wadas_results/st_monenergy_1
  python scripts/analyze_stress.py ~/wadas_results/st_* --out stress_charts

Charts (PNG, in --out):
  throughput.png   frames/s completed against frames/s offered, one line per configuration
  latency.png      95th percentile of the latency against the number of streams
  energy.png       joules above idle per completed frame against the number of streams
  placement.png    detection and classification devices of the streams, light and heavy step

The light step is the largest one at most half the capacity, the heavy step the capacity.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_run(run_dir):
    """Steps and streams of one run, with the label of its configuration."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    with open(run_dir / "steps.csv", newline="") as f:
        steps = list(csv.DictReader(f))
    with open(run_dir / "streams.csv", newline="") as f:
        streams = list(csv.DictReader(f))
    for step in steps:
        step["n_streams"] = int(step["n_streams"])
        step["sustainable"] = step["sustainable"] == "True"
        for key in ("offered_fps", "throughput_fps", "latency_p95_s", "j_per_frame"):
            step[key] = float(step[key]) if step.get(key) not in (None, "") else None
    args = meta["args"]
    return {"label": f"{args['use_case']} {args['scheduler_params']}", "name": run_dir.name,
            "steps": steps, "streams": streams}  # fmt: skip


def capacity(steps):
    """Largest number of streams sustainable before the first unsustainable step."""
    best = 0
    for step in steps:
        if not step["sustainable"]:
            break
        best = step["n_streams"]
    return best


def light_and_heavy(steps):
    """(light, heavy) numbers of streams: largest step <= capacity / 2, and the capacity."""
    heavy = capacity(steps)
    if not heavy:
        return None, None
    light = max((s["n_streams"] for s in steps if s["n_streams"] <= heavy / 2), default=None)
    return light, heavy


def placements(streams, n_streams, role):
    """Counter of node:device for one role ("det" or "cls") in the step with n_streams."""
    return Counter(
        f"{s[role + '_node']}:{s[role + '_device']}"
        for s in streams
        if int(s["n_streams"]) == n_streams and s.get(role + "_node")
    )


def print_table(runs):
    print(f"{'run':24s} {'capacity':>8s} {'light':>5s} {'p95 at cap.':>11s} {'J/frame':>8s}  "
          "configuration")  # fmt: skip
    for run in runs:
        light, heavy = light_and_heavy(run["steps"])
        at_cap = next((s for s in run["steps"] if s["n_streams"] == heavy), None)
        p95 = f"{at_cap['latency_p95_s']:.2f} s" if at_cap and at_cap["latency_p95_s"] else "-"
        jpf = f"{at_cap['j_per_frame']:.1f}" if at_cap and at_cap["j_per_frame"] else "-"
        print(f"{run['name']:24s} {heavy:>8d} {light or '-':>5} {p95:>11s} {jpf:>8s}  "
              f"{run['label']}")  # fmt: skip


def plot(runs, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    top = max(s["offered_fps"] for run in runs for s in run["steps"])
    ax.plot([0, top], [0, top], color="0.7", linestyle="--", label="offered = completed")
    for run in runs:
        steps = run["steps"]
        ax.plot([s["offered_fps"] for s in steps], [s["throughput_fps"] for s in steps],
                marker="o", label=run["name"])  # fmt: skip
    ax.set(xlabel="frames/s offered", ylabel="frames/s completed", title="Throughput")
    ax.legend()
    fig.savefig(out_dir / "throughput.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    for key, ylabel, name in (
        ("latency_p95_s", "p95 latency (s)", "latency.png"),
        ("j_per_frame", "J above idle per frame", "energy.png"),
    ):
        fig, ax = plt.subplots(figsize=(7, 5))
        for run in runs:
            points = [(s["n_streams"], s[key]) for s in run["steps"] if s[key] is not None]
            if points:
                ax.plot(*zip(*points), marker="o", label=run["name"])
        ax.set(xlabel="streams at 1 frame/s", ylabel=ylabel)
        ax.legend()
        fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(len(runs), 2, figsize=(11, 2.6 * len(runs)), squeeze=False)
    for row, run in zip(axes, runs):
        light, heavy = light_and_heavy(run["steps"])
        for ax, role, title in zip(row, ("det", "cls"), ("detection", "classification")):
            labels, counts = [], []
            for n in (light, heavy):
                if n:
                    labels.append(f"{n} streams")
                    counts.append(placements(run["streams"], n, role))
            keys = sorted({k for c in counts for k in c})
            left = [0] * len(counts)
            for key in keys:
                values = [c.get(key, 0) for c in counts]
                ax.barh(labels, values, left=left, label=key)
                left = [a + b for a, b in zip(left, values)]
            ax.set_title(f"{run['name']}: {title}")
            ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "placement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("runs", nargs="+", type=Path, help="run_stress.py result directories")
    parser.add_argument("--out", type=Path, default=Path("stress_charts"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    runs = [read_run(run) for run in args.runs]
    print_table(runs)
    if not args.no_plots:
        plot(runs, args.out)
        print(f"\nCharts in {args.out}")


if __name__ == "__main__":
    main()
