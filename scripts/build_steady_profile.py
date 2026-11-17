# Description: Build the steady-state profile (time and energy per processed item, with the model
# already loaded) of every node and device from pinned run_stress.py runs.

"""Build the steady-state profile from pinned stress runs (see run_steady_profile.sh).

Every run pins detection on one node and device and classification on a different node, and sends
one stream at a fixed rate. For each run, inside its measured window:

  time per item      median of the detection (or classification) time of the frames
  energy per item    energy of the node above its idle power, divided by the items it processed

Detection and classification run on different nodes, so the energy of each node belongs to one
role. Of the runs of a device and role, the one that processed the most items is kept: with few
items the energy above idle is close to the measurement noise.

  python scripts/build_steady_profile.py ~/wadas_results/sp_* --out steady_profile.json
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import median

ROLES = (("DETECTION", "detection", "det_s"), ("CLASSIFICATION", "classification", "cls_s"))


def read_run(run_dir):
    """Placement, window, per-node energy and frames of a one-step pinned run."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    params = json.loads(meta["args"]["scheduler_params"])
    with open(run_dir / "steps.csv", newline="") as f:
        steps = list(csv.DictReader(f))
    if len(steps) != 1:
        raise ValueError(f"{run_dir}: expected one step, found {len(steps)}")
    step = steps[0]
    with open(run_dir / "frames.csv", newline="") as f:
        frames = [
            {key: float(row[key] or 0) for key in ("arrival", "start", "end", "det_s", "cls_s")}
            | {"n_animals": int(row["n_animals"] or 0)}
            for row in csv.DictReader(f)
        ]
    return {
        "name": run_dir.name,
        "placement": {"detection": params["detection"], "classification": params["classification"]},
        "fps": float(meta["args"]["fps"]),
        "driver_host": meta.get("host"),
        "window": float(meta["args"]["window"]),
        "measure_start": float(step["measure_start"]),
        "measure_end": float(step["measure_end"]),
        "energy_by_node_j": json.loads(step["energy_by_node_j"] or "{}"),
        "idle_by_node_w": json.loads(step["idle_by_node_w"] or "{}"),
        "frames": frames,
    }


def role_measures(run, role, time_key):
    """Items, time per item and energy per item of one role in one run."""
    host, _, device = run["placement"][role].partition(":")
    inside = [f for f in run["frames"] if run["measure_start"] <= f["end"] <= run["measure_end"]]
    items = inside if role == "detection" else [f for f in inside if f["n_animals"] > 0]
    times = [f[time_key] for f in items if f[time_key] > 0]
    joules = None
    if items and host in run["energy_by_node_j"] and host in run["idle_by_node_w"]:
        active = run["energy_by_node_j"][host] - run["idle_by_node_w"][host] * run["window"]
        joules = active / len(items)
    return {
        "key": f"{host}:{device}",
        "items": len(items),
        "s": median(times) if times else None,
        "J": joules,
        "busy": sum(times) / run["window"],
        "fps": run["fps"],
        "run": run["name"],
        "shares_driver_node": host == run["driver_host"],
    }


def build_profile(runs):
    """{"host:DEVICE": {"DETECTION": {...}, "CLASSIFICATION": {...}}} keeping, for every device and
    role, the run that processed the most items."""
    best = {}
    for run in runs:
        for role_name, role, time_key in ROLES:
            measures = role_measures(run, role, time_key)
            if not measures["items"]:
                continue
            slot = (measures["key"], role_name)
            if slot not in best or measures["items"] > best[slot]["items"]:
                best[slot] = measures
    profile = {}
    for (key, role_name), measures in sorted(best.items()):
        entry = {k: v for k, v in measures.items() if k != "key"}
        entry["s"] = round(entry["s"], 4) if entry["s"] is not None else None
        entry["J"] = round(entry["J"], 3) if entry["J"] is not None else None
        entry["busy"] = round(entry["busy"], 3)
        profile.setdefault(key, {})[role_name] = entry
    return profile


def print_profile(profile):
    print(f"{'device':16s} {'role':15s} {'s/item':>7s} {'J/item':>7s} {'items':>6s} {'busy':>5s} "
          f"{'fps':>4s}  note")  # fmt: skip
    for key, roles in profile.items():
        for role, m in roles.items():
            s = f"{m['s']:.3f}" if m["s"] is not None else "-"
            j = f"{m['J']:.2f}" if m["J"] is not None else "-"
            note = "node also runs the test driver" if m["shares_driver_node"] else ""
            print(f"{key:16s} {role:15s} {s:>7s} {j:>7s} {m['items']:>6d} {m['busy']:>5.0%} "
                  f"{m['fps']:>4g}  {note}")  # fmt: skip


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("runs", nargs="+", type=Path, help="pinned run_stress.py result directories")
    parser.add_argument("--out", type=Path, help="Write the profile as JSON here")
    args = parser.parse_args(argv)
    profile = build_profile([read_run(run) for run in args.runs])
    print_profile(profile)
    if args.out:
        args.out.write_text(json.dumps({"steady": profile}, indent=2))
        print(f"\nProfile written to {args.out}")


if __name__ == "__main__":
    main()
