#!/usr/bin/env python3
# Description: Print, for every alive Ray node, the raw metrics read by the schedulers and the
# individual terms of each scheduler's score, to see what actually drives a placement decision.

"""Show how each scheduler scores every node, term by term.

Run it on the Ray head node, with the cluster and the per-node exporters up. It only reads
metrics: nothing is scheduled. Use it on an idle cluster to see the baseline, or while a load
is running to see how the scores react.

  python scripts/dump_scores.py
  python scripts/dump_scores.py --repeats 5 --interval 2     # mean of five readings
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from wadas.ai.schedulers import (  # noqa: E402
    CPU_POWER_WEIGHT,
    DEFAULT_GPU_UTILIZATION,
    GPU_POWER_WEIGHT,
    MEMORY_USAGE_GUARD,
    NPU_POWER_WEIGHT,
    Scheduler,
)

RAY_METRICS = {
    "ray_node_cpu_utilization",
    "ray_node_mem_total",
    "ray_node_mem_used",
    "ray_node_disk_io_read_speed",
    "ray_node_disk_io_write_speed",
}
NETWORK_METRICS = {
    "network_tx_bytes_per_second",
    "network_rx_bytes_per_second",
    "network_tx_capacity_bytes_per_second",
    "network_rx_capacity_bytes_per_second",
}
RAW_FIELDS = (
    "cpu",
    "gpu",
    "npu",
    "mem_used",
    "mem_total",
    "tx",
    "rx",
    "cap_tx",
    "cap_rx",
    "disk_read",
    "disk_write",
)


class Reader(Scheduler):
    """A scheduler that only exposes the metric reading of the base class."""

    def best_nodes_det_class(self):
        raise NotImplementedError


def read_node(reader, info):
    """Read the raw metrics of one node exactly as the schedulers do."""
    base = f"http://{info['ip']}"
    ray_metrics = reader.url_to_metric(f"{base}:{info['metric_port']}/metrics", RAY_METRICS)
    network = reader.url_to_metric(f"{base}:8004", NETWORK_METRICS)

    gpu = DEFAULT_GPU_UTILIZATION
    if info["resources"].get("GPU", 0) > 0:
        gpu = reader.url_to_metric(f"{base}:8002", {"gpu_utilization"}).get(
            "gpu_utilization", DEFAULT_GPU_UTILIZATION
        )
    npu = 0
    if info["resources"].get("NPU", 0) > 0:
        npu = reader.url_to_metric(f"{base}:8000", {"npu_utilization"}).get("npu_utilization", 0)

    return {
        "cpu": ray_metrics.get("ray_node_cpu_utilization", 0),
        "gpu": gpu,
        "npu": npu,
        "mem_used": ray_metrics.get("ray_node_mem_used", 0),
        "mem_total": ray_metrics.get("ray_node_mem_total", 0),
        "tx": network.get("network_tx_bytes_per_second", 0),
        "rx": network.get("network_rx_bytes_per_second", 0),
        "cap_tx": network.get("network_tx_capacity_bytes_per_second", 0),
        "cap_rx": network.get("network_rx_capacity_bytes_per_second", 0),
        "disk_read": ray_metrics.get("ray_node_disk_io_read_speed", 0),
        "disk_write": ray_metrics.get("ray_node_disk_io_write_speed", 0),
    }


def mean_readings(readings):
    return {field: sum(r[field] for r in readings) / len(readings) for field in RAW_FIELDS}


def score_terms(raw):
    """Break the score of the three schedulers into its terms, from the raw metrics."""
    mem_ratio = raw["mem_used"] / raw["mem_total"] if raw["mem_total"] else 0
    throughput = Scheduler.throughput_score(raw["tx"], raw["rx"], raw["cap_tx"], raw["cap_rx"])
    cpu_term = CPU_POWER_WEIGHT * raw["cpu"]
    gpu_term = GPU_POWER_WEIGHT * raw["gpu"]
    npu_term = NPU_POWER_WEIGHT * raw["npu"]
    power = cpu_term + gpu_term + npu_term
    watt = power + mem_ratio

    disk_total = raw["disk_read"] + raw["disk_write"]
    performance = (
        throughput if raw["disk_read"] == 0 else throughput + raw["disk_write"] / disk_total
    )
    return {
        "mem_ratio": mem_ratio,
        "cpu_term": cpu_term,
        "gpu_term": gpu_term,
        "npu_term": npu_term,
        "monitoring": watt,
        "throughput": throughput,
        "memory_ok": mem_ratio < MEMORY_USAGE_GUARD,
        "performance": performance,
        "optimum": performance / watt if watt else 0.0,
    }


def rank(values, reverse):
    """Return {key: position} where 1 is the node the scheduler would prefer."""
    ordered = sorted(values, key=values.get, reverse=reverse)
    return {key: position + 1 for position, key in enumerate(ordered)}


def print_table(title, header, rows):
    print(f"\n{title}")
    widths = [max(len(str(row[i])) for row in [header, *rows]) for i in range(len(header))]
    for row in [header, *rows]:
        print("  " + "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--repeats", type=int, default=1, help="Readings to average")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between readings")
    args = parser.parse_args(argv)

    import ray

    ray.init(address="auto", logging_level="ERROR")
    reader = Reader()
    nodes = reader.find_nodes()
    hostnames = {n["NodeID"]: n["NodeManagerHostname"] for n in ray.nodes() if n["Alive"]}
    if not nodes:
        sys.exit("No alive Ray nodes found")

    readings = {node_id: [] for node_id in nodes}
    for i in range(args.repeats):
        for node_id, info in nodes.items():
            readings[node_id].append(read_node(reader, info))
        if i < args.repeats - 1:
            time.sleep(args.interval)

    raw = {nid: mean_readings(r) for nid, r in readings.items()}
    terms = {nid: score_terms(r) for nid, r in raw.items()}
    order = sorted(nodes, key=lambda nid: hostnames.get(nid, nid))
    name = lambda nid: hostnames.get(nid, nid[:8])  # noqa: E731

    print(f"{args.repeats} reading(s) per node; averages shown")
    print_table(
        "Raw metrics",
        ["node", "cpu%", "gpu%", "npu%", "mem used/total", "tx/cap_tx", "rx/cap_rx", "GPU", "NPU"],
        [
            [
                name(nid),
                f"{raw[nid]['cpu']:.2f}",
                f"{raw[nid]['gpu']:.1f}",
                f"{raw[nid]['npu']:.1f}",
                f"{terms[nid]['mem_ratio']:.3f}",
                f"{raw[nid]['tx'] / raw[nid]['cap_tx']:.1e}" if raw[nid]["cap_tx"] else "n/a",
                f"{raw[nid]['rx'] / raw[nid]['cap_rx']:.1e}" if raw[nid]["cap_rx"] else "n/a",
                int(nodes[nid]["resources"].get("GPU", 0)),
                int(nodes[nid]["resources"].get("NPU", 0)),
            ]
            for nid in order
        ],
    )

    monitoring = rank({n: terms[n]["monitoring"] for n in nodes}, reverse=False)
    print_table(
        "Resource-Aware (MONITORING): score = 3*cpu + 4*gpu + 2*npu + mem  (lowest wins)",
        ["node", "3*cpu", "4*gpu", "2*npu", "mem", "score", "rank"],
        [
            [
                name(nid),
                f"{terms[nid]['cpu_term']:.2f}",
                f"{terms[nid]['gpu_term']:.2f}",
                f"{terms[nid]['npu_term']:.2f}",
                f"{terms[nid]['mem_ratio']:.3f}",
                f"{terms[nid]['monitoring']:.3f}",
                monitoring[nid],
            ]
            for nid in order
        ],
    )

    actuator = rank(
        {n: terms[n]["throughput"] if terms[n]["memory_ok"] else -1 for n in nodes}, True
    )
    print_table(
        "Performance-Aware (ACTUATOR): score = network headroom, memory < 55 %  (highest wins)",
        ["node", "headroom", "memory ok", "rank"],
        [
            [
                name(nid),
                f"{terms[nid]['throughput']:.6f}",
                "yes" if terms[nid]["memory_ok"] else "NO",
                actuator[nid],
            ]
            for nid in order
        ],
    )

    optimum = rank({n: terms[n]["optimum"] for n in nodes}, reverse=True)
    print_table(
        "Trade-Off (CRITICAL): score = performance / (3*cpu + 4*gpu + 2*npu + mem)  (highest wins)",
        ["node", "performance", "watt", "score", "rank"],
        [
            [
                name(nid),
                f"{terms[nid]['performance']:.4f}",
                f"{terms[nid]['monitoring']:.3f}",
                f"{terms[nid]['optimum']:.4f}",
                optimum[nid],
            ]
            for nid in order
        ],
    )
    ray.shutdown()


if __name__ == "__main__":
    main()
