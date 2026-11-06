# This file is part of WADAS project.
#
# WADAS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# WADAS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with WADAS. If not, see <https://www.gnu.org/licenses/>.
#
# Description: Ray cluster schedulers used to pick the best node/device for
# detection and classification model placement, based on live node metrics
# exposed by Ray and by the per-node exporters in wadas.ai.gpu_exporter,
# wadas.ai.npu_exporter and wadas.ai.network_exporter.

import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

import ray
import requests

from wadas.ai.pending_load import PendingLoad, read_pending, reserve

logger = logging.getLogger(__name__)

# Correspondence between the scheduler class, the use-case string used in
# wadas.ai.pipeline.USECASE_TO_SCHEDULER, and the thesis-era naming, for
# anyone cross-referencing the original thesis material:
#   MonitoringScheduler <-> "MONITORING" <-> "Resource-oriented"
#   ActuatorScheduler   <-> "ACTUATOR"   <-> "Performance-oriented"
#   OptimumScheduler    <-> "CRITICAL"   <-> "Trade-off"

# Power model coefficients (watt-equivalent weight per device type), used by
# schedulers that need to estimate power consumption.
CPU_POWER_WEIGHT = 3
GPU_POWER_WEIGHT = 4
NPU_POWER_WEIGHT = 2

# Relative importance of the two throughput components used by schedulers
# that optimize for network throughput (upload vs download bandwidth).
ALPHA = 0.7
BETA = 0.3

# GPU utilization percentage assumed for nodes that do not expose GPU metrics
# (i.e. the node has no GPU): keeps GPU-less nodes from being favored purely
# because "no data" would otherwise look like "0% busy".
DEFAULT_GPU_UTILIZATION = 21

# Node memory usage above this fraction is considered saturated and the node
# is excluded from best-node selection where a memory guard is applied.
MEMORY_USAGE_GUARD = 0.55

# GPU utilization above this percentage is considered saturated.
GPU_UTILIZATION_GUARD = 75.0

# How long a placement decision keeps counting as pending load (seconds): the models it
# creates only show up in the node metrics a few seconds after the decision.
DEFAULT_PENDING_TTL = 10.0

# How long a placement keeps counting against the capacity of a node (seconds). A model stays
# in memory for as long as its pipeline runs, much longer than the metrics take to show it.
DEFAULT_CAPACITY_TTL = 60.0

# Disk throughput (bytes/s) taken as "fully busy" by the "headroom" disk term. It is an
# assumption of the order of a SATA SSD: no disk capacity is measured on the nodes.
DEFAULT_DISK_REFERENCE_BPS = 500e6
DISK_TERMS = ("legacy", "off", "headroom")


class Scheduler(ABC):
    """Base class for Ray cluster node schedulers.

    Subclasses implement `_best_node()` to select, among the nodes of the Ray
    cluster, the best node/device pair for one model; `best_nodes_det_class()`
    uses it for the detection and for the classification model.

    `pending_penalty` (0 = off) makes a scheduler count the placements decided in the
    last `pending_ttl` seconds, which the node metrics do not show yet: each of them
    worsens the score of its node by `pending_penalty`, in the unit of that scheduler's
    score, and a node whose NPU was just assigned no longer counts as having a free NPU.
    """

    def __init__(
        self,
        save_scores: bool = False,
        scores_path: str | Path | None = None,
        pending_penalty: float = 0.0,
        pending_ttl: float = DEFAULT_PENDING_TTL,
        capacity_ttl: float = DEFAULT_CAPACITY_TTL,
    ):
        self.save_scores = save_scores
        self.scores_path = Path(scores_path) if scores_path else None
        self.pending_penalty = pending_penalty
        self.pending_ttl = pending_ttl
        self.capacity_ttl = capacity_ttl
        self._pending = PendingLoad()
        self._metrics_cache = {}

    def _metrics(self, node_id, info):
        """Node metrics read once per decision: the same reading serves detection and
        classification, and it is taken before the pending load is read (see below)."""
        if node_id not in self._metrics_cache:
            self._metrics_cache[node_id] = self._node_metrics(node_id, info)
        return self._metrics_cache[node_id]

    def save_score(self, node_id, score, model):
        """Append a node score to the scores file, if score saving is enabled."""
        if not self.save_scores:
            return
        if not self.scores_path:
            logger.warning("Score saving enabled but no scores_path configured; skipping.")
            return
        with open(self.scores_path, "a") as f:
            f.write("-------SCORE-------\n")
            f.write(f"{model}\n")
            f.write(f"Node {node_id} score = {score}\n")

    def save_best_score(self, node_id, best_score, model):
        """Append the best node score to the scores file, if score saving is enabled."""
        if not self.save_scores:
            return
        if not self.scores_path:
            logger.warning("Score saving enabled but no scores_path configured; skipping.")
            return
        with open(self.scores_path, "a") as f:
            f.write("-------BEST SCORE-------\n")
            f.write(f"{model}\n")
            f.write(f"Chosen node {node_id} score = {best_score}\n")

    def find_nodes(self):
        """Scan the Ray cluster and return per-node IP, metrics port and resources.

        Dead nodes are excluded: ray.nodes() keeps recently-departed nodes in
        its listing (with stale IP/metrics), which would otherwise leave them
        eligible for selection until Ray eventually forgets them.
        """
        nodes = {}
        for n in ray.nodes():
            if not n["Alive"]:
                continue
            node_id = n["NodeID"]
            nodes[node_id] = {
                "ip": n["NodeManagerAddress"],
                "hostname": n.get("NodeManagerHostname"),
                "metric_port": n["MetricsExportPort"],
                "resources": n["Resources"],
            }
            logger.debug(
                "Node %s at %s, resources: %s", node_id, nodes[node_id]["ip"], n["Resources"]
            )
        return nodes

    def url_to_metric(self, url, target_metrics):
        """Fetch a Prometheus metrics endpoint and extract the target metrics."""
        metrics = {}
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            metrics_text = response.text
        except requests.RequestException as e:
            logger.warning("Failed to fetch metrics from %s: %s", url, e)
            return metrics

        for line in metrics_text.splitlines():
            if line.startswith("#") or "{" not in line or "}" not in line:
                continue

            metric_match = re.match(r"^([a-zA-Z0-9_]+)\{", line)
            if not metric_match:
                continue

            metric_name = metric_match.group(1)
            if metric_name not in target_metrics:
                continue

            value_match = re.search(r"}\s+([0-9.eE+-]+)", line)
            if value_match:
                metrics[metric_name] = float(value_match.group(1)) if value_match.group(1) else 0

        return metrics

    @staticmethod
    def throughput_score(send_speed, receive_speed, capacity_tx, capacity_rx):
        """Weighted "headroom" score in [0, 1]: 1 = idle link, 0 = at/over capacity.

        Ratios are clamped to [0, 1] before combining, so a momentary reading
        where actual traffic exceeds the estimated capacity (the capacity
        comes from a possibly stale/approximate link-speed reading) can never
        push the score negative.
        """
        send_ratio = min(max(send_speed / capacity_tx, 0), 1) if capacity_tx else 1
        receive_ratio = min(max(receive_speed / capacity_rx, 0), 1) if capacity_rx else 1
        return ALPHA * (1 - send_ratio) + BETA * (1 - receive_ratio)

    def pending_on_node(self, node_id):
        """Placements decided recently on this node, weighted by `pending_penalty`."""
        return self.pending_penalty * self._pending.on_node(node_id, max_age=self.pending_ttl)

    def npu_is_pending(self, node_id):
        """Whether a model was just assigned to the NPU of this node."""
        return self._pending.on_device(node_id, "NPU", max_age=self.pending_ttl) > 0

    def _uses_pending(self):
        """Whether the decisions read and record the pending load."""
        return self.pending_penalty > 0

    def _reservation_ttl(self):
        """How long a decision stays in the record."""
        return self.pending_ttl

    @abstractmethod
    def _best_node(self, nodes, role):
        """Return {"id_node": <ray node id>, "device": "CPU"|"GPU"|"NPU"} for one model."""

    def best_nodes_det_class(self):
        """Return (best_detection_node, best_classification_node), each a dict as above."""
        nodes = self.find_nodes()
        self._metrics_cache = {}
        if self._uses_pending():
            # Reading the metrics takes a few hundred ms. The pending load is read only after
            # it, and the decision is recorded right after taking it: requests arriving
            # together would otherwise all read the same empty record.
            for node_id, info in nodes.items():
                self._metrics(node_id, info)
            self._pending = read_pending()
        best_detection = self._best_node(nodes, "DETECTION")
        best_classification = self._best_node(nodes, "CLASSIFICATION")
        if self._uses_pending():
            for node, role in (
                (best_detection, "DETECTION"),
                (best_classification, "CLASSIFICATION"),
            ):
                if node:
                    reserve(node["id_node"], node["device"], role, self._reservation_ttl())
        return best_detection, best_classification


def disk_score(read_speed, write_speed, mode="legacy", reference_bps=DEFAULT_DISK_REFERENCE_BPS):
    """Disk term added to the network score of the Trade-Off scheduler.

    "legacy" is the original term, write / (read + write), which is 0 when nothing is read and
    about 1 as soon as any read is seen next to heavier writes: a jump of the whole score for
    a negligible traffic, and a ratio that says nothing about how busy the disk is.
    "off" leaves the term out. "headroom" is 1 - busy / reference, between 0 (disk saturated)
    and 1 (disk idle), continuous like the network headroom.
    """
    if mode == "off":
        return 0.0
    if mode == "headroom":
        return max(0.0, 1.0 - (read_speed + write_speed) / reference_bps)
    if mode == "legacy":
        return write_speed / (read_speed + write_speed) if read_speed else 0.0
    raise ValueError(f"disk_term must be one of {DISK_TERMS}, got {mode!r}")


def load_energy_profile(profile):
    """Return {"host:DEVICE": {"DETECTION": joules, "CLASSIFICATION": joules}}, from a dict or
    from the path of a JSON file, or None."""
    if isinstance(profile, (str, Path)):
        path = Path(profile)
        if not path.is_absolute() and not path.exists():
            # Relative paths are relative to the repository: tasks change working directory
            path = Path(__file__).resolve().parents[2] / path
        return json.loads(path.read_text())
    return profile or None


def load_latency_profile(profile):
    """Return {"host:DEVICE": {"DETECTION": seconds, "CLASSIFICATION": seconds}}, from a dict
    or from the path of a JSON file, or None. A profile file that also holds the energy
    profile keeps the latencies in its "latency" section, which is what is returned."""
    loaded = load_energy_profile(profile)
    if loaded and "latency" in loaded:
        return loaded["latency"]
    return loaded


def candidate_devices(scheduler, node_id, info, role, npu_usage):
    """Devices a task of `role` may run on in this node: the NPU only accelerates detection,
    and only one task at a time."""
    devices = ["CPU"]
    if info["resources"].get("GPU", 0) > 0:
        devices.append("GPU")
    if (
        role == "DETECTION"
        and info["resources"].get("NPU", 0) > 0
        and npu_usage == 0
        and not scheduler.npu_is_pending(node_id)
    ):
        devices.append("NPU")
    return devices


class MonitoringScheduler(Scheduler):
    """Picks nodes minimizing estimated power consumption, favoring NPU/CPU
    over GPU: suited for continuous, low-power monitoring workloads.

    With `energy_profile` (the measured energy of one task on each node and device, see
    scripts/energy_profile.json) it picks the node/device where the task costs the least
    energy, instead of estimating power from the device loads. Use it with `pending_penalty`
    (here in joules): the cheapest device alone would receive every request of a burst.

    The profile may also hold a "capacity": {hostname: n}, the number of models a node can
    keep in memory. A node with that many placements in the last `capacity_ttl` seconds
    receives no more (unless no node is left), so the penalty can stay low without
    overloading the nodes with little memory.
    """

    def __init__(self, *args, energy_profile=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.energy_profile = load_energy_profile(energy_profile)
        self.capacity = (self.energy_profile or {}).get("capacity", {})

    def _uses_pending(self):
        return super()._uses_pending() or bool(self.capacity)

    def _reservation_ttl(self):
        # The capacity counts placements for longer than the penalty does
        if self.capacity:
            return max(self.pending_ttl, self.capacity_ttl)
        return super()._reservation_ttl()

    def _at_capacity(self, node_id, info):
        limit = self.capacity.get(info.get("hostname"))
        if limit is None:
            return False
        return self._pending.on_node(node_id, max_age=self.capacity_ttl) >= limit

    target_metrics = {
        "ray_node_cpu_utilization",
        "ray_node_mem_total",
        "ray_node_mem_used",
    }

    def _node_metrics(self, node_id, info):
        url = f"http://{info['ip']}:{info['metric_port']}/metrics"
        metrics = self.url_to_metric(url, self.target_metrics)
        cpu_usage = metrics.get("ray_node_cpu_utilization", 0)
        mem_total = metrics.get("ray_node_mem_total", 0)
        mem_used = metrics.get("ray_node_mem_used", 0)

        if info["resources"].get("GPU", 0) > 0:
            gpu_metrics = self.url_to_metric(f"http://{info['ip']}:8002", {"gpu_utilization"})
            gpu_usage = gpu_metrics.get("gpu_utilization", DEFAULT_GPU_UTILIZATION)
        else:
            gpu_usage = DEFAULT_GPU_UTILIZATION

        npu_usage = 0
        if info["resources"].get("NPU", 0) > 0:
            npu_metrics = self.url_to_metric(f"http://{info['ip']}:8000", {"npu_utilization"})
            npu_usage = npu_metrics.get("npu_utilization", 0)

        return cpu_usage, gpu_usage, npu_usage, mem_total, mem_used

    def _best_node(self, nodes, role):
        if self.energy_profile:
            if best_node := self._best_node_by_energy(nodes, role):
                return best_node
            # Every node is full: better an overloaded node than none
            if best_node := self._best_node_by_energy(nodes, role, respect_capacity=False):
                return best_node
            logger.warning("No node of the energy profile is alive: using the power estimate")
        return self._best_node_by_power(nodes, role)

    def _best_node_by_energy(self, nodes, role, respect_capacity=True):
        best_node = {}
        best_cost = float("inf")

        for node_id, info in nodes.items():
            if respect_capacity and self._at_capacity(node_id, info):
                continue
            _, _, npu_usage, _, _ = self._metrics(node_id, info)
            for device in candidate_devices(self, node_id, info, role, npu_usage):
                joules = self.energy_profile.get(f"{info.get('hostname')}:{device}", {}).get(role)
                if joules is None:
                    continue
                cost = joules + self.pending_on_node(node_id)
                if cost < best_cost:
                    best_cost = cost
                    best_node = {"id_node": node_id, "device": device}

        if best_node:
            self.save_best_score(best_node["id_node"], best_cost, role)
        return best_node

    def _best_node_by_power(self, nodes, role):
        best_node = {}
        best_score = float("inf")
        npu_selected = False

        for node_id, info in nodes.items():
            cpu_usage, gpu_usage, npu_usage, mem_total, mem_used = self._metrics(node_id, info)
            mem_ratio = (mem_used / mem_total) if mem_total else 0
            power = (
                CPU_POWER_WEIGHT * cpu_usage
                + GPU_POWER_WEIGHT * gpu_usage
                + NPU_POWER_WEIGHT * npu_usage
            )
            score = power + mem_ratio + self.pending_on_node(node_id)

            # The NPU only accelerates detection: classification is GPU/CPU only.
            npu_free = (
                role == "DETECTION"
                and info["resources"].get("NPU", 0) > 0
                and npu_usage == 0
                and not self.npu_is_pending(node_id)
            )

            if npu_free:
                if not npu_selected or score <= best_score:
                    best_node = {"id_node": node_id, "device": "NPU"}
                    best_score = score
                    npu_selected = True
            elif not npu_selected and score <= best_score:
                best_score = score
                if gpu_usage <= cpu_usage and info["resources"].get("GPU", 0) > 0:
                    best_node = {"id_node": node_id, "device": "GPU"}
                else:
                    best_node = {"id_node": node_id, "device": "CPU"}

            self.save_score(node_id, score, role)

        if best_node:
            self.save_best_score(best_node["id_node"], best_score, role)
        return best_node


class ActuatorScheduler(Scheduler):
    """Picks nodes maximizing available network throughput, guarded by a
    memory-usage ceiling: suited for actuation workloads that must stream
    results with low latency.

    With `latency_profile` (the measured completion time of one task on each node and
    device, see scripts/energy_profile_mdv6b_dfv14.json) it picks instead the node/device
    where the task is expected to finish first, which is what low latency means; the
    memory ceiling still applies. Use it with `pending_penalty` (here in seconds, the
    expected delay each recent placement causes on its node). Without a profile the
    behaviour is the original one."""

    def __init__(self, *args, latency_profile=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.latency_profile = load_latency_profile(latency_profile)

    target_metrics = {
        "ray_node_disk_io_write_speed",
        "ray_node_disk_io_read_speed",
        "ray_node_mem_total",
        "ray_node_mem_used",
    }

    network_metrics = {
        "network_tx_bytes_per_second",
        "network_rx_bytes_per_second",
        "network_tx_capacity_bytes_per_second",
        "network_rx_capacity_bytes_per_second",
    }

    def _node_metrics(self, node_id, info):
        url = f"http://{info['ip']}:{info['metric_port']}/metrics"
        metrics = self.url_to_metric(url, self.target_metrics)
        mem_total = metrics.get("ray_node_mem_total", 0)
        mem_used = metrics.get("ray_node_mem_used", 0)

        network_metrics = self.url_to_metric(f"http://{info['ip']}:8004", self.network_metrics)
        send_speed = network_metrics.get("network_tx_bytes_per_second", 0)
        receive_speed = network_metrics.get("network_rx_bytes_per_second", 0)
        capacity_tx = network_metrics.get("network_tx_capacity_bytes_per_second", 0)
        capacity_rx = network_metrics.get("network_rx_capacity_bytes_per_second", 0)

        if info["resources"].get("GPU", 0) > 0:
            gpu_metrics = self.url_to_metric(f"http://{info['ip']}:8002", {"gpu_utilization"})
            gpu_usage = gpu_metrics.get("gpu_utilization", DEFAULT_GPU_UTILIZATION)
        else:
            gpu_usage = DEFAULT_GPU_UTILIZATION

        npu_usage = 0
        if info["resources"].get("NPU", 0) > 0:
            npu_metrics = self.url_to_metric(f"http://{info['ip']}:8000", {"npu_utilization"})
            npu_usage = npu_metrics.get("npu_utilization", 0)

        throughput = self.throughput_score(send_speed, receive_speed, capacity_tx, capacity_rx)
        mem_ratio = (mem_used / mem_total) if mem_total else 0

        return throughput, gpu_usage, npu_usage, mem_ratio

    def _best_node(self, nodes, role):
        if self.latency_profile:
            if best_node := self._best_node_by_latency(nodes, role):
                return best_node
            logger.warning("No node of the latency profile is usable: using the network score")
        return self._best_node_by_network(nodes, role)

    def _best_node_by_latency(self, nodes, role):
        best_node = {}
        best_cost = float("inf")

        for node_id, info in nodes.items():
            _, _, npu_usage, mem_ratio = self._metrics(node_id, info)
            if mem_ratio >= MEMORY_USAGE_GUARD:
                continue
            for device in candidate_devices(self, node_id, info, role, npu_usage):
                seconds = self.latency_profile.get(f"{info.get('hostname')}:{device}", {}).get(
                    role
                )
                if seconds is None:
                    continue
                cost = seconds + self.pending_on_node(node_id)
                if cost < best_cost:
                    best_cost = cost
                    best_node = {"id_node": node_id, "device": device}

        if best_node:
            self.save_best_score(best_node["id_node"], best_cost, role)
        return best_node

    def _best_node_by_network(self, nodes, role):
        best_node = {}
        # Pending placements can push the score below zero, which must not rule a node out.
        best_score = float("-inf")

        for node_id, info in nodes.items():
            score, gpu_usage, npu_usage, mem_ratio = self._metrics(node_id, info)
            score -= self.pending_on_node(node_id)

            if score >= best_score and mem_ratio < MEMORY_USAGE_GUARD:
                best_score = score
                npu_free = (
                    role == "DETECTION"
                    and info["resources"].get("NPU", 0) > 0
                    and npu_usage == 0
                    and not self.npu_is_pending(node_id)
                )
                if gpu_usage < GPU_UTILIZATION_GUARD and info["resources"].get("GPU", 0) > 0:
                    best_node = {"id_node": node_id, "device": "GPU"}
                elif npu_free:
                    best_node = {"id_node": node_id, "device": "NPU"}
                else:
                    best_node = {"id_node": node_id, "device": "CPU"}

            self.save_score(node_id, score, role)

        if best_node:
            self.save_best_score(best_node["id_node"], best_score, role)
        return best_node


class OptimumScheduler(Scheduler):
    """Picks nodes maximizing a performance-per-watt score: balances network
    throughput/disk I/O against estimated power consumption."""

    target_metrics = {
        "ray_node_cpu_utilization",
        "ray_node_mem_total",
        "ray_node_mem_used",
        "ray_node_disk_io_write_speed",
        "ray_node_disk_io_read_speed",
    }

    network_metrics = {
        "network_tx_bytes_per_second",
        "network_rx_bytes_per_second",
        "network_tx_capacity_bytes_per_second",
        "network_rx_capacity_bytes_per_second",
    }

    def __init__(
        self,
        *args,
        disk_term="legacy",
        disk_reference_bps=DEFAULT_DISK_REFERENCE_BPS,
        energy_profile=None,
        latency_profile=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if disk_term not in DISK_TERMS:
            raise ValueError(f"disk_term must be one of {DISK_TERMS}, got {disk_term!r}")
        self.disk_term = disk_term
        self.disk_reference_bps = disk_reference_bps
        self.energy_profile = load_energy_profile(energy_profile)
        self.latency_profile = load_latency_profile(latency_profile)
        if bool(self.energy_profile) != bool(self.latency_profile):
            raise ValueError("energy_profile and latency_profile must be given together")

    def _node_metrics(self, node_id, info):
        url = f"http://{info['ip']}:{info['metric_port']}/metrics"
        metrics = self.url_to_metric(url, self.target_metrics)
        cpu_usage = metrics.get("ray_node_cpu_utilization", 0)
        mem_total = metrics.get("ray_node_mem_total", 0)
        mem_used = metrics.get("ray_node_mem_used", 0)
        write_speed = metrics.get("ray_node_disk_io_write_speed", 0)
        read_speed = metrics.get("ray_node_disk_io_read_speed", 0)

        network_metrics = self.url_to_metric(f"http://{info['ip']}:8004", self.network_metrics)
        send_speed = network_metrics.get("network_tx_bytes_per_second", 0)
        receive_speed = network_metrics.get("network_rx_bytes_per_second", 0)
        capacity_tx = network_metrics.get("network_tx_capacity_bytes_per_second", 0)
        capacity_rx = network_metrics.get("network_rx_capacity_bytes_per_second", 0)

        if info["resources"].get("GPU", 0) > 0:
            gpu_metrics = self.url_to_metric(f"http://{info['ip']}:8002", {"gpu_utilization"})
            gpu_usage = gpu_metrics.get("gpu_utilization", DEFAULT_GPU_UTILIZATION)
        else:
            gpu_usage = DEFAULT_GPU_UTILIZATION

        npu_usage = 0
        if info["resources"].get("NPU", 0) > 0:
            npu_metrics = self.url_to_metric(f"http://{info['ip']}:8000", {"npu_utilization"})
            npu_usage = npu_metrics.get("npu_utilization", 0)

        throughput = self.throughput_score(send_speed, receive_speed, capacity_tx, capacity_rx)
        performance = throughput + disk_score(
            read_speed, write_speed, self.disk_term, self.disk_reference_bps
        )

        power = (
            CPU_POWER_WEIGHT * cpu_usage
            + GPU_POWER_WEIGHT * gpu_usage
            + NPU_POWER_WEIGHT * npu_usage
        )
        watt = power + ((mem_used / mem_total) if mem_total else 0)

        return performance, watt, cpu_usage, gpu_usage, npu_usage

    def _best_node(self, nodes, role):
        if self.energy_profile and self.latency_profile:
            if best_node := self._best_node_by_energy_delay(nodes, role):
                return best_node
            logger.warning("No node of the profiles is usable: using the performance/watt score")
        return self._best_node_by_ratio(nodes, role)

    def _best_node_by_energy_delay(self, nodes, role):
        """Minimize energy x time, the classic trade-off between the two: `pending_penalty`
        is in seconds, added to the expected time of the node."""
        best_node = {}
        best_cost = float("inf")

        for node_id, info in nodes.items():
            _, _, cpu_usage, gpu_usage, npu_usage = self._metrics(node_id, info)
            for device in candidate_devices(self, node_id, info, role, npu_usage):
                key = f"{info.get('hostname')}:{device}"
                joules = self.energy_profile.get(key, {}).get(role)
                seconds = self.latency_profile.get(key, {}).get(role)
                if joules is None or seconds is None:
                    continue
                cost = joules * (seconds + self.pending_on_node(node_id))
                if cost < best_cost:
                    best_cost = cost
                    best_node = {"id_node": node_id, "device": device}

        if best_node:
            self.save_best_score(best_node["id_node"], best_cost, role)
        return best_node

    def _best_node_by_ratio(self, nodes, role):
        best_node = {}
        best_score = 0.0

        for node_id, info in nodes.items():
            performance, watt, cpu_usage, gpu_usage, npu_usage = self._metrics(node_id, info)
            watt += self.pending_on_node(node_id)
            score = performance / watt if watt else 0

            if score >= best_score:
                best_score = score
                has_gpu = info["resources"].get("GPU", 0) > 0
                has_npu = info["resources"].get("NPU", 0) > 0

                if (
                    role == "DETECTION"
                    and has_npu
                    and npu_usage == 0
                    and not self.npu_is_pending(node_id)
                ):
                    best_device = "NPU"
                elif has_gpu:
                    usage_by_device = {"CPU": cpu_usage, "GPU": gpu_usage}
                    best_device = min(usage_by_device, key=usage_by_device.get)
                else:
                    best_device = "CPU"
                best_node = {"id_node": node_id, "device": best_device}

            self.save_score(node_id, score, role)

        if best_node:
            self.save_best_score(best_node["id_node"], best_score, role)
        return best_node


class PinnedScheduler(Scheduler):
    """Always picks the node and device it is told to: for calibration, to measure what a
    given node/device costs (time, energy) with the placement under control.

    `detection` and `classification` are "hostname:DEVICE", e.g. "wadas1:NPU" or "wadas3:GPU".
    """

    DEVICES = ("CPU", "GPU", "NPU")

    def __init__(self, detection, classification, **kwargs):
        super().__init__(**kwargs)
        self.targets = {
            "DETECTION": self._parse(detection),
            "CLASSIFICATION": self._parse(classification),
        }

    def _parse(self, spec):
        hostname, _, device = spec.partition(":")
        if not hostname or device not in self.DEVICES:
            raise ValueError(
                f"Expected hostname:DEVICE with DEVICE in {self.DEVICES}, got {spec!r}"
            )
        return hostname, device

    def _best_node(self, nodes, role):
        hostname, device = self.targets[role]
        for node_id, info in nodes.items():
            if info.get("hostname") == hostname:
                return {"id_node": node_id, "device": device}
        raise ValueError(f"No alive Ray node is named {hostname}")
