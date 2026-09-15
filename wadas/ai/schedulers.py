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

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

import ray
import requests

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


class Scheduler(ABC):
    """Base class for Ray cluster node schedulers.

    Subclasses implement `best_nodes_det_class()` to select, among the nodes
    of the Ray cluster, the best node/device pair for running the detection
    model and the best node/device pair for running the classification model.
    """

    def __init__(self, save_scores: bool = False, scores_path: str | Path | None = None):
        self.save_scores = save_scores
        self.scores_path = Path(scores_path) if scores_path else None

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
        """Scan the Ray cluster and return per-node IP, metrics port and resources."""
        nodes = {}
        for n in ray.nodes():
            node_id = n["NodeID"]
            nodes[node_id] = {
                "ip": n["NodeManagerAddress"],
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

    @abstractmethod
    def best_nodes_det_class(self):
        """Return (best_detection_node, best_classification_node).

        Each is a dict {"id_node": <ray node id>, "device": "CPU"|"GPU"|"NPU"}.
        """


class MonitoringScheduler(Scheduler):
    """Picks nodes minimizing estimated power consumption, favoring NPU/CPU
    over GPU: suited for continuous, low-power monitoring workloads."""

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
        best_node = {}
        best_score = float("inf")
        npu_selected = False

        for node_id, info in nodes.items():
            cpu_usage, gpu_usage, npu_usage, mem_total, mem_used = self._node_metrics(
                node_id, info
            )
            mem_ratio = (mem_used / mem_total) if mem_total else 0

            if info["resources"].get("NPU", 0) > 0 and npu_usage == 0:
                power = CPU_POWER_WEIGHT * cpu_usage + GPU_POWER_WEIGHT * gpu_usage
                score = power + mem_ratio
                if not npu_selected or score <= best_score:
                    best_node = {"id_node": node_id, "device": "NPU"}
                    best_score = score
                    npu_selected = True
            elif not npu_selected:
                power = CPU_POWER_WEIGHT * cpu_usage + GPU_POWER_WEIGHT * gpu_usage
                if npu_usage > 0:
                    power += NPU_POWER_WEIGHT * npu_usage
                score = power + mem_ratio
                if score <= best_score:
                    best_score = score
                    if gpu_usage <= cpu_usage and info["resources"].get("GPU", 0) > 0:
                        best_node = {"id_node": node_id, "device": "GPU"}
                    else:
                        best_node = {"id_node": node_id, "device": "CPU"}

            self.save_score(node_id, best_score, role)

        if best_node:
            self.save_best_score(best_node["id_node"], best_score, role)
        return best_node

    def best_nodes_det_class(self):
        nodes = self.find_nodes()
        best_detection = self._best_node(nodes, "DETECTION")
        best_classification = self._best_node(nodes, "CLASSIFICATION")
        return best_detection, best_classification


class ActuatorScheduler(Scheduler):
    """Picks nodes maximizing available network throughput, guarded by a
    memory-usage ceiling: suited for actuation workloads that must stream
    results with low latency."""

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
        best_node = {}
        best_score = 0.0

        for node_id, info in nodes.items():
            score, gpu_usage, npu_usage, mem_ratio = self._node_metrics(node_id, info)

            if score >= best_score and mem_ratio < MEMORY_USAGE_GUARD:
                best_score = score
                if gpu_usage < GPU_UTILIZATION_GUARD and info["resources"].get("GPU", 0) > 0:
                    best_node = {"id_node": node_id, "device": "GPU"}
                elif npu_usage == 0 and info["resources"].get("NPU", 0) > 0:
                    best_node = {"id_node": node_id, "device": "NPU"}
                else:
                    best_node = {"id_node": node_id, "device": "CPU"}

            self.save_score(node_id, score, role)

        if best_node:
            self.save_best_score(best_node["id_node"], best_score, role)
        return best_node

    def best_nodes_det_class(self):
        nodes = self.find_nodes()
        best_detection = self._best_node(nodes, "DETECTION")
        best_classification = self._best_node(nodes, "CLASSIFICATION")
        return best_detection, best_classification


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
        performance = throughput if read_speed == 0 else throughput + (write_speed / (read_speed + write_speed))

        power = CPU_POWER_WEIGHT * cpu_usage + GPU_POWER_WEIGHT * gpu_usage + NPU_POWER_WEIGHT * npu_usage
        watt = power + ((mem_used / mem_total) if mem_total else 0)

        return performance / watt if watt else 0, cpu_usage, gpu_usage, npu_usage

    def _best_node(self, nodes, role):
        best_node = {}
        best_score = 0.0

        for node_id, info in nodes.items():
            score, cpu_usage, gpu_usage, npu_usage = self._node_metrics(node_id, info)

            if score >= best_score:
                best_score = score
                has_gpu = info["resources"].get("GPU", 0) > 0
                has_npu = info["resources"].get("NPU", 0) > 0

                if has_npu and npu_usage == 0:
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

    def best_nodes_det_class(self):
        nodes = self.find_nodes()
        best_detection = self._best_node(nodes, "DETECTION")
        best_classification = self._best_node(nodes, "CLASSIFICATION")
        return best_detection, best_classification
