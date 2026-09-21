import re
from unittest.mock import patch
from urllib.parse import urlparse

import pytest

from wadas.ai.schedulers import (
    ActuatorScheduler,
    MonitoringScheduler,
    OptimumScheduler,
    Scheduler,
)

ALL_SCHEDULERS = [MonitoringScheduler, ActuatorScheduler, OptimumScheduler]

# Per-node fake telemetry, keyed by IP address.
TELEMETRY = {}


def ray_node(node_id, ip, resources, alive=True):
    return {
        "NodeID": node_id,
        "Alive": alive,
        "NodeManagerAddress": ip,
        "MetricsExportPort": 9000,
        "Resources": resources,
    }


def fake_url_to_metric(self, url, target_metrics):
    parsed = urlparse(url)
    node = TELEMETRY[parsed.hostname]
    if parsed.port == 8000:
        return {"npu_utilization": node["npu"]}
    if parsed.port == 8002:
        return {"gpu_utilization": node["gpu"]}
    if parsed.port == 8004:
        return {
            "network_tx_bytes_per_second": node["net"],
            "network_rx_bytes_per_second": node["net"],
            "network_tx_capacity_bytes_per_second": 1e6,
            "network_rx_capacity_bytes_per_second": 1e6,
        }
    return {
        "ray_node_cpu_utilization": node["cpu"],
        "ray_node_mem_total": 100.0,
        "ray_node_mem_used": node["mem"],
        "ray_node_disk_io_write_speed": 0.0,
        "ray_node_disk_io_read_speed": 0.0,
    }


@pytest.fixture
def cluster():
    TELEMETRY.clear()
    with patch.object(Scheduler, "url_to_metric", fake_url_to_metric):
        yield TELEMETRY
    TELEMETRY.clear()


def idle(**overrides):
    node = {"cpu": 5.0, "gpu": 5.0, "npu": 0.0, "net": 1e4, "mem": 30.0}
    node.update(overrides)
    return node


@pytest.mark.parametrize("scheduler_cls", ALL_SCHEDULERS)
def test_npu_is_used_for_detection_only(cluster, scheduler_cls):
    cluster["10.0.0.1"] = idle()
    nodes = [ray_node("npu-node", "10.0.0.1", {"CPU": 6, "NPU": 1})]

    with patch("wadas.ai.schedulers.ray.nodes", return_value=nodes):
        detection, classification = scheduler_cls().best_nodes_det_class()

    assert detection == {"id_node": "npu-node", "device": "NPU"}
    assert classification["id_node"] == "npu-node"
    assert classification["device"] in ("CPU", "GPU")


@pytest.mark.parametrize("scheduler_cls", ALL_SCHEDULERS)
def test_dead_nodes_are_never_selected(cluster, scheduler_cls):
    cluster["10.0.0.1"] = idle(cpu=90.0, gpu=90.0, mem=50.0, net=9e5)
    cluster["10.0.0.2"] = idle(cpu=0.0, gpu=0.0, mem=1.0, net=0.0)
    nodes = [
        ray_node("alive", "10.0.0.1", {"CPU": 6, "GPU": 1}),
        ray_node("dead", "10.0.0.2", {"CPU": 6, "GPU": 1}, alive=False),
    ]

    with patch("wadas.ai.schedulers.ray.nodes", return_value=nodes):
        detection, classification = scheduler_cls().best_nodes_det_class()

    assert detection["id_node"] == "alive"
    assert classification["id_node"] == "alive"


def test_monitoring_saves_each_nodes_own_score(cluster, tmp_path):
    cluster["10.0.0.1"] = idle(cpu=10.0)
    cluster["10.0.0.2"] = idle(cpu=60.0)
    nodes = [
        ray_node("light", "10.0.0.1", {"CPU": 6}),
        ray_node("busy", "10.0.0.2", {"CPU": 6}),
    ]
    scores_file = tmp_path / "scores.txt"

    with patch("wadas.ai.schedulers.ray.nodes", return_value=nodes):
        MonitoringScheduler(save_scores=True, scores_path=scores_file).best_nodes_det_class()

    node_scores = re.findall(r"^Node (\S+) score = (\S+)$", scores_file.read_text(), re.M)
    assert {node_id for node_id, _ in node_scores} == {"light", "busy"}
    assert len({score for _, score in node_scores}) == 2
