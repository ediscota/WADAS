import re
from unittest.mock import patch
from urllib.parse import urlparse

import pytest

from wadas.ai.pending_load import PendingLoad
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


# --- pending load: placements decided but not yet visible in the node metrics ---

# Penalty per pending placement, in the unit of each scheduler's score.
PENALTY = {MonitoringScheduler: 20.0, ActuatorScheduler: 0.05, OptimumScheduler: 20.0}


def two_nodes(cluster):
    cluster["10.0.0.1"] = idle(cpu=5.0, net=1e4)  # preferred by every scheduler when idle
    cluster["10.0.0.2"] = idle(cpu=6.0, net=2e4)
    return [
        ray_node("a", "10.0.0.1", {"CPU": 6, "GPU": 1}),
        ray_node("b", "10.0.0.2", {"CPU": 6, "GPU": 1}),
    ]


@pytest.mark.parametrize("scheduler_cls", ALL_SCHEDULERS)
def test_pending_load_is_ignored_unless_enabled(cluster, scheduler_cls):
    nodes = two_nodes(cluster)

    with (
        patch("wadas.ai.schedulers.ray.nodes", return_value=nodes),
        patch("wadas.ai.schedulers.read_pending") as read_pending,
        patch("wadas.ai.schedulers.reserve") as reserve,
    ):
        detection, _ = scheduler_cls().best_nodes_det_class()

    assert detection["id_node"] == "a"
    read_pending.assert_not_called()
    reserve.assert_not_called()


@pytest.mark.parametrize("scheduler_cls", ALL_SCHEDULERS)
def test_a_node_with_pending_placements_is_avoided(cluster, scheduler_cls):
    nodes = two_nodes(cluster)
    pending = PendingLoad([("a", "GPU", "DETECTION")])

    with (
        patch("wadas.ai.schedulers.ray.nodes", return_value=nodes),
        patch("wadas.ai.schedulers.read_pending", return_value=pending),
        patch("wadas.ai.schedulers.reserve"),
    ):
        detection, classification = scheduler_cls(
            pending_penalty=PENALTY[scheduler_cls]
        ).best_nodes_det_class()

    assert detection["id_node"] == "b"
    assert classification["id_node"] == "b"


@pytest.mark.parametrize("scheduler_cls", ALL_SCHEDULERS)
def test_decisions_are_recorded_with_the_configured_ttl(cluster, scheduler_cls):
    nodes = two_nodes(cluster)

    with (
        patch("wadas.ai.schedulers.ray.nodes", return_value=nodes),
        patch("wadas.ai.schedulers.read_pending", return_value=PendingLoad()),
        patch("wadas.ai.schedulers.reserve") as reserve,
    ):
        scheduler_cls(
            pending_penalty=PENALTY[scheduler_cls], pending_ttl=7.0
        ).best_nodes_det_class()

    recorded = [(call.args[0], call.args[2], call.args[3]) for call in reserve.call_args_list]
    assert recorded == [("a", "DETECTION", 7.0), ("a", "CLASSIFICATION", 7.0)]


@pytest.mark.parametrize("scheduler_cls", ALL_SCHEDULERS)
def test_a_pending_npu_assignment_makes_the_npu_busy(cluster, scheduler_cls):
    cluster["10.0.0.1"] = idle()
    nodes = [ray_node("npu-node", "10.0.0.1", {"CPU": 6, "NPU": 1})]

    def detection_device(pending):
        with (
            patch("wadas.ai.schedulers.ray.nodes", return_value=nodes),
            patch("wadas.ai.schedulers.read_pending", return_value=pending),
            patch("wadas.ai.schedulers.reserve"),
        ):
            detection, _ = scheduler_cls(
                pending_penalty=PENALTY[scheduler_cls]
            ).best_nodes_det_class()
        return detection["device"]

    assert detection_device(PendingLoad()) == "NPU"
    assert detection_device(PendingLoad([("npu-node", "NPU", "DETECTION")])) == "CPU"


def test_actuator_still_answers_when_pending_placements_push_scores_below_zero(cluster):
    nodes = two_nodes(cluster)
    pending = PendingLoad([("a", "GPU", "DETECTION")] * 40 + [("b", "GPU", "DETECTION")] * 40)

    with (
        patch("wadas.ai.schedulers.ray.nodes", return_value=nodes),
        patch("wadas.ai.schedulers.read_pending", return_value=pending),
        patch("wadas.ai.schedulers.reserve"),
    ):
        detection, _ = ActuatorScheduler(pending_penalty=0.05).best_nodes_det_class()

    assert detection["id_node"] in ("a", "b")


@pytest.mark.parametrize("scheduler_cls", ALL_SCHEDULERS)
def test_metrics_are_read_once_per_node_for_both_models(cluster, scheduler_cls):
    nodes = two_nodes(cluster)
    original = scheduler_cls._node_metrics

    with (
        patch("wadas.ai.schedulers.ray.nodes", return_value=nodes),
        patch.object(scheduler_cls, "_node_metrics", autospec=True, side_effect=original) as spy,
    ):
        scheduler_cls().best_nodes_det_class()

    assert spy.call_count == len(nodes)


@pytest.mark.parametrize("scheduler_cls", ALL_SCHEDULERS)
def test_pending_load_is_read_after_the_metrics(cluster, scheduler_cls):
    nodes = two_nodes(cluster)
    events = []
    original = scheduler_cls._node_metrics

    def record_metrics(self, node_id, info):
        events.append("metrics")
        return original(self, node_id, info)

    def record_pending():
        events.append("pending")
        return PendingLoad()

    with (
        patch("wadas.ai.schedulers.ray.nodes", return_value=nodes),
        patch.object(scheduler_cls, "_node_metrics", autospec=True, side_effect=record_metrics),
        patch("wadas.ai.schedulers.read_pending", side_effect=record_pending),
        patch("wadas.ai.schedulers.reserve"),
    ):
        scheduler_cls(pending_penalty=PENALTY[scheduler_cls]).best_nodes_det_class()

    assert events == ["metrics"] * len(nodes) + ["pending"]
