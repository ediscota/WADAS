import json
import re
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlparse

import pytest

from wadas.ai.pending_load import PendingLoad
from wadas.ai.schedulers import (
    ActuatorScheduler,
    MonitoringScheduler,
    OptimumScheduler,
    PinnedScheduler,
    Scheduler,
    load_energy_profile,
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


# --- pinned scheduler, used to calibrate what a node/device costs ---


def named_nodes():
    return [
        {**ray_node("id-1", "10.0.0.1", {"CPU": 6, "NPU": 1}), "NodeManagerHostname": "wadas1"},
        {**ray_node("id-3", "10.0.0.3", {"CPU": 6, "GPU": 1}), "NodeManagerHostname": "wadas3"},
    ]


def test_pinned_scheduler_returns_the_requested_node_and_device():
    with patch("wadas.ai.schedulers.ray.nodes", return_value=named_nodes()):
        detection, classification = PinnedScheduler(
            detection="wadas1:NPU", classification="wadas3:GPU"
        ).best_nodes_det_class()

    assert detection == {"id_node": "id-1", "device": "NPU"}
    assert classification == {"id_node": "id-3", "device": "GPU"}


def test_pinned_scheduler_refuses_an_unknown_node():
    with patch("wadas.ai.schedulers.ray.nodes", return_value=named_nodes()):
        with pytest.raises(ValueError, match="wadas9"):
            PinnedScheduler(
                detection="wadas9:GPU", classification="wadas3:GPU"
            ).best_nodes_det_class()


@pytest.mark.parametrize("spec", ["wadas1", "wadas1:TPU", ":GPU"])
def test_pinned_scheduler_refuses_a_malformed_target(spec):
    with pytest.raises(ValueError):
        PinnedScheduler(detection=spec, classification="wadas3:GPU")


# --- energy profile: pick the node/device where a task costs the least energy ---

PROFILE = {
    "wadas1:NPU": {"DETECTION": 20.0},
    "wadas1:GPU": {"DETECTION": 30.0, "CLASSIFICATION": 30.0},
    "wadas3:GPU": {"DETECTION": 100.0, "CLASSIFICATION": 50.0},
}


def profile_cluster(cluster):
    cluster["10.0.0.1"] = idle()
    cluster["10.0.0.3"] = idle()
    return [
        {
            **ray_node("id-1", "10.0.0.1", {"CPU": 6, "GPU": 1, "NPU": 1}),
            "NodeManagerHostname": "wadas1",
        },
        {**ray_node("id-3", "10.0.0.3", {"CPU": 6, "GPU": 1}), "NodeManagerHostname": "wadas3"},
    ]


def energy_decision(cluster, pending=None, **kwargs):
    nodes = profile_cluster(cluster)
    kwargs.setdefault("energy_profile", PROFILE)
    with (
        patch("wadas.ai.schedulers.ray.nodes", return_value=nodes),
        patch("wadas.ai.schedulers.read_pending", return_value=pending or PendingLoad()),
        patch("wadas.ai.schedulers.reserve"),
    ):
        return MonitoringScheduler(**kwargs).best_nodes_det_class()


def test_energy_profile_picks_the_cheapest_device_for_each_model(cluster):
    detection, classification = energy_decision(cluster)
    assert detection == {"id_node": "id-1", "device": "NPU"}
    assert classification == {"id_node": "id-1", "device": "GPU"}


def test_without_an_energy_profile_the_power_estimate_is_used(cluster):
    nodes = profile_cluster(cluster)
    cluster["10.0.0.1"] = idle(cpu=80.0, gpu=80.0)
    cluster["10.0.0.3"] = idle(cpu=1.0, gpu=1.0)
    with patch("wadas.ai.schedulers.ray.nodes", return_value=nodes):
        _, classification = MonitoringScheduler().best_nodes_det_class()
    assert classification["id_node"] == "id-3"


def test_pending_placements_move_work_to_the_next_cheapest_option(cluster):
    # The NPU is taken and the GPU of wadas1 carries 2 pending placements (20 J each):
    # 30 + 40 = 70 J there, against 100 J for detection on wadas3.
    pending = PendingLoad([("id-1", "NPU", "DETECTION")] + [("id-1", "GPU", "CLASSIFICATION")])
    detection, _ = energy_decision(cluster, pending, pending_penalty=20.0)
    assert detection == {"id_node": "id-1", "device": "GPU"}

    more = PendingLoad([("id-1", "NPU", "DETECTION")] + [("id-1", "GPU", "CLASSIFICATION")] * 4)
    detection, _ = energy_decision(cluster, more, pending_penalty=20.0)
    assert detection == {"id_node": "id-3", "device": "GPU"}


def test_a_node_missing_from_the_profile_is_never_chosen(cluster):
    only_wadas3 = {"wadas3:GPU": PROFILE["wadas3:GPU"]}
    detection, classification = energy_decision(cluster, energy_profile=only_wadas3)
    assert detection["id_node"] == classification["id_node"] == "id-3"


def test_an_unusable_profile_falls_back_to_the_power_estimate(cluster):
    detection, _ = energy_decision(cluster, energy_profile={"nowhere:GPU": {"DETECTION": 1.0}})
    assert detection["id_node"] in ("id-1", "id-3")


def test_the_energy_profile_can_be_loaded_from_a_json_file(cluster, tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(PROFILE))
    detection, _ = energy_decision(cluster, energy_profile=str(path))
    assert detection == {"id_node": "id-1", "device": "NPU"}


def test_the_shipped_energy_profile_is_valid():
    path = Path(__file__).resolve().parents[1] / "scripts" / "energy_profile.json"
    profile = load_energy_profile(path)
    assert profile["wadas1:NPU"]["DETECTION"] < profile["wadas3:GPU"]["DETECTION"]
    costs = {key: value for key, value in profile.items() if key != "capacity"}
    assert all(set(value) <= {"DETECTION", "CLASSIFICATION"} for value in costs.values())
    # The capacity is optional: it made the measured mix worse when set to 12 / 10 models
    assert all(
        isinstance(limit, int) and limit > 0 for limit in profile.get("capacity", {}).values()
    )


def test_a_relative_profile_path_is_looked_up_in_the_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a different working directory, as in the experiment tasks
    profile = load_energy_profile("scripts/energy_profile.json")
    assert "wadas1:NPU" in profile


# --- capacity: how many models a node may be given before it gets no more ---

CAPACITY_PROFILE = {**PROFILE, "capacity": {"wadas1": 3}}


def models_on_wadas1(count, age=20.0):
    return PendingLoad([("id-1", "GPU", "CLASSIFICATION", age)] * count)


def test_a_node_at_its_capacity_receives_no_more_work(cluster):
    # wadas1 is far cheaper (20 J against 100 J), but it already holds 3 models
    detection, _ = energy_decision(
        cluster, models_on_wadas1(3), energy_profile=CAPACITY_PROFILE, pending_penalty=1.0
    )
    assert detection["id_node"] == "id-3"


def test_a_node_below_its_capacity_is_still_used(cluster):
    detection, _ = energy_decision(
        cluster, models_on_wadas1(2), energy_profile=CAPACITY_PROFILE, pending_penalty=1.0
    )
    assert detection["id_node"] == "id-1"


def test_the_capacity_counts_models_for_longer_than_the_penalty_does(cluster):
    # 3 models placed 20 s ago: too old for the 10 s penalty window, still in memory
    detection, _ = energy_decision(
        cluster, models_on_wadas1(3, age=20.0), energy_profile=CAPACITY_PROFILE, pending_penalty=1.0
    )
    assert detection["id_node"] == "id-3"

    forgotten = energy_decision(
        cluster, models_on_wadas1(3, age=90.0), energy_profile=CAPACITY_PROFILE, pending_penalty=1.0
    )[0]
    assert forgotten["id_node"] == "id-1"


def test_when_every_node_is_full_the_cheapest_is_used_anyway(cluster):
    full = PendingLoad(
        [("id-1", "GPU", "CLASSIFICATION", 5.0)] * 3 + [("id-3", "GPU", "DETECTION", 5.0)] * 3
    )
    profile = {**PROFILE, "capacity": {"wadas1": 3, "wadas3": 3}}
    detection, classification = energy_decision(cluster, full, energy_profile=profile)
    assert detection["id_node"] and classification["id_node"]


def test_decisions_are_recorded_for_as_long_as_the_capacity_window(cluster):
    nodes = profile_cluster(cluster)
    with (
        patch("wadas.ai.schedulers.ray.nodes", return_value=nodes),
        patch("wadas.ai.schedulers.read_pending", return_value=PendingLoad()),
        patch("wadas.ai.schedulers.reserve") as reserve,
    ):
        MonitoringScheduler(
            energy_profile=CAPACITY_PROFILE, pending_penalty=1.0, pending_ttl=10, capacity_ttl=45
        ).best_nodes_det_class()
    assert {call.args[3] for call in reserve.call_args_list} == {45}


def test_a_capacity_alone_is_enough_to_track_the_placements(cluster):
    nodes = profile_cluster(cluster)
    with (
        patch("wadas.ai.schedulers.ray.nodes", return_value=nodes),
        patch("wadas.ai.schedulers.read_pending", return_value=PendingLoad()) as read_pending,
        patch("wadas.ai.schedulers.reserve") as reserve,
    ):
        MonitoringScheduler(energy_profile=CAPACITY_PROFILE).best_nodes_det_class()
    read_pending.assert_called_once()
    assert reserve.call_count == 2
