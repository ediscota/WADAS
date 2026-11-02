import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dump_scores import mean_readings, rank, score_terms  # noqa: E402

IDLE = {
    "cpu": 1.0,
    "gpu": 0.0,
    "npu": 0.0,
    "mem_used": 10.0,
    "mem_total": 100.0,
    "tx": 0.0,
    "rx": 0.0,
    "cap_tx": 125e6,
    "cap_rx": 125e6,
    "disk_read": 0.0,
    "disk_write": 0.0,
}


def raw(**changes):
    return dict(IDLE, **changes)


def test_monitoring_score_is_weighted_power_plus_memory_ratio():
    terms = score_terms(raw(cpu=2.0, gpu=10.0, npu=5.0))
    assert terms["cpu_term"] == 6.0
    assert terms["gpu_term"] == 40.0
    assert terms["npu_term"] == 10.0
    assert terms["monitoring"] == pytest.approx(56.1)


def test_idle_network_has_full_headroom():
    assert score_terms(raw())["throughput"] == pytest.approx(1.0)


def test_memory_guard():
    assert score_terms(raw(mem_used=54.0))["memory_ok"]
    assert not score_terms(raw(mem_used=55.0))["memory_ok"]


def test_trade_off_adds_the_disk_term_only_when_there_is_disk_traffic():
    assert score_terms(raw())["performance"] == pytest.approx(1.0)
    busy = score_terms(raw(disk_read=3.0, disk_write=1.0))
    assert busy["performance"] == pytest.approx(1.25)


def test_trade_off_score_is_performance_over_power():
    terms = score_terms(raw(cpu=1.0))
    assert terms["optimum"] == pytest.approx(terms["performance"] / terms["monitoring"])


def test_a_lighter_node_scores_higher_in_the_trade_off():
    assert score_terms(raw(cpu=0.5))["optimum"] > score_terms(raw(cpu=5.0))["optimum"]


def test_rank_orders_by_preference():
    values = {"a": 3.0, "b": 1.0, "c": 2.0}
    assert rank(values, reverse=False) == {"b": 1, "c": 2, "a": 3}
    assert rank(values, reverse=True) == {"a": 1, "c": 2, "b": 3}


def test_mean_readings_averages_every_field():
    mean = mean_readings([raw(cpu=1.0), raw(cpu=3.0)])
    assert mean["cpu"] == 2.0
    assert mean["mem_total"] == 100.0
