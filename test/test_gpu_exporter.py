from pathlib import Path
from unittest.mock import patch

import pytest

from wadas.ai import gpu_exporter
from wadas.ai.gpu_exporter import (
    FrequencySource,
    find_frequency_source,
    sample_utilization,
    utilization_percent,
)


def write(path: Path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(content))


@pytest.fixture
def render_node(tmp_path):
    return tmp_path / "renderD128"


def test_finds_i915_counters_in_the_gt0_directory(render_node):
    card = render_node / "device/drm/card0"
    write(card / "gt/gt0/rps_act_freq_mhz", 300)
    write(card / "gt/gt0/rps_max_freq_mhz", 1550)
    source = find_frequency_source(render_node)
    assert source == FrequencySource(
        "i915", card / "gt/gt0/rps_act_freq_mhz", card / "gt/gt0/rps_max_freq_mhz"
    )


def test_finds_i915_counters_in_the_legacy_card_directory(render_node):
    card = render_node / "device/drm/card1"
    write(card / "gt_act_freq_mhz", 300)
    write(card / "gt_max_freq_mhz", 1550)
    assert find_frequency_source(render_node).actual == card / "gt_act_freq_mhz"


def test_finds_xe_counters(render_node):
    freq = render_node / "device/tile0/gt0/freq0"
    write(freq / "act_freq", 400)
    write(freq / "max_freq", 2000)
    assert find_frequency_source(render_node) == FrequencySource(
        "xe", freq / "act_freq", freq / "max_freq"
    )


def test_no_counters(render_node):
    write(render_node / "device/power/runtime_active_time", 123)
    assert find_frequency_source(render_node) is None


def test_a_missing_maximum_is_not_a_source(render_node):
    write(render_node / "device/tile0/gt0/freq0/act_freq", 400)
    assert find_frequency_source(render_node) is None


@pytest.mark.parametrize(
    "readings, maximum, expected",
    [
        ([0, 0, 0], 1550, 0.0),
        ([1550, 1550], 1550, 100.0),
        ([0, 1000], 2000, 25.0),
        ([3000], 2000, 100.0),  # never above 100
        ([500], 0, 0.0),  # no usable maximum
        ([], 1550, 0.0),
    ],
)
def test_utilization_percent(readings, maximum, expected):
    assert utilization_percent(readings, maximum) == expected


def test_idle_and_busy_gpu_read_differently(render_node):
    freq = render_node / "device/tile0/gt0/freq0"
    write(freq / "max_freq", 1000)
    source = FrequencySource("xe", freq / "act_freq", freq / "max_freq")
    with patch.object(gpu_exporter.time, "sleep"):
        write(freq / "act_freq", 0)
        idle = sample_utilization(source, samples=4)
        write(freq / "act_freq", 900)
        busy = sample_utilization(source, samples=4)
    assert idle == 0.0
    assert busy == 90.0
