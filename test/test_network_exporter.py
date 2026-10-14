from unittest.mock import patch

import pytest

from wadas.ai import network_exporter
from wadas.ai.network_exporter import (
    MBIT_TO_BPS,
    link_capacity_bps,
    parse_default_iface,
    wired_capacity_bps,
)

GIGABIT = 1000 * MBIT_TO_BPS


@pytest.fixture
def sysfs(tmp_path):
    def add(iface, speed_text):
        (tmp_path / iface).mkdir()
        (tmp_path / iface / "speed").write_text(speed_text)

    return tmp_path, add


def test_wired_capacity_comes_from_sysfs(sysfs):
    root, add = sysfs
    add("enp3s0", "1000\n")
    assert wired_capacity_bps("enp3s0", root) == (GIGABIT, GIGABIT)


@pytest.mark.parametrize("content", ["-1\n", "0\n", "not a number\n"])
def test_wired_capacity_ignores_invalid_speed(sysfs, content):
    root, add = sysfs
    add("enp3s0", content)
    assert wired_capacity_bps("enp3s0", root) == (None, None)


def test_wired_capacity_missing_interface(sysfs):
    root, _ = sysfs
    assert wired_capacity_bps("wlan0", root) == (None, None)


def test_link_capacity_prefers_wired_over_wifi(sysfs):
    root, add = sysfs
    add("enp3s0", "2500\n")
    with patch.object(network_exporter, "wifi_capacity_bps") as wifi:
        assert link_capacity_bps("enp3s0", sysfs_root=root) == (2500 * MBIT_TO_BPS,) * 2
    wifi.assert_not_called()


def test_link_capacity_uses_wifi_when_not_wired(sysfs):
    root, _ = sysfs
    with patch.object(network_exporter, "wifi_capacity_bps", return_value=(300.0, 200.0)):
        assert link_capacity_bps("wlan0", sysfs_root=root) == (300.0, 200.0)


def test_link_capacity_reuses_the_only_wifi_direction(sysfs):
    root, _ = sysfs
    with patch.object(network_exporter, "wifi_capacity_bps", return_value=(None, 200.0)):
        assert link_capacity_bps("wlan0", sysfs_root=root) == (200.0, 200.0)


def test_link_capacity_always_publishes_a_value(sysfs):
    root, _ = sysfs
    with patch.object(network_exporter, "wifi_capacity_bps", return_value=(None, None)):
        assert (
            link_capacity_bps("veth0", fallback_mbit=100, sysfs_root=root)
            == (100 * MBIT_TO_BPS,) * 2
        )


def test_parse_default_iface():
    out = "default via 10.10.95.1 dev enp3s0 proto dhcp src 10.10.95.62 metric 100\n"
    assert parse_default_iface(out) == "enp3s0"


@pytest.mark.parametrize("out", ["", "default via 10.0.0.1\n", "dev"])
def test_parse_default_iface_without_device(out):
    assert parse_default_iface(out) is None
