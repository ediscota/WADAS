import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import energy  # noqa: E402
from energy import energy_between, parse_energy, read_energy  # noqa: E402

TEXT = """# HELP node_energy_joules_total Energy used by a RAPL domain
# TYPE node_energy_joules_total counter
node_energy_joules_total{domain="package-0"} 1234.5
node_energy_joules_total{domain="core"} 800.25
node_power_watts{domain="package-0"} 3.2
"""


class Response:
    def __init__(self, text, status=200):
        self.text = text
        self.status = status

    def raise_for_status(self):
        if self.status != 200:
            raise requests.HTTPError("boom")


def test_parse_energy_picks_the_requested_domain():
    assert parse_energy(TEXT) == 1234.5
    assert parse_energy(TEXT, "core") == 800.25
    assert parse_energy(TEXT, "psys") is None


def test_parse_energy_ignores_the_power_gauge():
    assert parse_energy('node_power_watts{domain="package-0"} 3.2') is None


def test_read_energy_collects_every_host():
    with patch.object(energy.requests, "get", return_value=Response(TEXT)):
        assert read_energy(["10.0.0.1", "10.0.0.2"]) == {"10.0.0.1": 1234.5, "10.0.0.2": 1234.5}


def test_read_energy_refuses_a_partial_reading():
    def fake_get(url, timeout):
        if "10.0.0.2" in url:
            raise requests.ConnectionError("down")
        return Response(TEXT)

    with patch.object(energy.requests, "get", side_effect=fake_get):
        assert read_energy(["10.0.0.1", "10.0.0.2"]) is None


def test_read_energy_refuses_a_host_without_the_domain():
    with patch.object(energy.requests, "get", return_value=Response("nothing useful")):
        assert read_energy(["10.0.0.1"]) is None


def test_energy_between_sums_the_nodes():
    per_node, total = energy_between({"a": 100.0, "b": 50.0}, {"a": 130.0, "b": 80.0})
    assert per_node == {"a": 30.0, "b": 30.0}
    assert total == 60.0


@pytest.mark.parametrize(
    "before, after",
    [(None, {"a": 1.0}), ({"a": 1.0}, None), ({"a": 1.0}, {"b": 2.0})],
)
def test_energy_between_without_comparable_readings(before, after):
    assert energy_between(before, after) == (None, None)


def test_idle_watts_is_energy_over_time():
    readings = iter([{"a": 100.0}, {"a": 130.0}])
    with (
        patch.object(energy, "read_energy", side_effect=lambda *a, **k: next(readings)),
        patch.object(energy.time, "sleep"),
    ):
        assert energy.measure_idle_watts(["a"], seconds=10) == {"a": 3.0}
