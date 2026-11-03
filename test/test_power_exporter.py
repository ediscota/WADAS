import sys
from pathlib import Path

import pytest

from wadas.ai.power_exporter import energy_delta_uj, find_domains, power_watts

# sysfs directory names contain a colon, which Windows does not allow
needs_colons = pytest.mark.skipif(sys.platform == "win32", reason="sysfs names contain colons")


def make_domain(root: Path, directory: str, name: str, energy="1000", max_range="262143328850"):
    path = root / directory
    path.mkdir(parents=True)
    (path / "name").write_text(name + "\n")
    if energy is not None:
        (path / "energy_uj").write_text(energy + "\n")
    (path / "max_energy_range_uj").write_text(max_range + "\n")


@needs_colons
def test_finds_the_readable_domains(tmp_path):
    make_domain(tmp_path, "intel-rapl:0", "package-0")
    make_domain(tmp_path, "intel-rapl:0:0", "core")
    make_domain(tmp_path, "intel-rapl:1", "psys")
    assert [d.name for d in find_domains(tmp_path)] == ["package-0", "core", "psys"]


@needs_colons
def test_the_mmio_copies_of_the_domains_are_skipped(tmp_path):
    make_domain(tmp_path, "intel-rapl:0", "package-0")
    make_domain(tmp_path, "intel-rapl-mmio:0", "package-0")
    assert len(find_domains(tmp_path)) == 1


@needs_colons
def test_a_domain_without_a_readable_counter_is_skipped(tmp_path):
    make_domain(tmp_path, "intel-rapl:0", "package-0", energy=None)
    assert find_domains(tmp_path) == []


@needs_colons
def test_a_counter_with_garbage_is_skipped(tmp_path):
    make_domain(tmp_path, "intel-rapl:0", "package-0", energy="not a number")
    assert find_domains(tmp_path) == []


def test_no_domains_at_all(tmp_path):
    assert find_domains(tmp_path) == []


def test_energy_delta_between_two_readings():
    assert energy_delta_uj(1_000, 4_000, 10_000) == 3_000


def test_energy_delta_when_the_counter_wraps_around():
    assert energy_delta_uj(9_000, 500, 10_000) == 1_500


@pytest.mark.parametrize(
    "delta_uj, seconds, expected",
    [(5_000_000, 1.0, 5.0), (5_000_000, 2.0, 2.5), (0, 1.0, 0.0), (1_000, 0.0, 0.0)],
)
def test_power_watts(delta_uj, seconds, expected):
    assert power_watts(delta_uj, seconds) == expected
