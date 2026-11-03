# Description: Standalone Prometheus exporter for the energy used by a node, run on Ray cluster
# nodes. It reads the Intel RAPL energy counters exposed by the kernel under
# /sys/class/powercap and publishes both the cumulative energy and the mean power.

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from prometheus_client import Counter, Gauge, start_http_server

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8006
DEFAULT_ROOT = "/sys/class/powercap"
DEFAULT_SAMPLING_PERIOD = 1.0  # seconds

# Cumulative energy exposes as node_energy_joules_total{domain=...}
energy_metric = Counter("node_energy_joules", "Energy used by a RAPL domain", ["domain"])
power_metric = Gauge("node_power_watts", "Mean power of a RAPL domain", ["domain"])


@dataclass(frozen=True)
class RaplDomain:
    name: str  # package-0, core, uncore, dram, psys
    energy: Path  # cumulative energy counter, microjoules
    max_range_uj: int  # value after which the counter wraps around to zero


def find_domains(root: Path):
    """Return the readable RAPL domains under `root`.

    The "intel-rapl-mmio" entries repeat the same domains through another interface and
    are skipped. A domain whose counter cannot be read (most desktops restrict it to root)
    is skipped too.
    """
    domains = []
    for path in sorted(root.glob("intel-rapl:*")):
        energy = path / "energy_uj"
        name_file = path / "name"
        if not name_file.is_file():
            continue
        try:
            int(energy.read_text())
            max_range = int((path / "max_energy_range_uj").read_text())
        except (OSError, ValueError):
            continue
        domains.append(RaplDomain(name_file.read_text().strip(), energy, max_range))
    return domains


def energy_delta_uj(previous: int, current: int, max_range_uj: int) -> int:
    """Energy used between two readings of a counter that wraps around at `max_range_uj`."""
    if current >= previous:
        return current - previous
    return current + max_range_uj - previous


def power_watts(delta_uj: int, seconds: float) -> float:
    return delta_uj / 1e6 / seconds if seconds > 0 else 0.0


def read_uj(domain: RaplDomain) -> int:
    return int(domain.energy.read_text())


def expose_metrics(
    domains, sampling_period: float = DEFAULT_SAMPLING_PERIOD, port: int = DEFAULT_PORT
):
    """Start the HTTP server exposing energy and power and loop forever."""
    start_http_server(port)
    logger.info(
        "Power metrics exporter listening on port %d (%s)", port, ", ".join(d.name for d in domains)
    )
    previous = {domain.name: read_uj(domain) for domain in domains}
    last = time.monotonic()
    while True:
        time.sleep(sampling_period)
        now = time.monotonic()
        for domain in domains:
            current = read_uj(domain)
            delta = energy_delta_uj(previous[domain.name], current, domain.max_range_uj)
            previous[domain.name] = current
            energy_metric.labels(domain=domain.name).inc(delta / 1e6)
            power_metric.labels(domain=domain.name).set(power_watts(delta, now - last))
        last = now


def main():
    parser = argparse.ArgumentParser(description="Node power Prometheus exporter")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--sampling-period", type=float, default=DEFAULT_SAMPLING_PERIOD)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    domains = find_domains(Path(args.root))
    if not domains:
        parser.error(
            f"no readable RAPL energy counter under {args.root}: on most desktops they are "
            "readable by root only (see sudo chmod a+r on the energy_uj files)"
        )
    expose_metrics(domains, args.sampling_period, args.port)


if __name__ == "__main__":
    main()
