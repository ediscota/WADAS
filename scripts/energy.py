# Description: Read the energy counters published by wadas.ai.power_exporter on every node, and
# compute the energy used by the cluster between two readings.

import re
import time

import requests

DOMAIN = "package-0"
POWER_PORT = 8006

_ENERGY_LINE = re.compile(r'^node_energy_joules_total\{domain="([^"]+)"\}\s+([0-9.eE+-]+)', re.M)


def parse_energy(text, domain=DOMAIN):
    """Cumulative joules of `domain` in a Prometheus text, or None if it is not there."""
    for name, value in _ENERGY_LINE.findall(text):
        if name == domain:
            return float(value)
    return None


def read_energy(hosts, port=POWER_PORT, domain=DOMAIN, timeout=5.0):
    """Return {host: cumulative joules}, or None if any host cannot be read.

    A partial reading would make the cluster total look smaller than it is, so it is refused.
    """
    readings = {}
    for host in hosts:
        try:
            response = requests.get(f"http://{host}:{port}/metrics", timeout=timeout)
            response.raise_for_status()
        except requests.RequestException:
            return None
        value = parse_energy(response.text, domain)
        if value is None:
            return None
        readings[host] = value
    return readings


def energy_between(before, after):
    """Return ({host: joules}, cluster joules) used between two readings, or (None, None)."""
    if before is None or after is None or before.keys() != after.keys():
        return None, None
    per_node = {host: after[host] - before[host] for host in before}
    return per_node, sum(per_node.values())


def measure_idle_watts(hosts, seconds, **kwargs):
    """Mean cluster power over `seconds` while nothing is running, or None."""
    before = read_energy(hosts, **kwargs)
    time.sleep(seconds)
    _, joules = energy_between(before, read_energy(hosts, **kwargs))
    return None if joules is None else joules / seconds
