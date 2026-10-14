# Description: Standalone Prometheus exporter for network throughput and
# capacity, run on Ray cluster nodes. Consumed by wadas.ai.schedulers.
#
# Exposed metrics (all in bytes/second):
#   network_rx_bytes_per_second           actual throughput
#   network_tx_bytes_per_second
#   network_rx_capacity_bytes_per_second  instantaneous link capacity
#   network_tx_capacity_bytes_per_second

from __future__ import annotations

import argparse
import functools
import logging
import re
import subprocess
import time
from pathlib import Path

import psutil
from prometheus_client import Gauge, start_http_server

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8004
DEFAULT_SAMPLING_PERIOD = 1.0  # seconds
# Used only when neither sysfs nor `iw` can tell the link speed.
DEFAULT_FALLBACK_MBIT = 1000.0
MBIT_TO_BPS = 1_000_000 / 8  # 1 Mbit/s = 125000 B/s
SYS_CLASS_NET = Path("/sys/class/net")

rx_bps_gauge = Gauge("network_rx_bytes_per_second", "Bytes received per second", ("iface",))
tx_bps_gauge = Gauge("network_tx_bytes_per_second", "Bytes sent per second", ("iface",))
rx_cap_bps_gauge = Gauge(
    "network_rx_capacity_bytes_per_second", "RX capacity (bytes/s)", ("iface",)
)
tx_cap_bps_gauge = Gauge(
    "network_tx_capacity_bytes_per_second", "TX capacity (bytes/s)", ("iface",)
)


def parse_default_iface(route_output: str) -> str | None:
    """Return the interface of the default route from `ip -o -4 route show default`."""
    for line in route_output.splitlines():
        tokens = line.split()
        if "dev" in tokens and tokens.index("dev") + 1 < len(tokens):
            return tokens[tokens.index("dev") + 1]
    return None


def default_iface() -> str | None:
    """Interface used for the default route, or None when it cannot be determined."""
    try:
        out = subprocess.check_output(
            ["ip", "-o", "-4", "route", "show", "default"], text=True, stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return parse_default_iface(out)


def wired_capacity_bps(
    iface: str, sysfs_root: Path = SYS_CLASS_NET
) -> tuple[float | None, float | None]:
    """Return (rx_cap_Bps, tx_cap_Bps) of a wired link, from /sys/class/net/<iface>/speed.

    The file holds the negotiated speed in Mbit/s; it is unreadable on Wi-Fi and -1 when the
    link is down. The link is full duplex, so both directions get the same capacity.
    """
    try:
        speed_mbit = int((sysfs_root / iface / "speed").read_text().strip())
    except (OSError, ValueError):
        return None, None
    if speed_mbit <= 0:
        return None, None
    capacity = speed_mbit * MBIT_TO_BPS
    return capacity, capacity


def wifi_capacity_bps(iface: str) -> tuple[float | None, float | None]:
    """Return (rx_cap_Bps, tx_cap_Bps) for a WiFi interface, via `iw`.

    Returns (None, None) for wired interfaces or when `iw` is unavailable.
    """
    try:
        out = subprocess.check_output(
            ["iw", "dev", iface, "link"], text=True, stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None

    m_tx = re.search(r"tx bitrate:\s*([\d.]+)\s*MBit/s", out, re.I)
    m_rx = re.search(r"rx bitrate:\s*([\d.]+)\s*MBit/s", out, re.I)
    tx_bps = float(m_tx.group(1)) * MBIT_TO_BPS if m_tx else None
    rx_bps = float(m_rx.group(1)) * MBIT_TO_BPS if m_rx else None
    return rx_bps, tx_bps


@functools.lru_cache(maxsize=None)
def _warn_capacity_fallback(iface: str, fallback_mbit: float) -> None:
    logger.warning(
        "Cannot determine the link speed of %s: assuming %.0f Mbit/s", iface, fallback_mbit
    )


def link_capacity_bps(
    iface: str,
    fallback_mbit: float = DEFAULT_FALLBACK_MBIT,
    sysfs_root: Path = SYS_CLASS_NET,
) -> tuple[float, float]:
    """Return (rx_cap_Bps, tx_cap_Bps): wired speed, else Wi-Fi bitrate, else the fallback.

    A capacity is always published. Without one the schedulers cannot compute the network
    headroom and every node ends up with the same (zero) network score.
    """
    rx_cap, tx_cap = wired_capacity_bps(iface, sysfs_root)
    if rx_cap is None:
        rx_cap, tx_cap = wifi_capacity_bps(iface)
    if rx_cap is None and tx_cap is None:
        _warn_capacity_fallback(iface, fallback_mbit)
        fallback = fallback_mbit * MBIT_TO_BPS
        return fallback, fallback
    # `iw` can report a single direction: reuse it for the other one.
    return (rx_cap or tx_cap), (tx_cap or rx_cap)


def update_metrics(
    iface: str, prev, prev_t: float, fallback_mbit: float = DEFAULT_FALLBACK_MBIT
) -> tuple[dict, float]:
    now = psutil.net_io_counters(pernic=True)[iface]
    now_t = time.time()
    dt = max(now_t - prev_t, 1e-3)

    rx_bps = (now.bytes_recv - prev.bytes_recv) / dt
    tx_bps = (now.bytes_sent - prev.bytes_sent) / dt
    rx_bps_gauge.labels(iface).set(rx_bps)
    tx_bps_gauge.labels(iface).set(tx_bps)

    rx_cap_bps, tx_cap_bps = link_capacity_bps(iface, fallback_mbit)
    rx_cap_bps_gauge.labels(iface).set(rx_cap_bps)
    tx_cap_bps_gauge.labels(iface).set(tx_cap_bps)

    logger.debug(
        "RX %.0f B/s | TX %.0f B/s | CAP_RX %.0f B/s | CAP_TX %.0f B/s",
        rx_bps,
        tx_bps,
        rx_cap_bps,
        tx_cap_bps,
    )

    return now, now_t


def expose_metrics(
    iface: str,
    sampling_period: float = DEFAULT_SAMPLING_PERIOD,
    port: int = DEFAULT_PORT,
    fallback_mbit: float = DEFAULT_FALLBACK_MBIT,
):
    """Start the HTTP server exposing network throughput metrics and loop forever."""
    start_http_server(port)
    logger.info("Network metrics exporter for %s listening on port %d", iface, port)

    prev = psutil.net_io_counters(pernic=True)[iface]
    prev_t = time.time()
    while True:
        prev, prev_t = update_metrics(iface, prev, prev_t, fallback_mbit)
        time.sleep(sampling_period)


def main():
    parser = argparse.ArgumentParser(description="Network throughput Prometheus exporter")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--iface",
        default=None,
        help="Network interface to monitor (default: the one of the default route)",
    )
    parser.add_argument("--sampling-period", type=float, default=DEFAULT_SAMPLING_PERIOD)
    parser.add_argument(
        "--fallback-mbit",
        type=float,
        default=DEFAULT_FALLBACK_MBIT,
        help="Link speed to assume when it cannot be read",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    iface = args.iface or default_iface()
    if not iface:
        parser.error("cannot detect the default network interface, pass --iface")
    expose_metrics(iface, args.sampling_period, args.port, args.fallback_mbit)


if __name__ == "__main__":
    main()
