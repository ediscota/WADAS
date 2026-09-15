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
import logging
import re
import subprocess
import time

import psutil
from prometheus_client import Gauge, start_http_server

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8004
DEFAULT_IFACE = "eth0"
DEFAULT_SAMPLING_PERIOD = 1.0  # seconds
MBIT_TO_BPS = 1_000_000 / 8  # 1 Mbit/s = 125000 B/s

rx_bps_gauge = Gauge("network_rx_bytes_per_second", "Bytes received per second", ("iface",))
tx_bps_gauge = Gauge("network_tx_bytes_per_second", "Bytes sent per second", ("iface",))
rx_cap_bps_gauge = Gauge(
    "network_rx_capacity_bytes_per_second", "RX capacity (bytes/s)", ("iface",)
)
tx_cap_bps_gauge = Gauge(
    "network_tx_capacity_bytes_per_second", "TX capacity (bytes/s)", ("iface",)
)


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


def update_metrics(iface: str, prev, prev_t: float) -> tuple[dict, float]:
    now = psutil.net_io_counters(pernic=True)[iface]
    now_t = time.time()
    dt = max(now_t - prev_t, 1e-3)

    rx_bps = (now.bytes_recv - prev.bytes_recv) / dt
    tx_bps = (now.bytes_sent - prev.bytes_sent) / dt
    rx_bps_gauge.labels(iface).set(rx_bps)
    tx_bps_gauge.labels(iface).set(tx_bps)

    rx_cap_bps, tx_cap_bps = wifi_capacity_bps(iface)
    if rx_cap_bps is not None:
        rx_cap_bps_gauge.labels(iface).set(rx_cap_bps)
    if tx_cap_bps is not None:
        tx_cap_bps_gauge.labels(iface).set(tx_cap_bps)

    logger.debug(
        "RX %.0f B/s | TX %.0f B/s | CAP_RX %s B/s | CAP_TX %s B/s",
        rx_bps,
        tx_bps,
        rx_cap_bps if rx_cap_bps is not None else "-",
        tx_cap_bps if tx_cap_bps is not None else "-",
    )

    return now, now_t


def expose_metrics(
    iface: str = DEFAULT_IFACE,
    sampling_period: float = DEFAULT_SAMPLING_PERIOD,
    port: int = DEFAULT_PORT,
):
    """Start the HTTP server exposing network throughput metrics and loop forever."""
    start_http_server(port)
    logger.info("Network metrics exporter listening on port %d", port)

    prev = psutil.net_io_counters(pernic=True)[iface]
    prev_t = time.time()
    while True:
        prev, prev_t = update_metrics(iface, prev, prev_t)
        time.sleep(sampling_period)


def main():
    parser = argparse.ArgumentParser(description="Network throughput Prometheus exporter")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--iface", default=DEFAULT_IFACE, help="Network interface to monitor")
    parser.add_argument("--sampling-period", type=float, default=DEFAULT_SAMPLING_PERIOD)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    expose_metrics(args.iface, args.sampling_period, args.port)


if __name__ == "__main__":
    main()
