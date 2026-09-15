# Description: Standalone Prometheus exporter for NPU utilization, run on
# Ray cluster nodes equipped with an Intel NPU. Consumed by wadas.ai.schedulers.

import argparse
import logging
import time

from prometheus_client import Gauge, start_http_server

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8000
DEFAULT_NPU_BUSY_TIME_PATH = "/sys/devices/pci0000:00/0000:00:0b.0/npu_busy_time_us"
DEFAULT_SAMPLING_PERIOD = 2  # seconds

npu_utilization_metric = Gauge("npu_utilization", "NPU utilization percentage (0-100)", ["empty"])


def read_npu_busy_time(busy_time_path: str) -> int:
    with open(busy_time_path, "r") as f:
        return int(f.read().strip())


def compute_npu_utilization(busy_time_path: str, sampling_period: float) -> float:
    time_1 = read_npu_busy_time(busy_time_path)
    time.sleep(sampling_period)
    time_2 = read_npu_busy_time(busy_time_path)

    delta = time_2 - time_1
    utilization = 100 * delta / (sampling_period * 1_000_000)
    logger.debug("NPU utilization: %.2f%%", utilization)

    npu_utilization_metric.labels(empty="").set(utilization)
    return utilization


def expose_metrics(
    busy_time_path: str = DEFAULT_NPU_BUSY_TIME_PATH,
    sampling_period: float = DEFAULT_SAMPLING_PERIOD,
    port: int = DEFAULT_PORT,
):
    """Start the HTTP server exposing NPU utilization metrics and loop forever."""
    start_http_server(port)
    logger.info("NPU metrics exporter listening on port %d", port)
    while True:
        compute_npu_utilization(busy_time_path, sampling_period)


def main():
    parser = argparse.ArgumentParser(description="NPU utilization Prometheus exporter")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--busy-time-path", default=DEFAULT_NPU_BUSY_TIME_PATH)
    parser.add_argument("--sampling-period", type=float, default=DEFAULT_SAMPLING_PERIOD)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    expose_metrics(args.busy_time_path, args.sampling_period, args.port)


if __name__ == "__main__":
    main()
