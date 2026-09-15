# Description: Standalone Prometheus exporter for integrated-GPU utilization,
# run on Ray cluster nodes. Consumed by wadas.ai.schedulers.

import argparse
import logging
import os
import time

from prometheus_client import Gauge, start_http_server

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8002
DEFAULT_GPU_BUSY_TIME_PATH = "/sys/class/drm/renderD128/device/power/runtime_active_time"
DEFAULT_SAMPLING_PERIOD = 1  # seconds
MAX_CONCURRENT_ACCESSES = 10

gpu_utilization_metric = Gauge(
    "gpu_utilization", "GPU active-time percentage (0-100)", ["empty"]
)
gpu_access_count_metric = Gauge(
    "gpu_access_count", "Estimated number of concurrent accesses (0-10)"
)
gpu_access_ratio_metric = Gauge("gpu_access_ratio", "Normalized utilization (0-1)")


def read_runtime_us(busy_time_path: str) -> int:
    with open(busy_time_path) as f:
        return int(f.read().strip())


def update_metrics(busy_time_path: str, sampling_period: float, prev_us: int) -> int:
    time.sleep(sampling_period)
    cur_us = read_runtime_us(busy_time_path)
    delta = max(cur_us - prev_us, 0)

    busy_ratio = delta / (sampling_period * 1000)
    access_count = busy_ratio
    access_ratio = access_count / MAX_CONCURRENT_ACCESSES

    gpu_utilization_metric.labels(empty="").set(access_ratio * 100)
    gpu_access_count_metric.set(access_count)
    gpu_access_ratio_metric.set(access_ratio)

    logger.debug(
        "GPU busy ratio %.2f%% | accesses %.2f/%d | ratio %.2f",
        busy_ratio * 100,
        access_count,
        MAX_CONCURRENT_ACCESSES,
        access_ratio,
    )
    return cur_us


def expose_metrics(
    busy_time_path: str = DEFAULT_GPU_BUSY_TIME_PATH,
    sampling_period: float = DEFAULT_SAMPLING_PERIOD,
    port: int = DEFAULT_PORT,
):
    """Start the HTTP server exposing GPU utilization metrics and loop forever."""
    if not os.path.isfile(busy_time_path):
        raise FileNotFoundError(busy_time_path)

    start_http_server(port)
    logger.info("GPU metrics exporter listening on port %d", port)

    prev_us = read_runtime_us(busy_time_path)
    while True:
        prev_us = update_metrics(busy_time_path, sampling_period, prev_us)


def main():
    parser = argparse.ArgumentParser(description="GPU utilization Prometheus exporter")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--busy-time-path", default=DEFAULT_GPU_BUSY_TIME_PATH)
    parser.add_argument("--sampling-period", type=float, default=DEFAULT_SAMPLING_PERIOD)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    expose_metrics(args.busy_time_path, args.sampling_period, args.port)


if __name__ == "__main__":
    main()
