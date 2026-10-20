# Description: Standalone Prometheus exporter for integrated-GPU load, run on Ray cluster
# nodes. Consumed by wadas.ai.schedulers.
#
# The load is estimated from the actual GPU frequency relative to its maximum: it is 0 while
# the GPU is idle and grows towards 100 as the GPU works harder. The "runtime active time" of
# the device is deliberately not used: it only says whether the GPU is powered, so it reads the
# same for an idle and for a saturated GPU.

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from prometheus_client import Gauge, start_http_server

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8002
DEFAULT_RENDER_NODE = "/sys/class/drm/renderD128"
DEFAULT_SAMPLING_PERIOD = 1.0  # seconds
SAMPLES_PER_PERIOD = 10  # the frequency is sampled this many times per period and averaged

gpu_utilization_metric = Gauge(
    "gpu_utilization", "GPU load estimated from its actual frequency (0-100 %)", ["empty"]
)


@dataclass(frozen=True)
class FrequencySource:
    driver: str
    actual: Path  # actual GPU frequency, MHz
    maximum: Path  # maximum GPU frequency, MHz


def find_frequency_source(render_node: Path) -> FrequencySource | None:
    """Locate the frequency counters of the GPU behind a DRM render node, or None."""
    device = render_node / "device"

    # i915 (e.g. Intel UHD 770): under the card directory, in the gt0 subdirectory on recent
    # kernels or directly in the card directory on older ones.
    for card in sorted((device / "drm").glob("card*")):
        for actual, maximum in (
            (card / "gt/gt0/rps_act_freq_mhz", card / "gt/gt0/rps_max_freq_mhz"),
            (card / "gt_act_freq_mhz", card / "gt_max_freq_mhz"),
        ):
            if actual.exists() and maximum.exists():
                return FrequencySource("i915", actual, maximum)

    # xe (e.g. Intel Arc graphics of Core Ultra): under the PCI device, render GT of tile 0.
    freq = device / "tile0/gt0/freq0"
    if (freq / "act_freq").exists() and (freq / "max_freq").exists():
        return FrequencySource("xe", freq / "act_freq", freq / "max_freq")
    return None


def read_mhz(path: Path) -> int:
    return int(path.read_text().strip())


def utilization_percent(readings_mhz, max_mhz) -> float:
    """Mean of the frequency readings relative to the maximum frequency, as 0-100."""
    if not readings_mhz or max_mhz <= 0:
        return 0.0
    mean = sum(readings_mhz) / len(readings_mhz)
    return max(0.0, min(100.0, 100.0 * mean / max_mhz))


def sample_utilization(
    source: FrequencySource,
    sampling_period: float = DEFAULT_SAMPLING_PERIOD,
    samples: int = SAMPLES_PER_PERIOD,
) -> float:
    """Average the actual frequency over one sampling period and return the load (0-100)."""
    readings = []
    for _ in range(samples):
        readings.append(read_mhz(source.actual))
        time.sleep(sampling_period / samples)
    return utilization_percent(readings, read_mhz(source.maximum))


def expose_metrics(
    source: FrequencySource,
    sampling_period: float = DEFAULT_SAMPLING_PERIOD,
    port: int = DEFAULT_PORT,
):
    """Start the HTTP server exposing the GPU load and loop forever."""
    start_http_server(port)
    logger.info("GPU metrics exporter (%s) listening on port %d", source.driver, port)
    while True:
        utilization = sample_utilization(source, sampling_period)
        gpu_utilization_metric.labels(empty="").set(utilization)
        logger.debug("GPU load %.1f%%", utilization)


def main():
    parser = argparse.ArgumentParser(description="GPU load Prometheus exporter")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--render-node", default=DEFAULT_RENDER_NODE)
    parser.add_argument("--sampling-period", type=float, default=DEFAULT_SAMPLING_PERIOD)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    source = find_frequency_source(Path(args.render_node))
    if source is None:
        parser.error(f"no GPU frequency counters found under {args.render_node}/device")
    expose_metrics(source, args.sampling_period, args.port)


if __name__ == "__main__":
    main()
