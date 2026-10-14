# Description: Pure helpers shared by the camera load-generation scripts. No WADAS or Ray
# imports, so they can be unit tested without a cluster.

import random
from dataclasses import dataclass

PHOTO = "photo"
VIDEO = "video"

# An event that waited longer than this before being served was queued behind another one.
QUEUE_THRESHOLD_S = 1.0


@dataclass(frozen=True)
class Event:
    event_id: int
    offset_s: float  # scheduled arrival, in seconds after the common start of the run
    kind: str  # PHOTO or VIDEO
    media: str  # file name inside the photo / video set


def build_schedule(
    camera_id, duration_s, mean_interval_s, video_probability, photos, videos, seed=0
):
    """Arrival times of one camera: a Poisson process (exponential gaps) over duration_s.

    Deterministic for a given (seed, camera_id), so every scheduler sees the same workload.
    """
    if mean_interval_s <= 0:
        raise ValueError("mean_interval_s must be positive")
    if not 0 <= video_probability <= 1:
        raise ValueError("video_probability must be between 0 and 1")
    if video_probability < 1 and not photos:
        raise ValueError("no photos available")
    if video_probability > 0 and not videos:
        raise ValueError("no videos available")

    rng = random.Random(seed * 100_003 + camera_id)
    events = []
    t = rng.expovariate(1 / mean_interval_s)
    while t < duration_s:
        is_video = rng.random() < video_probability
        media = rng.choice(videos if is_video else photos)
        events.append(Event(len(events), round(t, 3), VIDEO if is_video else PHOTO, media))
        t += rng.expovariate(1 / mean_interval_s)
    return events


def plan_runs(labels, repeats, seed):
    """Return [(repeat, label)], shuffling the order of the labels within each repeat."""
    plan = []
    for repeat in range(repeats):
        order = list(labels)
        random.Random(seed + repeat).shuffle(order)
        plan.extend((repeat, label) for label in order)
    return plan


def percentile(values, q):
    """Percentile q (0-100) with linear interpolation; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def summarize_events(events):
    """Latency summary of event records (dicts with status, kind, latency_s, wait_s)."""
    served = [e for e in events if e["status"] == "ok"]
    summary = {
        "events": len(events),
        "ok": len(served),
        "errors": len(events) - len(served),
        "queued_share": (
            sum(1 for e in served if e["wait_s"] > QUEUE_THRESHOLD_S) / len(served)
            if served
            else None
        ),
    }
    for kind in (PHOTO, VIDEO):
        latencies = [e["latency_s"] for e in served if e["kind"] == kind]
        summary[f"{kind}_count"] = len(latencies)
        summary[f"{kind}_p50_s"] = percentile(latencies, 50)
        summary[f"{kind}_p95_s"] = percentile(latencies, 95)
        summary[f"{kind}_max_s"] = max(latencies) if latencies else None
    return summary
