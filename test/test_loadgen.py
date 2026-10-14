import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from loadgen import (  # noqa: E402
    PHOTO,
    VIDEO,
    build_schedule,
    percentile,
    plan_runs,
    summarize_events,
)

PHOTOS = ["a.jpg", "b.jpg", "c.jpg"]
VIDEOS = ["x.mp4", "y.mp4"]


def schedule(camera_id=0, **overrides):
    params = dict(
        duration_s=600,
        mean_interval_s=10,
        video_probability=0.0,
        photos=PHOTOS,
        videos=VIDEOS,
        seed=0,
    )
    params.update(overrides)
    return build_schedule(camera_id, **params)


def test_schedule_is_deterministic_and_camera_specific():
    assert schedule(3) == schedule(3)
    assert schedule(3) != schedule(4)
    assert schedule(3, seed=1) != schedule(3, seed=2)


def test_schedule_is_ordered_and_within_the_duration():
    events = schedule(duration_s=120)
    offsets = [e.offset_s for e in events]
    assert offsets == sorted(offsets)
    assert all(0 < offset < 120 for offset in offsets)
    assert [e.event_id for e in events] == list(range(len(events)))


def test_schedule_rate_matches_the_mean_interval():
    total = sum(len(schedule(camera, duration_s=3000)) for camera in range(20))
    expected = 20 * 3000 / 10
    assert expected * 0.9 < total < expected * 1.1


def test_photos_only_when_video_probability_is_zero():
    assert {e.kind for e in schedule()} == {PHOTO}
    assert {e.media for e in schedule()} <= set(PHOTOS)


def test_videos_only_when_video_probability_is_one():
    events = schedule(video_probability=1.0, photos=[])
    assert {e.kind for e in events} == {VIDEO}
    assert {e.media for e in events} <= set(VIDEOS)


def test_video_share_matches_the_probability():
    events = [e for c in range(20) for e in schedule(c, duration_s=3000, video_probability=0.2)]
    share = sum(1 for e in events if e.kind == VIDEO) / len(events)
    assert 0.17 < share < 0.23
    assert all(e.media in (VIDEOS if e.kind == VIDEO else PHOTOS) for e in events)


@pytest.mark.parametrize(
    "overrides",
    [
        {"mean_interval_s": 0},
        {"video_probability": 1.5},
        {"photos": []},
        {"video_probability": 0.5, "videos": []},
    ],
)
def test_schedule_rejects_inconsistent_parameters(overrides):
    with pytest.raises(ValueError):
        schedule(**overrides)


def test_plan_runs_covers_every_label_each_repeat():
    plan = plan_runs(["A", "B", "C"], repeats=3, seed=0)
    assert len(plan) == 9
    for repeat in range(3):
        assert sorted(label for r, label in plan if r == repeat) == ["A", "B", "C"]
    assert plan == plan_runs(["A", "B", "C"], repeats=3, seed=0)


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([7], 95) == 7
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([4, 1, 3, 2], 100) == 4
    assert percentile(list(range(101)), 95) == 95


def record(kind, latency, wait=0.0, status="ok"):
    return {"kind": kind, "latency_s": latency, "wait_s": wait, "status": status}


def test_summarize_events():
    events = [
        record(PHOTO, 1.0),
        record(PHOTO, 3.0, wait=2.0),
        record(VIDEO, 20.0),
        record(PHOTO, 99.0, status="error"),
    ]
    summary = summarize_events(events)
    assert (summary["events"], summary["ok"], summary["errors"]) == (4, 3, 1)
    assert summary["photo_count"] == 2
    assert summary["photo_p50_s"] == 2.0
    assert summary["photo_max_s"] == 3.0
    assert summary["video_p95_s"] == 20.0
    assert summary["queued_share"] == pytest.approx(1 / 3)


def test_summarize_events_without_any_served_event():
    summary = summarize_events([record(PHOTO, 1.0, status="error")])
    assert summary["ok"] == 0
    assert summary["queued_share"] is None
    assert summary["photo_p50_s"] is None
