import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_camera_test import (  # noqa: E402
    add_queue_lengths,
    camera_schedule,
    delay_growth,
    parse_args,
    parse_steps,
    run_queue,
    summarize_step,
)


class FakeClock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


# --- the cameras ---


def test_photo_cameras_send_one_photo_per_second_out_of_step():
    arrivals = camera_schedule(0.0, 3.0, [["a"], ["b"]], [], 1.0)
    assert [(t, cam) for t, cam, _, _ in arrivals] == [
        (0.0, "photo00"),
        (0.5, "photo01"),
        (1.0, "photo00"),
        (1.5, "photo01"),
        (2.0, "photo00"),
        (2.5, "photo01"),
    ]


def test_each_photo_camera_cycles_through_its_photos():
    arrivals = camera_schedule(0.0, 3.0, [["a", "b"]], [], 1.0)
    assert [media for _, _, _, media in arrivals] == ["a", "b", "a"]


def test_video_cameras_send_each_clip_when_it_has_been_recorded():
    arrivals = camera_schedule(0.0, 40.0, [], [[("v1", 10.0), ("v2", 15.0)], [("v2", 15.0)]], 1.0)
    assert [(t, cam, media) for t, cam, _, media in arrivals] == [
        (10.0, "video00", "v1"),
        (18.0, "video01", "v2"),  # starts recording 3 s after the first camera
        (25.0, "video00", "v2"),
        (33.0, "video01", "v2"),
        (35.0, "video00", "v1"),
    ]


def test_no_item_arrives_at_or_after_the_stop():
    arrivals = camera_schedule(0.0, 2.0, [["a"]], [[("v", 2.0)]], 1.0)
    assert all(t < 2.0 for t, *_ in arrivals)
    assert [kind for _, _, kind, _ in arrivals] == ["photo", "photo"]


# --- the queue: one item at a time, in arrival order ---


def queue(arrival_times, processing_s, stop_at):
    clock = FakeClock()

    def process(kind, media):
        clock.now += processing_s
        return {"n_animals": 1}

    arrivals = [(t, "photo00", "photo", f"p{i}.jpg") for i, t in enumerate(arrival_times)]
    return run_queue(arrivals, process, stop_at, clock=clock, sleep=clock.sleep)


def test_items_that_find_the_pipeline_free_start_on_arrival():
    records, backlog = queue([0.0, 1.0, 2.0], 0.3, 10.0)
    assert [r["start"] for r in records] == [0.0, 1.0, 2.0]
    assert [r["latency_s"] for r in records] == pytest.approx([0.3, 0.3, 0.3])
    assert backlog == 0


def test_items_wait_in_order_when_the_pipeline_is_slower_than_the_cameras():
    records, _ = queue([0.0, 0.5, 1.0], 1.0, 10.0)
    assert [r["start"] for r in records] == [0.0, 1.0, 2.0]
    assert [r["wait_s"] for r in records] == pytest.approx([0.0, 0.5, 1.0])


def test_items_still_queued_at_the_stop_are_counted_as_backlog():
    records, backlog = queue([0.0, 0.1, 0.2, 0.3, 0.4], 1.0, 2.0)
    # the second item starts at 1.0 and ends at 2.0: the last three never start
    assert len(records) == 2
    assert backlog == 3


def test_an_error_is_recorded_and_the_queue_goes_on():
    clock = FakeClock()

    def process(kind, media):
        clock.now += 0.1
        if media == "bad.jpg":
            raise RuntimeError("corrupted file")
        return {}

    arrivals = [(0.0, "photo00", "photo", "bad.jpg"), (1.0, "photo00", "photo", "ok.jpg")]
    records, _ = run_queue(arrivals, process, 5.0, clock=clock, sleep=clock.sleep)
    assert [r["status"] for r in records] == ["error", "ok"]
    assert "corrupted file" in records[0]["error"]


def test_the_queue_length_counts_the_items_waiting_ahead():
    records, _ = queue([0.0, 0.1, 0.2, 0.3], 1.0, 10.0)
    add_queue_lengths(records)
    assert [r["queue_at_arrival"] for r in records] == [0, 0, 1, 2]


def test_delay_growth_is_the_slope_of_the_waiting_time():
    records = [{"arrival": float(t), "wait_s": 0.5 * t} for t in range(10)]
    assert delay_growth(records, 0.0, 10.0) == pytest.approx(30.0)  # +0.5 s per s = 30 s/min
    flat = [{"arrival": float(t), "wait_s": 0.2} for t in range(10)]
    assert delay_growth(flat, 0.0, 10.0) == pytest.approx(0.0)


# --- one step: does WADAS keep up? ---


WINDOW = (10.0, 20.0)


def photo(t, latency, wait=0.0, status="ok"):
    return {"arrival": t, "start": t + wait, "end": t + latency, "latency_s": latency,
            "wait_s": wait, "kind": "photo", "media": "p.jpg", "status": status}  # fmt: skip


def clip(t, latency, media="v.mp4"):
    return {"arrival": t, "start": t, "end": t + latency, "latency_s": latency, "wait_s": 0.0,
            "kind": "video", "media": media, "status": "ok"}  # fmt: skip


def offered(records):
    return [(r["arrival"], "cam", r["kind"], r["media"]) for r in records]


def summary_of(records, arrivals=None, backlog=0, durations=None):
    return summarize_step(records, arrivals or offered(records), backlog, *WINDOW,
                          durations or {"v.mp4": 15.0}, 2.0, 0.95)  # fmt: skip


def test_photos_processed_quickly_keep_up():
    s = summary_of([photo(t, 0.5) for t in range(0, 30)])
    assert s["photos_offered"] == s["photos_done"] == 10
    assert s["keeps_up"]


def test_slow_photos_do_not_keep_up():
    s = summary_of([photo(t, 3.0) for t in range(0, 30)])
    assert not s["keeps_up"]
    assert "photo p95" in s["reasons"]


def test_photos_left_in_the_queue_do_not_keep_up():
    records = [photo(t, 0.5) for t in range(10, 15)]
    arrivals = offered(records) + [(t, "cam", "photo", "p.jpg") for t in range(15, 20)]
    s = summary_of(records, arrivals, backlog=5)
    assert s["photo_ratio"] == pytest.approx(0.5)
    assert "still queued" in s["reasons"]


def test_a_clip_slower_than_its_own_duration_does_not_keep_up():
    ok = summary_of([clip(12.0, 10.0)])
    late = summary_of([clip(12.0, 20.0)])
    assert ok["keeps_up"] and ok["clips_late"] == 0
    assert not late["keeps_up"] and late["clips_late"] == 1


def test_errors_do_not_keep_up():
    s = summary_of([photo(t, 0.5) for t in range(10, 20)] + [photo(15.5, 0.1, status="error")])
    assert s["errors"] == 1
    assert not s["keeps_up"]


# --- arguments ---


def test_steps_are_photo_plus_video_cameras():
    assert parse_steps("1+0, 17+0,5+5") == [(1, 0), (17, 0), (5, 5)]
    for bad in ("", "3", "0+0", "-1+2"):
        with pytest.raises(ValueError):
            parse_steps(bad)


def test_defaults_match_the_specification():
    args = parse_args(["--use-case", "MONITORING", "--steps", "17+0"])
    assert (args.photo_rate, args.fps, args.warmup, args.window) == (1.0, 1, 30.0, 180.0)
    assert (args.latency_limit, args.min_throughput) == (2.0, 0.95)


# --- analysis ---


def fake_run(tmp_path, name, steps, window=180.0):
    import csv
    import json

    from run_camera_test import STEP_FIELDS

    run_dir = tmp_path / name
    run_dir.mkdir()
    placement = {"det_node": "wadas1", "det_device": "NPU", "cls_node": "wadas1",
                 "cls_device": "GPU"}  # fmt: skip
    (run_dir / "meta.json").write_text(
        json.dumps({"args": {"window": str(window)}, "placement": placement})
    )
    with open(run_dir / "steps.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, STEP_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for photos, videos, ok in steps:
            writer.writerow({
                "photo_cameras": photos, "video_cameras": videos,
                "photos_offered": photos * window, "photos_done": photos * window if ok else 100,
                "photo_p95_s": 0.5 if ok else 40.0, "clips_offered": videos * 12,
                "clips_done": videos * 12, "clip_p95_s": 9.0 if videos else "", "backlog": 0,
                "delay_growth_s_per_min": 0.0, "power_active_w": 10.0, "keeps_up": ok,
            })  # fmt: skip
    return run_dir


def test_kept_up_cameras_stop_at_the_first_failure():
    from analyze_cameras import max_cameras_kept_up

    steps = [{"photo_cameras": n, "keeps_up": ok} for n, ok in ((1, True), (2, True), (4, False))]
    assert max_cameras_kept_up(steps) == 2


def test_the_analysis_reads_both_tests_and_draws_every_chart(tmp_path, capsys):
    from analyze_cameras import main

    run_a = fake_run(tmp_path, "cam_monorig_A", [(1, 0, True), (2, 0, True), (4, 0, False)])
    run_b = fake_run(tmp_path, "cam_monorig_B", [(5, 5, False)], window=300.0)
    main([str(run_a), str(run_b), "--out", str(tmp_path / "charts")])
    names = sorted(p.name for p in (tmp_path / "charts").iterdir())
    assert names == ["test_a_latency.png", "test_a_throughput.png", "test_b_latency.png"]
    out = capsys.readouterr().out
    assert "keeps up with at most 2 photo camera(s)" in out
    assert "det wadas1:NPU" in out
