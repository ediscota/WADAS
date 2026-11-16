import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_stream_task import run_paced  # noqa: E402
from run_stress import (  # noqa: E402
    capacity,
    parse_args,
    parse_steps,
    percentile,
    ramp_should_continue,
    rotated,
    summarize_step,
)


class FakeClock:
    """A clock that only moves when the code sleeps or a frame is processed."""

    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def paced(processing_s, stop_at=5.0, fps=1.0, deadline=None):
    clock = FakeClock()

    def process(frame):
        clock.now += processing_s
        return {"det_s": processing_s, "cls_s": 0.0, "n_animals": 0}

    records = []
    unprocessed = run_paced(
        itertools.count(),
        process,
        0.0,
        stop_at,
        fps,
        stop_at + 100 if deadline is None else deadline,
        records,
        clock=clock,
        sleep=clock.sleep,
    )
    return records, unprocessed


# --- one stream: frames on schedule, queued when late ---


def test_a_fast_stream_starts_every_frame_on_its_arrival():
    records, unprocessed = paced(0.2)
    assert [r["start"] for r in records] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert all(r["latency_s"] == pytest.approx(0.2) for r in records)
    assert not any(r["late"] for r in records)
    assert unprocessed == 0


def test_no_frame_is_sent_at_or_after_the_stop_time():
    records, _ = paced(0.1, stop_at=3.0)
    assert [r["frame"] for r in records] == [0, 1, 2]


def test_frames_of_a_slow_stream_queue_and_their_latency_grows():
    records, _ = paced(1.5, stop_at=4.0)
    assert [r["start"] for r in records] == [0.0, 1.5, 3.0, 4.5]
    assert [r["latency_s"] for r in records] == pytest.approx([1.5, 2.0, 2.5, 3.0])
    # frame 3 started 1.5 s after its arrival, after the next arrival: a live camera drops it
    assert [r["late"] for r in records] == [False, False, False, True]


def test_the_queue_is_abandoned_at_the_deadline_and_the_rest_is_counted():
    records, unprocessed = paced(3.0, stop_at=5.0, deadline=6.0)
    # frames start at 0 and 3; at 6 the deadline is reached with frames 2, 3 and 4 still queued
    assert [r["frame"] for r in records] == [0, 1]
    assert unprocessed == 3


def test_frames_done_before_a_crash_are_kept():
    records = []
    clock = FakeClock()

    def process(frame):
        clock.now += 0.1
        if frame == 2:
            raise RuntimeError("actor died")
        return {}

    with pytest.raises(RuntimeError):
        run_paced(itertools.count(), process, 0.0, 10.0, 1.0, 100.0, records, clock, clock.sleep)
    assert [r["frame"] for r in records] == [0, 1]


# --- one step: throughput, latency, sustainability ---


def frames(arrivals, latency, start_offset=0.0):
    return [
        {"arrival": a, "start": a + start_offset, "end": a + latency, "latency_s": latency,
         "late": False}  # fmt: skip
        for a in arrivals
    ]


def stream(arrivals, latency, status="ok", unprocessed=0):
    return {"status": status, "frames": frames(arrivals, latency), "unprocessed": unprocessed}


WINDOW = (10.0, 20.0)


def summary_of(streams, n_streams=None, latency_limit=2.0):
    return summarize_step(
        streams, n_streams or len(streams), *WINDOW, fps=1.0, latency_limit=latency_limit,
        min_throughput=0.95,
    )  # fmt: skip


def test_streams_that_keep_up_are_sustainable():
    s = summary_of([stream(range(0, 30), 0.3), stream(range(0, 30), 0.5)])
    assert s["offered_fps"] == 2
    assert s["throughput_fps"] == pytest.approx(2.0)
    assert s["latency_p95_s"] == pytest.approx(0.5)
    assert s["sustainable"]
    assert s["reasons"] == ""


def test_only_frames_arriving_inside_the_window_count_for_the_latency():
    warm = frames(range(0, 10), 8.0)  # slow first frames, during the warm-up
    s = summary_of([{"status": "ok", "frames": warm + frames(range(10, 30), 0.2)}])
    assert s["latency_max_s"] == pytest.approx(0.2)


def test_a_high_latency_makes_a_step_unsustainable():
    s = summary_of([stream(range(0, 30), 2.5)])
    assert not s["sustainable"]
    assert "p95 latency" in s["reasons"]


def test_a_failed_stream_makes_a_step_unsustainable():
    s = summary_of([stream(range(0, 30), 0.2), stream(range(0, 15), 0.2, status="error")])
    assert s["failed_streams"] == 1
    assert not s["sustainable"]


def test_a_stream_that_never_started_still_counts_in_the_offered_load():
    s = summary_of([stream(range(0, 30), 0.2)], n_streams=2)
    assert s["throughput_ratio"] == pytest.approx(0.5)
    assert not s["sustainable"]
    assert "throughput" in s["reasons"]


def test_a_step_without_frames_in_the_window_is_unsustainable():
    s = summary_of([stream(range(0, 5), 0.2)])
    assert s["latency_p95_s"] is None
    assert not s["sustainable"]


# --- the ramp ---


def test_the_ramp_runs_the_extra_steps_after_the_first_unsustainable_one():
    assert ramp_should_continue([True, True], extra_steps=1)
    assert ramp_should_continue([True, False], extra_steps=1)
    assert not ramp_should_continue([True, False, False], extra_steps=1)
    assert not ramp_should_continue([True, False], extra_steps=0)


def test_capacity_is_the_last_sustainable_step_before_the_first_failure():
    steps = [(1, True), (2, True), (4, False), (6, True)]
    assert capacity([{"n_streams": n, "sustainable": ok} for n, ok in steps]) == 2
    assert capacity([{"n_streams": 1, "sustainable": False}]) == 0


@pytest.mark.parametrize("q, expected", [(50, 5), (95, 10), (100, 10), (10, 1)])
def test_percentile_is_nearest_rank(q, expected):
    assert percentile(list(range(1, 11)), q) == expected


def test_streams_start_from_different_videos():
    videos = ["a", "b", "c"]
    assert rotated(videos, 0) == ["a", "b", "c"]
    assert rotated(videos, 4) == ["b", "c", "a"]


def test_steps_must_be_increasing_positive_integers():
    assert parse_steps("1, 2,4") == [1, 2, 4]
    for bad in ("", "2,1", "0,1", "1,1"):
        with pytest.raises(ValueError):
            parse_steps(bad)


def test_default_ramp_and_thresholds_match_the_specification():
    args = parse_args(["--use-case", "MONITORING"])
    assert args.steps == [1, 2, 4, 6, 8, 12, 16, 20, 24, 32]
    assert (args.fps, args.stagger, args.warmup, args.window) == (1.0, 1.0, 30.0, 120.0)
    assert (args.latency_limit, args.min_throughput, args.extra_steps) == (2.0, 0.95, 1)


# --- analysis ---


def fake_run(tmp_path, name, sustainable):
    import csv
    import json

    from run_stress import STEP_FIELDS, STREAM_FIELDS

    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(
        json.dumps({"args": {"use_case": "MONITORING", "scheduler_params": "{}"}})
    )
    with open(run_dir / "steps.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, STEP_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for n, ok in sustainable:
            writer.writerow({"n_streams": n, "offered_fps": n, "throughput_fps": n if ok else n / 2,
                             "latency_p95_s": 0.5 if ok else 4.0, "j_per_frame": 20.0,
                             "sustainable": ok})  # fmt: skip
    with open(run_dir / "streams.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, STREAM_FIELDS)
        writer.writeheader()
        for n, _ in sustainable:
            for i in range(n):
                writer.writerow({"n_streams": n, "stream_id": i, "status": "ok",
                                 "det_node": f"wadas{i % 5 + 1}", "det_device": "GPU",
                                 "cls_node": "wadas1", "cls_device": "GPU"})  # fmt: skip
    return run_dir


def test_light_step_is_at_most_half_the_capacity():
    from analyze_stress import light_and_heavy

    steps = [{"n_streams": n, "sustainable": n <= 12} for n in (1, 2, 4, 6, 8, 12, 16)]
    assert light_and_heavy(steps) == (6, 12)
    assert light_and_heavy([{"n_streams": 1, "sustainable": False}]) == (None, None)


def test_the_analysis_reads_runs_and_draws_every_chart(tmp_path, capsys):
    from analyze_stress import main

    runs = [
        fake_run(tmp_path, "st_a", [(1, True), (2, True), (4, True), (6, False)]),
        fake_run(tmp_path, "st_b", [(1, True), (2, False)]),
    ]
    main([str(r) for r in runs] + ["--out", str(tmp_path / "charts")])
    names = sorted(p.name for p in (tmp_path / "charts").iterdir())
    assert names == ["energy.png", "latency.png", "placement.png", "throughput.png"]
    table = capsys.readouterr().out
    assert "st_a" in table and "st_b" in table


def test_the_idle_power_is_measured_again_inside_every_cooldown():
    args = parse_args(["--use-case", "MONITORING"])
    assert args.step_idle <= args.cooldown
    with pytest.raises(SystemExit):
        parse_args(["--use-case", "MONITORING", "--cooldown", "10", "--step-idle", "15"])
