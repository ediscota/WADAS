import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_scenario import build_tasks, parse_args, print_summary  # noqa: E402

VIDEOS = [Path(f"v{i}.mp4") for i in range(3)]
PHOTOS = [Path(f"p{i}.jpg") for i in range(2)]


def test_build_tasks_puts_videos_first_and_cycles_through_the_files():
    tasks = build_tasks(VIDEOS, PHOTOS, 4, 3)
    assert [kind for kind, _ in tasks] == ["video"] * 4 + ["photo"] * 3
    assert [path.name for _, path in tasks] == [
        "v0.mp4",
        "v1.mp4",
        "v2.mp4",
        "v0.mp4",
        "p0.jpg",
        "p1.jpg",
        "p0.jpg",
    ]


def test_build_tasks_without_photos_does_not_need_them():
    assert build_tasks(VIDEOS, [], 2, 0) == [("video", VIDEOS[0]), ("video", VIDEOS[1])]


def test_n_tasks_still_means_videos():
    args = parse_args(["--n-tasks", "5"])
    assert (args.n_videos, args.n_photos) == (5, 0)


def test_explicit_mix():
    args = parse_args(["--n-videos", "10", "--n-photos", "10", "--stagger", "0"])
    assert (args.n_videos, args.n_photos, args.stagger) == (10, 10, 0)


def test_an_empty_scenario_is_refused():
    with pytest.raises(SystemExit):
        parse_args(["--n-videos", "0", "--n-photos", "0"])


def test_summary_reports_mean_per_kind(capsys):
    def row(kind, t):
        return {
            "use_case": "MONITORING",
            "kind": kind,
            "status": "ok",
            "t_total_s": t,
            "det_node": "n",
            "det_device": "GPU",
            "cls_node": "n",
            "cls_device": "GPU",
        }

    print_summary([row("video", 10), row("video", 20), row("photo", 4)])
    out = capsys.readouterr().out
    assert "MONITORING video: 2/2 tasks ok" in out
    assert "mean 15.0" in out
    assert "MONITORING photo: 1/1 tasks ok" in out


def test_only_photos_does_not_add_a_video():
    args = parse_args(["--n-photos", "10"])
    assert (args.n_videos, args.n_photos) == (0, 10)


def test_default_is_one_video():
    assert parse_args([]).n_videos == 1


def test_energy_measurement_is_off_by_default():
    assert not parse_args([]).measure_energy
    assert parse_args(["--measure-energy", "--idle-seconds", "5"]).idle_seconds == 5


def test_energy_summary_reports_energy_per_task(capsys):
    from run_scenario import print_energy_summary

    rows = [
        {"use_case": "MONITORING", "n_tasks": 10, "energy_j": 500.0, "energy_active_j": 300.0},
        {"use_case": "MONITORING", "n_tasks": 10, "energy_j": 700.0, "energy_active_j": 500.0},
        {"use_case": "CRITICAL", "n_tasks": 10, "energy_j": "", "energy_active_j": ""},
    ]
    print_energy_summary(rows)
    out = capsys.readouterr().out
    assert "MONITORING energy: scenario mean 600 J (400 J above idle), 40.0 J per task" in out
    assert "CRITICAL" not in out


def test_energy_summary_reports_the_mean_per_node(capsys):
    from run_scenario import print_energy_summary

    rows = [
        {
            "use_case": "MONITORING",
            "n_tasks": 10,
            "energy_j": 300.0,
            "energy_active_j": 200.0,
            "energy_by_node_j": '{"wadas1": 100.0, "wadas3": 200.0}',
        },
        {
            "use_case": "MONITORING",
            "n_tasks": 10,
            "energy_j": 500.0,
            "energy_active_j": 400.0,
            "energy_by_node_j": '{"wadas1": 300.0, "wadas3": 200.0}',
        },
    ]
    print_energy_summary(rows)
    assert "per node (mean J per scenario): wadas1 200, wadas3 200" in capsys.readouterr().out
