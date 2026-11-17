import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_steady_profile import build_profile, main, read_run  # noqa: E402
from run_stress import FRAME_FIELDS, STEP_FIELDS  # noqa: E402

WINDOW = 100.0


def fake_run(tmp_path, name, det, cls, fps, n_frames, det_j, cls_j, animals_every=2):
    """A pinned run: frames one per 1/fps s from t=0, window 10-110 s, idle 10 W per node."""
    run_dir = tmp_path / name
    run_dir.mkdir()
    params = json.dumps({"detection": det, "classification": cls})
    (run_dir / "meta.json").write_text(json.dumps({
        "args": {"scheduler_params": params, "fps": str(fps), "window": str(WINDOW)},
        "host": "wadas5",
    }))  # fmt: skip
    det_host, cls_host = det.split(":")[0], cls.split(":")[0]
    with open(run_dir / "steps.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, STEP_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerow({
            "n_streams": 1, "measure_start": 10.0, "measure_end": 10.0 + WINDOW,
            "energy_by_node_j": json.dumps({det_host: 10 * WINDOW + det_j,
                                            cls_host: 10 * WINDOW + cls_j}),
            "idle_by_node_w": json.dumps({det_host: 10.0, cls_host: 10.0}),
        })  # fmt: skip
    with open(run_dir / "frames.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, FRAME_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for k in range(n_frames):
            t = k / fps
            animals = 1 if k % animals_every == 0 else 0
            writer.writerow({"n_streams": 1, "stream_id": 0, "frame": k, "arrival": t,
                             "start": t, "end": t + 0.2, "latency_s": 0.2, "late": False,
                             "det_s": 0.1, "cls_s": 0.05 if animals else 0.0,
                             "n_animals": animals})  # fmt: skip
    return run_dir


def test_energy_per_item_is_the_node_energy_above_idle_over_its_items(tmp_path):
    # 4 frames/s for 120 s: 400 frames end inside the 10-110 s window, 200 with animals
    run = read_run(fake_run(tmp_path, "r", "wadas1:NPU", "wadas3:GPU", 4, 480, 800.0, 300.0))
    profile = build_profile([run])
    det = profile["wadas1:NPU"]["DETECTION"]
    cls = profile["wadas3:GPU"]["CLASSIFICATION"]
    assert det["items"] == 400 and cls["items"] == 200
    assert det["J"] == pytest.approx(2.0)  # 800 J above idle / 400 detections
    assert cls["J"] == pytest.approx(1.5)  # 300 J above idle / 200 classifications
    assert det["s"] == pytest.approx(0.1) and cls["s"] == pytest.approx(0.05)
    assert det["busy"] == pytest.approx(0.4)  # 400 x 0.1 s over 100 s


def test_the_run_with_the_most_items_is_kept(tmp_path):
    slow = read_run(fake_run(tmp_path, "f2", "wadas1:NPU", "wadas3:GPU", 2, 240, 900.0, 300.0))
    fast = read_run(fake_run(tmp_path, "f8", "wadas1:NPU", "wadas3:GPU", 8, 960, 1600.0, 600.0))
    profile = build_profile([slow, fast])
    assert profile["wadas1:NPU"]["DETECTION"]["run"] == "f8"
    assert profile["wadas1:NPU"]["DETECTION"]["J"] == pytest.approx(2.0)  # 1600 / 800


def test_a_device_on_the_node_of_the_test_driver_is_flagged(tmp_path):
    run = read_run(fake_run(tmp_path, "r", "wadas5:GPU", "wadas4:GPU", 4, 480, 800.0, 300.0))
    profile = build_profile([run])
    assert profile["wadas5:GPU"]["DETECTION"]["shares_driver_node"]
    assert not profile["wadas4:GPU"]["CLASSIFICATION"]["shares_driver_node"]


def test_the_profile_is_written_as_json(tmp_path, capsys):
    fake_run(tmp_path, "r", "wadas2:NPU", "wadas4:GPU", 4, 480, 800.0, 300.0)
    out = tmp_path / "profile.json"
    main([str(tmp_path / "r"), "--out", str(out)])
    profile = json.loads(out.read_text())["steady"]
    assert set(profile) == {"wadas2:NPU", "wadas4:GPU"}
    assert "wadas2:NPU" in capsys.readouterr().out
