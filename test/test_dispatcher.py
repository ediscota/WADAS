import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dispatcher import (  # noqa: E402
    WEIGHTS,
    DispatchedPipeline,
    Registry,
    run_dispatched,
    workers_from_profile,
)

PROFILE = {
    "wadas1:NPU": {"DETECTION": {"s": 0.06, "J": 0.5}},
    "wadas1:GPU": {"DETECTION": {"s": 0.09, "J": 1.0}, "CLASSIFICATION": {"s": 0.12, "J": 1.5}},
    "wadas3:GPU": {"DETECTION": {"s": 0.13, "J": 6.0}, "CLASSIFICATION": {"s": 0.18, "J": 9.0}},
}


def registry(policy="cost", slots=2, weights=WEIGHTS["CRITICAL"], **kwargs):
    return Registry(workers_from_profile(PROFILE, **kwargs), policy, slots, weights)


def names(workers):
    return [w.name for w in workers]


# --- workers from the profile ---


def test_one_worker_per_device_and_role_of_the_profile():
    workers = workers_from_profile(PROFILE)
    assert names(workers) == [
        "wadas1:GPU:DETECTION",
        "wadas1:GPU:CLASSIFICATION",
        "wadas1:NPU:DETECTION",
        "wadas3:GPU:DETECTION",
        "wadas3:GPU:CLASSIFICATION",
    ]


def test_workers_can_be_left_out_by_device_or_by_role():
    assert "wadas3:GPU:DETECTION" not in names(workers_from_profile(PROFILE, exclude={"wadas3:GPU"}))
    kept = names(workers_from_profile(PROFILE, exclude={"wadas1:GPU:DETECTION"}))
    assert "wadas1:GPU:DETECTION" not in kept and "wadas1:GPU:CLASSIFICATION" in kept


# --- choosing a worker ---


def test_the_cost_policy_prefers_the_fast_cheap_device_while_it_is_free():
    r = registry()
    assert r.acquire("DETECTION").name == "wadas1:NPU:DETECTION"


def test_work_in_flight_moves_the_next_items_to_other_devices():
    r = registry(slots=10, weights=WEIGHTS["ACTUATOR"])
    picked = [r.acquire("DETECTION").name for _ in range(6)]
    assert picked[0] == "wadas1:NPU:DETECTION"
    assert {"wadas1:GPU:DETECTION", "wadas3:GPU:DETECTION"} <= set(picked)


def test_energy_weight_keeps_work_on_the_cheap_devices_longer():
    def uhd_share(weights):
        r = registry(slots=10, weights=weights)
        picked = [r.acquire("DETECTION").name for _ in range(12)]
        return picked.count("wadas3:GPU:DETECTION")

    assert uhd_share(WEIGHTS["MONITORING"]) < uhd_share(WEIGHTS["ACTUATOR"])


def test_a_gpu_counts_detection_and_classification_together():
    r = registry(slots=10, weights=WEIGHTS["ACTUATOR"])
    before = r.cost(next(w for w in r.workers if w.name == "wadas1:GPU:DETECTION"), 1)
    r.acquire("CLASSIFICATION")  # goes to the GPU of wadas1, the fastest classifier
    after = r.cost(next(w for w in r.workers if w.name == "wadas1:GPU:DETECTION"), 1)
    assert after > before


def test_least_loaded_spreads_over_devices():
    r = registry("least_loaded", slots=5)
    picked = {r.acquire("DETECTION").device_key for _ in range(3)}
    assert picked == {"wadas1:NPU", "wadas1:GPU", "wadas3:GPU"}


def test_round_robin_takes_the_workers_in_turn():
    r = registry("round_robin", slots=5)
    first = [r.acquire("DETECTION").name for _ in range(3)]
    assert len(set(first)) == 3
    assert r.acquire("DETECTION").name == first[0]


def test_a_worker_takes_at_most_its_slots():
    r = registry(slots=1, weights=WEIGHTS["MONITORING"])
    taken = [r.acquire("DETECTION") for _ in range(3)]
    assert len({w.name for w in taken}) == 3
    assert r.choose("DETECTION", 1) is None


def test_acquire_waits_for_a_release_when_every_slot_is_taken():
    r = registry(slots=1)
    taken = [r.acquire("CLASSIFICATION") for _ in range(2)]
    got = []
    waiter = threading.Thread(target=lambda: got.append(r.acquire("CLASSIFICATION")))
    waiter.start()
    time.sleep(0.05)
    assert not got
    r.release(taken[0], 1, 0.1)
    waiter.join(timeout=1)
    assert got and got[0] is taken[0]


def test_release_keeps_the_statistics_and_frees_the_work():
    r = registry()
    w = r.acquire("DETECTION", n_items=10)
    assert r.work_s[w.device_key] == pytest.approx(10 * w.seconds_per_item)
    r.release(w, 10, 0.7)
    assert r.work_s[w.device_key] == 0
    assert r.stats()[w.name] == {"calls": 1, "items": 10, "busy_s": 0.7}
    r.reset_stats()
    assert r.stats()[w.name]["calls"] == 0


def test_an_unknown_policy_is_refused():
    with pytest.raises(ValueError):
        registry("fastest")


# --- the pipeline that AiModel sees ---


class FakePipeline:
    def __init__(self, worker):
        self.worker = worker

    def run_detection(self, img, threshold, filter_animals=True):
        return {"worker": self.worker.name, "n": len(img) if isinstance(img, list) else 1}

    def classify(self, img, results, threshold):
        return [{"worker": self.worker.name}]


def test_every_call_goes_to_the_worker_chosen_at_that_moment():
    r = registry()
    pipeline = DispatchedPipeline(r, FakePipeline, 1, lambda results: results)
    log = pipeline.start_item()
    detection = pipeline.run_detection("img", 0.5)
    pipeline.classify("img", detection, 0.5)
    assert detection["worker"] == "wadas1:NPU:DETECTION"
    assert log.det_worker == "wadas1:NPU:DETECTION"
    assert log.cls_worker == "wadas1:GPU:CLASSIFICATION"
    assert all(w.in_flight == 0 for w in r.workers)  # every slot was given back


def test_a_clip_is_one_call_counting_all_its_frames():
    r = registry()
    pipeline = DispatchedPipeline(r, FakePipeline, 1, lambda results: results)
    pipeline.start_item()
    assert pipeline.run_detection(["f"] * 15, 0.5)["n"] == 15
    npu = next(w for w in r.workers if w.name == "wadas1:NPU:DETECTION")
    assert (npu.calls, npu.items) == (1, 15)


def test_a_failed_call_still_gives_its_slot_back():
    class Broken(FakePipeline):
        def run_detection(self, *args, **kwargs):
            raise RuntimeError("actor died")

    r = registry()
    pipeline = DispatchedPipeline(r, Broken, 1, lambda results: results)
    with pytest.raises(RuntimeError):
        pipeline.run_detection("img", 0.5)
    assert all(w.in_flight == 0 for w in r.workers)


def test_the_animal_class_and_filter_are_those_of_the_real_pipeline():
    pipeline = DispatchedPipeline(registry(), FakePipeline, 7, lambda results: "filtered")
    assert pipeline.animal_class_idx == 7
    assert pipeline.filter_animal_detections({}) == "filtered"


# --- items in parallel ---


def test_items_run_in_parallel_up_to_the_limit():
    running, peak, lock = [0], [0], threading.Lock()

    def process(kind, media):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        return {}

    now = time.time()
    arrivals = [(now, f"photo{i}", "photo", f"p{i}.jpg") for i in range(8)]
    records, backlog = run_dispatched(arrivals, process, now + 5, max_parallel=3)
    assert len(records) == 8 and backlog == 0
    assert peak[0] == 3


def test_items_left_waiting_at_the_stop_are_the_backlog():
    def process(kind, media):
        time.sleep(0.2)
        return {}

    now = time.time()
    arrivals = [(now, "photo0", "photo", f"p{i}.jpg") for i in range(5)]
    records, backlog = run_dispatched(arrivals, process, now + 0.1, max_parallel=2)
    assert len(records) == 2 and backlog == 3


def test_an_error_is_recorded_and_the_others_go_on():
    def process(kind, media):
        if media == "bad.jpg":
            raise RuntimeError("corrupted")
        return {"n_animals": 1}

    now = time.time()
    arrivals = [(now, "c", "photo", "bad.jpg"), (now + 0.01, "c", "photo", "ok.jpg")]
    records, _ = run_dispatched(arrivals, process, now + 5, max_parallel=2)
    assert [r["status"] for r in records] == ["error", "ok"]
