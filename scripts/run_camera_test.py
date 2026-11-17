# Description: Stress test of WADAS as it works today: N cameras send photos and video clips into
# one queue, processed one at a time by one AiModel placed on the Ray cluster by the scheduler of
# the use case. Measures whether WADAS keeps up, with which latency, queue and energy.

"""Camera stress test: virtual cameras, one queue, one WADAS pipeline (see the specification).

Run it on the Ray head node, with the cluster and the per-node exporters up (the power exporter
too, for --measure-energy). The AiModel is created once, as when WADAS starts, then each step of
--steps runs P photo cameras and V video cameras ("P+V"):

  1. photo cameras send one photo every 1/--photo-rate seconds, video cameras one clip as soon as
     the previous one is "recorded" (the clip duration), all into the same queue;
  2. the queue is processed in arrival order by the same AiModel methods WADAS uses
     (AnimalDetectionAndClassificationMode): process_image + classify for a photo,
     process_video_offline + classification_from_video_tracking for a clip;
  3. --warmup seconds are left out, the next --window seconds are measured;
  4. at the end of the window the cameras stop, the item in progress is finished and the items
     still queued are counted as backlog and dropped, so that every step starts empty.

  python scripts/run_camera_test.py --use-case MONITORING --steps 1+0,2+0,4+0 --measure-energy
  python scripts/run_camera_test.py --use-case CRITICAL --steps 5+5,5+5 --window 300 \\
      --scheduler-params '{"energy_profile": "scripts/energy_profile_mdv6b_dfv14.json",
                           "pending_penalty": 5}'

With --dispatcher the single pipeline is replaced by the dispatcher prototype (dispatcher.py):
warm workers on every device of --profile, and a scheduler (--dispatcher round_robin,
least_loaded or cost, the cost weighted by --use-case) that picks a worker for every call. Items
then run in parallel, up to --max-parallel at a time, still through the unchanged AiModel methods.

  python scripts/run_camera_test.py --use-case MONITORING --dispatcher cost --steps 17+0
"""

import argparse
import bisect
import itertools
import csv
import json
import math
import os
import platform
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dispatcher import run_dispatched  # noqa: E402
from energy import energy_between, measure_idle_watts, read_energy  # noqa: E402
from run_scenario import alive_node_names, git_commit, package_version  # noqa: E402

DEFAULT_PROFILE = "scripts/steady_profile_mdv6b_dfv14.json"

USE_CASES = ("MONITORING", "ACTUATOR", "CRITICAL")
OUTPUT_DIRS = ("detection_output", "classification_output", "wadas_motion_detection", "video_frames")

ITEM_FIELDS = [
    "step",
    "photo_cameras",
    "video_cameras",
    "camera",
    "kind",
    "media",
    "arrival",
    "start",
    "end",
    "latency_s",
    "wait_s",
    "queue_at_arrival",
    "n_animals",
    "status",
    "error",
    "det_worker",
    "cls_worker",
    "det_call_s",
    "cls_call_s",
]
STEP_FIELDS = [
    "run_id",
    "use_case",
    "scheduler_params",
    "step",
    "photo_cameras",
    "video_cameras",
    "det_node",
    "det_device",
    "cls_node",
    "cls_device",
    "photos_offered",
    "photos_done",
    "photo_ratio",
    "photo_p50_s",
    "photo_p95_s",
    "photo_max_s",
    "clips_offered",
    "clips_done",
    "clip_p50_s",
    "clip_p95_s",
    "clip_max_s",
    "clips_late",
    "backlog",
    "delay_growth_s_per_min",
    "errors",
    "keeps_up",
    "reasons",
    "idle_w",
    "energy_j",
    "energy_active_j",
    "power_active_w",
    "energy_by_node_j",
    "dispatcher",
    "worker_stats",
]


def parse_steps(text):
    """'1+0,2+0,5+5' -> [(1, 0), (2, 0), (5, 5)]: photo and video cameras of every step."""
    steps = []
    for part in text.split(","):
        if not part.strip():
            continue
        photos, sep, videos = part.strip().partition("+")
        if not sep:
            raise ValueError(f"step {part!r} is not P+V")
        step = (int(photos), int(videos))
        if min(step) < 0 or sum(step) == 0:
            raise ValueError(f"step {part!r} needs at least one camera and no negative count")
        steps.append(step)
    if not steps:
        raise ValueError("no steps")
    return steps


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--use-case", required=True, choices=USE_CASES)
    parser.add_argument("--scheduler-params", default="{}", help="JSON with scheduler options")
    parser.add_argument("--detection-model", help="Detection model version")
    parser.add_argument("--classification-model", help="Classification model version")
    parser.add_argument("--photos-dir", type=Path, default=Path.home() / "PHOTO_TEST")
    parser.add_argument("--videos-dir", type=Path, default=Path.home() / "VIDEO_TEST")
    parser.add_argument("--steps", required=True, help="Cameras per step as P+V, e.g. 1+0,17+0,5+5")
    parser.add_argument("--photo-rate", type=float, default=1.0, help="Photos/s per photo camera")
    parser.add_argument("--fps", type=int, default=1, help="Video frames analysed per second")
    parser.add_argument("--warmup", type=float, default=30.0, help="Seconds left out per step")
    parser.add_argument("--window", type=float, default=180.0, help="Seconds measured per step")
    parser.add_argument("--cooldown", type=float, default=20.0, help="Seconds between steps")
    parser.add_argument(
        "--step-idle", type=float, default=15.0, help="Idle power measured in each cooldown"
    )
    parser.add_argument("--latency-limit", type=float, default=2.0, help="Max photo p95 (s)")
    parser.add_argument("--min-throughput", type=float, default=0.95, help="Min done / offered")
    parser.add_argument("--measure-energy", action="store_true")
    parser.add_argument("--idle-seconds", type=float, default=30.0)
    parser.add_argument("--expected-nodes", type=int, default=5)
    parser.add_argument("--results-dir", type=Path, default=Path.home() / "wadas_results")
    parser.add_argument("--run-id", default=datetime.now().strftime("cam-%Y%m%d-%H%M%S"))
    parser.add_argument(
        "--dispatcher",
        choices=("round_robin", "least_loaded", "cost"),
        help="Use the dispatcher prototype with this policy instead of the single pipeline",
    )
    parser.add_argument("--slots", type=int, default=2, help="Calls in flight per worker")
    parser.add_argument(
        "--max-parallel", type=int, help="Items in progress at once (default: all the slots)"
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="Steady-state profile (JSON)")
    parser.add_argument(
        "--exclude", nargs="*", default=[], help="Workers left out: host:DEVICE[:ROLE]"
    )
    args = parser.parse_args(argv)
    try:
        json.loads(args.scheduler_params)
        args.steps = parse_steps(args.steps)
    except ValueError as exc:
        parser.error(str(exc))
    if args.step_idle > args.cooldown:
        parser.error("--step-idle must fit in --cooldown")
    return args


# --- the cameras: when every photo and clip arrives ---


def rotated(items, start):
    start %= len(items)
    return items[start:] + items[:start]


def camera_schedule(start, stop, photos, videos, photo_rate):
    """Arrivals [(time, camera, kind, media)] between start and stop, sorted by time.

    `photos` is a list of photo files, `videos` a list of (file, duration_s), one list per
    camera already rotated. Photo camera i sends one photo every 1/photo_rate seconds, shifted by
    i / (photo_rate * n) so that the cameras are not in lockstep; video camera j records clips
    back to back, the first one starting j * 3 s after start, and sends each clip when it ends.
    """
    arrivals = []
    period = 1.0 / photo_rate
    for i, camera_photos in enumerate(photos):
        offset = i * period / len(photos)
        for k in range(int((stop - start - offset) / period) + 1):
            t = start + offset + k * period
            if t < stop:
                arrivals.append((t, f"photo{i:02d}", "photo", camera_photos[k % len(camera_photos)]))
    for j, camera_videos in enumerate(videos):
        t = start + j * 3.0
        k = 0
        while True:
            media, duration = camera_videos[k % len(camera_videos)]
            t += duration
            if t >= stop:
                break
            arrivals.append((t, f"video{j:02d}", "video", media))
            k += 1
    return sorted(arrivals, key=lambda a: a[0])


def run_queue(arrivals, process, stop_at, clock=time.time, sleep=time.sleep):
    """Process the arrivals one at a time in order, as WADAS does with its media queue.

    Items are processed as soon as they have arrived and the previous one is done. No item is
    started after stop_at; the ones that arrived by then but did not start are the backlog.
    Return (records, backlog). Every record has the arrival, start and end times plus what
    `process(kind, media)` returns; an exception in `process` is recorded, not raised.
    """
    records = []
    for index, (arrival, camera, kind, media) in enumerate(arrivals):
        now = clock()
        if now < arrival:
            if arrival >= stop_at:
                break
            sleep(arrival - now)
        if clock() >= stop_at:
            return records, sum(1 for a in arrivals[index:] if a[0] < stop_at)
        start = clock()
        try:
            extra = dict(process(kind, media), status="ok")
        except Exception as exc:  # noqa: BLE001 - one bad item must not stop the test
            extra = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:300]}
        end = clock()
        records.append(
            dict(extra, camera=camera, kind=kind, media=Path(media).name, arrival=arrival,
                 start=start, end=end, latency_s=end - arrival, wait_s=start - arrival)
        )  # fmt: skip
    return records, 0


def add_queue_lengths(records):
    """Queue length seen by each item when it arrived: earlier arrivals not yet started.

    Items are processed in arrival order, so those are the items between the one in progress
    and this one: the earlier arrivals minus the ones already started.
    """
    starts = sorted(r["start"] for r in records)
    for position, r in enumerate(records):
        started = bisect.bisect_right(starts, r["arrival"])
        r["queue_at_arrival"] = max(0, position - started)


# --- one step: summary and "keeps up" ---


def percentile(values, q):
    ordered = sorted(values)
    return ordered[max(1, math.ceil(q / 100 * len(ordered))) - 1]


def delay_growth(records, measure_start, measure_end):
    """Slope of the waiting time over the window, in seconds of delay gained per minute."""
    points = [(r["arrival"], r["wait_s"]) for r in records
              if measure_start <= r["arrival"] < measure_end]  # fmt: skip
    if len(points) < 2:
        return 0.0
    mean_t = sum(t for t, _ in points) / len(points)
    mean_w = sum(w for _, w in points) / len(points)
    var = sum((t - mean_t) ** 2 for t, _ in points)
    if var == 0:
        return 0.0
    slope = sum((t - mean_t) * (w - mean_w) for t, w in points) / var
    return slope * 60.0


def summarize_step(records, arrivals, backlog, measure_start, measure_end, clip_durations,
                   latency_limit, min_throughput):  # fmt: skip
    """Summary of one step; `arrivals` are all the items offered, `clip_durations` maps a clip
    file name to its duration (a clip is late if it takes longer than its own duration)."""

    def in_window(t):
        return measure_start <= t < measure_end

    offered = [a for a in arrivals if in_window(a[0])]
    done = [r for r in records if in_window(r["arrival"]) and r["status"] == "ok"]
    photos_offered = sum(1 for a in offered if a[2] == "photo")
    clips_offered = sum(1 for a in offered if a[2] == "video")
    photo_lat = [r["latency_s"] for r in done if r["kind"] == "photo"]
    clip_done = [r for r in done if r["kind"] == "video"]
    clip_lat = [r["latency_s"] for r in clip_done]
    clips_late = sum(1 for r in clip_done if r["latency_s"] > clip_durations.get(r["media"], 0))
    errors = sum(1 for r in records if r["status"] != "ok")
    growth = delay_growth(records, measure_start, measure_end)

    def stats(values):
        if not values:
            return None, None, None
        return median(values), percentile(values, 95), max(values)

    p = stats(photo_lat)
    c = stats(clip_lat)
    summary = {
        "photos_offered": photos_offered,
        "photos_done": len(photo_lat),
        "photo_ratio": len(photo_lat) / photos_offered if photos_offered else None,
        "photo_p50_s": p[0],
        "photo_p95_s": p[1],
        "photo_max_s": p[2],
        "clips_offered": clips_offered,
        "clips_done": len(clip_lat),
        "clip_p50_s": c[0],
        "clip_p95_s": c[1],
        "clip_max_s": c[2],
        "clips_late": clips_late,
        "backlog": backlog,
        "delay_growth_s_per_min": growth,
        "errors": errors,
    }
    reasons = []
    if photos_offered:
        if summary["photo_ratio"] < min_throughput:
            reasons.append(f"photos done {summary['photo_ratio']:.0%} < {min_throughput:.0%}")
        if p[1] is None or p[1] > latency_limit:
            reasons.append(f"photo p95 latency above {latency_limit:g} s")
    if clips_offered:
        if len(clip_lat) < clips_offered:
            reasons.append(f"{clips_offered - len(clip_lat)} clip(s) not done")
        if clips_late:
            reasons.append(f"{clips_late} clip(s) slower than their own duration")
    if backlog:
        reasons.append(f"{backlog} item(s) still queued at the end")
    if errors:
        reasons.append(f"{errors} error(s)")
    summary["keeps_up"] = not reasons
    summary["reasons"] = "; ".join(reasons)
    return summary


# --- WADAS ---


def make_ai_model(args):
    from wadas.domain.ai_model import AiModel

    AiModel.distributed_inference = True
    AiModel.use_case = args.use_case
    AiModel.scheduler_params = json.loads(args.scheduler_params)
    if args.detection_model:
        AiModel.detection_model_version = args.detection_model
    if args.classification_model:
        AiModel.classification_model_version = args.classification_model
    AiModel.video_fps = args.fps
    return AiModel()


def make_dispatched_model(args, photos):
    """AiModel whose pipeline is a DispatchedPipeline over warm workers on every device.

    Returns (ai_model, dispatched pipeline, registry). The workers are created on their node
    with their model loaded, then warmed up with one real call each (a photo with an animal),
    so that model compilation is not measured."""
    import ray
    from PIL import Image
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from dispatcher import WEIGHTS, DispatchedPipeline, Registry, workers_from_profile
    from wadas.ai.models import Classifier, OVMegaDetectorV5
    from wadas.ai.pipeline import NAME_TO_DETECTOR, DetectionPipeline
    from wadas.domain.ai_model import AiModel

    det_version = args.detection_model or AiModel.detection_model_version
    cls_version = args.classification_model or AiModel.classification_model_version
    AiModel.detection_model_version = det_version
    AiModel.classification_model_version = cls_version
    AiModel.distributed_inference = True
    AiModel.video_fps = args.fps

    profile_path = Path(args.profile)
    if not profile_path.is_absolute():
        profile_path = REPO_ROOT / profile_path
    profile = json.loads(profile_path.read_text())["steady"]
    workers = workers_from_profile(profile, exclude=set(args.exclude))
    ray.init(address="auto", logging_level="ERROR")
    node_ids = {n["NodeManagerHostname"]: n["NodeID"] for n in ray.nodes() if n["Alive"]}
    for w in workers:
        if w.host not in node_ids:
            raise RuntimeError(f"No alive Ray node {w.host} for worker {w.name}")
        if w.role == "DETECTION":
            cls, kwargs = NAME_TO_DETECTOR[det_version], {"model_name": det_version}
        else:
            cls, kwargs = Classifier, {"version": cls_version}
        strategy = NodeAffinitySchedulingStrategy(node_id=node_ids[w.host], soft=False)
        w.actor = ray.remote(cls).options(scheduling_strategy=strategy).remote(
            device=w.device, **kwargs
        )

    animal_idx = next(k for k, v in OVMegaDetectorV5.CLASS_NAMES.items() if v == "animal")
    template = object.__new__(DetectionPipeline)
    template.__dict__.update(distributed_inference=True, classifier_version=cls_version,
                             language=AiModel.language, animal_class_idx=animal_idx,
                             detection_model=None, classifier=None)  # fmt: skip
    bound = {}

    def bind(worker):
        """A DetectionPipeline whose remote model is this worker's actor."""
        if worker.name not in bound:
            pipeline = object.__new__(DetectionPipeline)
            pipeline.__dict__.update(template.__dict__)
            if worker.role == "DETECTION":
                pipeline.detection_model = worker.actor
            else:
                pipeline.classifier = worker.actor
            bound[worker.name] = pipeline
        return bound[worker.name]

    # Warm-up: one detection per detection worker, one classification per classification worker
    detections = None
    for photo in photos:
        img = Image.open(photo).convert("RGB")
        for w in [w for w in workers if w.role == "DETECTION"]:
            results = bind(w).run_detection(img, AiModel.detection_threshold)
        if len(results["detections"].xyxy):
            detections = (img, results)
            break
    if detections is None:
        raise RuntimeError("No test photo has an animal: cannot warm up the classifiers")
    for w in [w for w in workers if w.role == "CLASSIFICATION"]:
        bind(w).classify(detections[0], detections[1], AiModel.classification_threshold)

    registry = Registry(workers, args.dispatcher, args.slots, WEIGHTS[args.use_case])
    dispatched = DispatchedPipeline(registry, bind, animal_idx, template.filter_animal_detections)
    ai_model = object.__new__(AiModel)  # no __init__: it would place a single pipeline
    ai_model.detection_pipeline = dispatched
    ai_model.original_image = ""
    return ai_model, dispatched, registry


def make_processor(ai_model, dispatched=None):
    """process(kind, media, upload_dir): one item, exactly as AnimalDetectionAndClassificationMode
    does. Photos are copied into upload_dir first, under a name that is never reused.

    With `dispatched` (a DispatchedPipeline) items run in parallel: clips are copied too, since
    process_video_offline names its output after the clip, and every record says which workers
    the item used and how long its remote calls took."""
    counter = itertools.count()
    lock = threading.Lock()

    def unique(upload_dir, media):
        with lock:
            return upload_dir / f"{next(counter):06d}_{Path(media).name}"

    def process(kind, media, upload_dir):
        if dispatched is None:
            return process_item(kind, media, upload_dir)
        log = dispatched.start_item()
        result = process_item(kind, media, upload_dir)
        result.update(det_worker=log.det_worker, cls_worker=log.cls_worker,
                      det_call_s=log.det_s, cls_call_s=log.cls_s)  # fmt: skip
        return result

    def process_item(kind, media, upload_dir):
        if kind == "photo":
            # An FTP upload creates a new file; process_image may modify or delete it.
            path = unique(upload_dir, media)
            shutil.copyfile(media, path)
            results, detected_img_path = ai_model.process_image(str(path), True)
            n_animals = 0
            if results and detected_img_path:
                n_animals = len(results["detections"].xyxy)
                if n_animals:
                    ai_model.classify(str(path), results)
            return {"n_animals": n_animals}
        if dispatched is not None:
            clip = unique(upload_dir, media)
            shutil.copyfile(media, clip)
            media = clip
        tracked, video_path, _ = ai_model.process_video_offline(
            video_path=str(media), classification=True, save_processed_video=True,
            save_snapshot=False,
        )  # fmt: skip
        if video_path and tracked:
            ai_model.classification_from_video_tracking(tracked)
        return {"n_animals": len({a.get("id") for frame in tracked for a in frame})}

    return process


def clean_outputs(work_dir):
    for name in OUTPUT_DIRS:
        shutil.rmtree(work_dir / name, ignore_errors=True)
        (work_dir / name).mkdir()


def video_duration(path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if not fps:
        raise ValueError(f"cannot read the frame rate of {path}")
    return frames / fps


def node_name(node, ray_nodes):
    info = ray_nodes.get(node["id_node"], {})
    return info.get("NodeManagerHostname") or info.get("NodeManagerAddress") or node["id_node"]


def read_energy_at(times, node_ips, readings):
    """Read the energy counters at each of `times` (epoch), in the background."""

    def worker():
        for key, at in times.items():
            delay = at - time.time()
            if delay > 0:
                time.sleep(delay)
            readings[key] = read_energy(node_ips)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def run_step(args, index, step, process, photos, videos, durations, node_ips, idle_w, work_dir,
             registry=None):
    n_photo, n_video = step
    clean_outputs(work_dir)
    upload_dir = work_dir / "uploads"
    shutil.rmtree(upload_dir, ignore_errors=True)
    upload_dir.mkdir()

    start = time.time() + 1.0
    measure_start = start + args.warmup
    measure_end = measure_start + args.window
    arrivals = camera_schedule(
        start,
        measure_end,
        [rotated(photos, i) for i in range(n_photo)],
        [rotated([(v, durations[v.name]) for v in videos], j) for j in range(n_video)],
        args.photo_rate,
    )
    readings = {}
    energy_thread = None
    if args.measure_energy:
        times = {"before": measure_start, "after": measure_end}
        energy_thread = read_energy_at(times, node_ips, readings)

    def one(kind, media):
        return process(kind, media, upload_dir)

    if registry is None:
        records, backlog = run_queue(arrivals, one, measure_end)
    else:
        registry.reset_stats()
        max_parallel = args.max_parallel or args.slots * len(registry.workers)
        records, backlog = run_dispatched(arrivals, one, measure_end, max_parallel)
    add_queue_lengths(records)
    summary = summarize_step(
        records,
        arrivals,
        backlog,
        measure_start,
        measure_end,
        {name: d for name, d in durations.items()},
        args.latency_limit,
        args.min_throughput,
    )
    summary.update(step=index, photo_cameras=n_photo, video_cameras=n_video)
    if registry is not None:
        summary.update(dispatcher=registry.policy, worker_stats=json.dumps(registry.stats()))
    if energy_thread:
        energy_thread.join()
        per_node, joules = energy_between(readings.get("before"), readings.get("after"))
        if joules is not None:
            active = joules - idle_w * args.window
            summary.update(
                idle_w=idle_w,
                energy_j=joules,
                energy_active_j=active,
                power_active_w=active / args.window,
                energy_by_node_j=per_node,
            )
    for r in records:
        r.update(step=index, photo_cameras=n_photo, video_cameras=n_video)
    return summary, records


def write_metadata(args, run_dir, photos, videos, placement):
    info = {
        "args": {key: str(value) for key, value in vars(args).items()},
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "ray": package_version("ray"),
        "openvino": package_version("openvino"),
        "host": platform.node(),
        "photos": [p.name for p in photos],
        "videos": [v.name for v in videos],
        "placement": placement,
    }
    (run_dir / "meta.json").write_text(json.dumps(info, indent=2))


def main(argv=None):
    args = parse_args(argv)
    photos = sorted(
        p
        for pattern in ("*.jpg", "*.jpeg", "*.png")
        for p in args.photos_dir.expanduser().glob(pattern)
    )
    videos = sorted(args.videos_dir.expanduser().glob("*.mp4"))
    if any(n for n, _ in args.steps) and not photos:
        sys.exit(f"No photos in {args.photos_dir}")
    if any(v for _, v in args.steps) and not videos:
        sys.exit(f"No videos in {args.videos_dir}")
    durations = {v.name: video_duration(v) for v in videos}

    run_dir = args.results_dir.expanduser() / args.run_id
    run_dir.mkdir(parents=True)
    work_dir = run_dir / "work"
    work_dir.mkdir()
    (REPO_ROOT / "classification_output").mkdir(exist_ok=True)
    os.chdir(work_dir)  # AiModel writes its outputs relative to the working directory

    node_names = alive_node_names()
    node_ips = list(node_names)
    idle_w = None
    if args.measure_energy:
        print(f"Measuring the idle power of {len(node_ips)} node(s) for {args.idle_seconds:g}s...")
        idle_by_ip = measure_idle_watts(node_ips, args.idle_seconds)
        if idle_by_ip is None:
            sys.exit("Cannot read the energy counters of every node: is the power exporter up?")
        idle_w = sum(idle_by_ip.values())
        print(f"Idle power of the cluster: {idle_w:.1f} W", flush=True)

    import ray

    registry = None
    t0 = time.perf_counter()
    if args.dispatcher:
        print(f"Creating the warm workers ({args.dispatcher})...", flush=True)
        ai_model, dispatched, registry = make_dispatched_model(args, photos)
        placement = {"dispatcher": args.dispatcher, "slots": args.slots,
                     "workers": [w.name for w in registry.workers],
                     "t_init_s": time.perf_counter() - t0}  # fmt: skip
        process = make_processor(ai_model, dispatched)
    else:
        print("Creating the WADAS pipeline (one scheduling decision)...", flush=True)
        ai_model = make_ai_model(args)
        pipeline = ai_model.detection_pipeline
        ray_nodes = {n["NodeID"]: n for n in ray.nodes()}
        placement = {
            "det_node": node_name(pipeline.best_detection_node, ray_nodes),
            "det_device": pipeline.best_detection_node["device"],
            "cls_node": node_name(pipeline.best_classification_node, ray_nodes),
            "cls_device": pipeline.best_classification_node["device"],
            "t_init_s": time.perf_counter() - t0,
        }
        process = make_processor(ai_model)
    print(f"Placement: {placement}", flush=True)
    write_metadata(args, run_dir, photos, videos, placement)

    base = {"run_id": args.run_id, "use_case": args.use_case,
            "scheduler_params": args.scheduler_params,
            **{k: v for k, v in placement.items() if k in STEP_FIELDS}}  # fmt: skip
    try:
        with (
            open(run_dir / "steps.csv", "w", newline="") as steps_file,
            open(run_dir / "items.csv", "w", newline="") as items_file,
        ):
            steps_csv = csv.DictWriter(steps_file, STEP_FIELDS, extrasaction="ignore")
            items_csv = csv.DictWriter(items_file, ITEM_FIELDS, extrasaction="ignore")
            steps_csv.writeheader()
            items_csv.writeheader()
            for index, step in enumerate(args.steps):
                # The pipeline keeps Ray connected: ask it directly.
                alive = sum(1 for node in ray.nodes() if node["Alive"])
                if alive < args.expected_nodes:
                    print(f"ABORT: {alive} Ray nodes alive instead of {args.expected_nodes}")
                    break
                print(f"=== {datetime.now():%H:%M:%S} step {index}: "
                      f"{step[0]} photo + {step[1]} video camera(s)", flush=True)  # fmt: skip
                summary, records = run_step(
                    args, index, step, process, photos, videos, durations, node_ips, idle_w,
                    work_dir, registry,
                )  # fmt: skip
                if summary.get("energy_by_node_j"):
                    summary["energy_by_node_j"] = json.dumps(
                        {node_names.get(ip, ip): round(j, 1)
                         for ip, j in summary["energy_by_node_j"].items()}  # fmt: skip
                    )
                steps_csv.writerow(dict(base, **summary))
                items_csv.writerows(records)
                steps_file.flush()
                items_file.flush()
                print_step(summary)

                if index < len(args.steps) - 1:
                    time.sleep(args.cooldown - args.step_idle)
                    if args.measure_energy:
                        idle_by_ip = measure_idle_watts(node_ips, args.step_idle)
                        if idle_by_ip:
                            idle_w = sum(idle_by_ip.values())
                    else:
                        time.sleep(args.step_idle)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        clean_outputs(work_dir)
        if ray.is_initialized():
            ray.shutdown()
    print(f"\nResults: {run_dir}")


def print_step(s):
    parts = []
    if s["photos_offered"]:
        p95 = s["photo_p95_s"]
        parts.append(f"photos {s['photos_done']}/{s['photos_offered']}, p95 "
                     f"{'n/a' if p95 is None else f'{p95:.1f} s'}")  # fmt: skip
    if s["clips_offered"]:
        p95 = s["clip_p95_s"]
        parts.append(f"clips {s['clips_done']}/{s['clips_offered']}, p95 "
                     f"{'n/a' if p95 is None else f'{p95:.1f} s'}")  # fmt: skip
    parts.append(f"backlog {s['backlog']}, delay +{s['delay_growth_s_per_min']:.1f} s/min")
    verdict = "keeps up" if s["keeps_up"] else s["reasons"]
    print("    " + ", ".join(parts) + f": {verdict}", flush=True)


if __name__ == "__main__":
    main()
