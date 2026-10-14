#!/usr/bin/env python3
# Description: Simulate one camera sending photos and videos to the WADAS pipeline, and record
# when each event arrived, when it started being served and when its result was ready.

"""Simulate one camera. Usually launched by run_load.py, one process per camera.

The camera builds its own pipeline once (in distributed modes the scheduler picks the nodes
and devices at that moment), warms the models up, waits for the common start time written by
the launcher, then serves its events in order. An event that arrives while the previous one is
still being served waits: that wait is part of its latency.

  event latency = result ready - scheduled arrival
  event wait    = start of service - scheduled arrival
  event service = result ready - start of service
"""

import argparse
import json
import os
import shutil
import socket
import sys
import time
import traceback
from pathlib import Path

from loadgen import build_schedule

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MODES = ("MONITORING", "ACTUATOR", "CRITICAL", "LOCAL")
PHOTO_PATTERNS = ("*.jpg", "*.jpeg", "*.png")
VIDEO_PATTERNS = ("*.mp4",)
WARMUP_ATTEMPTS = 5


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-id", type=int, required=True)
    parser.add_argument("--mode", required=True, choices=MODES, help="LOCAL = no Ray, this host")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start-file", type=Path, required=True)
    parser.add_argument("--photos-dir", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--photo-interval", type=float, default=30.0, help="Mean seconds per event")
    parser.add_argument("--video-probability", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=1, help="Video frames sampled per second")
    parser.add_argument("--local-device", default="GPU", help="Device used in LOCAL mode")
    parser.add_argument("--start-timeout", type=float, default=900.0)
    return parser.parse_args(argv)


def list_media(directory, patterns):
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted({p.name for pattern in patterns for p in directory.glob(pattern)})


def node_name(node, ray_nodes):
    info = ray_nodes.get(node["id_node"], {})
    return info.get("NodeManagerHostname") or info.get("NodeManagerAddress") or node["id_node"]


def describe_placement(args, ai_model):
    if args.mode == "LOCAL":
        host = socket.gethostname()
        return {
            "det_node": host,
            "det_device": args.local_device,
            "cls_node": host,
            "cls_device": args.local_device,
        }
    import ray

    ray_nodes = {n["NodeID"]: n for n in ray.nodes()}
    pipeline = ai_model.detection_pipeline
    return {
        "det_node": node_name(pipeline.best_detection_node, ray_nodes),
        "det_device": pipeline.best_detection_node["device"],
        "cls_node": node_name(pipeline.best_classification_node, ray_nodes),
        "cls_device": pipeline.best_classification_node["device"],
    }


def discard(classification_result):
    """Delete the annotated image written by AiModel.classify: only its timing matters."""
    if isinstance(classification_result, tuple) and classification_result[0]:
        Path(classification_result[0]).unlink(missing_ok=True)


def process_photo(ai_model, source, work_dir, tag):
    # process_image blurs the file in place and deletes it when no animal is found, so it
    # must work on a copy of the shared photo.
    copy = work_dir / f"{tag}.jpg"
    shutil.copyfile(source, copy)

    started = time.perf_counter()
    results, detected_path = ai_model.process_image(str(copy), True)
    t_detection = time.perf_counter() - started
    if results is None:
        raise RuntimeError(f"process_image could not process {source.name}")

    t_classification = 0.0
    if detected_path:
        started = time.perf_counter()
        discard(ai_model.classify(detected_path, results))
        t_classification = time.perf_counter() - started
    return {
        "t_detection_s": t_detection,
        "t_classification_s": t_classification,
        "n_animals": len(results["detections"].xyxy),
    }


def process_video(ai_model, source, work_dir, tag):
    started = time.perf_counter()
    detections = list(ai_model.process_video(str(source), True))
    t_detection = time.perf_counter() - started

    started = time.perf_counter()
    for index, (results, detected_path, _) in enumerate(detections):
        if not detected_path:
            continue
        # classify names its output after the image: give every frame its own name.
        frame = work_dir / f"{tag}_f{index}.jpg"
        shutil.copyfile(detected_path, frame)
        discard(ai_model.classify(str(frame), results))
    return {
        "t_detection_s": t_detection,
        "t_classification_s": time.perf_counter() - started,
        "n_frames_with_animals": len(detections),
    }


def warm_up(ai_model, photos, videos, args, work_dir):
    """Run inferences until both models have processed an animal.

    The first inference of each model pays a device specific compile cost. A camera in
    production has been running for a long time, so that cost must not land on an event.
    """
    done = 0
    try:
        for name in photos[:WARMUP_ATTEMPTS]:
            done += 1
            detail = process_photo(ai_model, Path(args.photos_dir) / name, work_dir, "warmup")
            if detail["n_animals"] > 0:
                return done
        if not photos and videos:
            process_video(ai_model, Path(args.videos_dir) / videos[0], work_dir, "warmup")
            done += 1
        return done
    finally:
        for leftover in work_dir.glob("warmup*"):
            leftover.unlink(missing_ok=True)


def wait_for_start(start_file, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if start_file.exists():
            return float(start_file.read_text())
        time.sleep(0.2)
    raise TimeoutError(f"no start signal in {start_file} after {timeout:.0f}s")


def serve_events(args, ai_model, schedule, start_at, work_dir):
    done = 0
    with open("events.jsonl", "a") as sink:
        for event in schedule:
            scheduled_at = start_at + event.offset_s
            time.sleep(max(0.0, scheduled_at - time.time()))
            started_at = time.time()
            record = {
                "camera_id": args.camera_id,
                "event_id": event.event_id,
                "kind": event.kind,
                "media": event.media,
                "scheduled_at": scheduled_at,
            }
            tag = f"cam{args.camera_id}_e{event.event_id}"
            try:
                handler, directory = (
                    (process_video, args.videos_dir)
                    if event.kind == "video"
                    else (process_photo, args.photos_dir)
                )
                record.update(handler(ai_model, Path(directory) / event.media, work_dir, tag))
                record["status"] = "ok"
            except Exception as exc:  # noqa: BLE001 - a failed event must not stop the camera
                record["status"] = "error"
                record["error"] = f"{type(exc).__name__}: {exc}"[:300].replace("\n", " | ")
            finally:
                for leftover in work_dir.glob(f"{tag}*"):
                    leftover.unlink(missing_ok=True)
            finished_at = time.time()
            record.update(
                started_at=started_at,
                finished_at=finished_at,
                wait_s=started_at - scheduled_at,
                service_s=finished_at - started_at,
                latency_s=finished_at - scheduled_at,
            )
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            done += 1
    return done


def run_camera(args, camera):
    from wadas.domain.ai_model import AiModel

    photos = list_media(args.photos_dir, PHOTO_PATTERNS)
    videos = list_media(args.videos_dir, VIDEO_PATTERNS)
    schedule = build_schedule(
        args.camera_id,
        args.duration,
        args.photo_interval,
        args.video_probability,
        photos,
        videos,
        args.seed,
    )
    camera["n_events_planned"] = len(schedule)

    AiModel.video_fps = args.fps
    if args.mode == "LOCAL":
        AiModel.distributed_inference = False
        AiModel.detection_device = AiModel.classification_device = args.local_device
    else:
        AiModel.distributed_inference = True
        AiModel.use_case = args.mode

    started = time.perf_counter()
    ai_model = AiModel()
    camera["t_init_s"] = time.perf_counter() - started
    camera.update(describe_placement(args, ai_model))

    work_dir = Path("work")
    work_dir.mkdir(exist_ok=True)
    started = time.perf_counter()
    camera["warmup_inferences"] = warm_up(ai_model, photos, videos, args, work_dir)
    camera["t_warmup_s"] = time.perf_counter() - started

    Path("ready").write_text(str(time.time()))
    start_at = wait_for_start(args.start_file, args.start_timeout)
    camera["n_events_done"] = serve_events(args, ai_model, schedule, start_at, work_dir)


def main(argv=None):
    args = parse_args(argv)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # AiModel saves into folders relative to the working directory, and classify saves into
    # <repo>/classification_output: give each camera its own directory.
    (REPO_ROOT / "classification_output").mkdir(exist_ok=True)
    args.start_file = args.start_file.resolve()
    args.photos_dir = args.photos_dir.resolve()
    args.videos_dir = args.videos_dir.resolve()
    os.chdir(out_dir)

    camera = {"camera_id": args.camera_id, "mode": args.mode, "status": "error"}
    try:
        run_camera(args, camera)
        camera["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - any failure must end up in camera.json
        camera["error"] = f"{type(exc).__name__}: {exc}"[:500].replace("\n", " | ")
        traceback.print_exc()
    finally:
        try:
            import ray

            if ray.is_initialized():
                ray.shutdown()
        except ImportError:
            pass

    Path("camera.json").write_text(json.dumps(camera, indent=2))
    print(json.dumps(camera))
    return 0 if camera["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
