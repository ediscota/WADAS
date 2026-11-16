# Description: Simulate one camera streaming frames at a fixed rate through the distributed
# WADAS pipeline, and record the latency of every frame. Used by run_stress.py.

"""Simulate one continuous camera stream and write frames.csv and result.json.

The stream places its pipeline once, as a WADAS camera does, then sends one frame every
1/fps seconds, looping over the given videos forever (an "infinite video"), until --stop-at.
A frame that cannot start on time waits in the queue: frames are never dropped, so a stream
that cannot keep up shows a latency that grows over time.

Frames are processed in memory with DetectionPipeline.run_detection and .classify, the same
remote models WADAS uses, without the disk writes of AiModel (alarm images).

Run it on the Ray head node. Usually launched by run_stress.py, one process per stream.
"""

import argparse
import csv
import itertools
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

USE_CASES = ("MONITORING", "ACTUATOR", "CRITICAL", "PINNED")
FRAME_FIELDS = [
    "frame",
    "arrival",
    "start",
    "end",
    "latency_s",
    "late",
    "det_s",
    "cls_s",
    "n_animals",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", nargs="+", required=True, help="Videos to loop over, in order")
    parser.add_argument("--use-case", required=True, choices=USE_CASES)
    parser.add_argument("--scheduler-params", default="{}", help="JSON with scheduler options")
    parser.add_argument("--detection-model", help="Detection model version")
    parser.add_argument("--classification-model", help="Classification model version")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames sent per second")
    parser.add_argument(
        "--stop-at", type=float, required=True, help="Epoch time after which no frame is sent"
    )
    parser.add_argument(
        "--drain",
        type=float,
        default=10.0,
        help="Seconds after --stop-at to finish the queued frames before giving up",
    )
    parser.add_argument("--stream-id", default="0")
    parser.add_argument("--out-dir", required=True, help="Stream dir: frames.csv, result.json")
    return parser.parse_args(argv)


def run_paced(
    frames, process, start_at, stop_at, fps, deadline, records, clock=time.time, sleep=time.sleep
):
    """Send frames at start_at + k / fps until stop_at, process each one and append its record
    to `records` (filled as it goes, so that a crash keeps the frames done so far).

    Return the number of frames that arrived before stop_at but were never processed.

    `frames` is an iterator of frames, `process(frame)` returns a dict of extra fields. A frame
    whose arrival time is already past starts at once (it was queued). Processing stops at
    `deadline` even if frames arrived before stop_at are still queued; the result tells how
    many were left. Every record has frame, arrival, start, end, latency_s (end - arrival) and
    late: whether a newer frame had already arrived when this one started, i.e. whether a
    camera that keeps only the latest frame would have dropped it.
    """
    period = 1.0 / fps
    for k in itertools.count():
        arrival = start_at + k * period
        if arrival >= stop_at:
            return 0
        if clock() >= deadline:
            sent = int((stop_at - start_at) * fps - 1e-9) + 1
            return sent - k
        frame = next(frames)
        now = clock()
        if now < arrival:
            sleep(arrival - now)
        start = clock()
        extra = process(frame)
        end = clock()
        records.append(
            dict(
                extra,
                frame=k,
                arrival=arrival,
                start=start,
                end=end,
                latency_s=end - arrival,
                late=start - arrival > period,
            )
        )


def looping_frames(ai_model, videos):
    """Frames of the videos, sampled at AiModel.video_fps, starting over at the end."""
    while True:
        produced = False
        for video in videos:
            for frame, _ in ai_model.get_video_frames(str(video)):
                produced = True
                yield frame
        if not produced:
            raise RuntimeError("None of the videos produced a frame")


def make_processor(pipeline, detection_threshold, classification_threshold, clock=time.perf_counter):
    """Detection on every frame, classification when an animal is found, in memory."""

    def process(frame):
        t0 = clock()
        results = pipeline.run_detection(frame, detection_threshold, filter_animals=True)
        t1 = clock()
        n_animals = len(results["detections"].xyxy)
        if n_animals:
            pipeline.classify(frame, results, classification_threshold)
        t2 = clock()
        return {"det_s": t1 - t0, "cls_s": t2 - t1 if n_animals else 0.0, "n_animals": n_animals}

    return process


def write_frames(path, records):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, FRAME_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def node_name(node, ray_nodes):
    info = ray_nodes.get(node["id_node"], {})
    return info.get("NodeManagerHostname") or info.get("NodeManagerAddress") or node["id_node"]


def run_stream(args, videos, row, out_dir):
    import ray

    from wadas.domain.ai_model import AiModel

    try:
        AiModel.distributed_inference = True
        AiModel.use_case = args.use_case
        AiModel.scheduler_params = json.loads(args.scheduler_params)
        if args.detection_model:
            AiModel.detection_model_version = args.detection_model
        if args.classification_model:
            AiModel.classification_model_version = args.classification_model
        AiModel.video_fps = args.fps

        t0 = time.perf_counter()
        ai_model = AiModel()
        pipeline = ai_model.detection_pipeline
        ray_nodes = {n["NodeID"]: n for n in ray.nodes()}
        row.update(
            {
                "driver_host": socket.gethostname(),
                "det_node": node_name(pipeline.best_detection_node, ray_nodes),
                "det_device": pipeline.best_detection_node["device"],
                "cls_node": node_name(pipeline.best_classification_node, ray_nodes),
                "cls_device": pipeline.best_classification_node["device"],
                "t_init_s": time.perf_counter() - t0,
            }
        )

        process = make_processor(
            pipeline, AiModel.detection_threshold, AiModel.classification_threshold
        )
        start_at = time.time()
        row["first_arrival"] = start_at
        records = []
        try:
            row["unprocessed"] = run_paced(
                looping_frames(ai_model, videos),
                process,
                start_at,
                args.stop_at,
                args.fps,
                args.stop_at + args.drain,
                records,
            )
        finally:
            write_frames(out_dir / "frames.csv", records)
            row["n_frames"] = len(records)
    finally:
        if ray.is_initialized():
            ray.shutdown()


def main(argv=None):
    args = parse_args(argv)
    videos = [Path(v).resolve() for v in args.videos]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # AiModel creates its output folders relative to the cwd: keep streams apart.
    os.chdir(out_dir)

    row = {
        "stream_id": args.stream_id,
        "use_case": args.use_case,
        "scheduler_params": args.scheduler_params,
        "video": videos[0].name,
        "status": "error",
    }
    try:
        missing = [str(v) for v in videos if not v.is_file()]
        if missing:
            raise FileNotFoundError(", ".join(missing))
        run_stream(args, videos, row, out_dir)
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - any failure must end up in result.json
        row["error"] = f"{type(exc).__name__}: {exc}"[:500].replace("\n", " | ")
        traceback.print_exc()

    (out_dir / "result.json").write_text(json.dumps(row, indent=2))
    print(json.dumps(row))
    return 0 if row["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
