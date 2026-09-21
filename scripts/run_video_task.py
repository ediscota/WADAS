# Description: Run one video through the distributed detection + classification
# pipeline and record where the scheduler placed the models and how long it took.

"""Run one video through the distributed WADAS pipeline and write result.json.

Run it on the Ray head node (the pipeline connects to the local cluster).
Usually launched by run_scenario.py, one process per task.
"""

import argparse
import json
import os
import socket
import statistics
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

USE_CASES = ("MONITORING", "ACTUATOR", "CRITICAL")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--use-case", required=True, choices=USE_CASES)
    parser.add_argument("--out-dir", required=True, help="Task working dir; result.json goes here")
    parser.add_argument("--task-id", default="0")
    parser.add_argument("--fps", type=int, default=1, help="Video frames sampled per second")
    return parser.parse_args(argv)


def timed(func, latencies):
    """Wrap func so that the duration of every call is appended to latencies."""

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            latencies.append(time.perf_counter() - start)

    return wrapper


def node_name(node, ray_nodes):
    info = ray_nodes.get(node["id_node"], {})
    return info.get("NodeManagerHostname") or info.get("NodeManagerAddress") or node["id_node"]


def run_task(args, video, row):
    import ray

    from wadas.domain.ai_model import AiModel

    try:
        AiModel.distributed_inference = True
        AiModel.use_case = args.use_case
        AiModel.video_fps = args.fps

        started_at = time.time()
        t0 = time.perf_counter()
        ai_model = AiModel()
        # Scheduling decision plus asynchronous actor creation: the models are actually
        # loaded on the workers during the first detection call, not here.
        t_init = time.perf_counter() - t0

        pipeline = ai_model.detection_pipeline
        ray_nodes = {n["NodeID"]: n for n in ray.nodes()}
        row.update(
            {
                "driver_host": socket.gethostname(),
                "started_at": started_at,
                "det_node": node_name(pipeline.best_detection_node, ray_nodes),
                "det_device": pipeline.best_detection_node["device"],
                "cls_node": node_name(pipeline.best_classification_node, ray_nodes),
                "cls_device": pipeline.best_classification_node["device"],
                "t_init_s": t_init,
            }
        )

        det_latencies, cls_latencies = [], []
        pipeline.run_detection = timed(pipeline.run_detection, det_latencies)
        pipeline.classify = timed(pipeline.classify, cls_latencies)

        t1 = time.perf_counter()
        detections = list(ai_model.process_video(str(video), True))
        t_detection = time.perf_counter() - t1

        t2 = time.perf_counter()
        for results, detected_img_path, _ in detections:
            ai_model.classify(detected_img_path, results)
        t_classification = time.perf_counter() - t2

        row.update(
            {
                "n_frames": len(det_latencies),
                "n_frames_with_animals": len(detections),
                "t_detection_phase_s": t_detection,
                "t_classification_phase_s": t_classification,
                "t_total_s": time.perf_counter() - t0,
                "det_first_call_s": det_latencies[0] if det_latencies else None,
                "det_median_s": statistics.median(det_latencies) if det_latencies else None,
                "cls_median_s": statistics.median(cls_latencies) if cls_latencies else None,
                "det_latencies_s": det_latencies,
                "cls_latencies_s": cls_latencies,
            }
        )
    finally:
        if ray.is_initialized():
            ray.shutdown()


def main(argv=None):
    args = parse_args(argv)
    video = Path(args.video).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # AiModel.classify saves into <repo>/classification_output, which AiModel only creates
    # relative to the cwd.
    (REPO_ROOT / "classification_output").mkdir(exist_ok=True)
    # process_video saves frames under cwd-relative folders with names that do not include
    # the task: concurrent tasks need separate working directories or they overwrite each other.
    os.chdir(out_dir)

    row = {
        "task_id": args.task_id,
        "use_case": args.use_case,
        "video": video.name,
        "status": "error",
    }
    try:
        if not video.is_file():
            raise FileNotFoundError(video)
        run_task(args, video, row)
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - any failure must end up in result.json
        row["error"] = f"{type(exc).__name__}: {exc}"[:500].replace("\n", " | ")
        traceback.print_exc()

    (out_dir / "result.json").write_text(json.dumps(row, indent=2))
    print(json.dumps({k: v for k, v in row.items() if not k.endswith("_latencies_s")}))
    return 0 if row["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
