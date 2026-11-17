# Description: Prototype of a WADAS pipeline spread over the whole cluster: warm model workers on
# every device, a registry of the work in flight, and a scheduler that picks a worker for every
# detection and classification call. Used by run_camera_test.py --dispatcher.

"""Dispatcher prototype (see specifica_prototipo_dispatcher.md).

WADAS code is not changed. AiModel only uses four things of its DetectionPipeline:
run_detection, classify, filter_animal_detections and animal_class_idx. DispatchedPipeline offers
the same four, and every run_detection or classify call goes to the worker that the scheduler
chooses at that moment. Several items run in parallel, each in its own thread, through the
unchanged AiModel methods (process_image, classify, process_video_offline).

Workers are Ray actors, one per device and role, created once with their model loaded. The
registry counts, for every worker, the calls in flight and the seconds of work they represent;
a worker takes at most `slots` calls at a time, the rest wait in the WADAS queue.
"""

import itertools
import threading
import time
from dataclasses import dataclass, field

ROLES = ("DETECTION", "CLASSIFICATION")
POLICIES = ("round_robin", "least_loaded", "cost")
# (time weight, energy weight) of the cost policy, by use case (fixed before start-up)
WEIGHTS = {"ACTUATOR": (1.0, 0.0), "CRITICAL": (1.0, 1.0), "MONITORING": (0.3, 1.0)}


@dataclass
class Worker:
    name: str  # "wadas1:NPU:DETECTION"
    host: str
    device: str
    role: str
    seconds_per_item: float
    joules_per_item: float
    actor: object = None
    in_flight: int = 0
    calls: int = 0
    items: int = 0
    busy_s: float = 0.0

    @property
    def device_key(self):
        return f"{self.host}:{self.device}"


def workers_from_profile(profile, roles=ROLES, exclude=()):
    """Workers for every device and role of a steady-state profile {"host:DEVICE": {ROLE: {s, J}}}.
    `exclude` lists "host:DEVICE" or "host:DEVICE:ROLE" to leave out."""
    workers = []
    for key, by_role in sorted(profile.items()):
        host, _, device = key.partition(":")
        for role in roles:
            m = by_role.get(role)
            name = f"{key}:{role}"
            if not m or m.get("s") is None or m.get("J") is None:
                continue
            if key in exclude or name in exclude:
                continue
            workers.append(Worker(name, host, device, role, float(m["s"]), float(m["J"])))
    return workers


class Registry:
    """Calls in flight per worker and seconds of work in flight per device, under one lock.

    `acquire` blocks until the policy finds a worker of the role with a free slot, reserves it
    and returns it; `release` frees it. The work in flight of a device sums all its workers: a
    GPU running detection and classification is one device."""

    def __init__(self, workers, policy, slots=2, weights=(1.0, 1.0), clock=time.monotonic):
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
        self.workers = workers
        self.policy = policy
        self.slots = slots
        self.weights = weights
        self.clock = clock
        self.condition = threading.Condition()
        self.work_s = {w.device_key: 0.0 for w in workers}  # seconds of work in flight
        self.calls_on_device = {w.device_key: 0 for w in workers}
        self.round_robin = {role: itertools.cycle([w for w in workers if w.role == role])
                            for role in ROLES}  # fmt: skip
        self.refs = {}
        for role in ROLES:
            mine = [w for w in workers if w.role == role]
            if mine:
                # References that make the two terms of the cost comparable: the slowest and the
                # most expensive worker of the role are worth 1.
                self.refs[role] = (max(w.seconds_per_item for w in mine),
                                   max(w.joules_per_item for w in mine))  # fmt: skip

    def candidates(self, role):
        return [w for w in self.workers if w.role == role and w.in_flight < self.slots]

    def cost(self, worker, n_items):
        """Expected completion time and energy of n_items on this worker, weighted."""
        t_ref, j_ref = self.refs[worker.role]
        wait = self.work_s[worker.device_key] + n_items * worker.seconds_per_item
        energy = n_items * worker.joules_per_item
        w_time, w_energy = self.weights
        return w_time * wait / t_ref + w_energy * energy / j_ref

    def choose(self, role, n_items):
        """The worker the policy picks among those with a free slot, or None."""
        free = self.candidates(role)
        if not free:
            return None
        if self.policy == "round_robin":
            for _ in range(len(self.workers)):
                worker = next(self.round_robin[role])
                if worker in free:
                    return worker
        if self.policy == "least_loaded":
            return min(free, key=lambda w: (self.calls_on_device[w.device_key], w.in_flight))
        return min(free, key=lambda w: self.cost(w, n_items))

    def acquire(self, role, n_items=1):
        with self.condition:
            while (worker := self.choose(role, n_items)) is None:
                self.condition.wait()
            worker.in_flight += 1
            self.calls_on_device[worker.device_key] += 1
            self.work_s[worker.device_key] += n_items * worker.seconds_per_item
            return worker

    def release(self, worker, n_items, busy_s):
        with self.condition:
            worker.in_flight -= 1
            worker.calls += 1
            worker.items += n_items
            worker.busy_s += busy_s
            self.calls_on_device[worker.device_key] -= 1
            self.work_s[worker.device_key] = max(
                0.0, self.work_s[worker.device_key] - n_items * worker.seconds_per_item
            )
            self.condition.notify_all()

    def stats(self):
        return {w.name: {"calls": w.calls, "items": w.items, "busy_s": round(w.busy_s, 2)}
                for w in self.workers}  # fmt: skip

    def reset_stats(self):
        with self.condition:
            for w in self.workers:
                w.calls = w.items = 0
                w.busy_s = 0.0


@dataclass
class CallLog:
    """What the calls of one item did: workers used and seconds spent in remote calls."""

    det_worker: str = ""
    cls_worker: str = ""
    det_s: float = 0.0
    cls_s: float = 0.0
    calls: list = field(default_factory=list)


class DispatchedPipeline:
    """Stands in for DetectionPipeline inside AiModel: every call goes to a worker chosen now.

    `bind(worker)` returns an object with the DetectionPipeline methods bound to that worker's
    actor (see run_camera_test.py); in the tests it is a fake.
    """

    def __init__(self, registry, bind, animal_class_idx, filter_animals, clock=time.perf_counter):
        self.registry = registry
        self.bind = bind
        self.animal_class_idx = animal_class_idx
        self._filter = filter_animals
        self.clock = clock
        self.local = threading.local()

    def log(self):
        """The CallLog of the item processed by the current thread."""
        if not hasattr(self.local, "log"):
            self.local.log = CallLog()
        return self.local.log

    def start_item(self):
        self.local.log = CallLog()
        return self.local.log

    def _call(self, role, n_items, method, *args, **kwargs):
        worker = self.registry.acquire(role, n_items)
        t0 = self.clock()
        try:
            return getattr(self.bind(worker), method)(*args, **kwargs)
        finally:
            busy = self.clock() - t0
            self.registry.release(worker, n_items, busy)
            log = self.log()
            log.calls.append((worker.name, busy))
            if role == "DETECTION":
                log.det_worker, log.det_s = worker.name, log.det_s + busy
            else:
                log.cls_worker, log.cls_s = worker.name, log.cls_s + busy

    def run_detection(self, img, detection_threshold, filter_animals=True):
        n = len(img) if isinstance(img, (list, tuple)) else 1
        return self._call("DETECTION", n, "run_detection", img, detection_threshold,
                          filter_animals=filter_animals)  # fmt: skip

    def classify(self, img, results, classification_threshold):
        n = len(img) if isinstance(img, (list, tuple)) else 1
        return self._call("CLASSIFICATION", n, "classify", img, results, classification_threshold)

    def filter_animal_detections(self, results):
        return self._filter(results)


def run_dispatched(arrivals, process, stop_at, max_parallel, clock=time.time, sleep=time.sleep):
    """Process the arrivals in arrival order, up to max_parallel at a time, each in a thread.

    An item starts when it has arrived and fewer than max_parallel items are in progress; the
    others wait, in order, in the queue. No item starts after stop_at: the ones that arrived by
    then and did not start are the backlog. Return (records, backlog) once every started item
    is done; each record has arrival, start, end and what process(kind, media) returned.
    """
    records = []
    lock = threading.Lock()
    free = threading.Semaphore(max_parallel)
    threads = []

    def work(arrival, camera, kind, media, start):
        try:
            extra = dict(process(kind, media), status="ok")
        except Exception as exc:  # noqa: BLE001 - one bad item must not stop the test
            extra = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:300]}
        finally:
            free.release()
        end = clock()
        with lock:
            records.append(
                dict(extra, camera=camera, kind=kind, media=getattr(media, "name", str(media)),
                     arrival=arrival, start=start, end=end, latency_s=end - arrival,
                     wait_s=start - arrival)
            )  # fmt: skip

    backlog = 0
    for index, (arrival, camera, kind, media) in enumerate(arrivals):
        now = clock()
        if now < arrival:
            if arrival >= stop_at:
                break
            sleep(arrival - now)
        while not free.acquire(timeout=0.05):
            if clock() >= stop_at:
                break
        else:
            if clock() < stop_at:
                thread = threading.Thread(target=work, args=(arrival, camera, kind, media, clock()))
                thread.start()
                threads.append(thread)
                continue
            free.release()
        backlog = sum(1 for a in arrivals[index:] if a[0] < stop_at)
        break
    for thread in threads:
        thread.join()
    records.sort(key=lambda r: r["arrival"])
    return records, backlog
