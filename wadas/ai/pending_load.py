# This file is part of WADAS project.
#
# WADAS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# WADAS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with WADAS. If not, see <https://www.gnu.org/licenses/>.
#
# Description: Shared record of recent placement decisions. The models a decision creates only
# show up in the node metrics a few seconds later, so requests that arrive together would all
# see the same node as idle: the schedulers use this record to count what was just assigned.

import logging
import time

import ray

logger = logging.getLogger(__name__)

# The name changes with the record format, so that an actor left running by an older version
# of the code is never mistaken for this one.
REGISTRY_NAME = "wadas_pending_load_v2"
REGISTRY_NAMESPACE = "wadas"
REGISTRY_TIMEOUT = 5.0  # seconds to wait for the registry before ignoring it


class PendingRegistry:
    """Recent placement decisions. Runs as a detached Ray actor, so that every process taking
    placement decisions (one per camera) sees the same record."""

    def __init__(self, clock=time.time):
        self._clock = clock
        self._entries = []

    def add(self, node_id, device, role, ttl):
        """Record that a model for `role` was just placed on `device` of `node_id`."""
        now = self._clock()
        self._entries.append((node_id, device, role, now, now + ttl))

    def snapshot(self):
        """Return [(node_id, device, role, age_seconds)] of the decisions not expired yet."""
        now = self._clock()
        self._entries = [entry for entry in self._entries if entry[4] > now]
        return [
            (node, device, role, now - created) for node, device, role, created, _ in self._entries
        ]


class PendingLoad:
    """Recent placement decisions, counted per node and per node device.

    Entries are (node_id, device, role, age_seconds); the age may be left out, which counts
    the entry as brand new. The counts take `max_age` (seconds), to look only at the most
    recent decisions: a model keeps using memory long after its load shows in the metrics.
    """

    def __init__(self, entries=()):
        self._entries = [
            (entry[0], entry[1], entry[3] if len(entry) > 3 else 0.0) for entry in entries
        ]

    def on_node(self, node_id, max_age=None):
        return sum(1 for node, _, age in self._entries if node == node_id and _recent(age, max_age))

    def on_device(self, node_id, device, max_age=None):
        return sum(
            1
            for node, dev, age in self._entries
            if node == node_id and dev == device and _recent(age, max_age)
        )


def _recent(age, max_age):
    return max_age is None or age <= max_age


def _registry():
    """Return the shared registry, creating it the first time (several processes may race)."""
    options = {
        "name": REGISTRY_NAME,
        "namespace": REGISTRY_NAMESPACE,
        "lifetime": "detached",
        "get_if_exists": True,
        "num_cpus": 0,
    }
    try:
        return ray.remote(PendingRegistry).options(**options).remote()
    except ValueError:
        return ray.get_actor(REGISTRY_NAME, namespace=REGISTRY_NAMESPACE)


def read_pending():
    """Read the recent decisions; on any failure behave as if there were none."""
    try:
        # Keep the handle in a variable: a method only holds a weak reference to its actor.
        registry = _registry()
        entries = ray.get(registry.snapshot.remote(), timeout=REGISTRY_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - a broken registry must not stop scheduling
        logger.warning("Pending-load registry unavailable, ignoring it: %s", exc)
        return PendingLoad()
    return PendingLoad(entries)


def reserve(node_id, device, role, ttl):
    """Record a decision so that the next ones see it."""
    try:
        registry = _registry()
        ray.get(registry.add.remote(node_id, device, role, ttl), timeout=REGISTRY_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - a broken registry must not stop scheduling
        logger.warning("Could not record the placement decision: %s", exc)
