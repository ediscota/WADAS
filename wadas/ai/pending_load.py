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
from collections import Counter

import ray

logger = logging.getLogger(__name__)

REGISTRY_NAME = "wadas_pending_load"
REGISTRY_NAMESPACE = "wadas"
REGISTRY_TIMEOUT = 5.0  # seconds to wait for the registry before ignoring it


class PendingRegistry:
    """Decisions taken in the last few seconds. Runs as a detached Ray actor, so that every
    process taking placement decisions (one per camera) sees the same record."""

    def __init__(self, clock=time.time):
        self._clock = clock
        self._entries = []

    def add(self, node_id, device, role, ttl):
        """Record that a model for `role` was just placed on `device` of `node_id`."""
        self._entries.append((node_id, device, role, self._clock() + ttl))

    def snapshot(self):
        """Return [(node_id, device, role)] of the decisions that have not expired yet."""
        now = self._clock()
        self._entries = [entry for entry in self._entries if entry[3] > now]
        return [entry[:3] for entry in self._entries]


class PendingLoad:
    """Counts of recent decisions per node and per node device."""

    def __init__(self, entries=()):
        self._per_node = Counter(node_id for node_id, _, _ in entries)
        self._per_device = Counter((node_id, device) for node_id, device, _ in entries)

    def on_node(self, node_id):
        return self._per_node[node_id]

    def on_device(self, node_id, device):
        return self._per_device[(node_id, device)]


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
