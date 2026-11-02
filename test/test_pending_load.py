from unittest.mock import patch

import pytest

from wadas.ai import pending_load
from wadas.ai.pending_load import PendingLoad, PendingRegistry, read_pending, reserve


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_registry_returns_what_was_added():
    registry = PendingRegistry(clock=Clock())
    registry.add("n1", "GPU", "DETECTION", ttl=10)
    registry.add("n2", "NPU", "DETECTION", ttl=10)
    assert registry.snapshot() == [("n1", "GPU", "DETECTION"), ("n2", "NPU", "DETECTION")]


def test_entries_expire_after_their_ttl():
    clock = Clock()
    registry = PendingRegistry(clock=clock)
    registry.add("n1", "GPU", "DETECTION", ttl=10)
    registry.add("n2", "GPU", "DETECTION", ttl=30)
    clock.now += 11
    assert registry.snapshot() == [("n2", "GPU", "DETECTION")]
    clock.now += 20
    assert registry.snapshot() == []


def test_pending_load_counts_per_node_and_device():
    load = PendingLoad(
        [("n1", "NPU", "DETECTION"), ("n1", "GPU", "CLASSIFICATION"), ("n2", "GPU", "DETECTION")]
    )
    assert load.on_node("n1") == 2
    assert load.on_node("n3") == 0
    assert load.on_device("n1", "NPU") == 1
    assert load.on_device("n2", "NPU") == 0


def test_an_unreachable_registry_is_ignored():
    with patch.object(pending_load, "_registry", side_effect=RuntimeError("no ray")):
        assert read_pending().on_node("n1") == 0
        reserve("n1", "GPU", "DETECTION", ttl=10)  # must not raise


@pytest.mark.parametrize("empty", [(), []])
def test_empty_snapshot(empty):
    assert PendingLoad(empty).on_node("anything") == 0


class FakeRemote:
    def __init__(self, result, log):
        self._result = result
        self._log = log

    def remote(self, *args):
        self._log.append(args)
        return self._result


class FakeRegistry:
    def __init__(self, entries):
        self.log = []
        self.snapshot = FakeRemote(entries, self.log)
        self.add = FakeRemote(None, self.log)


def test_read_pending_goes_through_the_registry():
    registry = FakeRegistry([("n1", "GPU", "DETECTION"), ("n1", "NPU", "DETECTION")])
    with (
        patch.object(pending_load, "_registry", return_value=registry),
        patch.object(pending_load.ray, "get", side_effect=lambda ref, timeout=None: ref),
    ):
        load = read_pending()
    assert load.on_node("n1") == 2
    assert load.on_device("n1", "NPU") == 1


def test_reserve_sends_the_decision_to_the_registry():
    registry = FakeRegistry([])
    with (
        patch.object(pending_load, "_registry", return_value=registry),
        patch.object(pending_load.ray, "get", side_effect=lambda ref, timeout=None: ref),
    ):
        reserve("n1", "GPU", "DETECTION", ttl=10)
    assert registry.log == [("n1", "GPU", "DETECTION", 10)]
