"""Focused standard-library mocks for the upper-level GdsBackend adapter."""

import asyncio
import threading
import unittest

import lmcache_gds_backend_adapter as adapter
from lmcache_gds_policy_adapter import (
    ACTION_DEFER,
    ACTION_RECOMPUTE,
    ACTION_SUBMIT_NOW,
    FLAG_DEMAND,
    FLAG_RECOMPUTABLE,
    FLAG_SAFE_TO_DEFER,
    FLAG_SPECULATIVE,
    OP_READ,
    OP_WRITE,
    Decision,
    PolicyRequest,
)


class QueueDecider:
    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.requests = []
        self.closed = False

    def decide(self, request, request_id):
        self.requests.append((request, request_id))
        return self.decisions.pop(0)

    def close(self):
        self.closed = True


class Memory:
    def __init__(self, size=4096):
        self.tensor = object()
        self.size = size
        self.refs = 1

    def get_size(self):
        return self.size

    def ref_count_up(self):
        self.refs += 1

    def ref_count_down(self):
        self.refs -= 1


class Entry:
    size = 4096


class Backend:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever)
        self.thread.start()
        self.put_lock = threading.Lock()
        self.put_tasks = set()
        self.hot_lock = threading.Lock()
        self.hot_cache = {"a": Entry(), "b": Entry(), "c": Entry()}
        self.reads = []
        self.saves = []
        self.batches = []

    async def _async_save_bytes_to_disk(self, key, memory_obj, callback=None):
        try:
            self.saves.append(key)
        finally:
            memory_obj.ref_count_down()
            with self.put_lock:
                self.put_tasks.discard(key)
        if callback is not None:
            callback(key)

    def submit_put_task(self, key, memory_obj, on_complete_callback=None):
        memory_obj.ref_count_up()
        with self.put_lock:
            self.put_tasks.add(key)
        return asyncio.run_coroutine_threadsafe(
            self._async_save_bytes_to_disk(
                key, memory_obj, on_complete_callback
            ),
            self.loop,
        )

    def get_blocking(self, key):
        self.reads.append(key)
        return "loaded:" + key

    def get_non_blocking(self, key, location=None):
        raise AssertionError("the v0.5.4 dummy seam must be replaced")

    def batched_get_blocking(self, keys):
        self.batches.append(list(keys))
        return ["loaded:" + key for key in keys]

    def close_loop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        self.loop.close()


def provider(kind, key, memory_obj, backend):
    del key, backend
    if kind == adapter.WRITE:
        return PolicyRequest(
            op=OP_WRITE,
            flags=FLAG_SAFE_TO_DEFER,
            nbytes=memory_obj.get_size(),
        )
    if kind == adapter.READ_DEMAND:
        return PolicyRequest(op=OP_READ, flags=FLAG_DEMAND, nbytes=4096)
    return PolicyRequest(
        op=OP_READ,
        flags=FLAG_SPECULATIVE | FLAG_SAFE_TO_DEFER | FLAG_RECOMPUTABLE,
        nbytes=4096,
    )


class ControlledWait:
    def __init__(self):
        self.entered = threading.Event()
        self.release = None

    async def __call__(self, seconds):
        self.seconds = seconds
        self.release = asyncio.Event()
        self.entered.set()
        await self.release.wait()

    def allow(self, loop):
        self.entered.wait()
        loop.call_soon_threadsafe(self.release.set)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend()

    def tearDown(self):
        current = getattr(self.backend, adapter._ADAPTER_ATTR, None)
        if current is not None:
            current.close()
        self.backend.close_loop()

    def install(self, decider, **kwargs):
        return adapter.install_backend(
            self.backend,
            decider=decider,
            request_provider=provider,
            **kwargs,
        )

    def test_demand_read_submits_once(self):
        decider = QueueDecider(Decision(ACTION_SUBMIT_NOW, 0, 2, 1))
        self.install(decider)
        self.assertEqual(self.backend.get_blocking("a"), "loaded:a")
        self.assertEqual(self.backend.reads, ["a"])
        self.assertEqual(len(decider.requests), 1)
        self.assertEqual(decider.requests[0][0].flags, FLAG_DEMAND)

    def test_recompute_speculative_read_is_completed_miss_before_copy(self):
        decider = QueueDecider(Decision(ACTION_RECOMPUTE, 0, 2, 1))
        self.install(decider)
        future = self.backend.get_non_blocking("a")
        self.assertIsNone(future.result())
        self.assertEqual(self.backend.reads, [])
        self.assertEqual(len(decider.requests), 1)

    def test_deferred_speculative_read_uses_future_seam(self):
        waiter = ControlledWait()
        decider = QueueDecider(Decision(ACTION_DEFER, 7_000_000, 2, 1))
        self.install(decider, async_wait=waiter)
        future = self.backend.get_non_blocking("a")
        waiter.entered.wait()
        self.assertFalse(future.done())
        self.assertEqual(self.backend.reads, [])
        waiter.allow(self.backend.loop)
        self.assertEqual(future.result(), "loaded:a")
        self.assertEqual(self.backend.reads, ["a"])
        self.assertEqual(len(decider.requests), 1)

    def test_deferred_write_preserves_memory_reference_until_async_save(self):
        waiter = ControlledWait()
        decider = QueueDecider(Decision(ACTION_DEFER, 6_000_000, 2, 32))
        self.install(decider, async_wait=waiter)
        memory = Memory()
        future = self.backend.submit_put_task("a", memory)
        waiter.entered.wait()
        self.assertEqual(memory.refs, 2)
        self.assertIn("a", self.backend.put_tasks)
        self.assertEqual(self.backend.saves, [])
        waiter.allow(self.backend.loop)
        self.assertIsNone(future.result())
        self.assertEqual(memory.refs, 1)
        self.assertNotIn("a", self.backend.put_tasks)
        self.assertEqual(self.backend.saves, ["a"])
        self.assertEqual(len(decider.requests), 1)

    def test_batched_demand_read_decides_once_per_logical_key(self):
        decider = QueueDecider(
            Decision(ACTION_SUBMIT_NOW, 0, 0, 1),
            Decision(ACTION_SUBMIT_NOW, 0, 0, 1),
            Decision(ACTION_SUBMIT_NOW, 0, 0, 1),
        )
        self.install(decider)
        self.assertEqual(
            self.backend.batched_get_blocking(["a", "b", "c"]),
            ["loaded:a", "loaded:b", "loaded:c"],
        )
        self.assertEqual(self.backend.batches, [["a", "b", "c"]])
        self.assertEqual(len(decider.requests), 3)
        self.assertEqual([item[1] for item in decider.requests], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
