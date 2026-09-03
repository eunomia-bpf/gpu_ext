"""CPU-only slow-token and cancellation regressions for the SSE adapter."""

import ast
import asyncio
import json
import os
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from moe_infinity.entrypoints.openai import api_server_v2 as server
from moe_infinity.entrypoints.openai import revision_server
from moe_infinity.serving.stream import StreamManager


class CountedIterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def __next__(self):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self.active != 1:
                raise RuntimeError("concurrent next")
            return next(self.iterator)
        finally:
            with self.lock:
                self.active -= 1

    def close(self):
        self.iterator.close()


class Request:
    closed = False

    async def is_disconnected(self):
        return self.closed


class Runtime:
    def __init__(self):
        self.aborted = []

    def abort_request(self, request_id):
        self.aborted.append(request_id)


class StreamingRepairTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.manager = StreamManager()
        self.stream = CountedIterator(self.manager.create_stream("test", "model"))
        self.runtime = Runtime()
        self.request = Request()
        self.runtime_patch = patch.object(server, "_ensure_runtime_ready",
                                         return_value=(self.runtime, self.manager))
        self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)

    async def produce(self):
        await asyncio.sleep(0.25)  # Cross multiple 100-ms timeout boundaries.
        self.manager.push_token("test", "\n\n", False)
        await asyncio.sleep(0.02)
        self.manager.push_token("test", "ABC", True, "length")

    async def consume(self, handler=revision_server.lossless_completion_events):
        return [event async for event in handler(request_id="test", created=0,
                    model_name="model", stream=self.stream, raw_request=self.request)]

    async def test_delayed_first_token_survives_without_concurrent_next(self):
        producer = asyncio.create_task(self.produce())
        events = await asyncio.wait_for(self.consume(), 2)
        await producer
        texts = [json.loads(event[6:])["choices"][0]["text"]
                 for event in events if event != "data: [DONE]\n\n"]
        self.assertEqual(texts, ["\n\n", "ABC"])
        self.assertEqual(self.stream.peak, 1)
        self.assertEqual(self.stream.active, 0)
        self.assertEqual(self.manager._streams, {})
        self.assertEqual(self.runtime.aborted, [])

    async def test_disconnect_wakes_and_joins_pending_worker(self):
        task = asyncio.create_task(self.consume())
        await asyncio.sleep(0.15)
        self.request.closed = True
        self.assertEqual(await asyncio.wait_for(task, 1), [])
        self.assertEqual(self.runtime.aborted, ["test"])
        self.assertEqual(self.stream.active, 0)
        self.assertEqual(self.manager._streams, {})

    async def test_cancellation_wakes_and_joins_pending_worker(self):
        task = asyncio.create_task(self.consume())
        await asyncio.sleep(0.15)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        self.assertEqual(self.runtime.aborted, ["test"])
        self.assertEqual(self.stream.active, 0)
        self.assertEqual(self.manager._streams, {})

    async def test_original_timeout_adapter_reproduces_concurrent_next(self):
        tree = ast.parse(Path(server.__file__).read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.AsyncFunctionDef)
                        and node.name == "_completion_event_generator")
        namespace = dict(vars(server))
        exec(compile(ast.Module(body=[function], type_ignores=[]), "upstream-stream", "exec"), namespace)
        producer = asyncio.create_task(self.produce())
        try:
            with self.assertRaisesRegex(RuntimeError, "concurrent next"):
                await self.consume(namespace["_completion_event_generator"])
        finally:
            await producer
            await asyncio.sleep(0.05)
            self.stream.close()
        self.assertEqual(self.stream.active, 0)


if __name__ == "__main__":
    unittest.main()
