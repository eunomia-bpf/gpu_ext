"""Mocked tests for the UVM weights vLLM general plugin.

Never touch the GPU, never import real vLLM: the GPU worker class and the
torch allocator/mempool APIs are stubbed, and the plugin module is imported
fresh for every test.
"""

import importlib
import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VLLM_MODULES = (
    "vllm",
    "vllm.v1",
    "vllm.v1.worker",
    "vllm.v1.worker.gpu_worker",
)


def make_worker_cls():
    """Fresh stand-in class per test (a class-attribute wrap must not leak)."""

    class Worker:
        def __init__(self):
            self.calls: list[str] = []

        def _maybe_get_memory_pool_context(self, tag: str):
            self.calls.append(tag)
            return ("original", tag)

    return Worker


def make_fake_vllm(worker_cls):
    gpu_worker = types.ModuleType("vllm.v1.worker.gpu_worker")
    gpu_worker.Worker = worker_cls
    worker_pkg = types.ModuleType("vllm.v1.worker")
    worker_pkg.gpu_worker = gpu_worker
    v1 = types.ModuleType("vllm.v1")
    v1.worker = worker_pkg
    vllm = types.ModuleType("vllm")
    vllm.v1 = v1
    return {
        "vllm": vllm,
        "vllm.v1": v1,
        "vllm.v1.worker": worker_pkg,
        "vllm.v1.worker.gpu_worker": gpu_worker,
    }


def rendered(mock_logger):
    out = []
    for call in mock_logger.call_args_list:
        args = call.args
        out.append(args[0] % args[1:] if len(args) > 1 else args[0])
    return out


class PluginTestCase(unittest.TestCase):
    def setUp(self):
        self.worker_cls = make_worker_cls()
        self.worker = self.worker_cls()
        self.saved = {name: sys.modules.get(name) for name in VLLM_MODULES}
        sys.modules.update(make_fake_vllm(self.worker_cls))
        sys.modules.pop("uvm_weight_plugin", None)
        self.plugin = importlib.import_module("uvm_weight_plugin")

        self.alloc_cls = mock.MagicMock(name="CUDAPluggableAllocator")
        self.pool_cls = mock.MagicMock(name="MemPool")
        self.entered: list = []
        self.exited: list = []

        @contextmanager
        def fake_use_mem_pool(pool, device=None):
            self.entered.append(pool)
            try:
                yield
            finally:
                self.exited.append(pool)

        self.patches = [
            mock.patch.object(
                torch.cuda.memory, "CUDAPluggableAllocator", self.alloc_cls
            ),
            mock.patch.object(torch.cuda.memory, "MemPool", self.pool_cls),
            mock.patch.object(torch.cuda.memory, "use_mem_pool", fake_use_mem_pool),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        for name, module in self.saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        sys.modules.pop("uvm_weight_plugin", None)

    def make_so(self) -> str:
        fd, path = tempfile.mkstemp(suffix=".so")
        with os.fdopen(fd, "wb") as handle:
            handle.write(b"fake shared object")
        self.addCleanup(os.remove, path)
        return path

    def install_enabled(self, so_path: str, counters: str = ""):
        env = {"UVM_WEIGHT_PLUGIN": "1", "UVM_WEIGHT_PLUGIN_SO": so_path}
        if counters:
            env["UVM_WEIGHT_PLUGIN_COUNTERS"] = counters
        with mock.patch.dict(os.environ, env):
            self.plugin.install()
        self.assertTrue(
            hasattr(
                self.worker_cls._maybe_get_memory_pool_context,
                "_uvm_weight_plugin_wrapped",
            )
        )

    def assert_not_wrapped(self):
        self.assertFalse(
            hasattr(
                self.worker_cls._maybe_get_memory_pool_context,
                "_uvm_weight_plugin_wrapped",
            )
        )


class EntryPointTest(PluginTestCase):
    def test_general_plugin_entry_point_declared(self):
        import tomllib

        data = tomllib.loads((ROOT / "pyproject.toml").read_text())
        group = data["project"]["entry-points"]["vllm.general_plugins"]
        self.assertEqual(group["uvm_weight_plugin"], "uvm_weight_plugin:install")
        self.assertTrue(callable(self.plugin.install))


class GatingTest(PluginTestCase):
    def test_disabled_by_default_delegates(self):
        self.plugin.install()
        self.assert_not_wrapped()
        self.assertEqual(
            self.worker._maybe_get_memory_pool_context("weights"),
            ("original", "weights"),
        )
        self.assertEqual(self.worker.calls, ["weights"])

    def test_missing_so_env_stays_off(self):
        with mock.patch.dict(os.environ, {"UVM_WEIGHT_PLUGIN": "1"}):
            self.plugin.install()
        self.assert_not_wrapped()

    def test_relative_so_stays_off(self):
        with mock.patch.dict(
            os.environ,
            {
                "UVM_WEIGHT_PLUGIN": "1",
                "UVM_WEIGHT_PLUGIN_SO": "relative/uvm_allocator.so",
            },
        ):
            self.plugin.install()
        self.assert_not_wrapped()

    def test_nonexistent_so_stays_off(self):
        with mock.patch.dict(
            os.environ,
            {
                "UVM_WEIGHT_PLUGIN": "1",
                "UVM_WEIGHT_PLUGIN_SO": "/nonexistent/uvm_allocator.so",
            },
        ):
            self.plugin.install()
        self.assert_not_wrapped()

    def test_cuda_unavailable_stays_off(self):
        so = self.make_so()
        with (
            mock.patch.dict(
                os.environ,
                {
                    "UVM_WEIGHT_PLUGIN": "1",
                    "UVM_WEIGHT_PLUGIN_SO": so,
                },
            ),
            mock.patch.object(torch.cuda, "is_available", return_value=False),
        ):
            self.plugin.install()
        self.assert_not_wrapped()

    def test_install_is_idempotent(self):
        so = self.make_so()
        with mock.patch.object(self.plugin.atexit, "register") as register:
            with mock.patch.dict(
                os.environ,
                {"UVM_WEIGHT_PLUGIN": "1", "UVM_WEIGHT_PLUGIN_SO": so},
            ):
                self.plugin.install()
                first = self.worker_cls._maybe_get_memory_pool_context
                self.plugin.install()
                second = self.worker_cls._maybe_get_memory_pool_context
        self.assertIs(first, second)
        self.assertEqual(register.call_count, 1)


class WeightsPoolTest(PluginTestCase):
    def test_weights_tag_uses_uvm_pool(self):
        so = self.make_so()
        self.install_enabled(so)
        self.alloc_cls.return_value = types.SimpleNamespace(
            _allocator="fake-cpp-allocator"
        )
        self.pool_cls.return_value = "fake-pool"

        ctx = self.worker._maybe_get_memory_pool_context("weights")
        self.assertNotEqual(ctx, ("original", "weights"))
        with ctx:
            pass

        self.alloc_cls.assert_called_once_with(so, "uvm_malloc", "uvm_free")
        self.pool_cls.assert_called_once_with("fake-cpp-allocator")
        self.assertEqual(self.entered, ["fake-pool"])
        self.assertEqual(self.exited, ["fake-pool"])
        self.assertEqual(self.worker.calls, [])

    def test_non_weights_tags_delegate(self):
        so = self.make_so()
        self.install_enabled(so)
        for tag in ("kv_cache", "default"):
            self.assertEqual(
                self.worker._maybe_get_memory_pool_context(tag),
                ("original", tag),
            )
        self.assertEqual(self.worker.calls, ["kv_cache", "default"])
        self.alloc_cls.assert_not_called()
        self.pool_cls.assert_not_called()
        self.assertEqual(self.entered, [])

    def test_pool_objects_kept_alive_then_released(self):
        so = self.make_so()
        self.install_enabled(so)
        allocator = types.SimpleNamespace(_allocator="cpp-alloc")
        self.alloc_cls.return_value = allocator
        self.pool_cls.return_value = "pool"

        with self.worker._maybe_get_memory_pool_context("weights"):
            pass

        self.assertEqual(self.plugin._keep_alive, [(allocator, "pool")])
        self.plugin._release_keep_alive()
        self.assertEqual(self.plugin._keep_alive, [])

    def test_second_weights_entry_creates_new_pool(self):
        so = self.make_so()
        self.install_enabled(so)
        self.pool_cls.return_value = "pool"

        with self.worker._maybe_get_memory_pool_context("weights"):
            pass
        with self.worker._maybe_get_memory_pool_context("weights"):
            pass

        self.assertEqual(self.alloc_cls.call_count, 2)
        self.assertEqual(self.pool_cls.call_count, 2)
        self.assertEqual(len(self.plugin._keep_alive), 2)


class CountersTest(PluginTestCase):
    def make_stat_lib(self):
        stats = {
            "uvm_get_allocated_bytes": 4096,
            "uvm_get_peak_allocated_bytes": 8192,
            "uvm_get_num_allocs": 10,
            "uvm_get_num_frees": 3,
        }
        return types.SimpleNamespace(
            **{name: mock.Mock(return_value=value) for name, value in stats.items()}
        )

    def test_enter_exit_counters_logged(self):
        so = self.make_so()
        self.install_enabled(so, counters="1")
        self.alloc_cls.return_value = types.SimpleNamespace(_allocator="x")
        self.pool_cls.return_value = "p"

        with (
            mock.patch.object(
                self.plugin.ctypes, "CDLL", return_value=self.make_stat_lib()
            ),
            mock.patch.object(self.plugin.logger, "info") as info,
        ):
            with self.worker._maybe_get_memory_pool_context("weights"):
                pass

        lines = rendered(info)
        self.assertEqual(len(lines), 2)
        self.assertIn("enter UVM weights pool", lines[0])
        self.assertIn("exit UVM weights pool", lines[1])
        for line in lines:
            self.assertIn("allocs=10 frees=3 live=4096 bytes", line)
            self.assertIn("peak=8192 bytes", line)

    def test_counters_disabled_by_default(self):
        so = self.make_so()
        self.install_enabled(so)
        self.alloc_cls.return_value = types.SimpleNamespace(_allocator="x")
        self.pool_cls.return_value = "p"

        with mock.patch.object(self.plugin.ctypes, "CDLL") as cdll:
            with self.worker._maybe_get_memory_pool_context("weights"):
                pass
        cdll.assert_not_called()

    def test_counter_failure_does_not_break_pool(self):
        so = self.make_so()
        self.install_enabled(so, counters="1")
        self.alloc_cls.return_value = types.SimpleNamespace(_allocator="x")
        self.pool_cls.return_value = "p"

        with (
            mock.patch.object(
                self.plugin.ctypes, "CDLL", side_effect=OSError("no library")
            ),
            mock.patch.object(self.plugin.logger, "warning") as warning,
        ):
            with self.worker._maybe_get_memory_pool_context("weights"):
                pass

        self.assertEqual(self.entered, ["p"])
        self.assertEqual(self.exited, ["p"])
        self.assertEqual(warning.call_count, 2)


if __name__ == "__main__":
    unittest.main()
