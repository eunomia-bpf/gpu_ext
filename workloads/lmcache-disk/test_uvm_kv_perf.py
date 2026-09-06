#!/usr/bin/env python3
"""CPU-only tests for the UVM KV performance-only runner (no GPU, no server)."""

import importlib.util
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "run_uvm_kv_perf.py"
SPEC = importlib.util.spec_from_file_location("run_uvm_kv_perf", MODULE_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
ops = runner.ops

EXPECTED_DRIVER = "610.43.02"
CACHE_DIR = "/tmp/uvm-kv-cache-fixture"


class RunnerShapeTests(unittest.TestCase):
    def test_module_constants(self):
        self.assertEqual(runner.KIND, "lmcache_uvm_kv_perf")
        self.assertEqual(runner.CONFIGS,
                         ("recompute", "lmcache_disk", "lmcache_disk_uvm_kv"))
        self.assertEqual(runner.DEFAULT_BLOCKS, 1)
        self.assertEqual(runner.DEFAULT_PORT, 18080)
        self.assertEqual(runner.DEFAULT_STORE_BARRIER_TIMEOUT_S, 120.0)
        self.assertEqual(runner.REQUEST_LABELS, ("cold", "warm"))

    def test_parse_args_defaults(self):
        args = runner.parse_args([])
        self.assertEqual(args.blocks, 1)
        self.assertEqual(args.port, 18080)
        self.assertEqual(args.store_barrier_timeout_s, 120.0)
        self.assertEqual(args.expected_driver, EXPECTED_DRIVER)
        self.assertIs(args.dry_run, False)
        self.assertIsNone(args.output)

    def test_no_schedule_or_legacy_arm_in_source(self):
        src = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("SCHEDULE", src)
        self.assertNotIn("schedule.json", src)
        self.assertNotIn("ops.CONFIGS", src)
        self.assertNotIn("lmcache_cpu", src)


class RotationTests(unittest.TestCase):
    def test_default_one_block_is_the_base_order(self):
        self.assertEqual(runner.rotation_orders(1), [list(runner.CONFIGS)])

    def test_every_order_is_a_complete_three_arm_permutation(self):
        for order in runner.rotation_orders(3):
            self.assertEqual(sorted(order), sorted(runner.CONFIGS))
            self.assertEqual(len(order), 3)

    def test_three_blocks_cover_each_arm_in_each_position_once(self):
        orders = runner.rotation_orders(3)
        for position in range(3):
            column = [order[position] for order in orders]
            self.assertEqual(sorted(column), sorted(runner.CONFIGS))

    def test_rotation_is_deterministic_and_periodic(self):
        self.assertEqual(runner.rotation_orders(3), runner.rotation_orders(3))
        self.assertEqual(runner.rotation_orders(4)[:3], runner.rotation_orders(3))
        self.assertEqual(runner.rotation_orders(6)[3], runner.rotation_orders(3)[0])

    def test_blocks_below_one_rejected(self):
        with self.assertRaises(ValueError):
            runner.rotation_orders(0)


class EnvironmentTests(unittest.TestCase):
    def test_uvm_arm_is_disk_arm_plus_plugin_vars(self):
        disk = ops.server_environment("lmcache_disk", Path(CACHE_DIR))
        uvm = ops.server_environment("lmcache_disk_uvm_kv", Path(CACHE_DIR))
        lmc_keys = [key for key in disk if key.startswith("LMCACHE_")]
        self.assertTrue(lmc_keys)
        for key in lmc_keys:
            self.assertEqual(uvm[key], disk[key])
        self.assertEqual(set(uvm) - set(disk), {"UVM_KV_PLUGIN", "UVM_KV_PLUGIN_SO"})
        self.assertEqual(uvm["UVM_KV_PLUGIN"], "1")

    def test_uvm_plugin_so_is_absolute_prepared_allocator(self):
        uvm = ops.server_environment("lmcache_disk_uvm_kv", Path(CACHE_DIR))
        so = uvm["UVM_KV_PLUGIN_SO"]
        self.assertTrue(so.startswith("/"))
        self.assertEqual(Path(so), ops.UVM_KV_ALLOCATOR_SO)

    def test_allocator_constant_points_at_prepared_symlink_target(self):
        self.assertEqual(ops.UVM_KV_ALLOCATOR_SO.name, "uvm_allocator.so")
        self.assertTrue(ops.UVM_KV_ALLOCATOR_SO.is_absolute())
        self.assertEqual(ops.UVM_KV_ALLOCATOR_SO.parent,
                         ops.VLLM_WORKLOAD / "vllm" / "uvm_test")
        self.assertTrue(ops.UVM_KV_ALLOCATOR_SO.is_file())

    def test_other_arms_do_not_enable_plugin(self):
        for config in ("recompute", "lmcache_disk"):
            env = ops.server_environment(config, Path(CACHE_DIR))
            self.assertNotIn("UVM_KV_PLUGIN", env)
            self.assertNotIn("UVM_KV_PLUGIN_SO", env)

    def test_recompute_has_no_lmcache_or_uvm_env(self):
        env = ops.server_environment("recompute", Path(CACHE_DIR))
        self.assertFalse([key for key in env if key.startswith("LMCACHE_")])
        self.assertNotIn("UVM_KV_PLUGIN", env)

    def test_disk_arm_env_matches_lmcache_disk_runner_expectations(self):
        env = ops.server_environment("lmcache_disk_uvm_kv", Path(CACHE_DIR))
        self.assertEqual(env["LMCACHE_LOCAL_CPU"], "False")
        self.assertEqual(env["LMCACHE_LOCAL_DISK"], "file://" + CACHE_DIR)
        self.assertEqual(env["LMCACHE_USE_GPU_CONNECTOR_V3"], "True")
        self.assertIn("use_odirect", env["LMCACHE_EXTRA_CONFIG"])


class DryRunTests(unittest.TestCase):
    def capture_main(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = runner.main(argv)
        return rc, json.loads(buf.getvalue())

    def test_main_dry_run_defaults_to_one_block(self):
        rc, plan = self.capture_main(["--dry-run"])
        self.assertEqual(rc, 0)
        self.assertIs(plan["dry_run"], True)
        self.assertEqual(plan["blocks"], 1)
        self.assertEqual(plan["configs"], list(runner.CONFIGS))
        self.assertEqual(plan["block_orders"], [list(runner.CONFIGS)])
        self.assertEqual(plan["expected_driver_parameter"], EXPECTED_DRIVER)
        self.assertEqual(plan["port"], 18080)
        self.assertEqual(plan["store_barrier"]["timeout_s"], 120.0)

    def test_main_dry_run_three_blocks_matches_rotation(self):
        rc, plan = self.capture_main(["--dry-run", "--blocks", "3"])
        self.assertEqual(rc, 0)
        self.assertEqual(plan["blocks"], 3)
        self.assertEqual(plan["block_orders"], runner.rotation_orders(3))

    def test_main_dry_run_rejects_zero_blocks_without_output(self):
        with self.assertRaises(ValueError):
            self.capture_main(["--dry-run", "--blocks", "0"])


if __name__ == "__main__":
    unittest.main()
