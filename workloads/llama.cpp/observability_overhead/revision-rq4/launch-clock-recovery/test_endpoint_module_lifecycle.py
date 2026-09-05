#!/usr/bin/env python3
"""CPU-only tests for the bounded endpoint module lifecycle."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "run_endpoint_module_lifecycle.py"
SPEC = importlib.util.spec_from_file_location("endpoint_module_lifecycle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
lifecycle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lifecycle
SPEC.loader.exec_module(lifecycle)


def valid_probe_output() -> str:
    records = []
    for index in range(200):
        records.append({
            "record": "sample", "index": index,
            "control_transport": "direct",
            "correlation_command": "endpoints-v1", "valid": True,
            "rm_status": 0, "cpu_midpoint_regression": False,
            "ptimer_regression": False,
        })
    records.append({
        "record": "summary", "control_transport": "direct",
        "correlation_command": "endpoints-v1", "setup_error": 0,
        "cleanup_error": 0, "cleanup_rm_status": 0, "output_error": 0,
        "requested": 200, "attempted": 200, "accepted": 200,
        "rejected": 0, "cpu_midpoint_regressions": 0,
        "ptimer_regressions": 0, "gate_pass": True,
    })
    return "\n".join(json.dumps(value) for value in records) + "\n"


class EndpointLifecycleOfflineTests(unittest.TestCase):
    def test_fixed_scope(self) -> None:
        self.assertEqual(lifecycle.base.LOAD_ORDER,
                         ("nvidia", "nvidia_modeset", "nvidia_drm", "nvidia_uvm"))
        self.assertEqual(lifecycle.PROBE_ARGS,
                         ("--samples", "200", "--control-transport", "direct",
                          "--correlation-command", "endpoints-v1"))
        self.assertEqual(lifecycle.RESTORE_DIR.name,
                         "gpreempt-849ea75d-6.15.11")
        self.assertEqual(
            lifecycle.PREBUILT_CANDIDATE_DIR,
            Path("/opt/gpubpf/modules/575.57.08/"
                 "launchlate-endpoint-86e7e0dd-575-02"),
        )
        self.assertEqual(len(lifecycle.ENDPOINT_SYMBOLS), 2)

    def test_endpoint_symbol_gate_requires_both_symbols(self) -> None:
        lines = "\n".join(
            f"0000000000000001 T {symbol}"
            for symbol in lifecycle.ENDPOINT_SYMBOLS
        )
        with patch.object(lifecycle.base, "checked_stdout", return_value=lines):
            self.assertEqual(
                lifecycle.validate_endpoint_symbols(Path("/fixed/nvidia.ko")),
                list(lifecycle.ENDPOINT_SYMBOLS),
            )
        with patch.object(
                lifecycle.base, "checked_stdout",
                return_value=f"0000000000000001 T {lifecycle.ENDPOINT_SYMBOLS[0]}"):
            with self.assertRaisesRegex(lifecycle.EndpointLifecycleError,
                                        "lacks endpoint symbols"):
                lifecycle.validate_endpoint_symbols(Path("/fixed/nvidia.ko"))

    def test_candidate_path_rejects_source_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(lifecycle.EndpointLifecycleError,
                                        "exact admitted prebuilt path"):
                lifecycle.validate_paths(
                    Path(temp),
                    Path("/opt/gpubpf/modules/575.57.08/"
                         "launchlate-endpoint-stage-offline-test"),
                    HERE.parent / "raw/rm-correlation-575-offline-dry-run",
                )

    def test_child_paths_are_required_only_for_a_child(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with (patch.object(lifecycle, "BPFTIME_ROOT", root),
                  patch.object(lifecycle, "BPFTIME_BUILD", root / "missing")):
                lifecycle.validate_child_paths("none")
                with self.assertRaisesRegex(lifecycle.EndpointLifecycleError,
                                            "source/build is absent"):
                    lifecycle.validate_child_paths("preflight")

    def test_child_argv_is_fixed_and_inherits_both_leases(self) -> None:
        output = HERE.parent / "raw/example-preflight"
        argv = lifecycle.campaign_argv("preflight", output, (71, 72))
        self.assertEqual(argv.count("launchlate"), 1)
        self.assertEqual(argv[argv.index("--inherited-lease-fds") + 1:],
                         ["71", "72"])
        self.assertNotIn("--preflight-dir", argv)
        full = lifecycle.campaign_argv(
            "full", HERE.parent / "raw/example-full", (71, 72), output)
        self.assertEqual(full[full.index("--preflight-dir") + 1], str(output))

    def test_child_path_and_cuda_command_lookup_are_fixed(self) -> None:
        with patch.dict(os.environ, {"PATH": "/hostile", "BPFTIME_BAD": "1"}):
            environment = lifecycle.child_environment()
        self.assertEqual(environment["PATH"], lifecycle.CHILD_PATH)
        self.assertNotIn("BPFTIME_BAD", environment)
        commands = lifecycle.validate_child_command_lookup(environment)
        self.assertEqual(commands["cuobjdump"],
                         "/usr/local/cuda-12.9/bin/cuobjdump")
        self.assertEqual(commands["nvcc"], "/usr/local/cuda-12.9/bin/nvcc")

    def test_child_command_lookup_rejects_path_mutation(self) -> None:
        environment = lifecycle.child_environment()
        environment["PATH"] = "/usr/bin:/bin"
        with self.assertRaisesRegex(lifecycle.EndpointLifecycleError,
                                    "PATH differs"):
            lifecycle.validate_child_command_lookup(environment)
        with tempfile.TemporaryDirectory() as temp:
            fake = Path(temp) / "cuobjdump"
            fake.write_text("#!/bin/sh\nexit 0\n")
            fake.chmod(0o755)
            environment["PATH"] = f"{temp}:{lifecycle.CHILD_PATH}"
            with self.assertRaisesRegex(lifecycle.EndpointLifecycleError,
                                        "PATH differs"):
                lifecycle.validate_child_command_lookup(environment)

    def test_probe_gate_accepts_only_complete_fixed_run(self) -> None:
        result = lifecycle.validate_probe_output(valid_probe_output())
        self.assertEqual(result["samples"], 200)
        self.assertTrue(result["summary"]["gate_pass"])

    def test_probe_gate_rejects_tamper_and_truncation(self) -> None:
        records = valid_probe_output().splitlines()
        tampered = [json.loads(line) for line in records]
        tampered[17]["valid"] = False
        with self.assertRaises(lifecycle.EndpointLifecycleError):
            lifecycle.validate_probe_output(
                "\n".join(json.dumps(value) for value in tampered))
        with self.assertRaises(lifecycle.EndpointLifecycleError):
            lifecycle.validate_probe_output("\n".join(records[:-1]))

    def test_cpu_only_real_artifact_preflight(self) -> None:
        stage = Path(
            "/opt/gpubpf/modules/575.57.08/launchlate-endpoint-stage-575-09"
        )
        output = HERE.parent / "raw/rm-correlation-575-09-endpoint-lifecycle"
        self.assertFalse(stage.exists())
        self.assertFalse(output.exists())
        result = lifecycle.dry_run(
            lifecycle.PREBUILT_CANDIDATE_DIR,
            stage,
            output,
            child_mode="none",
        )
        self.assertTrue(result["complete"])
        self.assertEqual(result["mode"], "cpu-only-dry-run")
        self.assertEqual(result["stage"], str(stage))
        self.assertEqual(result["output"], str(output))
        self.assertFalse(stage.exists())
        self.assertFalse(output.exists())
        self.assertEqual(result["candidate_origin"]["path"],
                         str(lifecycle.PREBUILT_CANDIDATE_DIR))
        self.assertEqual(result["child_mode"], "none")
        self.assertEqual(set(result["candidate"]), set(lifecycle.base.LOAD_ORDER))
        for name, descriptor in result["candidate"].items():
            self.assertGreater(descriptor["inventory"]["size_bytes"], 0)
            self.assertEqual(descriptor["version"], lifecycle.base.EXPECTED_DRIVER)
            self.assertEqual(descriptor["vermagic"],
                             lifecycle.base.EXPECTED_VERMAGIC)
            self.assertTrue(descriptor["interface"])
            if name == "nvidia":
                self.assertEqual(descriptor["endpoint_symbols"],
                                 list(lifecycle.ENDPOINT_SYMBOLS))
            else:
                self.assertEqual(descriptor["endpoint_symbols"], [])


if __name__ == "__main__":
    unittest.main()
