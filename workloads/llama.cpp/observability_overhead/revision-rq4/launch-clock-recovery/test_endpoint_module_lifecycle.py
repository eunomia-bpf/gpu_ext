#!/usr/bin/env python3
"""CPU-only tests for the bounded endpoint module lifecycle."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


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
        result = lifecycle.dry_run(
            Path("/home/yunwei37/workspace/gpu/gpu_ext-kernel-575/kernel-open"),
            Path("/opt/gpubpf/modules/575.57.08/"
                 "launchlate-endpoint-86e7e0dd-offline-test"),
            HERE.parent / "raw/rm-correlation-575-offline-dry-run",
            child_mode="preflight-full",
        )
        self.assertTrue(result["complete"])
        self.assertEqual(result["mode"], "cpu-only-dry-run")
        self.assertEqual(result["source"]["revision"],
                         lifecycle.EXPECTED_CANDIDATE_REVISION)
        self.assertEqual(result["child_mode"], "preflight-full")
        self.assertEqual(set(result["candidate"]), set(lifecycle.base.LOAD_ORDER))


if __name__ == "__main__":
    unittest.main()
