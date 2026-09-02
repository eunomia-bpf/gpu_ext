#!/usr/bin/env python3
"""Offline parser and command-contract tests for run_safe_policy_smoke.py."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import run_safe_policy_smoke as smoke


class CommandTests(unittest.TestCase):
    def test_post_safety_waits_for_detached_module_reference_release(self) -> None:
        pending = {"struct_ops": {"maps": [], "links": []},
                   "gpu": {"compute_apps": []}, "uvm_refcount": 2}
        settled = {**pending, "uvm_refcount": 0}
        with patch.object(smoke.shared, "safety_snapshot", side_effect=[pending, settled]), \
             patch.object(smoke.shared, "validate_post_server_safety", side_effect=[
                 smoke.GateError("UVM reference count is not zero: 2"), None]), \
             patch.object(smoke.shared.time, "sleep"):
            self.assertIs(smoke.shared.wait_for_post_server_safety({}), settled)

    def test_post_safety_does_not_wait_through_kernel_anomaly(self) -> None:
        state = {"struct_ops": {"maps": [], "links": []},
                 "gpu": {"compute_apps": []}, "uvm_refcount": 0}
        with patch.object(smoke.shared, "safety_snapshot", return_value=state), \
             patch.object(smoke.shared, "validate_post_server_safety", side_effect=
                          smoke.GateError("current boot already contains a kernel/GPU abnormality")), \
             patch.object(smoke.shared.time, "sleep") as sleeper:
            with self.assertRaises(smoke.GateError):
                smoke.shared.wait_for_post_server_safety({})
            sleeper.assert_not_called()

    def test_driver_gate_is_explicit_and_forwarded(self) -> None:
        args = smoke.build_parser().parse_args(["admit", "--expected-driver", "575.57.08"])
        self.assertEqual(args.expected_driver, "575.57.08")
        with patch.object(smoke.shared, "verify_loaded_uvm_interface", return_value={
            "version": "575.57.08", "gpu_mem_ops_members": list(range(6)),
            "required_kfuncs": list(range(2)),
        }) as verify:
            result = smoke.verify_policy_interface("eviction_2q_approx", args.expected_driver)
        verify.assert_called_once_with("575.57.08")
        self.assertEqual(result["driver_revision"], "575.57.08")

    def test_markov_command_is_fixed_and_noninteractive(self) -> None:
        command = smoke.policy_command("prefetch_delta_markov")
        self.assertEqual(command[:2], ["sudo", "-n"])
        self.assertEqual(
            command[2:],
            [
                str(smoke.EXTENSION / "prefetch_delta_markov"),
                "-c",
                "2",
                "-n",
                "2",
                "-m",
                "128",
                "-i",
                "1",
            ],
        )

    def test_twoq_command_is_fixed_and_noninteractive(self) -> None:
        command = smoke.policy_command("eviction_2q_approx")
        self.assertEqual(command[:2], ["sudo", "-n"])
        self.assertEqual(
            command[2:],
            [
                str(smoke.EXTENSION / "eviction_2q_approx"),
                "-p",
                "2",
                "-g",
                "2",
                "-i",
                "1",
            ],
        )

    def test_workload_command_is_exact_8gib_64kib(self) -> None:
        output = Path("/tmp/offline-result.json")
        self.assertEqual(
            smoke.workload_command(output),
            [
                str(smoke.WORKLOAD),
                "--gib",
                "8",
                "--region-kib",
                "64",
                "--output",
                str(output),
            ],
        )
        self.assertEqual(smoke.WORKLOAD_TIMEOUT_SECONDS, 60)

    def test_cli_requires_explicit_run_and_policy(self) -> None:
        parser = smoke.build_parser()
        admitted = parser.parse_args(["admit"])
        self.assertEqual(admitted.action, "admit")
        self.assertIsNone(admitted.policy)
        running = parser.parse_args(
            [
                "run",
                "--policy",
                "prefetch_delta_markov",
                "--output",
                "/tmp/safe-smoke-offline",
            ]
        )
        self.assertEqual(running.action, "run")
        self.assertEqual(running.output, Path("/tmp/safe-smoke-offline"))


class ParserTests(unittest.TestCase):
    def test_noise_is_ignored_and_latest_event_wins(self) -> None:
        text = "\n".join(
            (
                "libbpf: harmless diagnostic",
                json.dumps({"event": "metrics", "callbacks": 1}),
                "not-json",
                json.dumps({"event": "metrics", "callbacks": 4}),
            )
        )
        self.assertEqual(smoke.latest_event(text, "metrics")["callbacks"], 4)

    def test_markov_ready_requires_all_owned_ids(self) -> None:
        ready = smoke.validate_ready(
            "prefetch_delta_markov",
            {
                "event": "ready",
                "pid": 91,
                "struct_map_id": 101,
                "struct_link_id": 102,
                "kprobe_link_id": 103,
                "confidence": 2,
                "prefetch_pages": 2,
                "maximum_delta": 128,
                "metrics_interval_seconds": 1,
            },
        )
        self.assertEqual(ready["kprobe_link_id"], 103)
        with self.assertRaises(smoke.GateError):
            smoke.validate_ready(
                "prefetch_delta_markov",
                {
                    "event": "ready",
                    "pid": 91,
                    "struct_map_id": 101,
                    "struct_link_id": 102,
                    "kprobe_link_id": 0,
                    "confidence": 2,
                    "prefetch_pages": 2,
                    "maximum_delta": 128,
                    "metrics_interval_seconds": 1,
                },
            )

    def test_twoq_ready_rejects_configuration_drift(self) -> None:
        ready = {
            "event": "ready",
            "pid": 91,
            "struct_map_id": 101,
            "struct_link_id": 102,
            "promote_after": 2,
            "maximum_generation_gap": 2,
            "metrics_interval_seconds": 1,
        }
        self.assertEqual(
            smoke.validate_ready("eviction_2q_approx", ready)["struct_map_id"],
            101,
        )
        ready["promote_after"] = 3
        with self.assertRaises(smoke.GateError):
            smoke.validate_ready("eviction_2q_approx", ready)

    def test_markov_auxiliary_link_ownership(self) -> None:
        ready = {
            "pid": 91,
            "struct_map_id": 101,
            "struct_link_id": 102,
            "kprobe_link_id": 103,
        }
        ownership = smoke.validate_kprobe_link_ownership(
            ready, [{"id": 103, "type": "perf_event", "pids": [{"pid": 91}]}]
        )
        self.assertTrue(ownership["owner_pid_enumerated"])
        with self.assertRaises(smoke.GateError):
            smoke.validate_kprobe_link_ownership(
                ready,
                [{"id": 103, "type": "perf_event", "pids": [{"pid": 92}]}],
            )

    def test_zero_mismatch_workload_is_validated(self) -> None:
        result = smoke.validate_workload_result(
            {
                "bytes": 8 * 1024**3,
                "region_bytes": 64 * 1024,
                "regions": 131072,
                "kernel_ms": 12.5,
                "mismatches": 0,
                "first_mismatch": None,
            }
        )
        self.assertEqual(result["mismatches"], 0)
        with self.assertRaises(smoke.GateError):
            smoke.validate_workload_result(
                {
                    **result,
                    "mismatches": 1,
                    "first_mismatch": 7,
                }
            )

    def test_twoq_engagement_requires_action_and_no_errors(self) -> None:
        event = {
            "event": "final_metrics",
            "activate_events": 10,
            "access_events": 10,
            "admissions": 10,
            "identity_resets": 0,
            "generation_resets": 0,
            "same_episode_events": 10,
            "probation_head_requests": 20,
            "promotions": 0,
            "protected_tail_requests": 0,
            "reorder_errors": 0,
            "eviction_prepares": 0,
        }
        metrics = smoke.validate_policy_metrics("eviction_2q_approx", event)
        self.assertEqual(metrics["probation_head_requests"], 20)
        event["reorder_errors"] = 1
        with self.assertRaises(smoke.GateError):
            smoke.validate_policy_metrics("eviction_2q_approx", event)

    def test_markov_engagement_requires_real_predictions(self) -> None:
        event = {
            "event": "final_metrics",
            "context_captures": 30,
            "callbacks": 30,
            "blocks_initialized": 2,
            "deltas_observed": 28,
            "invalid_deltas": 0,
            "transitions_created": 2,
            "transition_matches": 20,
            "transition_decays": 0,
            "transition_replacements": 0,
            "confident_predictions": 18,
            "prefetch_requests": 18,
            "empty_requests": 12,
            "map_errors": 0,
            "request_errors": 0,
        }
        metrics = smoke.validate_policy_metrics("prefetch_delta_markov", event)
        self.assertEqual(metrics["prefetch_requests"], 18)
        event["prefetch_requests"] = 0
        with self.assertRaises(smoke.GateError):
            smoke.validate_policy_metrics("prefetch_delta_markov", event)

    def test_active_snapshot_rechecks_idle_kernel_and_ownership(self) -> None:
        before = {
            "dmesg_abnormal": [],
            "journal_abnormal": [],
            "xids": [],
        }
        active = {
            **before,
            "power_limit_service": "active",
            "power_limit_w": 400.0,
            "gpu": {
                "compute_apps": [],
                "memory_used_mib": 0,
                "utilization_gpu_percent": 0,
            },
            "struct_ops": {
                "maps": [{"id": 101, "pids": [{"pid": 91}]}],
                "links": [{"id": 102}],
            },
        }
        ready = {"pid": 91, "struct_map_id": 101, "struct_link_id": 102}
        smoke.validate_active_policy_safety(before, active, ready)
        active["xids"] = ["new Xid"]
        with self.assertRaises(smoke.GateError):
            smoke.validate_active_policy_safety(before, active, ready)


if __name__ == "__main__":
    unittest.main()
