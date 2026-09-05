#!/usr/bin/env python3
"""CPU-only tests for the bounded endpoint module lifecycle."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
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


def completed(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, 0, "ok\n", "")


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
        self.assertEqual(lifecycle.FIXED_SM_CLOCK_MHZ, 2400)
        self.assertEqual(lifecycle.FIXED_MEMORY_CLOCK_MHZ, 14001)
        self.assertEqual(lifecycle.CLOCK_LOCK_COMMANDS, (
            ("nvidia-smi", "-i", "0", "--lock-gpu-clocks=2400,2400"),
            ("nvidia-smi", "-i", "0",
             "--lock-memory-clocks=14001,14001"),
        ))
        self.assertEqual(lifecycle.CLOCK_RESET_COMMANDS, (
            ("nvidia-smi", "-i", "0", "--reset-memory-clocks"),
            ("nvidia-smi", "-i", "0", "--reset-gpu-clocks"),
        ))

    def test_fixed_clock_lock_is_supported_exact_and_observed(self) -> None:
        query_results = [
            "14001, 2400\n14001, 2392\n",
            "P8, 22, 405, Not Active\n",
            "P0, 2400, 14001, Not Active\n",
        ]
        with (patch.object(lifecycle.base, "checked_stdout",
                           side_effect=query_results) as query,
              patch.object(lifecycle.base, "run_command",
                           side_effect=lambda argv: completed(argv)) as run):
            state = lifecycle.State()
            result = lifecycle.establish_fixed_clocks(state)
        self.assertTrue(result["supported_pair"])
        self.assertEqual(result["before"]["pstate"], "P8")
        self.assertEqual(result["after"]["pstate"], "P0")
        self.assertTrue(state.clock_lock_started)
        self.assertEqual(query.call_count, 3)
        self.assertEqual([item.args[0] for item in run.call_args_list],
                         [list(value) for value in lifecycle.CLOCK_LOCK_COMMANDS])

    def test_fixed_clock_lock_rejects_observation_mismatch(self) -> None:
        query_results = [
            "14001, 2400\n",
            "P8, 22, 405, Not Active\n",
            "P8, 22, 405, Not Active\n",
        ]
        with (patch.object(lifecycle.base, "checked_stdout",
                           side_effect=query_results),
              patch.object(lifecycle.base, "run_command",
                           side_effect=lambda argv: completed(argv))):
            with self.assertRaisesRegex(lifecycle.EndpointLifecycleError,
                                        "fixed GPU clocks are not observed"):
                lifecycle.establish_fixed_clocks(lifecycle.State())

    def test_fixed_clock_reset_attempts_both_commands_on_failure(self) -> None:
        calls = []

        def run(argv):
            calls.append(argv)
            if argv == list(lifecycle.CLOCK_RESET_COMMANDS[0]):
                raise RuntimeError("memory reset failed")
            return completed(argv)

        with patch.object(lifecycle.base, "run_command", side_effect=run):
            with self.assertRaisesRegex(lifecycle.EndpointLifecycleError,
                                        "memory reset failed"):
                lifecycle.reset_fixed_clocks()
        self.assertEqual(calls,
                         [list(value) for value in lifecycle.CLOCK_RESET_COMMANDS])

    def test_fixed_clock_lock_rejects_nonzero_even_if_observation_matches(self) -> None:
        query_results = [
            "14001, 2400\n",
            "P8, 22, 405, Not Active\n",
            "P0, 2400, 14001, Not Active\n",
        ]

        def run(argv):
            if argv == list(lifecycle.CLOCK_LOCK_COMMANDS[0]):
                return subprocess.CompletedProcess(
                    argv, 1, "ok\n", "lock denied\n")
            return completed(argv)

        with (patch.object(lifecycle.base, "checked_stdout",
                           side_effect=query_results),
              patch.object(lifecycle.base, "run_command",
                           side_effect=run) as run_mock):
            with self.assertRaisesRegex(lifecycle.EndpointLifecycleError,
                                        "clock lock command failed with status 1"):
                lifecycle.establish_fixed_clocks(lifecycle.State())
        self.assertEqual([item.args[0] for item in run_mock.call_args_list],
                         [list(lifecycle.CLOCK_LOCK_COMMANDS[0])])

    def test_fixed_clock_reset_runs_second_command_after_first_nonzero(self) -> None:
        calls = []

        def run(argv):
            calls.append(argv)
            if argv == list(lifecycle.CLOCK_RESET_COMMANDS[0]):
                return subprocess.CompletedProcess(
                    argv, 2, "", "memory reset denied\n")
            return completed(argv)

        with patch.object(lifecycle.base, "run_command", side_effect=run):
            with self.assertRaisesRegex(lifecycle.EndpointLifecycleError,
                                        "returned status 2"):
                lifecycle.reset_fixed_clocks()
        self.assertEqual(calls,
                         [list(value) for value in lifecycle.CLOCK_RESET_COMMANDS])

    def test_fixed_clock_reset_rejects_second_command_nonzero(self) -> None:
        def run(argv):
            if argv == list(lifecycle.CLOCK_RESET_COMMANDS[1]):
                return subprocess.CompletedProcess(
                    argv, 3, "", "gpu reset denied\n")
            return completed(argv)

        with patch.object(lifecycle.base, "run_command", side_effect=run):
            with self.assertRaisesRegex(lifecycle.EndpointLifecycleError,
                                        "returned status 3"):
                lifecycle.reset_fixed_clocks()

    def assert_scope_resets_after(self, error: BaseException) -> None:
        state = lifecycle.State()
        events = []
        recovery_errors = []

        def event(name, **details):
            events.append((name, details))

        def lock(active_state):
            active_state.clock_lock_started = True
            return {"after": {"sm_clock_mhz": 2400}}

        with (patch.object(lifecycle, "establish_fixed_clocks",
                           side_effect=lock),
              patch.object(lifecycle, "reset_fixed_clocks",
                           return_value={"after": {"sm_clock_mhz": 22}}) as reset):
            with self.assertRaises(type(error)):
                with lifecycle.fixed_clock_scope(state, recovery_errors, event):
                    raise error
        reset.assert_called_once_with()
        self.assertTrue(state.clock_lock_started)
        self.assertTrue(state.clocks_locked)
        self.assertTrue(state.clocks_reset)
        self.assertFalse(recovery_errors)
        self.assertEqual([name for name, _ in events],
                         ["fixed_clocks_established", "fixed_clocks_reset"])

    def test_fixed_clock_scope_resets_after_child_failure(self) -> None:
        self.assert_scope_resets_after(RuntimeError("child failed"))

    def test_fixed_clock_scope_resets_after_child_timeout(self) -> None:
        self.assert_scope_resets_after(
            subprocess.TimeoutExpired(["child"], timeout=1))

    def test_fixed_clock_scope_resets_after_child_interruption(self) -> None:
        self.assert_scope_resets_after(KeyboardInterrupt())

    def test_fixed_clock_scope_success_resets_symmetrically(self) -> None:
        state = lifecycle.State()
        events = []
        recovery_errors = []
        def lock(active_state):
            active_state.clock_lock_started = True
            return {"after": {"sm_clock_mhz": 2400}}
        with (patch.object(lifecycle, "establish_fixed_clocks",
                           side_effect=lock),
              patch.object(lifecycle, "reset_fixed_clocks",
                           return_value={"after": {"sm_clock_mhz": 22}}) as reset):
            with lifecycle.fixed_clock_scope(
                    state, recovery_errors,
                    lambda name, **details: events.append(name)):
                self.assertTrue(state.clocks_locked)
        reset.assert_called_once_with()
        self.assertTrue(state.clocks_reset)
        self.assertFalse(recovery_errors)
        self.assertEqual(events,
                         ["fixed_clocks_established", "fixed_clocks_reset"])

    def test_partial_clock_lock_failure_still_resets(self) -> None:
        state = lifecycle.State()
        recovery_errors = []
        def fail_after_first_lock(active_state):
            active_state.clock_lock_started = True
            raise RuntimeError("memory lock failed")
        with (patch.object(lifecycle, "establish_fixed_clocks",
                           side_effect=fail_after_first_lock),
              patch.object(lifecycle, "reset_fixed_clocks",
                           return_value={"after": {"sm_clock_mhz": 22}}) as reset):
            with self.assertRaisesRegex(RuntimeError, "memory lock failed"):
                with lifecycle.fixed_clock_scope(
                        state, recovery_errors, lambda name, **details: None):
                    self.fail("body must not run")
        reset.assert_called_once_with()
        self.assertTrue(state.clock_lock_started)
        self.assertFalse(state.clocks_locked)
        self.assertTrue(state.clocks_reset)

    def test_pre_mutation_clock_admission_failure_does_not_reset(self) -> None:
        state = lifecycle.State()
        recovery_errors = []
        with (patch.object(lifecycle, "establish_fixed_clocks",
                           side_effect=lifecycle.EndpointLifecycleError(
                               "unsupported clock pair")),
              patch.object(lifecycle, "reset_fixed_clocks") as reset):
            with self.assertRaisesRegex(lifecycle.EndpointLifecycleError,
                                        "unsupported clock pair"):
                with lifecycle.fixed_clock_scope(
                        state, recovery_errors, lambda name, **details: None):
                    self.fail("body must not run")
        reset.assert_not_called()
        self.assertFalse(state.clock_lock_started)
        self.assertFalse(recovery_errors)

    def test_reset_failure_is_retained_without_hiding_child_failure(self) -> None:
        state = lifecycle.State()
        recovery_errors = []

        def lock(active_state):
            active_state.clock_lock_started = True
            return {"after": {"sm_clock_mhz": 2400}}

        with (patch.object(lifecycle, "establish_fixed_clocks",
                           side_effect=lock),
              patch.object(lifecycle, "reset_fixed_clocks",
                           side_effect=RuntimeError("reset failed"))):
            with self.assertRaisesRegex(RuntimeError, "child failed"):
                with lifecycle.fixed_clock_scope(
                        state, recovery_errors, lambda name, **details: None):
                    raise RuntimeError("child failed")
        self.assertFalse(state.clocks_reset)
        self.assertEqual(len(recovery_errors), 1)
        self.assertIn("reset failed", recovery_errors[0])

    def test_failed_scope_reset_gets_one_final_retry(self) -> None:
        state = lifecycle.State(clock_lock_started=True, clocks_locked=True)
        recovery_errors = ["clock reset: first attempt failed"]
        events = []
        with patch.object(
                lifecycle, "reset_fixed_clocks",
                return_value={"after": {"sm_clock_mhz": 22}}) as reset:
            retried = lifecycle.retry_fixed_clock_reset(
                state, recovery_errors,
                lambda name, **details: events.append(name))
        self.assertTrue(retried)
        reset.assert_called_once_with()
        self.assertTrue(state.clocks_reset)
        self.assertEqual(events, ["fixed_clocks_reset_retry"])
        self.assertEqual(recovery_errors,
                         ["clock reset: first attempt failed"])

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
        suffix = str(os.getpid())
        stage = lifecycle.STAGE_ROOT / f"launchlate-endpoint-stage-test-{suffix}"
        output = lifecycle.RAW_ROOT / f"rm-correlation-575-test-{suffix}"
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
        self.assertEqual(result["fixed_clocks"]["lock_commands"],
                         [list(value) for value in lifecycle.CLOCK_LOCK_COMMANDS])
        self.assertEqual(result["fixed_clocks"]["reset_commands"],
                         [list(value) for value in lifecycle.CLOCK_RESET_COMMANDS])
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
