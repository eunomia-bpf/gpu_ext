"""CPU-only unit tests for the reversible Q2 UVM lifecycle."""
from __future__ import annotations

import copy
import fcntl
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import run_lifecycle as runner


def service(active: str, sub: str, result: str = "success") -> dict[str, str]:
    return {
        "LoadState": "loaded", "ActiveState": active, "SubState": sub,
        "Result": result, "UnitFileState": "enabled",
    }


class ParameterAndCommandTests(unittest.TestCase):
    def test_null_char_pointer_is_omitted_and_values_are_exact_argv(self):
        values = {
            "uvm_page_table_location": "(null)",
            "uvm_perf_prefetch_enable": "1",
            "uvm_disable_hmm": "N",
        }
        self.assertEqual(
            runner.insmod_command(Path("/stage/nvidia-uvm.ko"), values),
            ["sudo", "-n", "insmod", "/stage/nvidia-uvm.ko",
             "uvm_disable_hmm=N", "uvm_perf_prefetch_enable=1"],
        )

    def test_invalid_parameter_name_or_multiline_value_is_rejected(self):
        for values in ({"bad-name": "1"}, {"safe": "1\n2"}, {"safe": ""}):
            with self.subTest(values=values), self.assertRaises(runner.LifecycleError):
                runner.parameter_arguments(values)

    def test_commands_use_isolated_process_group(self):
        completed = mock.Mock(returncode=0)
        completed.communicate.return_value = ("ok", "")
        with mock.patch.object(runner.subprocess, "Popen", return_value=completed) as popen:
            result = runner.run_command(["true"])
        self.assertEqual(result.stdout, "ok")
        self.assertTrue(popen.call_args.kwargs["start_new_session"])


class LeaseTests(unittest.TestCase):
    def test_read_only_existing_inodes_are_locked_without_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = (Path(temporary) / "gpu0.lock", Path(temporary) / "struct.lock")
            for index, path in enumerate(paths):
                path.touch()
                path.chmod(0o444 if index == 0 else 0o644)
            before = [(path.stat().st_ino, path.stat().st_mode) for path in paths]
            lease = runner.LifecycleLeases(paths)
            try:
                with self.assertRaises(BlockingIOError):
                    runner.LifecycleLeases(paths)
            finally:
                lease.close()
            after = [(path.stat().st_ino, path.stat().st_mode) for path in paths]
            self.assertEqual(after, before)

    def test_partial_acquisition_failure_releases_earlier_descriptor(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.lock"
            second = Path(temporary) / "second.lock"
            first.touch()
            second.touch()
            blocker = os.open(second, os.O_RDONLY | os.O_CLOEXEC)
            fcntl.flock(blocker, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                with self.assertRaises(BlockingIOError):
                    runner.LifecycleLeases((first, second))
                proof = runner.LifecycleLeases((first,))
                proof.close()
            finally:
                os.close(blocker)

    def test_open_flags_never_include_create_or_write_access(self):
        real_open = os.open
        observed = []
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lease.lock"
            path.touch()

            def inspect_open(target, flags):
                observed.append(flags)
                return real_open(target, flags)

            with mock.patch.object(runner.os, "open", side_effect=inspect_open):
                lease = runner.LifecycleLeases((path,))
                lease.close()
        self.assertEqual(len(observed), 1)
        self.assertFalse(observed[0] & os.O_CREAT)
        self.assertEqual(observed[0] & os.O_ACCMODE, os.O_RDONLY)


class ServiceStateTests(unittest.TestCase):
    def test_stop_and_restore_only_original_active_services(self):
        initial = {
            "gdm.service": service("active", "running"),
            "nvidia-persistenced.service": service("inactive", "dead"),
        }
        self.assertEqual(runner.service_stop_plan(initial), ["gdm.service"])
        current = {
            "gdm.service": service("inactive", "dead"),
            "nvidia-persistenced.service": service("inactive", "dead"),
        }
        self.assertEqual(runner.service_restore_plan(initial, current), ["gdm.service"])

    def test_transitional_state_is_pollable_but_not_admitted(self):
        text = ("LoadState=loaded\nActiveState=deactivating\nSubState=stop-sigterm\n"
                "Result=success\nUnitFileState=enabled\n")
        with mock.patch.object(runner, "checked_stdout", return_value=text):
            with self.assertRaises(runner.LifecycleError):
                runner.service_state("gdm.service")
            self.assertEqual(
                runner.service_state("gdm.service", allow_transitional=True)["ActiveState"],
                "deactivating",
            )

    def test_set_service_polls_through_transition(self):
        states = [service("deactivating", "stop-sigterm"), service("inactive", "dead"),
                  service("inactive", "dead")]
        with mock.patch.object(runner, "run_command") as command, \
                mock.patch.object(runner, "service_state", side_effect=states):
            runner.set_service("gdm.service", "stop")
        command.assert_called_once_with(
            ["sudo", "-n", "systemctl", "--no-block", "stop", "gdm.service"])

    def test_final_service_gate_checks_substate_and_result(self):
        initial = {unit: service("active", "running") for unit in runner.SERVICES}
        bad = [service("active", "exited"), service("active", "running", "failed")]
        for value in bad:
            with self.subTest(value=value), \
                    mock.patch.object(runner, "service_state", return_value=value), \
                    self.assertRaises(runner.LifecycleError):
                runner.validate_services_restored(initial)

    def test_only_active_local_gdm_greeter_is_admitted(self):
        greeter = {"Id": "c9", "User": "120", "Name": "gdm", "Seat": "seat0",
                   "Class": "greeter", "Type": "x11",
                   "Service": "gdm-launch-environment", "Active": "yes",
                   "Remote": "no", "State": "active"}
        runner.validate_local_sessions([greeter])
        for changed in (dict(greeter, Class="user"), dict(greeter, Service="gdm-password")):
            with self.subTest(changed=changed), self.assertRaises(runner.LifecycleError):
                runner.validate_local_sessions([changed])
        ssh = dict(greeter, Id="21", User="1000", Name="yunwei37", Seat="",
                   Remote="yes", Class="user", Type="tty", Service="sshd",
                   State="closing")
        runner.validate_local_sessions([greeter, ssh])
        with self.assertRaises(runner.LifecycleError):
            runner.validate_local_sessions([{key: value for key, value in greeter.items()
                                              if key != "Remote"}])

    def test_second_local_session_gate_fails_before_service_stop(self):
        with mock.patch.object(runner, "service_stop_recheck",
                               side_effect=runner.LifecycleError("local user")), \
                mock.patch.object(runner, "set_service") as setter:
            with self.assertRaises(runner.LifecycleError):
                runner.stop_service_after_recheck("gdm.service", lambda _rows: None)
            setter.assert_not_called()


class FuserTests(unittest.TestCase):
    DEVICES = [Path("/dev/nvidia-uvm")]

    @staticmethod
    def result(code: int, stdout: str = "", stderr: str = ""):
        return subprocess.CompletedProcess(["fuser"], code, stdout, stderr)

    def test_only_silent_exit_one_means_no_holder(self):
        row = runner.validate_fuser_result(self.result(1), self.DEVICES)
        self.assertEqual(row["returncode"], 1)
        for result in (self.result(0, "123"), self.result(1, stderr="usage"),
                       self.result(2)):
            with self.subTest(result=result), self.assertRaises(runner.LifecycleError):
                runner.validate_fuser_result(result, self.DEVICES)

    def test_live_command_has_no_double_dash(self):
        with mock.patch.object(Path, "exists", return_value=True), \
                mock.patch.object(runner, "run_command", return_value=self.result(1)) as command:
            runner.no_uvm_holders()
        argv = command.call_args.args[0]
        self.assertNotIn("--", argv)
        self.assertEqual(argv[:4], ["sudo", "-n", "fuser", "-v"])

    def test_both_device_nodes_are_mandatory(self):
        with mock.patch.object(Path, "exists", side_effect=[True, False]), \
                mock.patch.object(runner, "run_command") as command, \
                self.assertRaises(runner.LifecycleError):
            runner.no_uvm_holders()
        command.assert_not_called()


class RecoveryStateTests(unittest.TestCase):
    def test_candidate_insmod_failure_with_no_module_routes_to_old_insert(self):
        state = runner.RuntimeState(
            destructive_started=True, old_unloaded=True, last_insert="candidate",
            candidate_loaded=False,
        )
        self.assertEqual(runner.module_recovery_action(state, False), "insert_old")

    def test_post_load_abi_failure_routes_candidate_removal(self):
        state = runner.RuntimeState(
            destructive_started=True, old_unloaded=True, last_insert="candidate",
            candidate_loaded=False,
        )
        self.assertEqual(runner.module_recovery_action(state, True), "remove_candidate")

    def test_old_still_loaded_is_validated_not_removed(self):
        state = runner.RuntimeState(destructive_started=True, last_insert="initial")
        self.assertEqual(runner.module_recovery_action(state, True), "validate_old")

    def test_every_failure_class_suppresses_complete_summary_gate(self):
        completed = list(runner.MODES)
        failures = {
            "cell_signal": (completed[:1], True, [], InterruptedError("signal")),
            "cell_failure": (completed[:2], True, [], RuntimeError("cell")),
            "candidate_unload": (completed, False, [{"stage": "remove", "error": "busy"}], None),
            "old_restore": (completed, False, [{"stage": "insert", "error": "failed"}], None),
            "service_stop": ([], True, [], RuntimeError("stop")),
            "service_start": (completed, False, [{"stage": "start", "error": "failed"}], None),
        }
        for name, arguments in failures.items():
            with self.subTest(name=name):
                self.assertFalse(runner.campaign_complete(*arguments))
        self.assertTrue(runner.campaign_complete(completed, True, [], None))

    def test_forward_remove_honors_signal_after_fuser_but_recovery_does_not(self):
        snapshots = {"marker": "quiet"}
        with mock.patch.object(runner, "quiet_snapshot", return_value=snapshots), \
                mock.patch.object(runner, "no_uvm_holders", return_value={}), \
                mock.patch.object(runner.cells, "raise_if_interrupted",
                                  side_effect=InterruptedError("signal")), \
                mock.patch.object(runner, "run_command") as command:
            with self.assertRaises(InterruptedError):
                runner.remove_uvm("boot", honor_interrupt=True)
            command.assert_not_called()

        fake = mock.Mock()
        fake.exists.return_value = False
        with mock.patch.object(runner, "quiet_snapshot", return_value=snapshots), \
                mock.patch.object(runner, "no_uvm_holders", return_value={}), \
                mock.patch.object(runner.cells, "raise_if_interrupted") as interrupted, \
                mock.patch.object(runner, "run_command"), \
                mock.patch.object(runner, "LOADED_MODULE", fake), \
                mock.patch.object(runner, "LOADED_UVM_BTF", fake):
            runner.remove_uvm("boot", honor_interrupt=False)
            interrupted.assert_not_called()

    def test_forward_candidate_insert_checks_signal_immediately_before_command(self):
        fake_module = mock.Mock()
        fake_module.exists.return_value = False
        with mock.patch.object(runner, "require_boot"), \
                mock.patch.object(runner, "LOADED_MODULE", fake_module), \
                mock.patch.object(runner.cells, "raise_if_interrupted",
                                  side_effect=InterruptedError("signal")), \
                mock.patch.object(runner, "run_command") as command:
            with self.assertRaises(InterruptedError):
                runner.insert_uvm(
                    Path("/stage/nvidia-uvm.ko"), {"uvm_perf_prefetch_enable": "1"},
                    diagnostic=True, expected_interface={}, initial_boot="boot",
                    honor_interrupt=True,
                )
            command.assert_not_called()


class FakeRecorder:
    def __init__(self, events=None, fail=False, fail_writes=()):
        self.value = {"complete": False, "transitions": []}
        self.events = events if events is not None else []
        self.fail = fail
        self.writes = []
        self.fail_writes = set(fail_writes)

    def transition(self, name, status, **details):
        self.events.append(f"record:{name}:{status}")
        if self.fail:
            raise OSError("record failure")
        self.value["transitions"].append((name, status, details))

    def write(self):
        self.events.append("write")
        self.writes.append(copy.deepcopy(self.value))
        if self.fail or len(self.writes) in self.fail_writes:
            raise OSError("write failure")


def recovery_fixture():
    descriptor = {"interface": {}}
    initial = {
        "boot_id": "boot", "core": {"core": "same"},
        "parameters": {"uvm_perf_prefetch_enable": "1"},
        "services": {unit: service("active", "running") for unit in runner.SERVICES},
        "safety": {"initial": True},
    }
    state = runner.RuntimeState(
        destructive_started=True, old_unloaded=True, last_insert="candidate",
        candidate_loaded=True,
    )
    return descriptor, initial, state


class RecoveryExecutionTests(unittest.TestCase):
    def run_restore(self, recorder, operations, *, remove_error=None,
                    insert_error=None, start_error=False, core_result=None):
        descriptor, initial, state = recovery_fixture()
        loaded_module = mock.Mock()
        loaded_module.is_dir.return_value = True

        def remove(*_args, **_kwargs):
            operations.append("remove_candidate")
            if remove_error:
                raise remove_error
            return {}

        def insert(*_args, **_kwargs):
            operations.append("insert_old")
            if insert_error:
                raise insert_error
            return {}

        def set_service(unit, action):
            operations.append(f"{action}:{unit}")
            if start_error and unit == "nvidia-persistenced.service":
                raise runner.LifecycleError("start failed")

        inactive = service("inactive", "dead")
        with mock.patch.object(runner, "require_unchanged_module", return_value={}), \
                mock.patch.object(runner, "LOADED_MODULE", loaded_module), \
                mock.patch.object(runner, "remove_uvm", side_effect=remove), \
                mock.patch.object(runner, "insert_uvm", side_effect=insert), \
                mock.patch.object(runner, "live_uvm_interface",
                                  return_value={"interface": {}}), \
                mock.patch.object(runner, "read_parameters",
                                  return_value=initial["parameters"]), \
                mock.patch.object(runner, "quiet_snapshot", return_value={"quiet": True}), \
                mock.patch.object(runner, "no_uvm_holders", return_value={}), \
                mock.patch.object(runner, "service_state", return_value=inactive), \
                mock.patch.object(runner, "set_service", side_effect=set_service), \
                mock.patch.object(runner, "validate_services_restored",
                                  return_value=initial["services"]), \
                mock.patch.object(runner, "capture_core",
                                  return_value=initial["core"] if core_result is None
                                  else core_result), \
                mock.patch.object(runner.safety, "validate_post_server_safety"):
            errors = runner.restore_runtime(
                recorder, state, initial, Path("/old/nvidia-uvm.ko"), descriptor,
                initial["parameters"], Path("/candidate/nvidia-uvm.ko"), descriptor,
                Path("/stage/nvidia-uvm.ko"), descriptor,
            )
        return state, errors

    def test_record_failures_do_not_block_physical_recovery_sequence(self):
        operations = []
        state, errors = self.run_restore(FakeRecorder(fail=True), operations)
        self.assertEqual(
            operations,
            ["remove_candidate", "insert_old", "start:nvidia-persistenced.service",
             "start:gdm.service"],
        )
        self.assertTrue(state.old_restored)
        self.assertTrue(state.services_restored)
        self.assertTrue(any(item["stage"].startswith("record_") for item in errors))

    def test_candidate_remove_failure_blocks_old_insert_and_services(self):
        operations = []
        state, errors = self.run_restore(
            FakeRecorder(), operations, remove_error=runner.LifecycleError("busy"))
        self.assertEqual(operations, ["remove_candidate"])
        self.assertFalse(state.old_restored)
        self.assertTrue(any(item["stage"] == "recovery_remove_candidate" for item in errors))

    def test_old_insert_failure_blocks_services(self):
        operations = []
        state, errors = self.run_restore(
            FakeRecorder(), operations, insert_error=runner.LifecycleError("insert"))
        self.assertEqual(operations, ["remove_candidate", "insert_old"])
        self.assertFalse(state.old_restored)
        self.assertTrue(any(item["stage"] == "recovery_insert_old" for item in errors))

    def test_core_mismatch_before_service_restore_blocks_service_start(self):
        operations = []
        state, errors = self.run_restore(
            FakeRecorder(), operations, core_result={"core": "changed"})
        self.assertEqual(operations, ["remove_candidate", "insert_old"])
        self.assertTrue(state.old_restored)
        self.assertFalse(state.services_restored)
        self.assertTrue(any(item["stage"] == "recovery_old_quiet" for item in errors))

    def test_recovery_ignores_queued_signal_but_result_remains_rejectable(self):
        operations = []
        runner.cells.INTERRUPTED_SIGNALS[:] = [runner.signal.SIGINT]
        try:
            state, errors = self.run_restore(FakeRecorder(), operations)
        finally:
            runner.cells.INTERRUPTED_SIGNALS.clear()
        self.assertTrue(state.old_restored and state.services_restored)
        self.assertEqual(operations[0:2], ["remove_candidate", "insert_old"])
        self.assertFalse(runner.campaign_complete(list(runner.MODES), True, errors,
                                                  InterruptedError("signal")))


class PublicationTests(unittest.TestCase):
    def call_publish(self, recorder, lease, events, *, pending=frozenset()):
        state = runner.RuntimeState(cells_completed=list(runner.MODES),
                                    old_restored=True, services_restored=True)
        expected = [{"lease": "same"}]
        handlers = {runner.signal.SIGINT: object(), runner.signal.SIGTERM: object()}

        def mask(operation, _signals):
            events.append(f"mask:{operation}")
            return frozenset()

        pending_values = pending if isinstance(pending, list) else [set(pending), set(pending)]
        with mock.patch.object(runner, "lease_inventory", return_value=expected), \
                mock.patch.object(runner.signal, "pthread_sigmask", side_effect=mask), \
                mock.patch.object(runner.signal, "sigpending", side_effect=pending_values), \
                mock.patch.object(runner.signal, "signal",
                                  side_effect=lambda *_args: events.append("restore_handler")), \
                mock.patch.object(runner.safety, "atomic_write_json",
                                  side_effect=lambda *_args: events.append("summary_candidate")), \
                mock.patch.object(runner, "promote_summary",
                                  side_effect=lambda *_args: events.append("summary_promote")):
            error = runner.close_lease_and_publish(
                lease, expected, recorder, state, True, [], [], None,
                Path("/tmp/fresh-q2-output"), handlers,
            )
        return error

    def test_lease_closes_before_true_records_and_summary(self):
        events = []
        lease = mock.Mock()
        lease.close.side_effect = lambda: events.append("close")
        recorder = FakeRecorder(events)
        self.assertIsNone(self.call_publish(recorder, lease, events))
        self.assertLess(events.index("close"), events.index("write"))
        self.assertLess(events.index("write"), events.index("summary_candidate"))
        self.assertLess(events.index("summary_candidate"), events.index("summary_promote"))
        self.assertFalse(recorder.writes[0]["complete"])
        self.assertTrue(recorder.writes[-1]["complete"])
        self.assertIn("commit_point_ns",
                      recorder.writes[-1]["completion_linearization"])
        self.assertTrue(recorder.value["complete"])
        restore_indices = [index for index, value in enumerate(events)
                           if value == "restore_handler"]
        self.assertEqual(len(restore_indices), 2)
        self.assertTrue(all(index < len(events) - 1 for index in restore_indices))
        self.assertEqual(events[-1], f"mask:{runner.signal.SIG_SETMASK}")

    def test_lease_close_failure_prevents_summary(self):
        events = []
        lease = mock.Mock()
        lease.close.side_effect = runner.LifecycleError("close")
        recorder = FakeRecorder(events)
        self.call_publish(recorder, lease, events)
        self.assertFalse(recorder.value["complete"])
        self.assertNotIn("summary_candidate", events)
        self.assertNotIn("summary_promote", events)

    def test_precommit_pending_signal_prevents_summary(self):
        events = []
        lease = mock.Mock()
        lease.close.side_effect = lambda: events.append("close")
        recorder = FakeRecorder(events)
        error = self.call_publish(recorder, lease, events,
                                  pending={runner.signal.SIGINT})
        self.assertIsInstance(error, InterruptedError)
        self.assertFalse(recorder.value["complete"])
        self.assertNotIn("summary_candidate", events)
        self.assertNotIn("summary_promote", events)
        # Original handlers are restored while the signal mask is still held;
        # the final mask restoration is the last signal-state operation.
        self.assertEqual(events[-1], f"mask:{runner.signal.SIG_SETMASK}")

    def test_publish_window_signal_rejects_true_records(self):
        events = []
        lease = mock.Mock()
        lease.close.side_effect = lambda: events.append("close")
        recorder = FakeRecorder(events)
        error = self.call_publish(
            recorder, lease, events,
            pending=[set(), {runner.signal.SIGTERM}],
        )
        self.assertIsInstance(error, InterruptedError)
        self.assertFalse(recorder.value["complete"])
        self.assertIn("summary_candidate", events)
        self.assertNotIn("summary_promote", events)
        self.assertGreaterEqual(events.count("write"), 2)

    def test_true_lifecycle_write_failure_never_promotes_summary(self):
        events = []
        lease = mock.Mock()
        lease.close.side_effect = lambda: events.append("close")
        recorder = FakeRecorder(events, fail_writes={2})
        self.call_publish(recorder, lease, events)
        self.assertFalse(recorder.value["complete"])
        self.assertIn("summary_candidate", events)
        self.assertNotIn("summary_promote", events)
        self.assertFalse(recorder.writes[0]["complete"])
        self.assertFalse(recorder.writes[-1]["complete"])


class PathAndDescriptorTests(unittest.TestCase):
    def test_descriptor_requires_exact_version_dependency_and_role(self):
        value = {
            "name": "nvidia_uvm", "version": runner.EXPECTED_DRIVER,
            "vermagic": runner.EXPECTED_VERMAGIC, "depends": ["nvidia"],
            "parameter_names": ["uvm_perf_prefetch_enable"],
            "diagnostic_present": True,
        }
        runner.validate_module_descriptor(value, diagnostic=True)
        for key, replacement in (("version", "610"), ("depends", []),
                                 ("diagnostic_present", False)):
            changed = dict(value, **{key: replacement})
            with self.subTest(key=key), self.assertRaises(runner.LifecycleError):
                runner.validate_module_descriptor(changed, diagnostic=True)

    def test_paths_are_fresh_scoped_and_exact_restore(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage_root = root / "stage-root"
            result_root = root / "results"
            candidate_dir = root / "candidate"
            restore_dir = stage_root / "gpreempt-849ea75d-6.15.11"
            for directory in (stage_root, result_root, candidate_dir, restore_dir):
                directory.mkdir(parents=True, exist_ok=True)
            candidate = candidate_dir / runner.MODULE_FILENAME
            restore = restore_dir / runner.MODULE_FILENAME
            candidate.write_text("candidate")
            restore.write_text("restore")
            stage = stage_root / "prefetch-diagnostic-0c109956-6.15.11"
            output = result_root / "prefetch-invalid-575-02"
            with mock.patch.object(runner, "STAGE_ROOT", stage_root), \
                    mock.patch.object(runner, "RESULT_ROOT", result_root), \
                    mock.patch.object(runner, "KNOWN_RESTORE", restore):
                runner.validate_paths(candidate, restore, stage, output)
                stage.mkdir()
                with self.assertRaises(runner.LifecycleError):
                    runner.validate_paths(candidate, restore, stage, output)


if __name__ == "__main__":
    unittest.main()
