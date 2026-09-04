#!/usr/bin/env python3
"""CPU-only failure-injection tests for the scheduler full-core lifecycle."""
from __future__ import annotations

import copy
import fcntl
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import tempfile
import unittest
from unittest import mock

import run_lifecycle as runner


def service(active: str, substate: str, result: str = "success") -> dict[str, str]:
    return {
        "LoadState": "loaded", "ActiveState": active, "SubState": substate,
        "Result": result, "UnitFileState": "enabled",
    }


class LeaseTests(unittest.TestCase):
    def test_existing_inodes_are_opened_read_only_and_locked(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = (Path(temporary) / "gpu.lock", Path(temporary) / "ops.lock")
            for path in paths:
                path.touch(mode=0o644)
            before = [(path.stat().st_dev, path.stat().st_ino) for path in paths]
            identity = (os.getuid(), os.getgid())
            lease = runner.LifecycleLeases(paths, expected_owner=identity)
            lease.acquire()
            try:
                self.assertEqual(
                    [fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE
                     for fd in lease.descriptors],
                    [os.O_RDONLY, os.O_RDONLY],
                )
                contender = runner.LifecycleLeases(paths, expected_owner=identity)
                with self.assertRaises(BlockingIOError):
                    contender.acquire()
            finally:
                lease.close()
            self.assertEqual([(path.stat().st_dev, path.stat().st_ino)
                              for path in paths], before)

    def test_missing_or_symlink_lease_is_never_created_or_followed(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.lock"
            lease = runner.LifecycleLeases((missing,))
            with self.assertRaises(FileNotFoundError):
                lease.acquire()
            self.assertFalse(missing.exists())
            target = Path(temporary) / "target.lock"
            target.touch(mode=0o644)
            missing.symlink_to(target)
            with self.assertRaises(runner.LifecycleError):
                runner.LifecycleLeases((missing,)).acquire()

    def test_matrix_reuses_validated_descriptors_without_reacquiring(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = (root / "gpu.lock", root / "ops.lock")
            for path in paths:
                path.touch(mode=0o644)
            lease = runner.LifecycleLeases(
                paths, expected_owner=(os.getuid(), os.getgid()))
            lease.acquire()
            try:
                with mock.patch.object(runner.cells, "LEASE_PATHS", paths), \
                        mock.patch.object(runner.cells, "matrix_plan", return_value=[]), \
                        mock.patch.object(runner.cells, "admission", return_value={"ok": True}):
                    result = runner.cells.run_matrix(
                        root / "matrix",
                        inherited_lease_fds=tuple(lease.descriptors),
                    )
                self.assertTrue(result["complete"])
                self.assertEqual(result["lease_mode"], "validated_inherited")
                self.assertEqual(result["passed_cells"], 0)
            finally:
                lease.close()


class ParameterTests(unittest.TestCase):
    CORE = runner.MODULE_BY_NAME["nvidia"]

    def test_core_report_is_converted_to_exact_insmod_values(self):
        raw = (
            "ResmanDebugLevel: 4294967295\n"
            "RmProfilingAdminOnly: 1\n"
            "RegistryDwords: \"\"\n"
            "RegistryDwordsPerDevice: \"pci=0000:01:00.0;A=1\"\n"
            "RmMsg: \"\"\nGpuBlacklist: \"\"\n"
            "TemporaryFilePath: \"\"\nExcludedGpus: \"\"\n"
        )
        values = runner.parse_core_parameters(raw)
        self.assertEqual(values["ResmanDebugLevel"], "-1")
        argv = runner.parameter_arguments(self.CORE, values)
        self.assertIn("NVreg_ResmanDebugLevel=-1", argv)
        self.assertIn("NVreg_RestrictProfilingToAdminUsers=1", argv)
        self.assertNotIn("NVreg_RegistryDwords=", argv)
        self.assertIn(
            "NVreg_RegistryDwordsPerDevice=pci=0000:01:00.0;A=1", argv)

    def test_core_report_rejects_duplicates_malformed_and_incomplete_state(self):
        valid_strings = (
            "RegistryDwords: \"\"\nRegistryDwordsPerDevice: \"\"\n"
            "RmMsg: \"\"\nGpuBlacklist: \"\"\n"
            "TemporaryFilePath: \"\"\nExcludedGpus: \"\"\n"
        )
        invalid = (
            valid_strings + "RmProfilingAdminOnly: 1\nRmProfilingAdminOnly: 0\n",
            valid_strings + "RmProfilingAdminOnly: -1\n",
            "RmProfilingAdminOnly: 1\n",
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(runner.LifecycleError):
                runner.parse_core_parameters(value)

    def test_null_sysfs_parameter_is_omitted_and_values_are_not_shell_parsed(self):
        module = runner.MODULE_BY_NAME["nvidia_modeset"]
        values = {"config_file": "(null)", "fail_malloc": "-1",
                  "debug_force_color_space": "0;touch /tmp/not-run"}
        self.assertEqual(
            runner.insmod_command(Path("/stage/nvidia-modeset.ko"), module, values),
            ["sudo", "-n", "insmod", "/stage/nvidia-modeset.ko",
             "debug_force_color_space=0;touch /tmp/not-run", "fail_malloc=-1"],
        )


class CommandAndOrderTests(unittest.TestCase):
    def test_only_explicit_nonforced_module_operations_are_permitted(self):
        forbidden = (
            ["modprobe", "nvidia"], ["sudo", "-n", "depmod", "-a"],
            ["make", "modules_install"], ["sudo", "rmmod", "-f", "nvidia"],
            ["reboot"], ["pkill", "Xorg"],
        )
        for argv in forbidden:
            with self.subTest(argv=argv), self.assertRaises(runner.LifecycleError):
                runner.validate_command(list(argv))
        runner.validate_command(["sudo", "-n", "rmmod", "nvidia_uvm"])
        runner.validate_command(["sudo", "-n", "insmod", "/stage/nvidia.ko"])

    def test_full_set_has_exact_forward_and_reverse_dependency_orders(self):
        self.assertEqual(
            runner.LOAD_ORDER,
            ("nvidia", "nvidia_modeset", "nvidia_drm", "nvidia_uvm"),
        )
        self.assertEqual(
            runner.REMOVE_ORDER,
            ("nvidia_uvm", "nvidia_drm", "nvidia_modeset", "nvidia"),
        )
        self.assertEqual(
            set(runner.leaf_module_names(runner.LOAD_ORDER)),
            {"nvidia_drm", "nvidia_uvm"},
        )
        self.assertEqual(
            set(runner.leaf_module_names(
                ("nvidia", "nvidia_modeset", "nvidia_uvm"))),
            {"nvidia_modeset", "nvidia_uvm"},
        )

    def test_each_unload_failure_stops_before_lower_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_modules = {}
            for name in runner.LOAD_ORDER:
                directory = root / name
                directory.mkdir()
                (directory / "refcnt").write_text("0\n")
                fake_modules[name] = SimpleNamespace(
                    name=name, sysfs=directory, loaded_btf=root / f"{name}.btf",
                )
            for failure_index in range(len(runner.REMOVE_ORDER)):
                calls = []

                def command(argv, **_kwargs):
                    calls.append(argv[-1])
                    if len(calls) - 1 == failure_index:
                        raise runner.LifecycleError("injected unload")
                    return subprocess.CompletedProcess(argv, 0, "", "")

                with self.subTest(failure_index=failure_index), \
                        mock.patch.object(runner, "MODULE_BY_NAME", fake_modules), \
                        mock.patch.object(runner, "removal_guard",
                                          return_value={"loaded_subset": list(runner.LOAD_ORDER)}), \
                        mock.patch.object(runner, "loaded_subset_unchecked", return_value=()), \
                        mock.patch.object(runner, "run_command", side_effect=command), \
                        mock.patch.object(runner, "wait_for"):
                    with self.assertRaisesRegex(runner.LifecycleError, "injected unload"):
                        runner.remove_loaded_subset("boot", [], honor_interrupt=False)
                self.assertEqual(calls, list(runner.REMOVE_ORDER[:failure_index + 1]))

    def test_each_insert_failure_stops_before_dependents(self):
        fake_modules = {
            name: SimpleNamespace(name=name, filename=f"{name}.ko")
            for name in runner.LOAD_ORDER
        }
        parameters = {name: {} for name in runner.LOAD_ORDER}
        for failure_index in range(len(runner.LOAD_ORDER)):
            calls = []

            def command(argv, **_kwargs):
                calls.append(Path(argv[3]).stem)
                if len(calls) - 1 == failure_index:
                    raise runner.LifecycleError("injected insert")
                return subprocess.CompletedProcess(argv, 0, "", "")

            with self.subTest(failure_index=failure_index), \
                    mock.patch.object(runner, "MODULE_BY_NAME", fake_modules), \
                    mock.patch.object(runner, "loaded_subset_unchecked",
                                      side_effect=[(), runner.LOAD_ORDER]), \
                    mock.patch.object(runner, "run_command", side_effect=command), \
                    mock.patch.object(runner, "wait_for"), \
                    mock.patch.object(runner, "live_module_descriptor", return_value={}):
                with self.assertRaisesRegex(runner.LifecycleError, "injected insert"):
                    runner.insert_subset(Path("/stage"), runner.LOAD_ORDER, parameters,
                                         candidate=True, honor_interrupt=False)
            self.assertEqual(len(calls), failure_index + 1)


class ServiceAndInputTests(unittest.TestCase):
    def test_restore_changes_only_initially_active_services_in_safe_order(self):
        initial = {
            "gdm.service": service("active", "running"),
            "nvidia-persistenced.service": service("active", "running"),
        }
        inactive = {unit: service("inactive", "dead") for unit in runner.SERVICES}
        final = copy.deepcopy(initial)
        observed = []
        states = iter((inactive["gdm.service"], inactive["nvidia-persistenced.service"],
                       final["gdm.service"], final["nvidia-persistenced.service"]))
        with mock.patch.object(runner, "service_state", side_effect=lambda _unit: next(states)), \
                mock.patch.object(runner, "set_service",
                                  side_effect=lambda unit, action: observed.append((unit, action))):
            runner.restore_services(initial)
        self.assertEqual(observed, [
            ("nvidia-persistenced.service", "start"),
            ("gdm.service", "start"),
        ])

    def test_local_user_session_is_rejected_before_display_stop(self):
        listing = "7 1000 user seat0 tty\n"
        detail = (
            "Id=7\nUser=1000\nName=user\nSeat=seat0\nClass=user\nType=tty\n"
            "Service=gdm-password\nActive=yes\nRemote=no\nState=active\n"
        )
        with mock.patch.object(runner, "checked_stdout", side_effect=[listing, detail]), \
                self.assertRaises(runner.LifecycleError):
            runner.local_sessions()

    def test_only_silent_fuser_exit_one_proves_no_holder(self):
        paths = [Path("/dev/nvidia0")]
        good = subprocess.CompletedProcess(["fuser"], 1, "", "")
        self.assertEqual(runner.validate_fuser_result(good, paths)["returncode"], 1)
        for value in (
            subprocess.CompletedProcess(["fuser"], 0, "1", ""),
            subprocess.CompletedProcess(["fuser"], 1, "", "header"),
            subprocess.CompletedProcess(["fuser"], 2, "", ""),
        ):
            with self.assertRaises(runner.LifecycleError):
                runner.validate_fuser_result(value, paths)


class FakeRecorder:
    def __init__(self, fail: bool = False, fail_writes: tuple[int, ...] = ()):
        self.value = {"complete": False, "transitions": []}
        self.fail = fail
        self.fail_writes = set(fail_writes)
        self.write_count = 0

    def transition(self, name, status, **details):
        if self.fail:
            raise OSError("injected record failure")
        self.value["transitions"].append((name, status, details))

    def write(self):
        self.write_count += 1
        if self.fail or self.write_count in self.fail_writes:
            raise OSError("injected write failure")


def recovery_initial() -> dict:
    subset = ("nvidia", "nvidia_uvm")
    return {
        "module_subset": list(subset), "parameters": {name: {} for name in subset},
        "boot_id": "boot", "device_nodes": [{"path": "/dev/nvidia0"}],
        "services": {unit: service("active", "running") for unit in runner.SERVICES},
        "safety": {"initial": True},
    }


class RecoveryTests(unittest.TestCase):
    def run_recovery(self, *, recorder=None, remove_error=None, insert_error=None,
                     before_service_error=None, service_error=None,
                     destructive=True):
        initial = recovery_initial()
        state = runner.RuntimeState(
            destructive_started=destructive,
            old_removal_complete=destructive,
        )
        events = []

        def remove(*_args, **_kwargs):
            events.append("remove")
            if remove_error:
                raise remove_error
            return []

        def insert(*_args, **_kwargs):
            events.append("insert")
            if insert_error:
                raise insert_error
            return []

        def power():
            events.append("power")
            if before_service_error:
                raise before_service_error
            return runner.POWER_LIMIT_W

        def services(_initial):
            events.append("services")
            if service_error:
                raise service_error
            return _initial

        descriptor = {name: {} for name in initial["module_subset"]}
        with mock.patch.object(runner, "require_artifacts_unchanged"), \
                mock.patch.object(runner, "stop_active_services",
                                  side_effect=lambda: events.append("stop") or []), \
                mock.patch.object(runner, "remove_loaded_subset", side_effect=remove), \
                mock.patch.object(runner, "insert_subset", side_effect=insert), \
                mock.patch.object(runner, "ensure_power_limit", side_effect=power), \
                mock.patch.object(runner, "wait_device_nodes", return_value=[]), \
                mock.patch.object(runner, "capture_runtime", return_value={}), \
                mock.patch.object(runner, "quiet_snapshot", return_value={}), \
                mock.patch.object(runner, "no_device_holders", return_value={}), \
                mock.patch.object(runner, "device_nodes",
                                  return_value=initial["device_nodes"]), \
                mock.patch.object(runner, "restore_services", side_effect=services), \
                mock.patch.object(runner, "require_boot"):
            errors = runner.recover(
                recorder or FakeRecorder(), state, initial,
                Path("/candidate"), Path("/restore"), Path("/stage"),
                descriptor, descriptor, descriptor,
            )
        return state, errors, events

    def test_recovery_prioritizes_stop_remove_insert_validate_then_services(self):
        state, errors, events = self.run_recovery()
        self.assertEqual(events[:4], ["stop", "remove", "insert", "power"])
        self.assertIn("services", events)
        self.assertTrue(state.old_restored and state.services_restored)
        self.assertEqual(errors, [])

    def test_record_failures_do_not_stop_physical_restoration(self):
        state, errors, events = self.run_recovery(recorder=FakeRecorder(fail=True))
        self.assertTrue(state.old_restored and state.services_restored)
        self.assertEqual(events[:3], ["stop", "remove", "insert"])
        self.assertTrue(any(item["stage"].startswith("record_") for item in errors))

    def test_unload_failure_blocks_insert_and_service_restart(self):
        state, errors, events = self.run_recovery(
            remove_error=runner.LifecycleError("injected remove"))
        self.assertEqual(events, ["stop", "remove"])
        self.assertFalse(state.old_restored or state.services_restored)
        self.assertTrue(any(item["stage"] == "recovery_remove_live_subset"
                            for item in errors))

    def test_insert_failure_blocks_validation_and_service_restart(self):
        state, errors, events = self.run_recovery(
            insert_error=runner.LifecycleError("injected insert"))
        self.assertEqual(events, ["stop", "remove", "insert"])
        self.assertFalse(state.old_restored or state.services_restored)
        self.assertTrue(any(item["stage"] == "recovery_insert_old_subset"
                            for item in errors))

    def test_validation_failure_withholds_services(self):
        state, errors, events = self.run_recovery(
            before_service_error=runner.LifecycleError("injected validation"))
        self.assertEqual(events[:4], ["stop", "remove", "insert", "power"])
        self.assertNotIn("services", events)
        self.assertTrue(state.old_restored)
        self.assertFalse(state.services_restored)
        self.assertTrue(any(item["stage"] == "recovery_validate_before_services"
                            for item in errors))

    def test_service_failure_is_a_hard_recovery_failure(self):
        state, errors, events = self.run_recovery(
            service_error=runner.LifecycleError("injected service"))
        self.assertIn("services", events)
        self.assertTrue(state.old_restored)
        self.assertFalse(state.services_restored)
        self.assertTrue(any(item["stage"] == "recovery_restore_services"
                            for item in errors))

    def test_failure_before_mutation_validates_old_without_unloading(self):
        state, errors, events = self.run_recovery(destructive=False)
        self.assertNotIn("remove", events)
        self.assertNotIn("insert", events)
        self.assertTrue(state.old_restored and state.services_restored)
        self.assertEqual(errors, [])

    def test_partial_old_removal_is_completed_without_unloading_survivors(self):
        initial = recovery_initial()
        state = runner.RuntimeState(destructive_started=True,
                                    old_removal_complete=False,
                                    candidate_insert_started=False)
        descriptor = {name: {} for name in initial["module_subset"]}
        events = []
        with mock.patch.object(runner, "require_artifacts_unchanged"), \
                mock.patch.object(runner, "stop_active_services", return_value=[]), \
                mock.patch.object(runner, "remove_loaded_subset") as remove, \
                mock.patch.object(runner, "restore_partial_old_subset",
                                  side_effect=lambda *_args: events.append("complete-old") or []), \
                mock.patch.object(runner, "ensure_power_limit", return_value=400.0), \
                mock.patch.object(runner, "wait_device_nodes", return_value=[]), \
                mock.patch.object(runner, "capture_runtime", return_value={}), \
                mock.patch.object(runner, "quiet_snapshot", return_value={}), \
                mock.patch.object(runner, "no_device_holders", return_value={}), \
                mock.patch.object(runner, "device_nodes",
                                  return_value=initial["device_nodes"]), \
                mock.patch.object(runner, "restore_services",
                                  return_value=initial["services"]), \
                mock.patch.object(runner, "require_boot"):
            errors = runner.recover(
                FakeRecorder(), state, initial, Path("/candidate"),
                Path("/restore"), Path("/stage"), descriptor, descriptor,
                descriptor,
            )
        remove.assert_not_called()
        self.assertEqual(events, ["complete-old"])
        self.assertEqual(errors, [])
        self.assertTrue(state.old_restored and state.services_restored)


class PublicationTests(unittest.TestCase):
    def complete_state(self):
        return runner.RuntimeState(
            native_preflight_complete=True, matrix_complete=True,
            old_restored=True, services_restored=True,
        )

    def test_success_closes_lease_before_publishing_true_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            recorder = FakeRecorder()
            recorder.value["final"] = {"restored": True}
            lease = mock.Mock()
            lease.inventory.return_value = [{"same": True}]
            events = []
            lease.close.side_effect = lambda: events.append("close")
            original_write = recorder.write

            def write():
                events.append(f"write:{recorder.value['complete']}")
                original_write()

            recorder.write = write
            with mock.patch.object(runner.signal, "sigpending",
                                  side_effect=[set(), set()]):
                error = runner.publish(
                    recorder, lease, [{"same": True}], self.complete_state(),
                    None, [], [], output,
                )
            self.assertIsNone(error)
            self.assertTrue(recorder.value["complete"])
            self.assertTrue((output / "summary.json").is_file())
            self.assertLess(events.index("close"), events.index("write:False"))
            self.assertLess(events.index("write:False"), events.index("write:True"))

    def test_pending_signal_or_lease_failure_suppresses_completion(self):
        cases = ("signal", "identity", "close")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                recorder = FakeRecorder()
                recorder.value["final"] = {}
                lease = mock.Mock()
                lease.inventory.return_value = ([] if case == "identity"
                                                else [{"same": True}])
                if case == "close":
                    lease.close.side_effect = runner.LifecycleError("close")
                pending = {runner.signal.SIGINT} if case == "signal" else set()
                with mock.patch.object(runner.signal, "sigpending",
                                      side_effect=[pending, pending]):
                    error = runner.publish(
                        recorder, lease, [{"same": True}], self.complete_state(),
                        None, [], [], Path(temporary),
                    )
                self.assertFalse(recorder.value["complete"])
                self.assertFalse((Path(temporary) / "summary.json").exists())
                if case == "signal":
                    self.assertIsInstance(error, InterruptedError)

    def test_record_write_failure_never_publishes_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = FakeRecorder(fail_writes=(1,))
            recorder.value["final"] = {}
            lease = mock.Mock()
            lease.inventory.return_value = [{"same": True}]
            with mock.patch.object(runner.signal, "sigpending", return_value=set()):
                runner.publish(recorder, lease, [{"same": True}], self.complete_state(),
                               None, [], [], Path(temporary))
            self.assertFalse(recorder.value["complete"])
            self.assertFalse((Path(temporary) / "summary.json").exists())


class PathTests(unittest.TestCase):
    def test_stage_output_are_fresh_scoped_and_restore_is_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage_root = root / "stages"
            result_root = root / "results"
            candidate = root / "candidate"
            restore = stage_root / "restore"
            for directory in (stage_root, result_root, candidate, restore):
                directory.mkdir(parents=True, exist_ok=True)
            stage = stage_root / "sched-init-candidate-one"
            output = result_root / "sched-init-live-575-one"
            with mock.patch.object(runner, "STAGE_ROOT", stage_root), \
                    mock.patch.object(runner, "KNOWN_RESTORE_DIR", restore), \
                    mock.patch.object(runner, "RESULT_ROOT", result_root):
                runner.validate_paths(candidate, restore, stage, output)
                stage.mkdir()
                with self.assertRaises(runner.LifecycleError):
                    runner.validate_paths(candidate, restore, stage, output)


class BtfAbiTests(unittest.TestCase):
    RAW = (
        "[100] STRUCT 'nv_gpu_sched_ops' size=32 vlen=4\n"
        "\t'on_task_init' type_id=1 bits_offset=0\n"
        "\t'on_bind' type_id=2 bits_offset=64\n"
        "\t'on_task_destroy' type_id=3 bits_offset=128\n"
        "\t'on_timeslice_control' type_id=4 bits_offset=192\n"
        "[200] FUNC_PROTO '(anon)' ret_type_id=6 vlen=2\n"
        "\t'hClient' type_id=14\n\t'hTsg' type_id=14\n"
        "[201] FUNC 'bpf_nv_gpu_preempt_tsg' type_id=200 linkage=static\n"
        "[202] DECL_TAG 'bpf_kfunc' type_id=201 component_idx=-1\n"
    )
    C = "extern int bpf_nv_gpu_preempt_tsg(u32 hClient, u32 hTsg) __weak __ksym;\n"

    def test_scheduler_base_abi_is_exact_and_address_free(self):
        observed = runner.validate_scheduler_base_interface(self.RAW, self.C)
        self.assertEqual(observed["struct_size"], 32)
        self.assertEqual(observed["preempt_signature"], "s32(u32,u32)")
        with self.assertRaises(runner.LifecycleError):
            runner.validate_scheduler_base_interface(
                self.RAW.replace("bits_offset=192", "bits_offset=224"), self.C)
        with self.assertRaises(runner.LifecycleError):
            runner.validate_scheduler_base_interface(
                self.RAW, self.C.replace("u32 hTsg", "u64 hTsg"))


if __name__ == "__main__":
    unittest.main()
