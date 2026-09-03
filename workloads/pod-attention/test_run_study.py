"""CPU-only coordinator checks; no process in this suite executes CUDA."""
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack
from types import SimpleNamespace

import bench
import run_study as run
from test_bench import sparse_fallback_case


def report(arm='pod_bpf', preflight=True):
    cells = []
    for name, bs in bench.shape_order(1, preflight):
        meta, counters, contexts = sparse_fallback_case()
        meta.update(mode={'pod_inline': 0, 'pod_cuda': 1, 'pod_bpf': 2}.get(arm, 0), trace=1,
                    smem_bytes=81920)
        engine = 2 if arm == 'pod_bpf' else 1
        for context in contexts:
            context['engine'] = engine
        samples = [{'cuda_ms': 2.5, 'host_wall_ms': 3.0}] * (3 if preflight else 100)
        before = dict(launches=0, runtime_redirects=0)
        after = dict(launches=11 + len(samples), runtime_redirects=11 + len(samples),
                     prepared_functions=1, requested_dynamic_bytes=81920, verified_dynamic_bytes=81920,
                     static_shared_bytes=0, device_optin_bytes=101376)
        diagnostic = dict(metadata=meta, counters=counters, contexts=contexts,
                          audit=bench.audit_decisions(meta, counters, contexts, engine))
        characterization = {}
        for phase, shape in (('prefill', [1, 8192, 32, 128]), ('decode', [bs, 1, 32, 128])):
            characterization[phase] = dict(numeric_protocol=bench.NUMERIC_PROTOCOL, phase=phase,
                role='characterization_not_cross_precision_pass_gate', finite=True, shape_checked=True,
                mask='causal_prefix' if phase == 'prefill' else 'valid_kv', output_shape=shape,
                checked_elements=math.prod(shape), exceeding_elements=0,
                max_abs_error=0.0003, mean_abs_error=0.00001, rms_error=0.00002,
                atol=1e-3, rtol=1e-5, diagnostic_directory=None)
        cells.append(dict(numeric_protocol=bench.NUMERIC_PROTOCOL, fp32_characterization=characterization,
            model=name, decode_batch=bs, kv_heads=bench.MODEL_HEADS[name],
            query_heads=32, head_dim=128, prefill_batch=1, prefill_length=8192, decode_query_length=1,
            decode_cache_extent=8192, decode_valid_kv=8191, dtype='float16', warmups=10,
            seed=20260904 + (0 if name == 'llama-3-8b' else 1000) + bs, atol=1e-3, rtol=1e-5,
            samples=samples, mean_cuda_ms=2.5, mean_host_wall_ms=3.0, max_abs_vs_official=0.0002,
            official_max_abs_vs_fp32=0.0003, fused_params=15 if arm.startswith('pod_') else None,
            diagnostic=diagnostic if arm.startswith('pod_') else None,
            launch_bridge=dict(before=before, after=after, expected_launches=11 + len(samples))
                          if arm in ('pod_cuda', 'pod_bpf') else None))
    return dict(complete=True, numeric_protocol=bench.NUMERIC_PROTOCOL, arm=arm, block=1, preflight=preflight,
                fp32_characterizations={f"{c['model']}:bs{c['decode_batch']}": c['fp32_characterization'] for c in cells},
                shape_order=[list(x) for x in bench.shape_order(1, preflight)], cells=cells)


class CoordinatorTests(unittest.TestCase):
    def test_fixed_matrix_and_orders(self):
        self.assertEqual(len(run.orders('preflight')), 5)
        self.assertEqual(len(run.orders('full')), 25)
        self.assertEqual(run.orders('full')[:5], run.orders('preflight'))
        for block in range(1, 6):
            self.assertEqual({x['arm'] for x in run.orders('full') if x['block'] == block}, set(bench.ARMS))

    def test_clean_environment_and_exact_injection_paths(self):
        with patch.dict(os.environ, {'LD_PRELOAD': 'unrelated.so', 'BPFTIME_RUN_WITH_KERNEL': '1'}):
            for arm in ('official_serial', 'official_streams', 'pod_inline'):
                env = run.environment(arm, Path('/ptx'))
                self.assertNotIn('LD_PRELOAD', env)
                self.assertFalse(any(x.startswith('BPFTIME') for x in env))
            cuda = run.environment('pod_cuda', Path('/ptx'))
            self.assertEqual(cuda['LD_PRELOAD'], str(run.BRIDGE))
            self.assertFalse(any(x.startswith('BPFTIME') for x in cuda))
            bpf = run.environment('pod_bpf', Path('/ptx'), 'pod_attention_owned')
            self.assertEqual(bpf['LD_PRELOAD'], f'{run.BRIDGE}:{run.AGENT}')
            self.assertEqual(bpf['BPFTIME_CUDA_LATE_PTX_DIR'], '/ptx/device')
            self.assertNotIn('BPFTIME_RUN_WITH_KERNEL', bpf)
            loader = run.environment('pod_bpf', Path('/ptx'), 'pod_attention_owned', True)
            self.assertEqual(loader['LD_PRELOAD'], str(run.SERVER))
            self.assertNotIn('POD_LAUNCH_BRIDGE', loader)
        for invalid in (None, 'bpftime_maps_shm', 'pod_attention_/bad'):
            with self.subTest(name=invalid), self.assertRaises(ValueError):
                run.environment('pod_bpf', Path('/ptx'), invalid)

    def test_all_five_arm_reports_and_complete_shape_sweep(self):
        for arm in bench.ARMS:
            for preflight in (False, True):
                value = report(arm, preflight)
                self.assertIs(run.validate_report(value, arm, 1, preflight), value)

    def test_report_rejects_missing_samples_numerics_and_mechanism(self):
        changes = [lambda r: r.update(complete=False),
                   lambda r: r['cells'][0]['samples'].pop(),
                   lambda r: r['cells'][0].update(atol=0.1),
                   lambda r: r['cells'][0].update(head_dim=64),
                   lambda r: r['cells'][0].update(mean_cuda_ms=0.5),
                   lambda r: r['cells'][0].update(official_max_abs_vs_fp32=float('nan')),
                   lambda r: r['cells'][0]['diagnostic']['contexts'][0].update(engine=1),
                   lambda r: r['cells'][0]['launch_bridge']['after'].update(launches=13)]
        for change in changes:
            value = report()
            change(value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                run.validate_report(value, 'pod_bpf', 1, True)

    def test_private_segment_removal_is_exact_and_identity_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pod_attention_fixture'
            path.write_bytes(b'owned')
            identity = run.segment_identity(path)
            self.assertFalse(run.remove_owned_segment(path, None))
            with self.assertRaises(RuntimeError):
                run.remove_owned_segment(path, (identity[0], identity[1] + 1, identity[2]))
            self.assertTrue(path.exists())
            self.assertTrue(run.remove_owned_segment(path, identity))
            self.assertFalse(run.remove_owned_segment(path, identity))
            target = Path(directory) / 'unrelated'
            target.write_bytes(b'keep')
            path.symlink_to(target)
            with self.assertRaises(RuntimeError):
                run.segment_identity(path)
            self.assertTrue(target.exists())

    def test_formal_requires_all_five_preflight_cells_same_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = {'fixture': {'bytes': 10, 'mtime_ns': 20}}
            manifest = dict(complete=True, numeric_protocol=bench.NUMERIC_PROTOCOL, mode='preflight', order=run.orders('preflight'), runtime=runtime)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            for item in run.orders('preflight'):
                cell = root / f"block-01-{item['arm']}"
                cell.mkdir()
                (cell / 'execution.json').write_text(json.dumps(dict(status='passed', numeric_protocol=bench.NUMERIC_PROTOCOL, runtime_before=runtime,
                                                                     runtime_after=runtime)))
                (cell / 'operator.json').write_text(json.dumps(report(item['arm'])))
            run.validate_preflight(root, runtime)
            with self.assertRaises(ValueError):
                run.validate_preflight(root, {'changed': 1})
            broken = root / 'block-01-pod_bpf/operator.json'
            value = json.loads(broken.read_text())
            value['complete'] = False
            broken.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                run.validate_preflight(root, runtime)

    def test_v1_and_missing_characterization_are_not_admitted(self):
        for change in (lambda r: r.pop('numeric_protocol'),
                       lambda r: r['cells'][0]['fp32_characterization'].pop('decode'),
                       lambda r: r['cells'][0]['fp32_characterization']['prefill'].update(finite=False),
                       lambda r: r['cells'][0]['fp32_characterization']['prefill'].update(checked_elements=1)):
            value = report()
            change(value)
            with self.assertRaises(ValueError):
                run.validate_report(value, 'pod_bpf', 1, True)

    def test_v2_characterizes_excess_but_does_not_change_matching_threshold(self):
        value = report()
        cell = value['cells'][0]
        stats = cell['fp32_characterization']['prefill']
        stats.update(exceeding_elements=1, max_abs_error=.001328,
                     diagnostic_directory=bench.fp32_diagnostic_name(cell['model'], cell['decode_batch'], 'prefill'))
        cell['official_max_abs_vs_fp32'] = .001328
        run.validate_report(value, 'pod_bpf', 1, True)
        cell['atol'] = .002
        with self.assertRaises(ValueError):
            run.validate_report(value, 'pod_bpf', 1, True)

    def test_build_guard_is_read_only_and_does_not_kill(self):
        with tempfile.TemporaryDirectory() as directory:
            comm = Path(directory) / '1234/comm'
            comm.parent.mkdir()
            comm.write_text('nvcc\n')
            with patch.object(Path, 'glob', return_value=[comm]), self.assertRaises(RuntimeError):
                run.require_no_build()
            comm.write_text('python\n')
            with patch.object(Path, 'glob', return_value=[comm]):
                run.require_no_build()

    def lifecycle(self, root, *, preexisting=False, loader_fails=False, changed_between_cells=False,
                  closing_fails=False):
        cell = root / 'cell'
        shm_root = root / 'shm'
        shm_root.mkdir()
        segment = shm_root / f'pod_attention_{os.getpid()}_12345'
        if preexisting:
            segment.write_bytes(b'not ours')
        events = []
        class Process:
            def __init__(self, kind, rc=None):
                self.pid = {'client': 901, 'loader': 902, 'telemetry': 903}[kind]
                self.kind, self.returncode = kind, rc
            def poll(self):
                return self.returncode
            def wait(self, timeout):
                if self.returncode is None:
                    self.returncode = 0
                return self.returncode
        def popen(command, **kwargs):
            self.assertTrue(kwargs['start_new_session'])
            if 'libbpftime-syscall-server.so' in kwargs['env'].get('LD_PRELOAD', ''):
                events.append('loader start')
                segment.write_bytes(b'owned loader state')
                kwargs['stdout'].write('failed\n' if loader_fails else 'POD_LOADER_READY kernels=6\n')
                kwargs['stdout'].flush()
                process = Process('loader', 1 if loader_fails else None)
                def close():
                    events.append('loader stdin close')
                    kwargs['stdout'].write('POD_LOADER_CLOSED\n')
                    kwargs['stdout'].flush()
                process.stdin = SimpleNamespace(close=close)
                return process
            events.append('client start')
            (cell / 'operator.json').write_text(json.dumps(report()))
            return Process('client', 0)
        def stop(process):
            if process is not None:
                events.append(process.kind + ' stop')
                process.returncode = process.returncode or 0
        def json_write(path, value):
            path.write_text(json.dumps(value))
        snapshot = {'gpu': {'driver': '575.57.08'}}
        with ExitStack() as stack:
            stack.enter_context(patch.object(run, 'require_no_build'))
            stack.enter_context(patch.object(run, 'file_inventory', return_value={'binary': 1}))
            stack.enter_context(patch.object(run.time, 'monotonic_ns', return_value=12345))
            stack.enter_context(patch.object(run, 'Path', side_effect=lambda p: shm_root if p == '/dev/shm' else Path(p)))
            stack.enter_context(patch.object(run.subprocess, 'Popen', side_effect=popen))
            stack.enter_context(patch.object(run.shared, 'stop_owned', side_effect=stop))
            stack.enter_context(patch.object(run.shared, 'group_members', return_value=[]))
            for function in ('validate_pre_server_safety',):
                stack.enter_context(patch.object(run.safety, function))
            stack.enter_context(patch.object(run.safety, 'safety_snapshot', return_value=snapshot))
            stack.enter_context(patch.object(run.safety, 'wait_for_post_server_safety', return_value=snapshot))
            stack.enter_context(patch.object(run.safety, 'validate_gpu_telemetry', return_value={'ok': True}))
            stack.enter_context(patch.object(run.safety, 'atomic_write_json', side_effect=json_write))
            if closing_fails:
                def close_failed():
                    raise OSError('test stream close failure')
                stream = SimpleNamespace(close=close_failed)
            else:
                stream = stack.enter_context((root / 'telemetry.txt').open('w'))
            stack.enter_context(patch.object(run.safety, 'start_gpu_telemetry',
                return_value=(Process('telemetry'), stream, root / 'telemetry.txt')))
            frozen = {'old_binary': 1} if changed_between_cells else {'binary': 1}
            if preexisting or loader_fails or changed_between_cells or closing_fails:
                with self.assertRaises(RuntimeError):
                    run.run_cell(cell, {'block': 1, 'arm': 'pod_bpf'}, 'preflight', Path('/ptx'), [], frozen)
            else:
                run.run_cell(cell, {'block': 1, 'arm': 'pod_bpf'}, 'preflight', Path('/ptx'), [], frozen)
        return events, segment, json.loads((cell / 'execution.json').read_text())

    def test_loader_is_kept_until_client_exit_and_exact_segment_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            events, segment, execution = self.lifecycle(Path(directory))
            self.assertLess(events.index('client stop'), events.index('loader stdin close'))
            self.assertLess(events.index('loader stdin close'), events.index('loader stop'))
            self.assertEqual(execution['status'], 'passed')
            self.assertTrue(execution['private_segment_removed'])
            self.assertFalse(segment.exists())

    def test_existing_segment_is_never_started_over_or_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            events, segment, execution = self.lifecycle(Path(directory), preexisting=True)
            self.assertNotIn('loader start', events)
            self.assertNotIn('client start', events)
            self.assertEqual(segment.read_bytes(), b'not ours')
            self.assertEqual(execution['status'], 'failed')

    def test_loader_early_failure_preserves_failure_and_cleans_own_created_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            events, segment, execution = self.lifecycle(Path(directory), loader_fails=True)
            self.assertNotIn('client start', events)
            self.assertFalse(segment.exists())
            self.assertEqual(execution['status'], 'failed')
            self.assertIn('private BPF loader did not become ready', execution['error'])

    def test_change_between_cells_is_rejected_before_any_gpu_activity(self):
        with tempfile.TemporaryDirectory() as directory:
            events, _, execution = self.lifecycle(Path(directory), changed_between_cells=True)
            self.assertEqual(events, [])
            self.assertEqual(execution['status'], 'failed')
            self.assertIn('frozen campaign inventory', execution['error'])

    def test_stream_close_error_does_not_skip_post_safety_or_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, execution = self.lifecycle(Path(directory), closing_fails=True)
            self.assertEqual(execution['status'], 'failed')
            self.assertIn('test stream close failure', execution['cleanup_errors'])
            self.assertIn('safety_after', execution)
            self.assertEqual(execution['telemetry'], {'ok': True})


if __name__ == '__main__':
    unittest.main()
