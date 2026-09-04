"""CPU-only POD phase-study protocol tests; no CUDA process is launched."""
import io
import json
from contextlib import ExitStack
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

import bench
import run_phase_study as phase
import run_study as base
from test_run_study import report as study_report


TARGETS = [f'_Z{i}_true_fused_tb_fwd_kernel_fixture' for i in range(6)]
OPERATOR_TIMES = {
    'process_main_ns': 110,
    'stdlib_imports_done_ns': 120,
    'runtime_imports_start_ns': 130,
    'runtime_imports_done_ns': 140,
    'pre_first_diagnostic_ns': 150,
    'post_first_sync_ns': 160,
    'warmup_done_ns': 170,
    'steady_complete_ns': 180,
}


def phase_report(arm='pod_bpf', preflight=True):
    value = study_report(arm, True)
    value.update(preflight=preflight, phase_study=True,
                 phase_timestamps=dict(OPERATOR_TIMES))
    cell = value['cells'][0]
    cell['diagnostic']['metadata']['threads'] = 128
    count = 3 if preflight else 100
    cell['samples'] = [{'cuda_ms': 2.5, 'host_wall_ms': 3.0}] * count
    cell['mean_cuda_ms'] = 2.5
    cell['mean_host_wall_ms'] = 3.0
    if arm != 'pod_inline':
        bridge = cell['launch_bridge']
        bridge['expected_launches'] = 11 + count
        bridge['before'].update(first_launches=0)
        bridge['after'].update(launches=11 + count,
                               runtime_redirects=11 + count,
                               first_launches=1)
        bridge['first_launches'] = [
            {'kernel': TARGETS[0], 'monotonic_ns': 155}
        ]
    return value


def execution(arm='pod_bpf', runtime=None):
    runtime = runtime or {'fixture': {'bytes': 1, 'mtime_ns': 2}}
    is_bpf = arm == 'pod_bpf'
    return {
        'status': 'passed',
        'numeric_protocol': bench.NUMERIC_PROTOCOL,
        'arm': arm,
        'block': 1,
        'phase_study': True,
        'runtime_before': runtime,
        'runtime_after': runtime,
        'private_segment': 'pod_attention_fixture' if is_bpf else None,
        'private_segment_removed': True if is_bpf else None,
        'loader_command': ['/pod-loader', '/selector.bpf.o', '/ptx/exact-kernels.txt'] if is_bpf else None,
        'environment': {'BPFTIME_CUDA_LATE_PTX_DIR': '/ptx/device'} if is_bpf else {},
        'phase_timestamps': {
            'cell_start_ns': 10,
            'loader_spawn_ns': 20 if is_bpf else None,
            'loader_ready_ns': 30 if is_bpf else None,
            'client_spawn_ns': 100,
            'client_exit_ns': 200,
            'cleanup_complete_ns': 210,
        },
    }


class PhaseStudyTests(unittest.TestCase):
    def test_three_arm_preflight_and_five_interleaved_blocks(self):
        self.assertEqual(len(phase.orders('preflight')), 3)
        self.assertEqual(len(phase.orders('full')), 15)
        self.assertEqual(phase.orders('full')[:3], phase.orders('preflight'))
        self.assertEqual(phase.orders('full'), phase.orders('full'))
        for block in range(1, 6):
            cells = [item for item in phase.orders('full') if item['block'] == block]
            self.assertEqual({item['arm'] for item in cells}, set(phase.ARMS))
            self.assertEqual(len(cells), 3)

    def test_phase_report_accepts_all_arms_and_modes(self):
        for arm in phase.ARMS:
            for preflight in (True, False):
                value = phase_report(arm, preflight)
                self.assertIs(base.validate_report(value, arm, 1, preflight,
                                                   phase_study=True), value)

    def test_phase_report_fails_closed_on_markers_engine_and_first_launch(self):
        mutations = [
            lambda value: value['phase_timestamps'].pop('runtime_imports_done_ns'),
            lambda value: value['phase_timestamps'].update(warmup_done_ns=149),
            lambda value: value['phase_timestamps'].update(post_first_sync_ns=True),
            lambda value: value.update(shape_order=[['yi-6b', 32]]),
            lambda value: value['cells'][0]['diagnostic']['contexts'][0].update(engine=1),
            lambda value: value['cells'][0]['launch_bridge'].update(first_launches=[]),
            lambda value: value['cells'][0]['launch_bridge']['first_launches'][0].update(monotonic_ns=161),
        ]
        for mutation in mutations:
            value = phase_report()
            mutation(value)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                base.validate_report(value, 'pod_bpf', 1, True, phase_study=True)

    def test_cross_process_timeline_and_derived_durations_fail_closed(self):
        value = phase_report()
        run = execution()
        measured = phase.durations(run, value)
        self.assertEqual(measured['loader_ready_ns'], 10)
        self.assertEqual(measured['client_lifetime_ns'], 100)
        self.assertEqual(measured['pre_runtime_imports_ns'], 10)
        self.assertEqual(measured['whole_cell_ns'], 200)
        base.validate_phase_execution(run, value, 'pod_bpf')
        run['phase_timestamps']['client_spawn_ns'] = 111
        with self.assertRaises(ValueError):
            base.validate_phase_execution(run, value, 'pod_bpf')
        run = execution('pod_cuda')
        run['phase_timestamps']['loader_ready_ns'] = 20
        with self.assertRaises(ValueError):
            base.validate_phase_execution(run, phase_report('pod_cuda'), 'pod_cuda')

    def write_cell(self, root, arm, preflight=True, runtime=None):
        item = {'block': 1, 'arm': arm}
        directory = root / f'block-01-{arm}'
        directory.mkdir()
        (directory / 'execution.json').write_text(json.dumps(execution(arm, runtime)))
        (directory / 'operator.json').write_text(json.dumps(phase_report(arm, preflight)))
        if arm == 'pod_bpf':
            (directory / 'loader.log').write_text(
                'POD_LOADER_READY kernels=6\nPOD_LOADER_CLOSED\n')
        return directory, item

    def test_cell_summary_distinguishes_registration_from_actual_first_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for arm in phase.ARMS:
                directory, item = self.write_cell(root, arm)
                summary = phase.validate_cell(directory, item, True,
                                              {'fixture': {'bytes': 1, 'mtime_ns': 2}}, TARGETS)
                self.assertTrue(summary['complete'])
                self.assertEqual(summary['launched_target_count'], 0 if arm == 'pod_inline' else 1)
                if arm == 'pod_bpf':
                    self.assertEqual(summary['registered_target_count'], 6)
                    self.assertEqual(sum(x['first_launch_ns'] is not None
                                         for x in summary['attachment_targets']), 1)
                    self.assertIn('not evidence of six-kernel launch coverage', summary['scope'])

    def test_cell_rejects_non_target_launch_dirty_cleanup_and_loader_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, item = self.write_cell(root, 'pod_bpf')
            value = json.loads((directory / 'operator.json').read_text())
            value['cells'][0]['launch_bridge']['first_launches'][0]['kernel'] = '_Z_unregistered_true_fused_tb_fwd_kernel'
            (directory / 'operator.json').write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                phase.validate_cell(directory, item, True,
                                    {'fixture': {'bytes': 1, 'mtime_ns': 2}}, TARGETS)
            (directory / 'operator.json').write_text(json.dumps(phase_report()))
            run = json.loads((directory / 'execution.json').read_text())
            run['cleanup_errors'] = ['surviving loader']
            (directory / 'execution.json').write_text(json.dumps(run))
            with self.assertRaises(ValueError):
                phase.validate_cell(directory, item, True,
                                    {'fixture': {'bytes': 1, 'mtime_ns': 2}}, TARGETS)
            run.pop('cleanup_errors')
            (directory / 'execution.json').write_text(json.dumps(run))
            (directory / 'loader.log').write_text('POD_LOADER_READY kernels=5\nPOD_LOADER_CLOSED\n')
            with self.assertRaises(ValueError):
                phase.validate_cell(directory, item, True,
                                    {'fixture': {'bytes': 1, 'mtime_ns': 2}}, TARGETS)

    def test_full_requires_all_three_unchanged_preflight_cells(self):
        runtime = {'fixture': {'bytes': 1, 'mtime_ns': 2}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                'complete': True,
                'protocol': phase.PROTOCOL,
                'numeric_protocol': bench.NUMERIC_PROTOCOL,
                'mode': 'preflight',
                'order': phase.orders('preflight'),
                'completed': phase.orders('preflight'),
                'matched_blocks': [],
                'arms': list(phase.ARMS),
                'fixed_shape': list(phase.FIXED_SHAPE),
                'warmups': 10,
                'samples_per_cell': 3,
                'fresh_process_per_cell': True,
                'runtime': runtime,
                'exact_targets': TARGETS,
                'excluded_from_formal': True,
            }
            (root / 'manifest.json').write_text(json.dumps(manifest))
            for item in phase.orders('preflight'):
                directory, _ = self.write_cell(root, item['arm'], runtime=runtime)
                summary = phase.validate_cell(directory, item, True, runtime, TARGETS)
                (directory / 'phase.json').write_text(json.dumps(summary))
            manifest['matched_blocks'] = [phase.validate_matched_block(root, 1)]
            (root / 'manifest.json').write_text(json.dumps(manifest))
            phase.validate_preflight(root, runtime, TARGETS)
            manifest['runtime'] = {'changed': 1}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                phase.validate_preflight(root, runtime, TARGETS)
            manifest['runtime'] = runtime
            manifest['mode'] = 'full'
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                phase.validate_preflight(root, runtime, TARGETS)

    def test_exact_target_inventory_is_six_unique_named_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'exact-kernels.txt').write_text('\n'.join(TARGETS) + '\n')
            self.assertEqual(phase.exact_targets(root), TARGETS)
            (root / 'exact-kernels.txt').write_text('\n'.join(TARGETS[:5]) + '\n')
            with self.assertRaises(ValueError):
                phase.exact_targets(root)

    def test_entrypoint_creates_one_fresh_cell_directory_per_order_item(self):
        class Lease:
            def __init__(self):
                self.closed = False
            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extraction = root / 'ptx'
            extraction.mkdir()
            (extraction / 'exact-kernels.txt').write_text('\n'.join(TARGETS) + '\n')
            runtime = {'fixture': {'bytes': 1, 'mtime_ns': 2}}
            calls = []

            def json_write(path, value):
                path.write_text(json.dumps(value))

            def run_cell(directory, item, mode, selected, paths, inventory, phase_study=False):
                self.assertTrue(phase_study)
                self.assertEqual(selected, extraction)
                self.assertEqual(inventory, runtime)
                directory.mkdir()
                calls.append((directory, dict(item), mode))

            minimal = {'complete': True, 'protocol': phase.PROTOCOL}
            common = [patch.object(base, 'preparation', return_value=[]),
                      patch.object(base, 'file_inventory', return_value=runtime),
                      patch.object(base, 'require_no_build'),
                      patch.object(base.shared, 'Leases', Lease),
                      patch.object(base, 'run_cell', side_effect=run_cell),
                      patch.object(base.safety, 'atomic_write_json', side_effect=json_write),
                      patch.object(phase, 'validate_cell', return_value=minimal),
                      patch.object(phase, 'validate_matched_block',
                                   side_effect=lambda _, block: {'block': block})]
            for mode, count in (('preflight', 3), ('full', 15)):
                calls.clear()
                output = root / mode
                argv = ['run_phase_study.py', mode, '--output', str(output),
                        '--ptx', str(extraction)]
                if mode == 'full':
                    argv += ['--preflight', str(root / 'prior')]
                with ExitStack() as stack:
                    stack.enter_context(patch.object(sys, 'argv', argv))
                    stack.enter_context(patch.object(phase, 'validate_preflight'))
                    for context in common:
                        stack.enter_context(context)
                    phase.main()
                self.assertEqual(len(calls), count)
                self.assertEqual(len({directory for directory, _, _ in calls}), count)
                self.assertEqual([item for _, item, _ in calls], phase.orders(mode))
                manifest = json.loads((output / 'manifest.json').read_text())
                self.assertTrue(manifest['complete'])
                self.assertEqual(manifest['completed'], phase.orders(mode))

    def test_dry_run_prints_exact_matrix_without_runtime_or_output_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            forbidden = AssertionError('dry-run crossed the offline boundary')
            for mode, count, samples in (('preflight', 3, 3), ('full', 15, 100)):
                output = root / f'{mode}-must-not-exist'
                argv = ['run_phase_study.py', mode, '--dry-run', '--output', str(output),
                        '--ptx', str(root / 'absent-ptx')]
                if mode == 'full':
                    argv += ['--preflight', str(root / 'absent-preflight')]
                stream = io.StringIO()
                with ExitStack() as stack:
                    stack.enter_context(patch.object(sys, 'argv', argv))
                    stack.enter_context(patch.object(base, 'preparation', side_effect=forbidden))
                    stack.enter_context(patch.object(base, 'file_inventory', side_effect=forbidden))
                    stack.enter_context(patch.object(base, 'require_no_build', side_effect=forbidden))
                    stack.enter_context(patch.object(base.shared, 'Leases', side_effect=forbidden))
                    stack.enter_context(patch.object(base, 'run_cell', side_effect=forbidden))
                    stack.enter_context(patch.object(phase, 'validate_preflight', side_effect=forbidden))
                    stack.enter_context(redirect_stdout(stream))
                    phase.main()
                plan = json.loads(stream.getvalue())
                self.assertFalse(output.exists())
                self.assertTrue(plan['dry_run'])
                self.assertFalse(plan['executes_gpu_work'])
                self.assertFalse(plan['writes_output'])
                self.assertFalse(plan['experiment_evidence'])
                self.assertEqual(plan['cell_count'], count)
                self.assertEqual(plan['samples_per_cell'], samples)
                self.assertEqual(plan['fixed_shape'], list(phase.FIXED_SHAPE))
                self.assertEqual([(cell['block'], cell['arm']) for cell in plan['cells']],
                                 [(item['block'], item['arm']) for item in phase.orders(mode)])
                self.assertTrue(all(cell['fresh_client_process'] for cell in plan['cells']))
                self.assertEqual(sum(cell['owned_private_loader'] for cell in plan['cells']),
                                 1 if mode == 'preflight' else 5)
                self.assertIn('generic attachment cost', plan['claim_boundary'])

    def test_full_dry_run_still_requires_preflight_argument(self):
        with tempfile.TemporaryDirectory() as temporary:
            argv = ['run_phase_study.py', 'full', '--dry-run', '--output',
                    str(Path(temporary) / 'out')]
            with patch.object(sys, 'argv', argv), self.assertRaises(SystemExit):
                phase.main()


if __name__ == '__main__':
    unittest.main()
