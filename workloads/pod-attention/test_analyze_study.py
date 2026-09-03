"""Synthetic CPU-only audit fixtures; no attention workload or GPU is executed."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze_study as audit
import bench
import run_study as run
from test_run_study import report as fixture_report

VALUES = dict(official_serial=6.0, official_streams=5.0, pod_inline=3.0, pod_cuda=3.1, pod_bpf=3.2)


def write(path, data):
    path.write_text(json.dumps(data))


def fixture(root):
    campaign, preflight = root / 'full', root / 'preflight'
    campaign.mkdir()
    preflight.mkdir()
    runtime = {'fixture-runtime': {'bytes': 1, 'mtime_ns': 2}}
    write(preflight / 'manifest.json', dict(complete=True, mode='preflight', numeric_protocol=bench.NUMERIC_PROTOCOL,
        order=run.orders('preflight'), runtime=runtime))
    for item in run.orders('preflight'):
        directory = preflight / f"block-01-{item['arm']}"
        directory.mkdir()
        write(directory / 'operator.json', fixture_report(item['arm'], True))
        write(directory / 'execution.json', dict(status='passed', numeric_protocol=bench.NUMERIC_PROTOCOL,
            runtime_before=runtime, runtime_after=runtime))
    order = run.orders('full')
    write(campaign / 'manifest.json', dict(complete=True, mode='full', excluded_from_formal=False,
        numeric_protocol=bench.NUMERIC_PROTOCOL, order=order, completed=order, seed=20260903,
        lease_paths=['/tmp/gpubpf-revision-gpu0.lock', '/tmp/gpubpf-revision-struct-ops.lock'],
        runtime=runtime, preflight=str(preflight)))
    for number, item in enumerate(order):
        arm, block = item['arm'], item['block']
        directory = campaign / f'block-{block:02d}-{arm}'
        directory.mkdir()
        report = fixture_report(arm, False)
        report['block'] = block
        shapes = bench.shape_order(block, False)
        report['shape_order'] = [list(x) for x in shapes]
        cells = {(c['model'], c['decode_batch']): c for c in report['cells']}
        report['cells'] = [cells[shape] for shape in shapes]
        value = VALUES[arm] * (1 + block / 100)
        for cell in report['cells']:
            cell['samples'] = [dict(cuda_ms=value, host_wall_ms=value + 1) for _ in range(100)]
            cell['mean_cuda_ms'] = value
            cell['mean_host_wall_ms'] = value + 1
        write(directory / 'operator.json', report)
        (directory / 'client.log').write_text(''.join(
            f"POD_CELL arm={arm} model={c['model']} bs={c['decode_batch']} mean_cuda_ms={value:.6f}\n"
            for c in report['cells']))
        name = f'pod_attention_123_{number}' if arm == 'pod_bpf' else None
        execution = dict(status='passed', numeric_protocol=bench.NUMERIC_PROTOCOL, arm=arm, block=block,
            returncode=0, runtime_before=runtime, runtime_after=runtime, private_segment=name,
            command=['taskset', '-c', '8-15', '/fixture/python', '/fixture/bench.py', '--arm', arm,
                     '--block', str(block), '--output', str(directory / 'operator.json')],
            environment=run.environment(arm, Path('/fixture/ptx'), name),
            safety_before={'timestamp_ns': 1000 + 100 * number, 'gpu': {'driver': '575.57.08'}},
            safety_after={'timestamp_ns': 1010 + 100 * number, 'gpu': {'driver': '575.57.08'}},
            telemetry={'synthetic': True})
        execution['launch_environment'] = dict(execution['environment'])
        preload = execution['launch_environment'].pop('LD_PRELOAD', None)
        if preload:
            execution['command'][3:3] = ['/usr/bin/env', 'LD_PRELOAD=' + preload]
        if arm == 'pod_bpf':
            execution.update(private_segment_removed=True,
                loader_environment=run.environment(arm, Path('/fixture/ptx'), name, True),
                loader_command=['/fixture/pod-loader', '/fixture/selector.bpf.o', '/fixture/exact-kernels.txt'])
            (directory / 'loader.log').write_text('POD_LOADER_READY kernels=6\nPOD_LOADER_CLOSED\n')
        write(directory / 'execution.json', execution)
        (directory / 'gpu-telemetry.csv').write_text('synthetic CPU fixture, not telemetry\n')
    return campaign


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.campaign = fixture(Path(self.directory.name))
        for name in ('validate_pre_server_safety', 'validate_post_server_safety'):
            mocked = patch.object(run.safety, name)
            mocked.start()
            self.addCleanup(mocked.stop)
        mocked = patch.object(run.safety, 'validate_gpu_telemetry', return_value={'synthetic': True})
        mocked.start()
        self.addCleanup(mocked.stop)

    def first(self, name='operator.json'):
        item = run.orders('full')[0]
        return self.campaign / f"block-{item['block']:02d}-{item['arm']}" / name

    def test_complete_raw_matrix_and_constant_paired_ratios(self):
        result = audit.analyze(self.campaign)
        self.assertTrue(result['formal_complete'])
        self.assertEqual(result['operator_cells'], 250)
        self.assertEqual(result['raw_cuda_event_observations'], 25000)
        self.assertEqual(len(result['fp32_characterizations']), 250)
        self.assertEqual(len(result['results']), 10)
        ratio = result['results'][0]['comparisons']['device_bpf_vs_original_inline']
        self.assertAlmostEqual(ratio['geometric_mean_ratio'], 3.2 / 3.0)
        for value in ratio['confidence_interval_95']:
            self.assertAlmostEqual(value, 3.2 / 3.0)
        self.assertTrue(ratio['lower_is_better'])

    def test_reject_preflight_old_protocol_and_failed_campaign(self):
        path = self.campaign / 'manifest.json'
        original = audit.read_json(path)
        for update in (dict(mode='preflight'), dict(numeric_protocol='v1'), dict(complete=False),
                       dict(error='failed adverse cell')):
            write(path, {**original, **update})
            with self.assertRaises(ValueError):
                audit.audit_campaign(self.campaign)

    def test_reject_missing_duplicate_and_nested_attempt(self):
        extra = self.campaign / 'block-01-pod_bpf-attempt-02'
        extra.mkdir()
        write(extra / 'operator.json', audit.read_json(self.first()))
        with self.assertRaises(ValueError):
            audit.audit_campaign(self.campaign)
        (extra / 'operator.json').unlink()
        extra.rmdir()
        self.first().unlink()
        with self.assertRaises(ValueError):
            audit.audit_campaign(self.campaign)

    def test_does_not_trust_producer_mean_or_drop_bad_sample(self):
        path = self.first()
        original = audit.read_json(path)
        for field, value in (('mean_cuda_ms', .01), ('samples', original['cells'][0]['samples'][:-1])):
            report = json.loads(json.dumps(original))
            report['cells'][0][field] = value
            write(path, report)
            with self.assertRaises(ValueError):
                audit.audit_campaign(self.campaign)

    def test_reaudits_recorded_preflight(self):
        preflight = Path(audit.read_json(self.campaign / 'manifest.json')['preflight'])
        path = preflight / 'block-01-pod_bpf/operator.json'
        report = audit.read_json(path)
        report['complete'] = False
        write(path, report)
        with self.assertRaises(ValueError):
            audit.audit_campaign(self.campaign)

    def test_reject_cleanup_failure_and_runtime_change(self):
        path = self.first('execution.json')
        original = audit.read_json(path)
        for update in (dict(cleanup_errors=['surviving CUDA process']), dict(runtime_after={'changed': True}),
                       dict(numeric_protocol='v1')):
            write(path, {**original, **update})
            with self.assertRaises(ValueError):
                audit.audit_campaign(self.campaign)

    def test_reject_wrapper_injection_or_changed_target_command(self):
        path = self.campaign / 'block-01-pod_bpf/execution.json'
        original = audit.read_json(path)
        changes = (lambda e: e['launch_environment'].update(LD_PRELOAD=e['environment']['LD_PRELOAD']),
                   lambda e: e['command'].__delitem__(slice(3, 5)),
                   lambda e: e['command'].__setitem__(4, 'LD_PRELOAD=wrong.so'),
                   lambda e: e['command'].__setitem__(2, '0-7'))
        for change in changes:
            value = json.loads(json.dumps(original))
            change(value)
            write(path, value)
            with self.assertRaises(ValueError):
                audit.audit_campaign(self.campaign)

    def test_reject_overlapping_and_nonpositive_cell_windows(self):
        first, second = run.orders('full')[:2]
        first_path = self.first('execution.json')
        original = audit.read_json(first_path)
        broken = json.loads(json.dumps(original))
        broken['safety_after']['timestamp_ns'] = broken['safety_before']['timestamp_ns']
        write(first_path, broken)
        with self.assertRaises(ValueError):
            audit.audit_campaign(self.campaign)
        write(first_path, original)
        second_path = self.campaign / f"block-{second['block']:02d}-{second['arm']}" / 'execution.json'
        broken = audit.read_json(second_path)
        broken['safety_before']['timestamp_ns'] = original['safety_after']['timestamp_ns'] - 1
        write(second_path, broken)
        with self.assertRaises(ValueError):
            audit.audit_campaign(self.campaign)

    def test_reject_missing_real_bpf_detach_or_wrong_engine(self):
        directory = self.campaign / 'block-01-pod_bpf'
        loader = directory / 'loader.log'
        loader.write_text('POD_LOADER_READY kernels=6\n')
        with self.assertRaises(ValueError):
            audit.audit_campaign(self.campaign)
        loader.write_text('POD_LOADER_READY kernels=6\nPOD_LOADER_CLOSED\n')
        path = directory / 'operator.json'
        report = audit.read_json(path)
        report['cells'][0]['diagnostic']['contexts'][0]['engine'] = 1
        write(path, report)
        with self.assertRaises(ValueError):
            audit.audit_campaign(self.campaign)

    def test_fixed_whole_block_bootstrap_is_reproducible(self):
        draws = audit.bootstrap_indices()
        self.assertEqual(draws, audit.bootstrap_indices())
        self.assertEqual(len(draws), 10000)
        ratio = audit.paired_ratio([2, 4, 6, 8, 10], [1, 2, 3, 4, 5], draws)
        self.assertEqual(ratio['geometric_mean_ratio'], 2)
        self.assertEqual(ratio['confidence_interval_95'], [2, 2])


if __name__ == '__main__':
    unittest.main()
