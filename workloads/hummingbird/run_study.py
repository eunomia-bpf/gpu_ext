#!/usr/bin/env python3
"""Run the real DISB clients with fixed arrivals; no builds or driver changes."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'gpreempt'))
import run_load_study as gp
from analyze_three_way import estimate_ratios
from analyze_load_study import policy_action_coverage

base, safety = gp.original, gp.safety
ARMS = ('native', 'fixed', 'timeslice_control', 'idle_c', 'idle_bpf')
SCENARIOS = ('periodic', 'burstgpt')
SEED = 20260903
NS = 1_000_000_000


def config(scenario, seconds, isolated=False):
    if scenario not in SCENARIOS or type(seconds) is not int or seconds not in (10, 60):
        raise ValueError('use the fixed 10 s preflight/calibration or 60 s study')
    value = gp.make_config('be_continuous', seconds)
    for task in value['tasks']:
        task['client']['use_cuda_graph'] = False
    if scenario == 'burstgpt':
        trace = json.loads((HERE / 'arrivals-burstgpt.json').read_text())
        offsets = trace['offsets_ns']
        if len(offsets) != 6000 or offsets[0] != 0 or offsets[-1] != 59_990_000_000:
            raise ValueError('unexpected frozen public trace')
        if any(type(t) is not int or t < 0 for t in offsets) or offsets != sorted(offsets):
            raise ValueError('invalid trace offsets')
        value['tasks'][0]['load'] = {'type': 'trace_fifo',
                                    'offsets_ns': [t for t in offsets if t < seconds * NS]}
    if isolated:
        value['tasks'] = value['tasks'][:1]
    return value


def records(log, prefix):
    return [json.loads(line[len(prefix) + 1:]) for line in log.splitlines()
            if line.startswith(prefix + ' ')]


def one(log, prefix):
    values = records(log, prefix)
    if len(values) != 1:
        raise ValueError(f'expected exactly one {prefix} record')
    return values[0]


def measurement(log, cfg, slo_ns):
    """Recompute arrival/coverage/goodput from each real request, not summaries."""
    decoder, reports = json.JSONDecoder(), []
    for match in re.finditer(r'(?m)^\{', log):
        try:
            item, _ = decoder.raw_decode(log[match.start():])
        except ValueError:
            continue
        if isinstance(item, dict) and 'benchmarkTime(s)' in item:
            reports.append(item)
    if len(reports) != 1 or reports[0].get('benchmarkTime(s)') != cfg['time']:
        raise ValueError('missing unique source-native DISB report')
    report = reports[0]
    begin = gp.integer(report.get('loadStudyBeginNs'), 'begin', 1)
    end = gp.integer(report.get('loadStudyEndNs'), 'end', 1)
    if end - begin != cfg['time'] * NS or report.get('loadStudyClock') != 'steady_clock':
        raise ValueError('invalid common measurement window')
    names = {task['id'] for task in cfg['tasks']}
    sources = {}
    for label, rows, key in (
            ('loads', records(log, 'GPREEMPT_LOAD_STUDY'), 'task'),
            ('checks', records(log, 'GPREEMPT_VALIDATION'), 'task'),
            ('results', report.get('results', []), 'clientName')):
        if len(rows) != len(names) or {row.get(key) for row in rows} != names:
            raise ValueError(f'missing or duplicate {label}')
        sources[label] = {row[key]: row for row in rows}
    metrics = {}
    for task in cfg['tasks']:
        name, arrival = task['id'], task['load']
        row, check = sources['loads'][name], sources['checks'][name]
        interval = NS // arrival['frequency'] if arrival['type'] == 'periodic_fifo' else 0
        offsets = (list(range(0, end - begin, interval)) if interval else
                   arrival.get('offsets_ns'))
        offered = len(offsets) if offsets is not None else None
        mode = arrival['type'] if offsets is not None else 'continuous_closed_loop'
        expected = {'clock': 'steady_clock', 'mode': mode, 'phase_ns': 0,
                    'begin_ns': begin, 'end_ns': end, 'interval_ns': interval,
                    'offered': offered,
                    'request_fields': 'id,scheduled_ns,started_ns,verified_ready_ns'}
        if any(type(row.get(k)) is not type(v) or row[k] != v for k, v in expected.items()):
            raise ValueError(f'{name}: recorded arrivals differ from configured FIFO')
        count = gp.integer(row.get('started'), 'started')
        requests = row.get('requests')
        if not isinstance(requests, list) or len(requests) != count or (offered is not None and count > offered):
            raise ValueError('request count exceeds recorded/expected arrivals')
        previous, responses, inside, slo_met = begin, [], 0, 0
        for index, request in enumerate(requests):
            if not isinstance(request, list) or len(request) != 4:
                raise ValueError('malformed request timestamps')
            rid, scheduled, started, finish = [gp.integer(v, 'request timestamp/id') for v in request]
            expected_arrival = started if offsets is None else begin + offsets[index]
            if (rid != index or scheduled != expected_arrival or not begin <= scheduled <= started < end
                    or not previous <= started < finish):
                raise ValueError('FIFO order, arrival, positive work, or cutoff violated')
            previous = finish
            response = finish - scheduled
            responses.append(response)
            inside += finish < end
            slo_met += finish < end and slo_ns is not None and response <= slo_ns
        if count - inside > 1:
            raise ValueError('serial worker has multiple completions after cutoff')
        analyzers = [a for a in sources['results'][name].get('analyzers', []) if a.get('type') == 'basic']
        if len(analyzers) != 1:
            raise ValueError('missing original service analyzer')
        analyzer = analyzers[0]
        samples = analyzer.get('requestLatencyNs', [])
        if (gp.integer(analyzer.get('completedRequests'), 'service count') != count
                or len(samples) != count or any(type(x) is not int or x <= 0 for x in samples)
                or analyzer.get('latencyDefinition') != 'sum_of_original_six_recorded_stages'):
            raise ValueError('original service sample count disagrees with verified requests')
        rate = analyzer.get('avgThroughput(req/s)')
        error = check.get('max_absolute_error')
        if (not isinstance(rate, (int, float)) or not math.isfinite(rate) or abs(rate - count / cfg['time']) > 1e-6
                or gp.integer(check.get('timed_checked'), 'timed checks') != count
                or gp.integer(check.get('checked'), 'all checks') != count + 110
                or check.get('atol') != 1e-6 or check.get('rtol') != 1e-4
                or not isinstance(error, (int, float)) or not math.isfinite(error) or error < 0):
            raise ValueError('common full-output numerical checks or reported rate disagree')
        metrics[name] = {
            'offered': offered, 'started': count, 'completed_in_window': inside,
            'completed_after_window': count - inside, 'started_unfinished': 0,
            'never_started': None if offered is None else offered - count,
            'completion_coverage': None if offered is None else inside / offered,
            'goodput_rps': inside / cfg['time'],
            'response_p99_ns': sorted(responses)[math.ceil(len(responses) * .99) - 1] if responses else None,
            'conditional_p99': offered is not None and count < offered,
            'slo_ns': slo_ns, 'slo_met': slo_met if offered is not None and slo_ns is not None else None,
            'slo_attainment': slo_met / offered if offered is not None and slo_ns is not None else None,
            'numerics': check}
    return {'metrics': metrics, 'report': report, 'loads': sources['loads']}


def engagement(log, arm, measured, profile=None):
    if arm == 'fixed':
        result = base.check_engagement('original_gpreempt', log, '', 'host_mapped')
        result['per_request_coverage'] = policy_action_coverage(
            'original_gpreempt', result,
            {task: {'started_requests': values['started']} for task, values in measured['metrics'].items()})
        return result
    base.check_engagement('native', log, '')  # No GPreempt flags or driver BPF in these arms.
    setup = one(log, 'HUMMINGBIRD_SETUP')
    if setup['mode'] != arm or setup['graph'] is not False:
        raise ValueError('wrong actual mode/graph')
    names = set(measured['metrics'])
    cleanups = records(log, 'HUMMINGBIRD_CLEANUP')
    if len(cleanups) != len(names) or {r['task'] for r in cleanups} != names or any(r['complete'] is not True for r in cleanups):
        raise ValueError('missing owned-client cleanup')
    contexts = records(log, 'HUMMINGBIRD_CONTEXT')
    if arm == 'native':
        priorities = [dict(re.findall(r'(\w+)=([^\s]+)', line)) for line in log.splitlines()
                      if line.startswith('GPREEMPT_LOAD_PRIORITY ')]
        if len(priorities) != len(names) or {r['task'] for r in priorities} != names or contexts:
            raise ValueError('native context/priority evidence missing')
        for row in priorities:
            least, greatest, actual = (int(row[k]) for k in ('least', 'greatest', 'actual'))
            if greatest >= least or actual != (greatest if int(row['role']) == 0 else least):
                raise ValueError('native stream priority mismatch')
    elif (len(contexts) != 2 or {r['role'] for r in contexts} != {0, 1}
          or len({(r['hclient'], r['hobject']) for r in contexts}) != 2
          or any(r['timeslice_us'] != 1_000_000 or r['stream_priority'] != 0
                 or r['owned_query_ok'] is not True or r['timeslice_set_ok'] is not True for r in contexts)):
        raise ValueError('both independent equal-timeslice contexts must be observed')
    result = {'setup': setup, 'contexts': contexts, 'cleanups': cleanups}
    if arm.startswith('idle_'):
        executor, hp = one(log, 'HUMMINGBIRD_EXECUTOR'), one(log, 'HUMMINGBIRD_HP_EVENTS')
        lc_count = measured['metrics'][base.TASKS[0]]['started']
        be_count = measured['metrics'][base.TASKS[1]]['started']
        if (executor['requests_accepted'] != be_count or executor['requests_completed'] != be_count
                or hp['hp_enqueues'] != lc_count or hp['hp_completions'] != lc_count
                or executor['decisions'] <= 0 or executor['max_lp_inflight'] != 1
                or executor['completion_fence'] != 'event-query-before-next-launch'
                or executor['split_launches'] + executor['whole_launches'] <= 0
                or executor['small_launches'] + executor['large_launches'] != executor['split_launches'] + executor['whole_launches']
                or executor['jit_decisions'] != (executor['decisions'] if arm == 'idle_bpf' else 0)):
            raise ValueError('real scheduling, completion, or JIT engagement mismatch')
        if profile is None:
            raise ValueError('idle engagement needs the actual frozen profile')
        ctas = sum(math.prod(k['grid']) for k in profile['kernels'] if k['name'] != 'nop')
        nops = sum(k['name'] == 'nop' for k in profile['kernels'])
        if executor['ctas_submitted'] != ctas * be_count or executor['nop_copies'] != nops * be_count:
            raise ValueError('recorded CTA/copy count differs from complete real-model execution')
        if executor['input_small_launches'] + executor['output_small_launches'] != executor['small_launches']:
            raise ValueError('small-pattern launch counts disagree')
        for pattern in ('input', 'output'):
            enabled = profile[f'small_{pattern}_enabled']
            if (setup[f'small_{pattern}_enabled'] is not enabled
                    or hp[f'{pattern}_bubbles'] != (lc_count if enabled else 0)
                    or (not enabled and executor[f'{pattern}_small_launches'])):
                raise ValueError('actual copy-pattern observation differs from frozen eligibility')
        result.update(executor=executor, hp_events=hp)
    elif records(log, 'HUMMINGBIRD_EXECUTOR') or records(log, 'HUMMINGBIRD_HP_EVENTS'):
        raise ValueError('unsplit control unexpectedly used idle scheduling')
    return result


def run_cell(directory, cfg, arm, profile, slo_ns, timeout):
    directory.mkdir(parents=True, exist_ok=False)
    config_path = directory / 'config.json'
    safety.atomic_write_json(config_path, cfg)
    if arm == 'fixed':
        command = [str(HERE / 'build/fixed_client'), str(config_path), 'true', '--flag-transport', 'host_mapped']
    else:
        command = [str(HERE / 'build/hummingbird_client'), str(config_path), '--mode', arm]
        if arm.startswith('idle_'):
            command += ['--profile', str(profile), '--split-cubin', str(HERE / 'build/resnet152-split/mod.cubin')]
            if arm == 'idle_bpf':
                command += ['--bpf-program', str(HERE / 'build/idle_policy.bin')]
    result = execute(directory, command, timeout)
    try:
        log = (directory / 'client.log').read_text()
        measured = measurement(log, cfg, slo_ns)
        result['engagement'] = engagement(log, arm, measured,
                                          json.loads(profile.read_text()) if arm.startswith('idle_') else None)
        result.update(arm=arm, metrics=measured['metrics'])
        safety.atomic_write_json(directory / 'request-report.json', measured['report'])
        safety.atomic_write_json(directory / 'arrival-report.json', measured['loads'])
        result['status'] = 'passed'
    except BaseException as error:
        result.update(status='failed', validation_error=f'{type(error).__name__}: {error}')
        raise
    finally:
        safety.atomic_write_json(directory / 'result.json', result)
    return result


def execute(directory, command, timeout):
    """Shared GPU safety/telemetry and owned-process cleanup; caller holds leases."""
    before = process = telemetry = None
    streams = []
    env = gp.environment('original_gpreempt', Path('/sys/fs/bpf/unused-hummingbird'))
    result = {'status': 'failed', 'command': command, 'environment': env, 'timeout_seconds': timeout}
    binaries = [HERE / 'build' / name for name in ('hummingbird_client', 'fixed_client', 'hummingbird_profile',
                                                  'idle_policy.bin', 'resnet152-split/mod.cubin')]
    binaries += [gp.BUILD / name for name in ('libexecutor.so', 'libworkloads.so', 'libgpreempt.so', 'block.cubin')]
    def inventory():
        return {str(path): {'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
                for path in binaries}
    try:
        result['runtime_before'] = inventory()
        before = safety.safety_snapshot()
        safety.validate_pre_server_safety(before)
        if before['gpu']['driver'] != '575.57.08':
            raise RuntimeError('the prepared 575 driver is required; this runner never changes it')
        result['safety_before'] = before
        telemetry, stream, telemetry_path = safety.start_gpu_telemetry(directory)
        streams.append(stream)
        client_log = (directory / 'client.log').open('x')
        streams.append(client_log)
        start = time.monotonic()
        process = subprocess.Popen(command, stdout=client_log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        while process.poll() is None:
            if time.monotonic() - start > timeout:
                raise TimeoutError('owned client exceeded its bound')
            time.sleep(.2)
        result.update(returncode=process.returncode, wall_seconds=time.monotonic() - start)
        if process.returncode:
            raise RuntimeError(f'client exited {process.returncode}')
        result['status'] = 'executed'
    except BaseException as error:
        result['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        errors = []
        for owned in (process, telemetry):
            try:
                base.stop_owned(owned)
            except BaseException as error:
                errors.append(str(error))
        for stream in streams:
            stream.close()
        try:
            result['runtime_after'] = inventory()
            if result.get('runtime_before') != result['runtime_after']:
                errors.append('runtime binaries changed during the cell')
            if before is not None:
                result['safety_after'] = safety.wait_for_post_server_safety(before)
            if telemetry is not None:
                result['telemetry'] = safety.validate_gpu_telemetry(telemetry_path, allow_fixed_power_cap=True)
        except BaseException as error:
            errors.append(str(error))
        if errors:
            result.update(status='failed', cleanup_errors=errors)
        safety.atomic_write_json(directory / 'execution.json', result)
        if errors:
            raise RuntimeError('; '.join(errors))
    return result


def orders(mode):
    rng = random.Random(SEED)
    result = []
    for block in range(5 if mode in ('full', 'small-calibration') else 1):
        scenarios = list(SCENARIOS) if mode in ('full', 'preflight') else ['periodic']
        rng.shuffle(scenarios)
        for scenario in scenarios:
            arms = list(('none', 'input', 'output', 'both') if mode == 'small-calibration' else ARMS)
            rng.shuffle(arms)
            result.extend({'block': block, 'scenario': scenario, 'arm': arm} for arm in arms)
    return result


def choose_small_patterns(results):
    groups = {(r['block'], r['variant']): r for r in results}
    expected = {(b, v) for b in range(5) for v in ('none', 'input', 'output', 'both')}
    if len(results) != 20 or set(groups) != expected:
        raise ValueError('small-pattern decision needs all five complete four-way blocks')
    passed, estimates = [], {}
    for variant in ('input', 'output', 'both'):
        ratios, engaged, coverage = [], True, True
        for block in range(5):
            candidate, reference = groups[block, variant], groups[block, 'none']
            for row in (candidate, reference):
                coverage &= row['metrics'][base.TASKS[0]]['completion_coverage'] == 1
            above = candidate['metrics'][base.TASKS[0]]['response_p99_ns']
            below = reference['metrics'][base.TASKS[0]]['response_p99_ns']
            ratios.append(above / below if above is not None and below else None)
            counters = candidate['engagement']['executor']
            for pattern in ('input', 'output') if variant == 'both' else (variant,):
                engaged &= counters[f'{pattern}_small_launches'] > 0
        estimate = (estimate_ratios(ratios, draws=10000) if None not in ratios else
                    {'geometric_ratio': None, 'paired_block_bootstrap_ci95': None,
                     'unavailable_reason': 'no completed foreground request in a paired cell'})
        ok = coverage and engaged and estimate['paired_block_bootstrap_ci95'] is not None and estimate['paired_block_bootstrap_ci95'][1] <= 1.01
        estimates[variant] = {**estimate, 'full_coverage': coverage, 'actual_pattern_launches': engaged, 'eligible': ok}
        if ok:
            passed.append(variant)
    selected = ('both' if {'input', 'output', 'both'} <= set(passed) else
                next((name for name in ('input', 'output') if name in passed), 'none'))
    return selected, estimates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('profile', 'calibrate', 'small-calibration', 'preflight', 'full'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--slo-ns', type=int)
    parser.add_argument('--cell-timeout', type=int, default=240)
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error('actual runs use the same root privilege for all arms')
    if args.mode not in ('profile', 'calibrate') and (args.profile is None or not args.slo_ns or args.slo_ns < 0):
        parser.error('comparison needs the frozen --profile and positive isolated --slo-ns')
    if args.cell_timeout < 90:
        parser.error('client timeout must allow setup and drain')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    def interrupted(signum, _frame):
        raise InterruptedError(f'signal {signum}; cleaning owned processes')
    signal.signal(signal.SIGTERM, interrupted)
    lease = base.Leases()
    try:
        if args.mode == 'profile':
            execute(output, [str(HERE / 'build/hummingbird_profile'), '--split-cubin',
                            str(HERE / 'build/resnet152-split/mod.cubin'), '--output', str(output / 'profile.json')],
                    args.cell_timeout)
            print(f'PROFILE {output / "profile.json"}', flush=True)
            return
        if args.mode == 'calibrate':
            result = run_cell(output / 'isolated-lc', config('periodic', 60, True), 'native', None, None, args.cell_timeout)
            lc = result['metrics'][base.TASKS[0]]
            if lc['completion_coverage'] != 1:
                raise RuntimeError('isolated LC did not complete every offered request')
            print(f'ISOLATED_SLO_NS {lc["response_p99_ns"]}', flush=True)
            return
        profile = json.loads(args.profile.read_text())
        profiles = {}
        variants = ('none', 'input', 'output', 'both') if args.mode == 'small-calibration' else ('frozen',)
        for variant in variants:
            value = copy.deepcopy(profile)
            if variant != 'frozen':
                value.update(small_input_enabled=variant in ('input', 'both'),
                             small_output_enabled=variant in ('output', 'both'))
            profiles[variant] = output / f'profile-{variant}.json'
            safety.atomic_write_json(profiles[variant], value)
        sequence = orders(args.mode)
        safety.atomic_write_json(output / 'run-order.json', {'mode': args.mode, 'seed': SEED, 'orders': sequence,
                                                          'slo_ns': args.slo_ns, 'source_profile': str(args.profile.resolve())})
        safety.atomic_write_json(output / 'model-assets.json', base.model_assets())
        results = []
        for index, spec in enumerate(sequence):
            variant = spec['arm'] if args.mode == 'small-calibration' else 'frozen'
            arm = 'idle_c' if args.mode == 'small-calibration' else spec['arm']
            seconds = 60 if args.mode == 'full' else 10
            print(f'START {index + 1}/{len(sequence)} {spec}', flush=True)
            result = run_cell(output / f'block-{spec["block"]:02d}' / spec['scenario'] / spec['arm'],
                              config(spec['scenario'], seconds), arm, profiles[variant], args.slo_ns, args.cell_timeout)
            result.update(block=spec['block'], scenario=spec['scenario'], variant=variant)
            results.append(result)
            safety.atomic_write_json(output / 'completed-cells.json', results)
            print(f'PASS LC_p99_ns={result["metrics"][base.TASKS[0]]["response_p99_ns"]} '
                  f'BE_rps={result["metrics"][base.TASKS[1]]["goodput_rps"]}', flush=True)
            if index + 1 < len(sequence):
                time.sleep(5)
        if args.mode == 'small-calibration':
            selected, estimates = choose_small_patterns(results)
            chosen = json.loads(profiles[selected].read_text())
            safety.atomic_write_json(output / 'small-pattern-results.json', {'selected': selected, 'paired': estimates})
            safety.atomic_write_json(output / 'profile-selected.json', chosen)
            print(f'SMALL_PATTERNS {selected}', flush=True)
    finally:
        lease.close()


if __name__ == '__main__':
    main()
