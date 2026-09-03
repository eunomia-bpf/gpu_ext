#!/usr/bin/env python3
"""Read-only Hummingbird raw audit; stdout JSON, no CUDA or experiment changes.

Request/arrival arithmetic and the randomized matrix are independently decoded.
Existing read-only engagement and safety validators are reused, not represented
as a second implementation. Synthetic unit fixtures are not experiment results.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import re
import statistics

import run_study as runtime
from analyze_three_way import estimate_ratios

HERE = Path(__file__).resolve().parent
ARMS = ('native', 'fixed', 'timeslice_control', 'idle_c', 'idle_bpf')
SCENARIOS = ('periodic', 'burstgpt')
VARIANTS = ('none', 'input', 'output', 'both')
LC, BE = 'vgg_rt', 'resnet152_be'
NS, SEED, DRAWS = 1_000_000_000, 20260903, 10000
PAIRS = (('idle_bpf', 'idle_c'), ('idle_c', 'fixed'), ('idle_bpf', 'fixed'),
         ('idle_c', 'timeslice_control'), ('idle_bpf', 'timeslice_control'),
         ('timeslice_control', 'fixed'), ('fixed', 'native'))


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f'invalid {name}')
    return value


def same(left, right):
    if type(left) is not type(right):
        # JSON does not distinguish an integral floating-point metric from an int.
        return (type(left) in (int, float) and type(right) in (int, float)
                and math.isfinite(left) and math.isfinite(right) and left == right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def expected_config(scenario, seconds):
    require(scenario in SCENARIOS and type(seconds) is int and seconds in (10, 60), 'unknown workload')
    arrival = {'type': 'periodic_fifo', 'frequency': 100, 'priority': 0}
    if scenario == 'burstgpt':
        offsets = read(HERE / 'arrivals-burstgpt.json')['offsets_ns']
        require(len(offsets) == 6000 and offsets[0] == 0 and offsets[-1] == 59_990_000_000
                and all(type(t) is int and t >= 0 for t in offsets)
                and offsets == sorted(offsets), 'invalid frozen BurstGPT arrivals')
        arrival = {'type': 'trace_fifo', 'offsets_ns': [t for t in offsets if t < seconds * NS]}
    return {'time': seconds, 'tasks': [
        {'id': name, 'load': arrival if role == 0 else {'type': 'continuous'},
         'client': {'name': name, 'model_name': model, 'priority': role,
                    'batch_size': 1, 'use_cuda_graph': False, 'preprocess_time': 200}}
        for role, name, model in ((0, LC, 'vgg'), (1, BE, 'resnet152'))]}


def expected_orders(mode):
    require(mode in ('full', 'small-calibration'), 'only full or small-calibration is audited')
    rng, sequence = random.Random(SEED), []
    for block in range(5):
        scenarios = list(SCENARIOS) if mode == 'full' else ['periodic']
        rng.shuffle(scenarios)
        for scenario in scenarios:
            arms = list(ARMS if mode == 'full' else VARIANTS)
            rng.shuffle(arms)
            sequence.extend({'block': block, 'scenario': scenario, 'arm': arm} for arm in arms)
    return sequence


def parse_client(log, cfg, slo_ns):
    """Decode every accepted request, including the verified tail after cutoff."""
    integer(slo_ns, 'frozen SLO', 1)
    seconds = integer(cfg['time'], 'duration', 1)
    decoder, reports = json.JSONDecoder(), []
    for match in re.finditer(r'(?m)^\{', log):
        try:
            value, _ = decoder.raw_decode(log[match.start():])
        except ValueError:
            continue
        if isinstance(value, dict) and 'benchmarkTime(s)' in value:
            reports.append(value)
    require(len(reports) == 1, 'missing or duplicate DISB report')
    report = reports[0]
    begin = integer(report.get('loadStudyBeginNs'), 'window begin', 1)
    end = integer(report.get('loadStudyEndNs'), 'window end', 1)
    require(report.get('benchmarkTime(s)') == seconds and end - begin == seconds * NS
            and report.get('loadStudyClock') == 'steady_clock', 'measurement window mismatch')
    names = {task['id'] for task in cfg['tasks']}
    sources = {}
    for kind, prefix, key in (('loads', 'GPREEMPT_LOAD_STUDY', 'task'),
                              ('checks', 'GPREEMPT_VALIDATION', 'task'),
                              ('results', None, 'clientName')):
        rows = (report.get('results', []) if prefix is None else
                [json.loads(line[len(prefix) + 1:]) for line in log.splitlines() if line.startswith(prefix + ' ')])
        require(len(rows) == len(names) and {r.get(key) for r in rows} == names,
                f'missing or duplicate {kind}')
        sources[kind] = {row[key]: row for row in rows}
    metrics, latest = {}, end
    for task in cfg['tasks']:
        name, load = task['id'], task['load']
        row, check = sources['loads'][name], sources['checks'][name]
        interval = NS // load['frequency'] if load['type'] == 'periodic_fifo' else 0
        offsets = list(range(0, end - begin, interval)) if interval else load.get('offsets_ns')
        offered = None if offsets is None else len(offsets)
        expected = {'clock': 'steady_clock', 'mode': load['type'] if offsets is not None else 'continuous_closed_loop',
                    'begin_ns': begin, 'end_ns': end, 'phase_ns': 0, 'interval_ns': interval,
                    'offered': offered, 'request_fields': 'id,scheduled_ns,started_ns,verified_ready_ns'}
        require(all(type(row.get(k)) is type(v) and row[k] == v for k, v in expected.items()),
                f'{name}: actual arrivals differ from config')
        count = integer(row.get('started'), 'started')
        requests = row.get('requests')
        require(isinstance(requests, list) and len(requests) == count
                and (offered is None or count <= offered), 'started count mismatch')
        previous, responses, inside, met = begin, [], 0, 0
        for index, request in enumerate(requests):
            require(isinstance(request, list) and len(request) == 4, 'invalid request record')
            rid, scheduled, started, verified = [integer(x, 'request timestamp/id') for x in request]
            arrival = started if offsets is None else begin + offsets[index]
            require(rid == index and scheduled == arrival and begin <= scheduled <= started < end
                    and previous <= started < verified, 'FIFO prefix, arrival, cutoff or completion mismatch')
            previous = verified
            response = verified - scheduled
            responses.append(response)
            inside += verified < end
            met += verified < end and response <= slo_ns
        require(count - inside <= 1, 'serial worker has multiple late completions')
        latest = max(latest, previous)
        analyzers = [a for a in sources['results'][name].get('analyzers', []) if a.get('type') == 'basic']
        require(len(analyzers) == 1, 'missing original service samples')
        analyzer, error = analyzers[0], check.get('max_absolute_error')
        samples, rate = analyzer.get('requestLatencyNs'), analyzer.get('avgThroughput(req/s)')
        require(integer(analyzer.get('completedRequests'), 'service count') == count
                and isinstance(samples, list) and len(samples) == count
                and all(type(x) is int and x > 0 for x in samples)
                and analyzer.get('latencyDefinition') == 'sum_of_original_six_recorded_stages',
                'service samples disagree with verified requests')
        require(type(rate) in (int, float) and math.isfinite(rate) and abs(rate - count / seconds) <= 1e-6
                and integer(check.get('timed_checked'), 'timed numerics') == count
                and integer(check.get('checked'), 'all numerics') == count + 110
                and check.get('atol') == 1e-6 and check.get('rtol') == 1e-4
                and type(error) in (int, float) and math.isfinite(error) and error >= 0,
                'numerical validation or original rate mismatch')
        metrics[name] = {
            'offered': offered, 'started': count, 'completed_in_window': inside,
            'completed_after_window': count - inside, 'started_unfinished': 0,
            'never_started': None if offered is None else offered - count,
            'completion_coverage': None if offered is None else inside / offered,
            'goodput_rps': inside / seconds,
            'response_p99_ns': sorted(responses)[math.ceil(.99 * count) - 1] if count else None,
            'conditional_p99': offered is not None and count < offered, 'slo_ns': slo_ns,
            'slo_met': met if offered is not None else None,
            'slo_attainment': met / offered if offered is not None else None, 'numerics': check}
    return {'metrics': metrics, 'report': report, 'loads': sources['loads'],
            'begin_ns': begin, 'end_ns': end, 'latest_verified_ns': latest}


def audit_cell(directory, spec, plan, campaign):
    result, execution, cfg = (read(directory / name) for name in ('result.json', 'execution.json', 'config.json'))
    small = plan['mode'] == 'small-calibration'
    arm, variant = ('idle_c', spec['arm']) if small else (spec['arm'], 'frozen')
    seconds, profile_path = (10 if small else 60), campaign / f'profile-{variant}.json'
    # Config field types are frozen too: bool must not stand in for integer role.
    expected = expected_config(spec['scenario'], seconds)
    require(json.dumps(cfg, sort_keys=True) == json.dumps(expected, sort_keys=True), 'config drift')
    require(result.get('status') == 'passed' and result.get('returncode') == 0 and result.get('arm') == arm
            and not any(result.get(k) for k in ('error', 'validation_error', 'cleanup_errors')), 'cell failed or wrong arm')
    require(execution.get('status') == 'executed'
            and all(k == 'status' or same(value, result.get(k)) for k, value in execution.items()),
            'execution record disagrees with result')
    command = [str(HERE / 'build' / ('fixed_client' if arm == 'fixed' else 'hummingbird_client')),
               str(directory / 'config.json')]
    if arm == 'fixed':
        command += ['true', '--flag-transport', 'host_mapped']
    else:
        command += ['--mode', arm]
        if arm.startswith('idle_'):
            command += ['--profile', str(profile_path), '--split-cubin', str(HERE / 'build/resnet152-split/mod.cubin')]
            if arm == 'idle_bpf':
                command += ['--bpf-program', str(HERE / 'build/idle_policy.bin')]
    require(result['command'] == command, 'actual client command differs')
    require(result['environment'] == runtime.gp.environment('original_gpreempt', Path('/sys/fs/bpf/unused-hummingbird')),
            'actual runtime environment differs')
    log = (directory / 'client.log').read_text()
    measured = parse_client(log, cfg, plan['slo_ns'])
    require(same(measured['metrics'], result['metrics']), 'raw metrics differ from cell result')
    require(same(measured['report'], read(directory / 'request-report.json'))
            and same(measured['loads'], read(directory / 'arrival-report.json')), 'saved raw reports differ from log')
    observed = runtime.engagement(log, arm, measured, read(profile_path) if arm.startswith('idle_') else None)
    require(same(observed, result['engagement']), 'raw engagement differs from cell result')
    if arm.startswith('idle_'):
        setup, executor = observed['setup'], observed['executor']
        require(setup['profile_path'] == str(profile_path) and setup['split_cubin'] == str(HERE / 'build/resnet152-split/mod.cubin')
                and setup['bpf_program'] == (str(HERE / 'build/idle_policy.bin') if arm == 'idle_bpf' else '')
                and executor['configured_lp_inflight_bound'] == 1, 'actual idle profile or execution bound differs')
    inventory = result['runtime_before']
    require(inventory and same(inventory, result['runtime_after'])
            and all(integer(item['bytes'], 'binary bytes', 1) and integer(item['mtime_ns'], 'binary mtime', 1)
                    for item in inventory.values()), 'runtime inventory changed within cell')
    runtime.safety.validate_pre_server_safety(result['safety_before'])
    runtime.safety.validate_post_server_safety(result['safety_before'], result['safety_after'])
    require(result['safety_before']['gpu']['driver'] == '575.57.08', 'unexpected driver')
    require(same(runtime.safety.validate_gpu_telemetry(directory / 'gpu-telemetry.csv', allow_fixed_power_cap=True),
                 result['telemetry']), 'raw telemetry differs')
    return {**spec, 'actual_arm': arm, 'metrics': measured['metrics'], 'engagement': observed,
            'begin_ns': measured['begin_ns'], 'end_ns': measured['end_ns'],
            'latest_verified_ns': measured['latest_verified_ns'], 'runtime_inventory': inventory}


def ratio_estimate(numerators, denominators):
    ratios = [a / b if a is not None and b is not None and b > 0 else None
              for a, b in zip(numerators, denominators)]
    if ratios and all(value is not None and math.isfinite(value) and value > 0 for value in ratios):
        return estimate_ratios(ratios, draws=DRAWS)
    return {'block_ratios': ratios, 'numerators': numerators, 'denominators': denominators,
            'geometric_ratio': None, 'paired_block_bootstrap_ci95': None,
            'unavailable_reason': 'missing/nonpositive value; all paired cells retained, no geometric ratio defined'}


def difference_estimate(differences):
    result = {'block_differences_pp': differences, 'mean_difference_pp': statistics.mean(differences) if differences else None,
              'paired_block_bootstrap_ci95_pp': None}
    if len(differences) > 1:
        rng = random.Random(SEED)
        samples = sorted(sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences)
                         for _ in range(DRAWS))
        result['paired_block_bootstrap_ci95_pp'] = [samples[int(.025 * DRAWS)], samples[int(.975 * DRAWS)]]
    return result


def paired_comparison(pairs, formal):
    lc = ratio_estimate([a['metrics'][LC]['response_p99_ns'] for a, _ in pairs],
                        [b['metrics'][LC]['response_p99_ns'] for _, b in pairs])
    be = ratio_estimate([a['metrics'][BE]['goodput_rps'] for a, _ in pairs],
                        [b['metrics'][BE]['goodput_rps'] for _, b in pairs])
    slo = difference_estimate([100 * (a['metrics'][LC]['slo_attainment'] - b['metrics'][LC]['slo_attainment'])
                               for a, b in pairs])
    coverage = bool(pairs) and all(row['metrics'][LC]['completion_coverage'] == 1
                                  and not row['metrics'][LC]['conditional_p99'] for pair in pairs for row in pair)
    lc_ci, be_ci, slo_ci = lc['paired_block_bootstrap_ci95'], be['paired_block_bootstrap_ci95'], slo['paired_block_bootstrap_ci95_pp']
    classification = 'inconclusive'
    if not formal or len(pairs) != 5:
        classification = 'incomplete_campaign'
    elif not coverage:
        classification = 'incomplete_lc_coverage'
    elif lc_ci is None or be_ci is None or slo_ci is None:
        classification = 'undefined_ratio'
    elif lc_ci[1] <= 1.01 and slo_ci[0] >= -1 and be_ci[0] > 1:
        classification = 'win'
    elif be_ci[0] > 1 and (lc_ci[0] > 1.01 or slo_ci[1] < -1):
        classification = 'throughput_protection_tradeoff'
    elif be_ci[1] < 1:
        classification = 'throughput_loss'
    return {'paired_blocks': len(pairs), 'lc_response_p99_ratio': lc, 'be_goodput_ratio': be,
            'slo_attainment_difference_pp': slo, 'full_lc_window_coverage': coverage,
            'classification': classification, 'equivalence_claimed': False}


def small_selection(cells, complete):
    by_key = {(c['block'], c['arm']): c for c in cells}
    if not complete:
        return {'selected': None, 'classification': 'incomplete_campaign'}
    estimates = {}
    for variant in VARIANTS[1:]:
        pairs = [(by_key[b, variant], by_key[b, 'none']) for b in range(5)]
        estimate = ratio_estimate([a['metrics'][LC]['response_p99_ns'] for a, _ in pairs],
                                  [b['metrics'][LC]['response_p99_ns'] for _, b in pairs])
        coverage = all(c['metrics'][LC]['completion_coverage'] == 1 for pair in pairs for c in pair)
        engaged = all(a['engagement']['executor'][f'{pattern}_small_launches'] > 0
                      for a, _ in pairs for pattern in (('input', 'output') if variant == 'both' else (variant,)))
        ci = estimate['paired_block_bootstrap_ci95']
        estimates[variant] = {**estimate, 'full_coverage': coverage, 'actual_pattern_launches': engaged,
                              'eligible': bool(coverage and engaged and ci is not None and ci[1] <= 1.01)}
    eligible = {name for name, value in estimates.items() if value['eligible']}
    selected = 'both' if set(VARIANTS[1:]) <= eligible else next((v for v in ('input', 'output') if v in eligible), 'none')
    return {'selected': selected, 'paired': estimates}


def analyze(campaign):
    campaign = campaign.resolve()
    plan = read(campaign / 'run-order.json')
    mode, sequence = plan['mode'], expected_orders(plan['mode'])
    integer(plan.get('slo_ns'), 'frozen SLO', 1)
    require(type(plan.get('seed')) is int and plan['seed'] == SEED
            and plan.get('orders') == sequence, 'randomized matrix differs from frozen plan')
    variants = VARIANTS if mode == 'small-calibration' else ('frozen',)
    profiles = {v: read(campaign / f'profile-{v}.json') for v in variants}
    if mode == 'small-calibration':
        common = {k: v for k, v in profiles['none'].items() if k not in ('small_input_enabled', 'small_output_enabled')}
        for variant, profile in profiles.items():
            require({k: v for k, v in profile.items() if k not in ('small_input_enabled', 'small_output_enabled')} == common
                    and profile['small_input_enabled'] is (variant in ('input', 'both'))
                    and profile['small_output_enabled'] is (variant in ('output', 'both')), 'calibration profile drift')
    directories = [campaign / f'block-{s["block"]:02d}' / s['scenario'] / s['arm'] for s in sequence]
    unexpected = sorted(str(p.relative_to(campaign)) for p in campaign.glob('block-*/*/*') if p.is_dir() and p not in directories)
    accepted, rejected, pending, inventory, latest = [], [], [], None, 0
    for spec, directory in zip(sequence, directories):
        if not (directory / 'result.json').exists():
            pending.append(spec)
            continue
        try:
            cell = audit_cell(directory, spec, plan, campaign)
            require(cell['begin_ns'] > latest, 'cells overlap or differ from recorded serial order')
            require(inventory is None or same(inventory, cell['runtime_inventory']), 'runtime inventory differs across cells')
            inventory, latest = cell.pop('runtime_inventory'), cell['latest_verified_ns']
            accepted.append(cell)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            rejected.append({**spec, 'reason': f'{type(error).__name__}: {error}'})
    complete = len(accepted) == len(sequence) and not (rejected or unexpected or pending)
    output = {'campaign': str(campaign), 'mode': mode, 'required_cells': len(sequence), 'accepted_cells': len(accepted),
              'complete': complete, 'formal_complete': complete and mode == 'full',
              'pending': pending, 'rejected': rejected, 'unexpected': unexpected, 'cells': accepted,
              'slo_ns': plan['slo_ns'], 'statistics': {'paired_blocks': 5, 'draws': DRAWS, 'seed': SEED,
                 'method': 'paired geometric ratios and arithmetic SLO percentage-point differences; whole-block percentile bootstrap'},
              'definitions': {'p99': 'nearest-rank arrival-to-verified over all started, including the late verified tail',
                  'conditional_p99': 'true when some offered LC requests never started; cannot establish protection',
                  'completion_coverage': 'window-completed / all offered, not final verified / offered',
                  'slo': 'response <= frozen SLO AND verified < end; denominator is all offered LC requests',
                  'goodput': 'verified < end divided by configured seconds; zero is retained',
                  'audit_scope': 'independent raw measurement/matrix/statistics; shared read-only engagement/safety checks',
                  'fidelity': 'paper-described scheduling component port; not author binary or full-system reproduction'},
              'runtime_inventory': inventory, 'equivalence_claimed': False}
    if mode == 'small-calibration':
        output['small_patterns'] = small_selection(accepted, complete)
        if complete:
            saved = read(campaign / 'small-pattern-results.json')
            require(same(saved, output['small_patterns']), 'saved small-pattern selection differs from independent raw calculation')
            require(same(read(campaign / 'profile-selected.json'), profiles[saved['selected']]), 'selected profile differs')
    else:
        output['scenarios'] = {}
        by_key = {(c['block'], c['scenario'], c['arm']): c for c in accepted}
        for scenario in SCENARIOS:
            blocks = [b for b in range(5) if all((b, scenario, arm) in by_key for arm in ARMS)]
            output['scenarios'][scenario] = {'complete_blocks': blocks, 'paired': {
                f'{above}/{below}': paired_comparison([(by_key[b, scenario, above], by_key[b, scenario, below]) for b in blocks], complete)
                for above, below in PAIRS}}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.campaign), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
