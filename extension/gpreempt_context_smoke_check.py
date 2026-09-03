#!/usr/bin/env python3
"""Correlate real owned contexts, BPF engagement, and completed GSP timeslices.

Pure offline analysis; does not admit/load/attach or claim full-workload results.
"""
import argparse
import json
from pathlib import Path


def demand(condition, message):
    if not condition:
        raise ValueError(message)


def rows(text):
    return [json.loads(line) for line in text.splitlines() if line.startswith('{')]


def stats(text, prefix):
    matched = [line for line in text.splitlines() if line.startswith(prefix)]
    demand(len(matched) == 1, f"expected one {prefix} record")
    return {key: value for key, value in
            (field.split('=', 1) for field in matched[0][len(prefix):].split())}


def analyze(client_text, rpc_text, mode, policy_text=None):
    client, rpc = rows(client_text), rows(rpc_text) if rpc_text is not None else []
    completed = [r for r in client if r.get('event') == 'gpreempt_context_smoke']
    demand(len(completed) == 1 and completed[0].get('passed') is True, "client incomplete or failed")
    demand(completed[0]['policy'] == mode, "wrong client policy")
    demand(completed[0]['validated_values'] == 2048 and completed[0]['negative_cases'] == 17,
           "missing numerical/negative cases")
    roles = [r for r in client if r.get('event') == 'role_context']
    demand(len(roles) == 2 and {r['role'] for r in roles} == {0, 1}, "missing two role contexts")
    demand(len({r['pid'] for r in roles}) == 1 and len({r['tid'] for r in roles}) == 2,
           "roles must share a process but have different creators")
    demand(len({r['cuda_context'] for r in roles}) == 2 and
           len({(r['hclient'], r['htsg']) for r in roles}) == 2, "aliased role identities")
    summary = [r for r in rpc if r.get('event') == 'gpreempt_rpc_observer_summary']
    events = [r for r in rpc if r.get('event') == 'gsp_timeslice_rpc']
    if rpc_text is not None:
        demand(len(summary) == 1, "observer did not finish")
        summary = summary[0]
        demand(summary['observed'] > 0 and summary['observed'] == summary['completed'] ==
               summary['received'] == len(events), "incomplete RPC observations")
        demand(summary['read_errors'] == summary['ring_drops'] == 0, "RPC observation lost events")
    evidence = []
    for role in sorted(roles, key=lambda r: r['role']):
        demand(role['cuda_context'] and role['hclient'] and role['htsg'] and 1 <= role['channels'] <= 64,
               "bad role identity")
        if rpc_text is None:
            evidence.append({'role': role['role'], 'hclient': role['hclient'], 'htsg': role['htsg'],
                             'expected_timeslice_us': 1000000 if role['role'] == 0 else 1,
                             'direct_rpc_observation': 'unavailable_notrace_restriction'})
            continue
        matches = [e for e in events if e['pid'] == role['pid'] and e['hclient'] == role['hclient'] and
                   e['hobject'] == role['htsg'] and e['completed_ns'] <= role['kernel_begin_ns']]
        demand(matches, "no completed firmware timeslice for queried role before kernel")
        for event in matches:
            demand(event['command'] == 0xa06c0103 and event['params_size'] == 8, "wrong RPC command/size")
            demand(event['source'] == 'kernel_open_rpc_completion', "not a real RPC completion observation")
            demand(event['input_valid'] == event['gsp_status_valid'] == 1 and
                   event['transport_status'] == event['gsp_status'] == 0, "firmware/transport/input failure")
        last = max(matches, key=lambda e: e['completed_ns'])
        expected = 1000000 if role['role'] == 0 else 1
        demand(last['timeslice_us'] == expected, "role timeslice was missing or overwritten before execution")
        evidence.append({'role': role['role'], 'hclient': role['hclient'], 'htsg': role['htsg'],
                         'timeslice_us': expected, 'rpc_count_before_kernel': len(matches),
                         'last_rpc_completed_ns': last['completed_ns']})
    bridge = stats(client_text, 'gpreempt_bridge_stats:')
    demand(int(bridge['errors']) == 0, "bridge errors")
    if mode == 'bpf':
        demand(policy_text is not None, "BPF policy log missing")
        policy = stats(policy_text, 'gpreempt_policy_stats:')
        for name in ('scope_enter', 'scope_leave', 'gr_init', 'timeslice_ok', 'alloc_captured', 'registered', 'destroy'):
            demand(int(policy[name]) == 2, f"expected exactly two {name}")
        for name in ('unknown_engine', 'setter_error', 'alloc_error', 'register_error',
                     'bind_shadow_mismatch', 'map_error', 'scope_error'):
            demand(int(policy[name]) == 0, f"policy error {name}")
        demand(int(policy['bind_shadow_match']) > 0, "no bind observation")
        demand(bridge['backend'] == 'ubpf-jit' and all(int(bridge[k]) == 2 for k in ('scopes', 'registered', 'ended')),
               "missing actual JIT/context bridge engagement")
        demand(all(int(bridge[k]) > 0 for k in ('hint', 'block', 'release')), "hint decisions not exercised")
    return {'passed': True, 'mode': mode, 'roles': evidence, 'validated_values': 2048,
            'negative_cases': 17, 'firmware_timeslice_status_observed': rpc_text is not None,
            'physical_quantum_measured': False, 'interleave_tested': False,
            'gdr_actuator_tested': False, 'performance_measured': False,
            'external_safety_and_lease_checks_required': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--client-log', type=Path, required=True)
    parser.add_argument('--rpc-log', type=Path)
    parser.add_argument('--context-only', action='store_true',
                        help='explicitly omit direct firmware-RPC evidence; never a hardware/RPC pass')
    parser.add_argument('--policy-log', type=Path)
    parser.add_argument('--mode', choices=('original', 'bpf'), required=True)
    args = parser.parse_args()
    if args.context_only == (args.rpc_log is not None):
        parser.error('provide exactly one of --rpc-log or --context-only')
    try:
        result = analyze(args.client_log.read_text(), args.rpc_log.read_text() if args.rpc_log else None, args.mode,
                         args.policy_log.read_text() if args.policy_log else None)
    except (KeyError, ValueError, OSError) as error:
        parser.exit(1, f"canary validation failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
