#!/usr/bin/env python3
"""Minimal paired launcher + analysis for the Hummingbird host-device BPF arms.

Reuses the existing build/hummingbird_client, the existing frozen
profile-frozen.json, and the existing 60 s periodic (100 Hz) per-block configs.
No new frontend, preflight, correctness/audit gate, or timeout is added.

Paired matrix: N blocks x 4 arms
  original_inline : --mode idle_c  + original inline split cubin   (control)
  inline_bpfhost  : --mode idle_bpf + original inline split cubin
  native_adapter  : --mode idle_bpf + native-callable adapter cubin
  bpf_device      : --mode idle_bpf + device-BPF cubin            (BPF arm)

All arms share the same tile budget / host-policy family / arrival trace / DNN;
only the device coordinate-mapping mechanism differs (and, between
original_inline and the idle_bpf arms, the host policy). The existing unsplit
native no-policy baseline stays historical and is not relabeled.

Execution:
  * blocks run in order (block 0 first); within each block the 4 arms are in a
    seeded order so the paired cells of one block stay adjacent.
  * the manifest is persisted after every cell; cells already done/reused are
    skipped on re-invocation (resume).
  * each cell's client argv is stored as a list and run with the working
    directory at the repo root (the known-successful cwd).

Analysis (after launch, or --analyze-only):
  * recompute, per cell, the HP (vgg) scheduled-arrival response p99 and the
    BE (resnet152) goodput from the real GP_LOAD_STUDY request records using
    run_study.measurement (pure log parse; no execution gate).
  * report paired ratios (geometric ratio + paired block-bootstrap CI) for
    bpf_device vs original_inline (mechanism effect) and bpf_device vs
    native_adapter (wrapper overhead), with block 0 excluded for sensitivity.
"""
from __future__ import annotations

import argparse
import json
import random
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
HB = HERE.parent
REPO_ROOT = HERE.parents[2]  # workloads/hummingbird/hostdev -> repo root
HP_TASK = 'vgg_rt'
BE_TASK = 'resnet152_be'

ARM_DEFS = {
    'original_inline': ('idle_c', 'original', False),
    'inline_bpfhost': ('idle_bpf', 'original', True),
    'native_adapter': ('idle_bpf', 'adapter', True),
    'bpf_device': ('idle_bpf', 'bpf', True),
}


def q(parts: list) -> str:
    return ' '.join(shlex.quote(str(p)) for p in parts)


def load_json(path: Path):
    with open(path, 'rb') as handle:
        return json.loads(handle.read())


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def measure_cell(log: Path, cfg: dict) -> dict:
    """Recompute HP p99 / BE goodput / numerics from the real request records."""
    import run_study
    text = log.read_text(errors='replace')
    measured = run_study.measurement(text, cfg, None)
    metrics = measured['metrics']
    return {
        'hp_p99_ns': metrics[HP_TASK]['response_p99_ns'],
        'hp_coverage': metrics[HP_TASK]['completion_coverage'],
        'be_goodput_rps': metrics[BE_TASK]['goodput_rps'],
        'be_max_abs_error': metrics[BE_TASK]['numerics']['max_absolute_error'],
        'be_split_launches': None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', default=str(HB / 'raw/idle-study-575-01'))
    p.add_argument('--profile', default=None)
    p.add_argument('--client', default=str(HB / 'build/hummingbird_client'))
    p.add_argument('--bpf-program', default=str(HB / 'build/idle_policy.bin'))
    p.add_argument('--original-cubin',
                   default=str(HB / 'build/resnet152-split/mod.cubin'))
    p.add_argument('--adapter-cubin',
                   default=str(HERE / 'build/resnet152-bpf/mod-native.cubin'))
    p.add_argument('--bpf-cubin',
                   default=str(HERE / 'build/resnet152-bpf/mod-bpf.cubin'))
    p.add_argument('--blocks', default='0,1,2,3,4')
    p.add_argument('--arms',
                   default='original_inline,inline_bpfhost,native_adapter,bpf_device')
    p.add_argument('--seed', type=int, default=20260908)
    p.add_argument('--out', default=None, help='run dir (default runs/<ts>)')
    p.add_argument('--reuse-block0-bpf', default=None,
                   help='log of the already-completed block0 bpf_device cell')
    p.add_argument('--cwd', default=str(REPO_ROOT))
    p.add_argument('--dry-run', action='store_true',
                   help='validate + print commands; do not run the GPU client')
    p.add_argument('--analyze-only', action='store_true',
                   help='no launch; load an existing --out manifest and analyze')
    p.add_argument('--no-sudo', action='store_true',
                   help='run without `sudo -n` (default uses sudo -n)')
    a = p.parse_args()

    study = Path(a.study)
    profile = Path(a.profile) if a.profile else study / 'profile-frozen.json'
    blocks = [int(b) for b in a.blocks.split(',') if b.strip() != '']
    cubins = {'original': a.original_cubin, 'adapter': a.adapter_cubin,
              'bpf': a.bpf_cubin}
    arms = [x for x in a.arms.split(',') if x]
    unknown = [x for x in arms if x not in ARM_DEFS]
    if unknown:
        raise SystemExit(f'unknown arm(s): {", ".join(unknown)}')
    for label, path in [('client', Path(a.client)), ('profile', profile),
                        ('bpf-program', Path(a.bpf_program)),
                        ('original-cubin', Path(a.original_cubin)),
                        ('adapter-cubin', Path(a.adapter_cubin)),
                        ('bpf-cubin', Path(a.bpf_cubin))]:
        if not path.exists():
            raise SystemExit(f'missing required input ({label}): {path}')
    for b in blocks:
        if not (study / f'block-{b:02d}' / 'periodic' / 'idle_bpf' / 'config.json').exists():
            raise SystemExit(f'missing config for block {b}')
    if not Path(a.cwd).is_dir():
        raise SystemExit(f'cwd not a directory: {a.cwd}')

    out = Path(a.out) if a.out else HERE / 'runs' / datetime.now().strftime('%Y%m%d.%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    prefix = [] if a.no_sudo else ['sudo', '-n']
    manifest_path = out / 'manifest.json'

    # Resume: if this out dir already has a manifest, load it (no overwrite).
    prior = {}
    if a.analyze_only:
        if not manifest_path.exists():
            raise SystemExit(f'--analyze-only needs an existing manifest: {manifest_path}')
        manifest = load_json(manifest_path)
        for record in manifest.get('cells', []):
            prior[(record['block'], record['arm'])] = record
        manifest['cells'] = []
    else:
        manifest = {
            'generated': datetime.now().isoformat(),
            'client': str(Path(a.client)), 'profile': str(profile),
            'bpf_program': str(a.bpf_program),
            'original_cubin': a.original_cubin, 'adapter_cubin': a.adapter_cubin,
            'bpf_cubin': a.bpf_cubin, 'cwd': str(Path(a.cwd)),
            'privilege': 'sudo -n' if not a.no_sudo else 'unprivileged',
            'seed': a.seed, 'gpu_measurement': not a.dry_run, 'cells': [],
        }
        # block-major order; within a block the arms are seeded. block 0 first.
        rng = random.Random(a.seed)
        cells = []
        for b in blocks:
            block_arms = list(arms)
            rng.shuffle(block_arms)
            cells.extend((b, name) for name in block_arms)

    if not a.analyze_only:
        for order, (b, name) in enumerate(cells):
            mode, cubin_key, needs_bpf = ARM_DEFS[name]
            cubin = cubins[cubin_key]
            cfg_path = study / f'block-{b:02d}' / 'periodic' / 'idle_bpf' / 'config.json'
            log = out / f'block-{b:02d}-{name}.log'
            record = {'order': order, 'block': b, 'arm': name, 'mode': mode,
                      'split_cubin': str(cubin), 'config': str(cfg_path),
                      'log': str(log), 'status': 'pending',
                      'bpf_program': str(a.bpf_program) if needs_bpf else None}
            argv = list(prefix) + [str(Path(a.client)), str(cfg_path), '--mode', mode,
                                   '--profile', str(profile),
                                   '--split-cubin', str(cubin)]
            if needs_bpf:
                argv += ['--bpf-program', str(a.bpf_program)]
            record['argv'] = argv
            record['command'] = q(argv)
            if (b, name) == (0, 'bpf_device') and a.reuse_block0_bpf:
                record['status'] = 'reused'
                record['log'] = str(a.reuse_block0_bpf)
                record['command'] = f'reused: {a.reuse_block0_bpf}'
            manifest['cells'].append(record)
        atomic_write(manifest_path, json.dumps(manifest, indent=2) + '\n')

    records = manifest['cells']
    if a.dry_run:
        for record in records:
            print(f"[{record['order']:02d}] {record['status']:7} "
                  f"block-{record['block']:02d} {record['arm']:<16} -> "
                  f"{record.get('command', record['log'])}")
        print(f'manifest -> {manifest_path} ({len(records)} cells)')
        return

    if not a.analyze_only:
        for record in records:
            if record['status'] in ('reused', 'done'):
                print(f"[{record['order']:02d}] skip {record['status']}: "
                      f"block-{record['block']:02d} {record['arm']}")
                continue
            label = f"block-{record['block']:02d}-{record['arm']}"
            print(f"[{record['order']:02d}] running {label} ...", flush=True)
            with open(record['log'], 'wb') as handle:
                proc = subprocess.run(record['argv'], stdout=handle,
                                      stderr=subprocess.STDOUT, cwd=a.cwd)
            record['exit'] = proc.returncode
            record['status'] = 'done' if proc.returncode == 0 else 'failed'
            print(f"[{record['order']:02d}] {label} exit={proc.returncode}")
            atomic_write(manifest_path, json.dumps(manifest, indent=2) + '\n')
        failed = [f"block-{r['block']:02d}-{r['arm']}" for r in records
                  if r['status'] == 'failed']
        if failed:
            print(f'WARNING: failed cells: {", ".join(failed)}', file=sys.stderr)

    # ------------------------------------------------------------- analysis
    import analyze_three_way
    analysis = {'cells': [], 'paired_ratios': {}, 'estimate': {}}
    hp: dict = {'hp_p99_ns': {}, 'be_goodput_rps': {}}
    for record in records:
        entry = {'block': record['block'], 'arm': record['arm'],
                 'status': record.get('status')}
        if record.get('status') in ('done', 'reused'):
            try:
                m = measure_cell(Path(record['log']),
                                 load_json(Path(record['config'])))
                entry.update(m)
                hp['hp_p99_ns'].setdefault(record['arm'], {})[record['block']] = m['hp_p99_ns']
                hp['be_goodput_rps'].setdefault(record['arm'], {})[record['block']] = m['be_goodput_rps']
            except Exception as exc:  # a failed/unfinished cell has no valid records
                entry['analysis_error'] = f'{type(exc).__name__}: {exc}'
        analysis['cells'].append(entry)

    def block_ratios(arm_num, arm_den, metric, exclude_block0):
        ratios = []
        for entry in analysis['cells']:
            if entry['arm'] != arm_num:
                continue
            block = entry['block']
            if exclude_block0 and block == 0:
                continue
            den = next((e[metric] for e in analysis['cells']
                        if e['arm'] == arm_den and e['block'] == block
                        and e.get(metric) is not None), None)
            num = entry.get(metric)
            if num and den:
                ratios.append(num / den)
        return ratios

    for pair, (num, den) in {
            'bpf_vs_original': ('bpf_device', 'original_inline'),
            'bpf_vs_adapter': ('bpf_device', 'native_adapter')}.items():
        analysis['paired_ratios'][pair] = {
            'hp_p99_ns_excl_block0':
                block_ratios(num, den, 'hp_p99_ns', True),
            'be_goodput_rps_excl_block0':
                block_ratios(num, den, 'be_goodput_rps', True),
        }
        for metric in ('hp_p99_ns_excl_block0', 'be_goodput_rps_excl_block0'):
            values = analysis['paired_ratios'][pair][metric]
            analysis['estimate'].setdefault(pair, {})[metric] = (
                analyze_three_way.estimate_ratios(values) if len(values) >= 2 else
                {'n': len(values), 'note': 'need >=2 paired blocks'})

    atomic_write(out / 'analysis.json', json.dumps(analysis, indent=2) + '\n')

    print('\narm                block  status  HP_p99_ns     BE_goodput  max_err')
    for entry in sorted(analysis['cells'], key=lambda x: (x['arm'], x['block'])):
        print(f"{entry['arm']:<18} {entry['block']:>5}  {entry['status']:<7}  "
              f"{str(entry.get('hp_p99_ns', '-')):>11}  "
              f"{(f\"{entry['be_goodput_rps']:.2f}\" if entry.get('be_goodput_rps') is not None else '-'):>11}  "
              f"{str(entry.get('be_max_abs_error', '-')):>7}")
    for pair, stats in analysis['estimate'].items():
        for metric, est in stats.items():
            if est.get('geometric_ratio') is not None:
                print(f"{pair} {metric}: geometric_ratio={est['geometric_ratio']:.4f} "
                      f"ci95=[{est['paired_block_bootstrap_ci95'][0]:.4f},"
                      f"{est['paired_block_bootstrap_ci95'][1]:.4f}] n={est.get('n', len(est.get('block_ratios', [])))}")
            else:
                print(f"{pair} {metric}: {est.get('note', 'insufficient paired blocks')}")
    print(f'\nanalysis -> {out / "analysis.json"}')


if __name__ == '__main__':
    main()
