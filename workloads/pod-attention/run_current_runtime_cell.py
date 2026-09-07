#!/usr/bin/env python3
"""Performance-only single-cell POD launcher: one unchanged phase-study cell.

One invocation owns exactly one fresh @--output cell plus, for pod_bpf, the
private loader and its unique segment. It records the exact commands and
environments, monotonic spawn/exit stamps, loader READY marker, child return
codes, raw client/loader logs, and the operator's own phase timestamps
(derived pre-Python and complete client wall durations). It performs no build,
no preflight, no inventories, no output correctness gates, no validate_* calls,
and no artificial timeouts, and never fabricates performance on error; failed
logs stay. Import and --help are CPU-only; root's invocation starts the GPU
work under the shared exclusive-GPU leases.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import bench

HERE = Path(__file__).resolve().parent
PYTHON = HERE.parent / 'moe-infinity/.venv/bin/python'
BRIDGE = HERE / 'build/libpod_launch_bridge.so'
PTX_ADAPTER = HERE / 'build/libpod_ptx_adapter.so'
LOADER_BINARY = HERE / 'build/pod-loader'
SELECTOR_OBJECT = HERE / 'build/selector.bpf.o'
READY_PREFIX = 'POD_LOADER_READY'
CLOSED_MARKER = 'POD_LOADER_CLOSED\n'


def extract_operator(report, result, phases, path):
    """Copy only the operator's own evidence; nothing is recomputed or gated."""
    timestamps = report.get('phase_timestamps')
    if not isinstance(timestamps, dict):
        raise RuntimeError('operator report exposes no phase_timestamps to extract')
    result['operator_timestamps'] = timestamps
    spawn, exit_ns = phases['client_spawn_ns'], phases['client_exit_ns']
    process_main = timestamps.get('process_main_ns')
    if all(type(value) is int for value in (spawn, exit_ns, process_main)):
        result['pre_python_main_ns'] = process_main - spawn
        result['client_wall_ns'] = exit_ns - spawn
    cells = report.get('cells')
    if isinstance(cells, list):
        result['operator_cells'] = [
            {'model': cell.get('model'), 'decode_batch': cell.get('decode_batch'),
             'warmups': cell.get('warmups'),
             'timed_samples': len(cell['samples']) if isinstance(cell.get('samples'), list) else None,
             'mean_cuda_ms': cell.get('mean_cuda_ms'),
             'mean_host_wall_ms': cell.get('mean_host_wall_ms')}
            for cell in cells if isinstance(cell, dict)]
    del path


def run_cell(base, directory, specification, command, target_env, launch_env,
             loader_command, loader_env, build, ptx, lease_paths):
    arm, block = specification['arm'], specification['block']
    shm, segment = specification['private_segment'], specification['segment']
    result = dict(
        status='failed', numeric_protocol=bench.NUMERIC_PROTOCOL, arm=arm, block=block,
        phase_study=True, bpftime_build=str(build), ptx=str(ptx),
        command=command, environment=target_env, launch_environment=launch_env,
        private_segment=shm, loader_command=loader_command, loader_environment=loader_env,
        operator_output=str(directory / 'operator.json'), timeout_seconds=None,
        lease_paths=[str(path) for path in lease_paths],
        phase_timestamps=dict(cell_start_ns=time.monotonic_ns(), loader_spawn_ns=None,
                              loader_ready_ns=None, loader_ready_marker=None,
                              client_spawn_ns=None, client_exit_ns=None,
                              cleanup_complete_ns=None))
    directory.mkdir(parents=True, exist_ok=False)
    base.safety.atomic_write_json(directory / 'execution.json', result)
    phases = result['phase_timestamps']
    identity = None
    client = loader = started = None
    streams, cleanup = [], []
    try:
        if segment is not None:
            if segment.exists() or segment.is_symlink():
                raise RuntimeError('private segment already exists; refusing loader start')
            log = (directory / 'loader.log').open('x')
            streams.append(log)
            phases['loader_spawn_ns'] = time.monotonic_ns()
            loader = subprocess.Popen(loader_command, stdin=subprocess.PIPE, stdout=log,
                stderr=subprocess.STDOUT, env=loader_env, start_new_session=True, cwd=HERE)
            # Readiness observation only: the existing READY line or the loader's
            # actual exit ends the loop; there is no deadline.
            while not [line for line in (directory / 'loader.log').read_text(errors='replace')
                       .splitlines() if line.startswith(READY_PREFIX)]:
                if identity is None:
                    try:
                        identity = base.segment_identity(segment)
                    except FileNotFoundError:
                        pass
                if loader.poll() is not None:
                    raise RuntimeError(f'private BPF loader exited {loader.returncode} '
                                       'before READY')
                time.sleep(0.1)
            if identity is None:
                identity = base.segment_identity(segment)
            phases['loader_ready_ns'] = time.monotonic_ns()
            phases['loader_ready_marker'] = next(
                line for line in (directory / 'loader.log').read_text(errors='replace')
                .splitlines() if line.startswith(READY_PREFIX))
        log = (directory / 'client.log').open('x')
        streams.append(log)
        started = time.monotonic()
        phases['client_spawn_ns'] = time.monotonic_ns()
        client = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
            env=launch_env, start_new_session=True, cwd=HERE)
        while client.poll() is None:
            if loader is not None and loader.poll() is not None:
                raise RuntimeError(f'private loader exited {loader.returncode} '
                                   'while CUDA client was alive')
            time.sleep(0.2)
        phases['client_exit_ns'] = time.monotonic_ns()
        result.update(returncode=client.returncode, process_wall_seconds=time.monotonic() - started)
        if client.returncode:
            raise RuntimeError(f'operator client exited {client.returncode}')
        report = json.loads((directory / 'operator.json').read_text())
        if not isinstance(report, dict):
            raise RuntimeError('operator report is not an object')
        result['operator_complete_flag'] = report.get('complete')
        extract_operator(report, result, phases, directory / 'operator.json')
        result['status'] = 'passed'
    except BaseException as error:
        result['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            base.shared.stop_owned(client)
        except BaseException as error:
            cleanup.append(str(error))
        client_stopped = client is None or not base.shared.group_members(client.pid)
        if not client_stopped:
            cleanup.append('CUDA client still alive; cannot safely close its policy loader')
        if loader is not None and client_stopped:
            try:
                # The open stdin owns the policy lifetime, not a guessed sleep.
                loader.stdin.close()
                loader.wait()
                closed = (directory / 'loader.log').read_text().count(CLOSED_MARKER) == 1
                if loader.returncode != 0 or not closed:
                    raise RuntimeError('private loader failed its orderly detach')
                result['loader_returncode'] = loader.returncode
            except BaseException as error:
                cleanup.append(str(error))
            try:
                base.shared.stop_owned(loader)
            except BaseException as error:
                cleanup.append(str(error))
        try:
            if segment is not None:
                if any(p is not None and base.shared.group_members(p.pid) for p in (client, loader)):
                    raise RuntimeError('owned processes survive; refusing private segment removal')
                if identity is None:
                    if segment.exists() or segment.is_symlink():
                        raise RuntimeError('unidentified private segment survived; refusing removal')
                    result['private_segment_removed'] = False
                else:
                    result['private_segment_removed'] = base.remove_owned_segment(segment, identity)
                if segment.exists() or segment.is_symlink():
                    raise RuntimeError('private loader segment survived cleanup')
        except BaseException as error:
            cleanup.append(str(error))
        for stream in streams:
            try:
                stream.close()
            except BaseException as error:
                cleanup.append(str(error))
        phases['cleanup_complete_ns'] = time.monotonic_ns()
        if cleanup:
            result.update(status='failed', cleanup_errors=cleanup)
        base.safety.atomic_write_json(directory / 'execution.json', result)
        if cleanup:
            result['error'] = '; '.join(cleanup)
            raise RuntimeError(result['error'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='fresh cell directory (must not exist)')
    parser.add_argument('--arm', choices=('pod_inline', 'pod_cuda', 'pod_bpf'), required=True)
    parser.add_argument('--block', type=int, default=1,
                        help='passed through to bench.py; bench keeps its own choices')
    parser.add_argument('--bpftime-build', type=Path, required=True,
                        help='explicit existing bpftime build; becomes POD_BPFTIME_BUILD')
    parser.add_argument('--ptx', type=Path, default=HERE / 'build/ptx-runtime-01')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite an existing cell')
    try:
        build = args.bpftime_build.resolve(strict=True)
        ptx = args.ptx.resolve(strict=True)
    except OSError as error:
        parser.error(f'--bpftime-build and --ptx must be existing paths: {error}')
    # The selected runtime must be set before run_study is imported; its own
    # environment() then supplies the agent/server and BPF PTX-pass directory.
    os.environ['POD_BPFTIME_BUILD'] = str(build)
    import run_study as base
    import run_phase_study
    if base.BPFTIME != build:
        parser.error('the selected bpftime build was not adopted by run_study')
    required = [PYTHON, HERE / 'bench.py']
    if args.arm != 'pod_inline':
        required += [BRIDGE, ptx / 'device']
    if args.arm == 'pod_bpf':
        required += [PTX_ADAPTER, LOADER_BINARY, SELECTOR_OBJECT, base.AGENT, base.SERVER,
                     ptx / 'exact-kernels.txt']
    for path in required:
        if path.is_dir():
            continue
        if not path.is_file():
            parser.error(f'missing existing runtime file: {path}')
    shm = f'pod_attention_{os.getpid()}_{time.monotonic_ns()}' if args.arm == 'pod_bpf' else None
    segment = Path('/dev/shm') / shm if shm else None
    target_env = base.environment(args.arm, ptx, shm)
    launch_env = dict(target_env)
    preload = launch_env.pop('LD_PRELOAD', None)
    loader_env = None
    if args.arm == 'pod_bpf':
        loader_env = base.environment(args.arm, ptx, shm, loader=True)
        if loader_env.get('LD_PRELOAD') != str(base.SERVER):
            parser.error('the selected BPF syscall server is not the loader preload')
        if str(base.AGENT) not in (preload or ''):
            parser.error('the selected BPF agent is missing from the client preload')
        if 'libpod_ptx_adapter.so' not in launch_env.get('BPFTIME_PTXPASS_LIBRARIES', ''):
            parser.error('the PTX-pass library is missing from the selected BPF environment')
    elif preload and BRIDGE.name not in preload:
        parser.error('the CUDA launch bridge is missing from the client preload')
    command = ['taskset', '-c', '8-15', str(PYTHON), str(HERE / 'bench.py'),
               '--arm', args.arm, '--block', str(args.block), '--phase-study',
               '--output', str(args.output.absolute() / 'operator.json')]
    if preload:
        # taskset before the injected LD_PRELOAD: the wrapper is pinned before
        # any bpftime agent loads; the env element sits inside the pinned task.
        command[3:3] = ['/usr/bin/env', 'LD_PRELOAD=' + preload]
    loader_command = None
    if args.arm == 'pod_bpf':
        loader_command = [str(LOADER_BINARY), str(SELECTOR_OBJECT), str(ptx / 'exact-kernels.txt')]
    result = None
    lease = None
    code = 1
    try:
        try:
            lease = run_phase_study.ReadOnlyLeases()
        except FileNotFoundError as error:
            raise RuntimeError(f'root must hold the two exclusive GPU lease files '
                               f'{[str(path) for path in run_phase_study.LEASE_PATHS]}: '
                               f'{error}') from error
        result = run_cell(base, args.output.absolute(), dict(arm=args.arm, block=args.block,
                          private_segment=shm, segment=segment), command, target_env,
                          launch_env, loader_command, loader_env, build, ptx,
                          run_phase_study.LEASE_PATHS)
    except BaseException as error:
        print(f'FAILED arm={args.arm} block={args.block}: {type(error).__name__}: {error}',
              file=sys.stderr)
    else:
        code = 0 if result.get('status') == 'passed' else 1
        print(f"PASS arm={args.arm} block={args.block} "
              f"client_wall_ns={result.get('client_wall_ns', 'unavailable')} "
              f"pre_python_main_ns={result.get('pre_python_main_ns', 'unavailable')}")
        for cell in result.get('operator_cells', []):
            mean = cell.get('mean_cuda_ms')
            text = f'{mean:.6f}' if type(mean) in (int, float) else 'unavailable'
            print(f"CELL model={cell.get('model')} decode_batch={cell.get('decode_batch')} "
                  f"timed_samples={cell.get('timed_samples')} mean_cuda_ms={text}")
    finally:
        if lease is not None:
            lease.close()
    sys.exit(code)


if __name__ == '__main__':
    main()
