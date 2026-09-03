#!/usr/bin/env python3
"""Safety/lease wrapper for the bounded GPReempt context+RPC canary only."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'workloads/moe-infinity'))
import run_moe_head_to_head as shared
from gpreempt_context_smoke_check import analyze


@contextmanager
def leases():
    held = []
    try:
        for name in ('/tmp/gpubpf-revision-gpu0.lock', '/tmp/gpubpf-revision-struct-ops.lock'):
            path = Path(name)
            # O_CREAT on another user's existing /tmp file can be rejected by
            # protected_regular even as root. Preserve the SAME lock inode.
            try:
                stream = path.open('r+')
            except FileNotFoundError:
                try:
                    stream = os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o666), 'r+')
                except FileExistsError:
                    stream = path.open('r+')
            held.append(stream)
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        for stream in reversed(held):
            stream.close()


class Process:
    def __init__(self, name, command, directory, env):
        self.name = name
        self.path = directory / f'{name}.stdout'
        self.logs = [self.path.open('x'), (directory / f'{name}.stderr').open('x')]
        self.proc = subprocess.Popen(command, stdout=self.logs[0], stderr=self.logs[1],
                                     env=env, start_new_session=True)
        self.command = command

    def ready(self, marker, deadline):
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f'{self.name} exited {self.proc.returncode} before ready')
            if marker in self.path.read_text():
                return
            time.sleep(0.1)
        raise TimeoutError(f'{self.name} ready deadline')

    def stop(self):
        try:
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
                if self.proc.poll() is not None:
                    break
                pgid = os.getpgid(self.proc.pid)
                if pgid != self.proc.pid:
                    raise RuntimeError(f'{self.name}: refusing drifted owned process-group ID')
                os.killpg(pgid, signum)
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    continue
            if self.proc.poll() is None:
                raise RuntimeError(f'{self.name}: owned group survived 15-second cleanup')
        finally:
            for log in self.logs:
                log.close()


def interrupted(number, frame):
    raise InterruptedError(f'canary wrapper received signal {number}')


def run(args):
    if os.geteuid() != 0:
        raise RuntimeError('run the complete wrapper as root; all compared canary arms use the same privilege')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with leases():
        before = shared.safety_snapshot()
        (output / 'safety-before.json').write_text(json.dumps(before, indent=2) + '\n')
        shared.validate_pre_server_safety(before)
        if not set(range(8, 16)).issubset(os.sched_getaffinity(0)):
            raise RuntimeError('fixed canary CPU range 8-15 is unavailable')
        env = os.environ.copy()
        for name in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'GPUBPF_HPF_CODE', 'GPREEMPT_BPF_MAPS', 'GPREEMPT_HINT_CODE'):
            env.pop(name, None)
        env.update(GPREEMPT_POLICY=args.mode, CUDA_CACHE_DISABLE='1', CUDA_VISIBLE_DEVICES='0')
        pin = Path(f'/sys/fs/bpf/gpreempt-context-{os.getpid()}-{time.monotonic_ns()}')
        processes = []
        failure = None
        result = None
        deadline = time.monotonic() + args.timeout

        def start(name, command):
            process = Process(name, ['taskset', '-c', '8-15'] + list(map(str, command)), output, env)
            processes.append(process)
            return process

        try:
            if args.rpc_observer == 'required':
                observer = start('rpc', [HERE / '.output/gpreempt_context_smoke_rpc', '120'])
                observer.ready('gpreempt_rpc_observer_ready:', min(deadline, time.monotonic() + 15))
            if args.mode == 'bpf':
                env.update(GPREEMPT_BPF_MAPS=str(pin), GPREEMPT_HINT_CODE=str(HERE / '.output/gpreempt_hint.bin'))
                policy = start('policy', [HERE / '.output/gpreempt_policy', '--library',
                    HERE / '.output/libgpreempt_bridge.so', '--pin-dir', pin, '--duration', '120'])
                policy.ready('gpreempt_policy_ready:', min(deadline, time.monotonic() + 15))
            client = start('client', [HERE / '.output/gpreempt_context_smoke'])
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                raise TimeoutError('canary total startup deadline')
            if client.proc.wait(timeout=timeout):
                raise RuntimeError(f'canary client failed with exit {client.proc.returncode}')
        except BaseException as error:
            failure = f'{type(error).__name__}: {error}'
        finally:
            cleanup_errors = []
            for process in reversed(processes):
                try:
                    process.stop()
                except BaseException as error:
                    cleanup_errors.append(f'{process.name}: {type(error).__name__}: {error}')
            (output / 'processes.json').write_text(json.dumps([
                dict(name=p.name, command=p.command, pid=p.proc.pid, returncode=p.proc.returncode)
                for p in processes], indent=2) + '\n')
            if cleanup_errors:
                failure = (failure or '') + '; cleanup: ' + '; '.join(cleanup_errors)
            if not failure:
                try:
                    if any(p.proc.returncode != 0 for p in processes):
                        raise RuntimeError('one or more observer/policy processes failed')
                    result = analyze((output / 'client.stdout').read_text(),
                                     (output / 'rpc.stdout').read_text() if args.rpc_observer == 'required' else None,
                                     args.mode, (output / 'policy.stdout').read_text() if args.mode == 'bpf' else None)
                    if pin.exists():
                        raise RuntimeError('owned BPF pin directory still exists after loader cleanup')
                except BaseException as error:
                    failure = f'{type(error).__name__}: {error}'
            try:
                after = shared.wait_for_post_server_safety(before)
                (output / 'safety-after.json').write_text(json.dumps(after, indent=2) + '\n')
            except BaseException as error:
                failure = (failure or '') + f'; post-safety: {type(error).__name__}: {error}'
            record = dict(mode=args.mode, timeout_seconds=args.timeout, rpc_observer=args.rpc_observer,
                          result=result, failure=failure)
            (output / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record, indent=2), flush=True)
        return 1 if failure else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('original', 'bpf'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--rpc-observer', choices=('required', 'off'), default='required',
                        help='off is context-only diagnosis; cannot claim direct firmware RPC evidence')
    args = parser.parse_args()
    if not 1 <= args.timeout <= 60:
        parser.error('canary total startup/runtime deadline must be 1..60 seconds')
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
