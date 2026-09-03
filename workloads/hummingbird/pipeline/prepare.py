#!/usr/bin/env python3
"""Copy the recorded Hummingbird sources and apply only the private ablation."""
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE_REVISION = '995bc62'
FILES = ('idle_policy.c', 'idle_policy.h', 'idle_policy.bpf.c',
         'idle_executor.cpp', 'idle_executor.h', 'hummingbird_client.cpp',
         'test_idle_policy.cpp', 'test_profile.cpp', 'Makefile',
         'run_study.py', 'analyze_study.py')


def prepare():
    patches = [HERE / name for name in ('completion-fence.patch', 'runner.patch')]
    if not all(path.is_file() for path in patches):
        raise RuntimeError('missing private source patch')
    target = HERE / 'build/src'
    target.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        destination = target / name
        if not destination.exists():
            source = subprocess.check_output([
                'git', '-C', str(ROOT), 'show',
                f'{SOURCE_REVISION}:workloads/hummingbird/{name}'])
            with destination.open('xb') as output:
                output.write(source)
            print(f'copied {SOURCE_REVISION} {name} bytes={len(source)}', flush=True)
    command = ['git', '-C', str(ROOT), 'apply',
               '--directory=workloads/hummingbird/pipeline/build/src']
    for patch in patches:
        already = subprocess.run(command + ['--reverse', '--check', str(patch)],
                                 capture_output=True, text=True)
        if already.returncode == 0:
            print(f'{patch.name} reverse application check passed', flush=True)
        else:
            subprocess.run(command + ['--check', str(patch)], check=True)
            print(f'{patch.name} forward application check passed', flush=True)
            subprocess.run(command + [str(patch)], check=True)
            print(f'{patch.name} applied', flush=True)
    return target


if __name__ == '__main__':
    prepare()
