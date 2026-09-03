#!/usr/bin/env python3
"""Restricted TVM CUDA-source offset transform; no CUDA invocation or profiling.

This is a declared source-level substitute for Hummingbird's PTX transformer,
not an implementation of its opaque-library support. GPU numerical validation
is still necessary; syntax acceptance alone is not a proof of independence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re


SIGNATURE = re.compile(
    r'extern\s+"C"\s+__global__\s+void\s+'
    r'__launch_bounds__\s*\(\s*\d+(?:\s*,\s*\d+)?\s*\)\s+'
    r'(?P<name>[A-Za-z_]\w*)\s*\((?P<args>[^()]*)\)\s*(?P<end>[;{])')
POINTER_ARGUMENT = re.compile(r'float\s*\*\s*__restrict__\s+[A-Za-z_]\w*')
COMMENT = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/')
UNSUPPORTED = re.compile(
    r'\b(?:gridDim|cooperative_groups|atomic\w*|__threadfence\w*|asm|__asm__|'
    r'__device__|__managed__)\b|<<<|>>>')
BLOCK_INDEX = re.compile(r'\bblockIdx\s*\.\s*([xyz])\b')
OFFSET_PARAMETERS = ', '.join(f'unsigned int hb_offset_{axis}' for axis in 'xyz')


def code_without_comments(source: str) -> str:
    """Mask comments without changing positions used for edits."""
    return COMMENT.sub(lambda match: ''.join('\n' if c == '\n' else ' '
                                             for c in match.group()), source)


def transform(source: str) -> tuple[str, dict[str, int]]:
    """Add three offset arguments and translate each function's block indices.

    Only the generated float-pointer TVM entrypoint syntax is supported. No
    textual fallback silently rewrites unfamiliar CUDA constructs.
    """
    code = code_without_comments(source)
    if 'hb_offset_' in code:
        raise ValueError('source already uses reserved offset argument names')
    bad = UNSUPPORTED.search(code)
    if bad:
        raise ValueError(f'unsupported CUDA construct: {bad.group()}')
    entries = list(SIGNATURE.finditer(code))
    if not entries or len(entries) != len(re.findall(r'\b__global__\b', code)):
        raise ValueError('unrecognized CUDA entrypoint; only recorded TVM signatures are supported')
    definitions: Counter[str] = Counter()
    declarations: Counter[str] = Counter()
    argument_counts: dict[str, int] = {}
    edits: list[tuple[int, int, str]] = []
    body_intervals: list[tuple[int, int]] = []
    for entry in entries:
        name = entry['name']
        args = [arg.strip() for arg in entry['args'].split(',')]
        if not args or any(not POINTER_ARGUMENT.fullmatch(arg) for arg in args):
            raise ValueError(f'{name}: only nonempty float-pointer argument lists are supported')
        if name in argument_counts and argument_counts[name] != len(args):
            raise ValueError(f'{name}: conflicting declarations')
        argument_counts[name] = len(args)
        if entry['end'] == ';':
            declarations[name] += 1
        else:
            definitions[name] += 1
            start = entry.end() - 1
            depth, position = 1, start + 1
            while position < len(code) and depth:
                depth += (code[position] == '{') - (code[position] == '}')
                position += 1
            if depth:
                raise ValueError(f'{name}: unterminated function')
            body_intervals.append((start, position))
            body = code[start:position]
            if '"' in body or "'" in body:
                raise ValueError(f'{name}: embedded literals are outside the generated-source scope')
            matches = list(BLOCK_INDEX.finditer(body))
            if len(matches) != len(re.findall(r'\bblockIdx\b', body)):
                raise ValueError(f'{name}: unsupported use of blockIdx')
            for match in matches:
                axis = match[1]
                edits.append((start + match.start(), start + match.end(),
                              f'(blockIdx.{axis} + hb_offset_{axis})'))
        position = entry.end('args')
        edits.append((position, position, ', ' + OFFSET_PARAMETERS))
    if any(count != 1 for count in definitions.values()) or set(definitions) != set(argument_counts):
        raise ValueError('every CUDA function needs exactly one definition')
    if any(count > 1 for count in declarations.values()):
        raise ValueError('duplicate CUDA forward declaration')
    # No CUDA launch or block-index operation may hide outside recognized bodies.
    for match in BLOCK_INDEX.finditer(code):
        if not any(start <= match.start() < end for start, end in body_intervals):
            raise ValueError('blockIdx occurs outside a recognized entrypoint body')
    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source, argument_counts


def positive_integer(value: object, label: str) -> int:
    if type(value) is not int or not 1 <= value <= 2**31 - 1:
        raise ValueError(f'{label} must be an integer in [1, 2^31-1]')
    return value


def split_grid(grid: list[int] | tuple[int, int, int], max_blocks: int):
    """Yield disjoint rectangular (offset, grid) pieces in x-major CUDA order."""
    if len(grid) != 3:
        raise ValueError('a grid must have three dimensions')
    gx, gy, gz = [positive_integer(v, 'grid dimension') for v in grid]
    cap = positive_integer(max_blocks, 'max_blocks')
    tx = min(gx, cap)
    ty = min(gy, cap // tx)
    tz = min(gz, cap // (tx * ty))
    for z in range(0, gz, tz):
        for y in range(0, gy, ty):
            for x in range(0, gx, tx):
                yield ((x, y, z), (min(tx, gx - x), min(ty, gy - y), min(tz, gz - z)))


def launch_inventory(host: dict, arguments: dict[str, int]) -> list[dict]:
    if not isinstance(host, dict) or not isinstance(host.get('funcs'), list):
        raise ValueError('missing original host funcs metadata')
    temporaries = host.get('temp_args', [])
    if not isinstance(temporaries, list):
        raise ValueError('invalid original temporary-storage metadata')
    result = []
    for function in host['funcs']:
        if not isinstance(function, dict) or not isinstance(function.get('kernels'), list):
            raise ValueError('invalid host function metadata')
        for kernel in function['kernels']:
            name = kernel.get('name')
            if name == 'nop':
                result.append({'ordinal': len(result), 'kernel': name, 'operation': 'device_copy'})
                continue
            dims, args = kernel.get('launch_params'), kernel.get('args')
            if name not in arguments or not isinstance(dims, list) or len(dims) != 6:
                raise ValueError(f'{name}: missing function or six-dimensional launch metadata')
            for value in dims:
                positive_integer(value, 'launch dimension')
            if dims[3] * dims[4] * dims[5] > 1024:
                raise ValueError(f'{name}: unsupported block with more than 1024 threads')
            # The upstream TVM recorder uses -1, -2, ... for temporary storage;
            # the original Model loader resolves these, not this transformer.
            if (not isinstance(args, list) or len(args) != arguments[name]
                    or any(type(arg) is not int or not -len(temporaries) <= arg <= 2**31 - 1
                           for arg in args)):
                raise ValueError(f'{name}: recorded kernel argument count/index is invalid')
            result.append({'ordinal': len(result), 'kernel': name, 'operation': 'kernel',
                           'grid': dims[:3], 'block': dims[3:], 'argument_count': len(args)})
    if not result:
        raise ValueError('empty model launch inventory')
    return result


def prepare(source_path: Path, host_path: Path, output: Path | None) -> dict:
    source = source_path.read_text()
    transformed, arguments = transform(source)
    launches = launch_inventory(json.loads(host_path.read_text()), arguments)
    report = {'source': str(source_path.resolve()), 'source_bytes': source_path.stat().st_size,
              'host_metadata': str(host_path.resolve()), 'host_bytes': host_path.stat().st_size,
              'transform': 'restricted CUDA-source blockIdx xyz offset arguments',
              'entrypoints': len(arguments), 'recorded_launches': len(launches),
              'original_argument_counts': arguments, 'launches': launches,
              'gpu_correctness_validated': False, 'profiled': False}
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'mod.cu').write_text(transformed)
        (output / 'launches.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--host', type=Path, required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--check', action='store_true', help='CPU-only syntax/metadata check; no files written')
    choice.add_argument('--output', type=Path, help='new output directory; no overwrite')
    args = parser.parse_args()
    report = prepare(args.source, args.host, args.output)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('launches', 'original_argument_counts')}, indent=2))


if __name__ == '__main__':
    main()
