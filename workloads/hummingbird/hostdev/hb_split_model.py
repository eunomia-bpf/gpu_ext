#!/usr/bin/env python3
"""Hummingbird host-device BPF coordinate-mapping source transform.

This is the BPF-variant companion to the original restricted TVM offset
transform in split_model.py. It produces mod-bpf.cu, a drop-in replacement for
the original split mod.cu in which the logical-block coordinate remap

    out = block(local blockIdx) + off(hb_offset_*)

is computed by a device engine call instead of being inlined as
`(blockIdx.{axis} + hb_offset_{axis})`.

Three device arms are built from the ORIGINAL resnet152 source:
  * original inline  -- split_model.py -> mod.cu   (frozen control, reused as-is)
  * native callable  -- this file, unpatched  (adapter control: callable device
                       function + 48-byte context, native computation)
  * BPF              -- this file, patched     (the eBPF engine; same call +
                       context, BPF computation)

The host is unchanged: it still tiles the launch and appends the same three
hb_offset_* arguments to every non-nop kernel, so all arms share the identical
tile budget, host policy, arrival trace and DNN. Only the device mechanism that
turns (blockIdx, hb_offset_*) into the mapped coordinate differs.

CPU-only: this transform invokes no CUDA workload and produces no performance
measurements. Actual kernel execution is a separate integration step.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import split_model as sm  # noqa: E402  (shared entrypoint/metadata logic)


CUDAH_INCLUDE = '#include "hb_map_cuda.cuh"\n'
BPF_CALL = 'hb_device_map'          # native callable wrapper symbol
# BPF variant of the mapping: body uses hb_m* (BPF-computed) instead of the
# inline (blockIdx + hb_offset) sum. The setup block below materializes the
# 48-byte HbMapContext in kernel local memory and calls the device engine.


def setup_block(used_axes: set[str]) -> str:
    """Return the CUDA text that computes hb_m* through the device engine.

    Declares only the hb_m* coordinates the kernel actually uses, then fills
    the shared context with the local split-block index and the host tile
    offsets, invokes the engine, and reads back the mapped coordinates.
    """
    lines = []
    for axis, idx in zip('xyz', (0, 1, 2)):
        if axis in used_axes:
            lines.append(f'    unsigned int hb_m{idx};')
    lines.append('    {')
    lines.append('        struct HbMapContext hb_ctx;')
    for axis in 'xyz':
        lines.append(f'        hb_ctx.block_{axis} = blockIdx.{axis};')
    for axis in 'xyz':
        lines.append(f'        hb_ctx.off_{axis} = hb_offset_{axis};')
    lines.append(f'        {BPF_CALL}(&hb_ctx, sizeof(hb_ctx));')
    for axis, idx in zip('xyz', (0, 1, 2)):
        if axis in used_axes:
            lines.append(f'        hb_m{idx} = hb_ctx.out_{axis};')
    lines.append('    }')
    return '\n'.join(lines)


def transform(source: str) -> tuple[str, dict[str, int]]:
    """Add three offset arguments and route each used coordinate through the
    device engine (BPF-variant), unlike the original inline sum.
    """
    code = sm.code_without_comments(source)
    if 'hb_offset_' in code or 'hb_m' in code or 'HbMapContext' in code:
        raise ValueError('source already uses reserved host-device names')
    bad = sm.UNSUPPORTED.search(code)
    if bad:
        raise ValueError(f'unsupported CUDA construct: {bad.group()}')
    entries = list(sm.SIGNATURE.finditer(code))
    if not entries or len(entries) != len(sm.re.findall(r'\b__global__\b', code)):
        raise ValueError('unrecognized CUDA entrypoint; only recorded TVM signatures are supported')

    definitions: Counter[str] = Counter()
    declarations: Counter[str] = Counter()
    argument_counts: dict[str, int] = {}
    edits: list[tuple[int, int, str]] = []
    body_intervals: list[tuple[int, int]] = []

    for entry in entries:
        name = entry['name']
        args = [arg.strip() for arg in entry['args'].split(',')]
        if not args or any(not sm.POINTER_ARGUMENT.fullmatch(arg) for arg in args):
            raise ValueError(f'{name}: only nonempty float-pointer argument lists are supported')
        if name in argument_counts and argument_counts[name] != len(args):
            raise ValueError(f'{name}: conflicting declarations')
        argument_counts[name] = len(args)

        # Add the three offset arguments to every entry (declaration or
        # definition) so the launch ABI matches the host for all arms.
        position = entry.end('args')
        edits.append((position, position, ', ' + sm.OFFSET_PARAMETERS))

        if entry['end'] == ';':
            declarations[name] += 1
            continue

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

        matches = list(sm.BLOCK_INDEX.finditer(body))
        if len(matches) != len(sm.re.findall(r'\bblockIdx\b', body)):
            raise ValueError(f'{name}: unsupported use of blockIdx')

        used_axes = {match[1] for match in matches}
        # Replace each original coordinate with the engine-computed hb_m*
        # (applied before the setup block is inserted, so the setup block's own
        # literal blockIdx references are left untouched).
        for match in matches:
            axis = match[1]
            idx = 'xyz'.index(axis)
            edits.append((start + match.start(), start + match.end(),
                          f'hb_m{idx}'))
        if used_axes:
            setup = setup_block(used_axes)
            # Insert the setup block immediately after the opening brace.
            edits.append((start + 1, start + 1, '\n' + setup + '\n'))

    if any(count != 1 for count in definitions.values()) or set(definitions) != set(argument_counts):
        raise ValueError('every CUDA function needs exactly one definition')
    if any(count > 1 for count in declarations.values()):
        raise ValueError('duplicate CUDA forward declaration')
    # No block-index operation may hide outside recognized bodies.
    for match in sm.BLOCK_INDEX.finditer(code):
        if not any(start <= match.start() < end for start, end in body_intervals):
            raise ValueError('blockIdx occurs outside a recognized entrypoint body')

    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    # The device engine + context ABI come from hostdev; include it up front.
    source = CUDAH_INCLUDE + source
    return source, argument_counts


def prepare(source_path: Path, host_path: Path, output: Path | None) -> dict:
    source = source_path.read_text()
    transformed, arguments = transform(source)
    launches = sm.launch_inventory(json.loads(host_path.read_text()), arguments)
    report = {'source': str(source_path.resolve()),
              'source_bytes': source_path.stat().st_size,
              'host_metadata': str(host_path.resolve()),
              'host_bytes': host_path.stat().st_size,
              'transform': 'host-device BPF coordinate mapping (hb_m* via HbMapContext)',
              'device_engine': 'hb_device_map (native adapter) -> hb_device_bpf_map (BPF) via PTX patch',
              'entrypoints': len(arguments), 'recorded_launches': len(launches),
              'original_argument_counts': arguments, 'launches': launches,
              'gpu_correctness_validated': False, 'profiled': False}
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'mod-bpf.cu').write_text(transformed)
        (output / 'launches.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='original resnet152 mod.cu (read-only input)')
    parser.add_argument('--host', type=Path, required=True,
                        help='original host.json launch metadata')
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--check', action='store_true',
                        help='CPU-only syntax/metadata check; no files written')
    choice.add_argument('--output', type=Path,
                        help='new output directory; no overwrite')
    args = parser.parse_args()
    report = prepare(args.source, args.host, args.output)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('launches', 'original_argument_counts')}, indent=2))


if __name__ == '__main__':
    main()
