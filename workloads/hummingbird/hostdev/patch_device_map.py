#!/usr/bin/env python3
"""Patch a callable-native ResNet PTX so it actually consumes the eBPF mapping.

nvcc specializes the called device mapping into a private clone named
`hb_device_map$NN` and makes every kernel call that clone; the visible
`hb_device_map` is left unused. A naive rename of the visible symbol would
therefore leave the BPF engine unconsumed. This patcher instead:

  1. rewrites each specialized-clone CALL TARGET (`hb_device_map$NN` appearing
     as a bare call-target line) to `hb_device_bpf_map`; and
  2. injects the standalone eBPF-derived `.func hb_device_bpf_map`
     (generic load/store; the 48-byte context lives in kernel local memory).

The call sites pass a `cvta.local`'d generic context pointer (param0) and an
unused param1; the BPF function reads only param0, so the existing argument
wiring is left untouched. The native clone and visible function become dead
code and are discarded by `ptxas -O`.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

BPF_FUNC = 'hb_device_bpf_map'
# A line whose only content is a specialized-clone call target: optional
# indent, `hb_device_map$<digits>`, optional trailing comma/whitespace. This
# deliberately does NOT match `.func ...` definitions or `..._param_` names.
CALL_TARGET = re.compile(r'^(\s*)hb_device_map\$(\d+)\s*(,?\s*)$')


def patch(kernel_ptx: str, bpf_ptx: str) -> tuple[str, int]:
    lines = kernel_ptx.splitlines()

    # 1) rewrite the specialized-clone call targets.
    rewritten = 0
    for i, line in enumerate(lines):
        match = CALL_TARGET.match(line)
        if match:
            prefix, suffix = match.group(1), match.group(3)
            lines[i] = f'{prefix}{BPF_FUNC}{suffix}'
            rewritten += 1
    if rewritten == 0:
        raise SystemExit('no hb_device_map$NN call target found; '
                         'the BPF engine would remain unconsumed')

    # 2) extract the BPF .func, dropping its standalone .version/.target/
    #    .address_size header (the kernel TU already carries its own).
    bpf_lines = bpf_ptx.splitlines()
    try:
        start = next(i for i, l in enumerate(bpf_lines)
                     if l.strip().startswith(f'.visible .func {BPF_FUNC}'))
    except StopIteration:
        raise SystemExit(f'no `.visible .func {BPF_FUNC}` found in BPF PTX')
    bpf_func = '\n'.join(bpf_lines[start:]).rstrip() + '\n'

    # 3) inject the BPF function at top level, just after the kernel header.
    out: list[str] = []
    injected = False
    for line in lines:
        out.append(line)
        if not injected and line.strip() == '.address_size 64':
            out.append('')
            out.append(bpf_func.rstrip('\n'))
            injected = True
    if not injected:
        out.append('')
        out.append(bpf_func.rstrip('\n'))
    return '\n'.join(out) + '\n', rewritten


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel', type=Path, required=True,
                        help='nvcc -ptx output (callable-native ResNet)')
    parser.add_argument('--bpf', type=Path, required=True,
                        help='exporter BPF PTX (standalone .func)')
    parser.add_argument('--output', type=Path, required=True,
                        help='patched PTX to feed ptxas')
    args = parser.parse_args()
    patched, count = patch(args.kernel.read_text(), args.bpf.read_text())
    args.output.write_text(patched)
    print(f'rewrote {count} specialized-clone call target(s) to {BPF_FUNC}; '
          f'injected BPF .func -> {args.output}')


if __name__ == '__main__':
    main()
