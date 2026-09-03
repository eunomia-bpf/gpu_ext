#!/usr/bin/env python3
"""POD-only host link fix for NVCC's unused strong device-function stub.

Original objects remain untouched. Only the exact host symbol is localized in
link copies; embedded GPU bytes and every other defined symbol must be intact.
No device function body, CUDA registration, or broad linker option is changed.
"""
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent
SYMBOL = 'pod_device_selector'
EXPECTED_TUS = {f'truefused_fwd_hdim128_fp16_{causal}{split}fo{op}_sm80.o'
                for causal in ('', 'causal_') for split in ('', 'split_') for op in (8, 9, 10, 11, 64)}


def defined_symbols(path):
    rows = subprocess.check_output(['nm', '--defined-only', '--format=posix', str(path)], text=True)
    return {fields[0]: fields[1:] for line in rows.splitlines() if (fields := line.split())}


def fatbin_extent(path):
    sections = subprocess.check_output(['readelf', '-SW', str(path)], text=True)
    matches = re.findall(r'\[\s*\d+\]\s+\.nv_fatbin\s+\S+\s+[0-9a-fA-F]+\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)', sections)
    if len(matches) != 1:
        raise ValueError('expected exactly one original embedded .nv_fatbin section')
    return tuple(int(x, 16) for x in matches[0])


def equal_fatbin(original, localized):
    offset_a, size_a = fatbin_extent(original)
    offset_b, size_b = fatbin_extent(localized)
    if size_a != size_b or size_a == 0:
        raise ValueError('localization changed embedded GPU extent')
    # Direct bytes, in bounded buffers; no derived content identifier is made.
    with original.open('rb') as a, localized.open('rb') as b:
        a.seek(offset_a)
        b.seek(offset_b)
        remaining = size_a
        while remaining:
            amount = min(1 << 20, remaining)
            left, right = a.read(amount), b.read(amount)
            if len(left) != amount or len(right) != amount or left != right:
                raise ValueError('localization changed embedded GPU bytes')
            remaining -= amount
    return size_a


def relocations(path):
    text = subprocess.check_output(['readelf', '--relocs', '--wide', str(path)], text=True)
    if re.search(r'\b' + SYMBOL + r'\b', text):
        raise ValueError('host code/registration refers to the device-only host stub')
    # ELF symbol-table indices may move after localization; semantic relocation
    # offset/type/value/name/addend must not. Keep complete semantic rows.
    rows = []
    section = None
    for line in text.splitlines():
        if line.startswith('Relocation section '):
            section = line.split("'")[1]
        match = re.match(r'\s*([0-9a-fA-F]+)\s+[0-9a-fA-F]+\s+(R_\S+)\s+(.*)', line)
        if match:
            rows.append((section, *match.groups()))
    return rows


def localize_object(original, output):
    if output.exists():
        raise ValueError('refusing to overwrite an existing localized link object')
    symbols = defined_symbols(original)
    fields = symbols.get(SYMBOL, [])
    if len(fields) != 3 or fields[0] != 'T' or int(fields[2], 16) != 25:
        raise ValueError('expected the actual NVCC 12.9 global 25-byte host stub')
    disassembly = subprocess.check_output(['objdump', '-dr', '--section=.text',
        '--disassemble=' + SYMBOL, str(original)], text=True)
    if 'exit-0x4' not in disassembly or '$0x1,' not in disassembly:
        raise ValueError('host stub is not the observed NVCC fail-fast exit(1) body')
    original_relocations = relocations(original)
    registrations = sum('__cudaRegisterFunction' in row[-1] for row in original_relocations)
    if registrations == 0:
        raise ValueError('official numerical object has no actual CUDA registration references')
    subprocess.run(['objcopy', '--localize-symbol=' + SYMBOL, str(original), str(output)], check=True)
    expected = {**symbols, SYMBOL: ['t', *fields[1:]]}
    if defined_symbols(output) != expected:
        raise ValueError('localization changed another defined symbol')
    if relocations(output) != original_relocations:
        raise ValueError('localization changed host/CUDA registration relocations')
    embedded_bytes = equal_fatbin(original, output)
    return dict(original=str(original), original_bytes=original.stat().st_size,
                localized=str(output), localized_bytes=output.stat().st_size,
                host_symbol=SYMBOL, original_binding='T', link_binding='t',
                host_stub_disassembly=disassembly, host_references_to_stub=0,
                cuda_registration_references=registrations,
                other_defined_symbols_unchanged=len(symbols) - 1,
                unchanged_embedded_gpu_bytes=embedded_bytes)


def localize_for_link(objects, output):
    selected = [Path(x).resolve() for x in objects if Path(x).name in EXPECTED_TUS]
    if {p.name for p in selected} != EXPECTED_TUS or len(selected) != 20:
        raise ValueError('fused link must retain all twenty official numerical TUs')
    link_root = ROOT / 'build/link'
    link_root.mkdir(exist_ok=True)
    # Each retry gets a new explicit inventory and copies, never rewrites the
    # original inputs or a previously reviewed link attempt.
    attempt = 1
    while (link_root / f'attempt-{attempt:02d}').exists():
        attempt += 1
    directory = link_root / f'attempt-{attempt:02d}'
    directory.mkdir()
    manifest = dict(complete=False, output=str(output), records=[])
    manifest_path = directory / 'inventory.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    replacements = {}
    for original in selected:
        localized = directory / original.name
        record = localize_object(original, localized)
        replacements[str(original)] = str(localized)
        manifest['records'].append(record)
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    manifest['complete'] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'POD_HOST_LINK localized_objects=20 original_objects_preserved=20 inventory={manifest_path}', flush=True)
    return [replacements.get(str(Path(p).resolve()), p) for p in objects]


def install():
    # Only this setup invocation, never a persistent mutation of torch/bpftime.
    from torch.utils.cpp_extension import BuildExtension
    original_build = BuildExtension.build_extensions
    def build_extensions(self):
        original_link = self.compiler.link_shared_object
        def link(objects, output, *args, **kwargs):
            if Path(output).name.startswith('fused_attn.'):
                objects = localize_for_link(objects, output)
            return original_link(objects, output, *args, **kwargs)
        self.compiler.link_shared_object = link
        try:
            return original_build(self)
        finally:
            self.compiler.link_shared_object = original_link
    BuildExtension.build_extensions = build_extensions


if __name__ == '__main__':
    import runpy
    import sys
    install()
    source = ROOT / 'deps/vattention/pod_attn/setup.py'
    sys.argv[0] = str(source)
    runpy.run_path(str(source), run_name='__main__')
