#!/usr/bin/env python3
"""Extract the four planned official POD TUs after the complete extension links.

Preserve original PTX and explicit inventories, conservatively remove only
unreferenced functions, and exercise the existing real BPF/PTX adapter's complete
64-MiB transport on the resulting numerical kernels. This is CPU preparation,
not device engagement or an attention-correctness experiment.
"""
import argparse
import ctypes
import json
from pathlib import Path
import re
import struct
import subprocess

from ptx_prune import prune
from ptx_partition import partition_ptx

ROOT = Path(__file__).resolve().parent
TUS = tuple(f'truefused_fwd_hdim128_fp16_causal_{split}fo{op}_sm80'
            for op in (9, 11) for split in ('', 'split_'))
CALL_ARGS = r'\s*,\s*\(\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*,\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*\)\s*;'
NATIVE_CALL = re.compile(r'\bcall(?:\.uni)?\s+pod_device_selector(?:\$[0-9]+)?' + CALL_ARGS)
BPF_CALL = re.compile(r'\bcall(?:\.uni)?\s+pod_device_bpf_selector' + CALL_ARGS)
ENTRY = re.compile(r'(?m)^\s*(?:\.visible\s+)?\.entry\s+([A-Za-z_$][A-Za-z0-9_$]*)')


def transform_pruned(ptx, representative=None):
    entries = set(ENTRY.findall(ptx))
    candidates = sorted(name for name in entries if 'true_fused_tb_fwd_kernel' in name)
    representative = representative or (candidates[0] if candidates else None)
    if not representative or representative not in entries:
        raise ValueError('missing exact official POD kernel representative')
    original_arguments = NATIVE_CALL.findall(ptx)
    if not original_arguments:
        raise ValueError('official numerical module has no real two-argument selector call')
    library = ctypes.CDLL(str(ROOT / 'build/libpod_ptx_adapter.so'))
    library.process_input.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]
    library.process_input.restype = ctypes.c_int
    bytecode = (ROOT / 'build/selector.bin').read_bytes()
    if not bytecode or len(bytecode) % 8:
        raise ValueError('invalid actual BPF bytecode')
    words = [{'upper_32bit': value >> 32, 'lower_32bit': value & 0xffffffff}
             for (value,) in struct.iter_unpack('<Q', bytecode)]
    request = {'input': {'full_ptx': ptx, 'to_patch_kernel': representative},
               'ebpf_instructions': words}
    capacity = 64 << 20  # The unmodified existing agent's actual transport limit.
    response = ctypes.create_string_buffer(capacity)
    code = library.process_input(json.dumps(request, separators=(',', ':')).encode(), capacity, response)
    if code != 0 or not response.value:
        raise ValueError('actual BPF PTX adapter failed or exceeded its transport capacity')
    encoded = response.value
    payload = json.loads(encoded)
    transformed = payload['output_ptx']
    if payload['modified'] is not True or NATIVE_CALL.search(transformed):
        raise ValueError('adapter did not replace every original typed selector call')
    if BPF_CALL.findall(transformed) != original_arguments:
        raise ValueError('adapter did not preserve every actual two-argument call')
    if set(ENTRY.findall(transformed)) != entries:
        raise ValueError('adapter changed the official entry inventory')
    if ('pod_device_bpf_selector_param_1' not in transformed or
        not re.search(r'atom\.[^\n]*add', transformed)):
        raise ValueError('generated device selector lacks typed arguments or atomic operations')
    return transformed, dict(representative=representative, official_entries=len(entries),
                             typed_calls=len(original_arguments), response_json_bytes=len(encoded),
                             modified_ptx_bytes=len(transformed.encode()), transport_capacity=capacity)


def extract_one(obj, cuobjdump):
    raw = subprocess.check_output([str(cuobjdump), '--dump-ptx', str(obj)], text=True)
    starts = list(re.finditer(r'(?m)^\.version\s+', raw))
    if len(starts) != 1:
        raise ValueError('official TU must contain exactly one original PTX module')
    source = raw[starts[0].start():]
    if not re.search(r'(?m)^\.target\s+sm_120\s*$', source):
        raise ValueError('official TU was not built for sm_120')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extension', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cuobjdump', type=Path, default=Path('/usr/local/cuda-12.9/bin/cuobjdump'))
    args = parser.parse_args()
    extension = args.extension.resolve(strict=True)
    if args.output_dir.exists():
        parser.error('refusing to overwrite an existing extraction directory')
    objects = [ROOT / 'build/torch/pod_attn' / (tu + '.o') for tu in TUS]
    for obj, tu in zip(objects, TUS):
        source = ROOT / 'deps/vattention/pod_attn/pod_attn' / (tu + '.cu')
        if not obj.is_file() or obj.stat().st_size == 0 or source.stat().st_mtime_ns > obj.stat().st_mtime_ns:
            raise ValueError('missing/stale official input object: ' + str(obj))
        if obj.stat().st_mtime_ns > extension.stat().st_mtime_ns:
            raise ValueError('official extension has not linked after every required object')
    symbols = set()
    for line in subprocess.check_output(['nm', '--defined-only', '--format=posix', str(extension)], text=True).splitlines():
        if line:
            name = line.split()[0]
            if name.startswith('__device_stub__'):
                name = name[len('__device_stub__'):]
                if not name.startswith('_'):
                    name = '_' + name
            symbols.add(name)
    args.output_dir.mkdir(parents=True)
    for subdir in ('original', 'device', 'bpf-proof', 'functions'):
        (args.output_dir / subdir).mkdir()
    manifest = dict(complete=False, extension=str(extension), extension_bytes=extension.stat().st_size,
                    planned_tus=TUS, records=[], cpu_preparation_only=True)
    (args.output_dir / 'inventory.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for obj, tu in zip(objects, TUS):
        source = extract_one(obj, args.cuobjdump)
        (args.output_dir / 'original' / (tu + '.ptx')).write_text(source)
        reduced, inventory = prune(source)
        # The generated selector adds less than 64 KiB here; the actual adapter
        # response is checked below, never inferred from this sizing estimate.
        if inventory['retained_response_json_bytes'] >= (64 << 20) - 65536:
            packets, partition = partition_ptx(reduced)
        else:
            packets, partition = [reduced], None
        proofs = []
        for number, packet in enumerate(packets, 1):
            candidates = sorted(name for name in ENTRY.findall(packet)
                                if 'true_fused_tb_fwd_kernel' in name and name in symbols)
            if not candidates:
                raise ValueError('no exact representative present in fully linked official extension: ' + tu)
            transformed, proof = transform_pruned(packet, candidates[0])
            filename = f'{tu}.part-{number:02d}.ptx'
            (args.output_dir / 'device' / filename).write_text(packet)
            (args.output_dir / 'bpf-proof' / filename).write_text(transformed)
            proofs.append(dict(filename=filename, **proof))
        if sum(p['official_entries'] for p in proofs) != len(inventory['entry_names']):
            raise ValueError('complete packet entry count differs from original official TU')
        if sum(p['typed_calls'] for p in proofs) != len(NATIVE_CALL.findall(reduced)):
            raise ValueError('complete packet call count differs from original official TU')
        inventory['partition'] = partition
        (args.output_dir / 'functions' / (tu + '.json')).write_text(json.dumps(inventory, indent=2) + '\n')
        manifest['records'].append(dict(tu=tu, object=str(obj), object_bytes=obj.stat().st_size,
            source_bytes=inventory['source_bytes'], retained_bytes=inventory['retained_bytes'], packets=proofs))
        (args.output_dir / 'inventory.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(f"POD_PTX tu={tu} entries={sum(p['official_entries'] for p in proofs)} "
              f"typed_calls={sum(p['typed_calls'] for p in proofs)} packets={len(proofs)} "
              f"source_bytes={inventory['source_bytes']} retained_bytes={inventory['retained_bytes']} "
              f"actual_response_bytes={[p['response_json_bytes'] for p in proofs]}", flush=True)
    (args.output_dir / 'exact-kernels.txt').write_text('\n'.join(p['representative']
        for r in manifest['records'] for p in r['packets']) + '\n')
    manifest['complete'] = True
    (args.output_dir / 'inventory.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
