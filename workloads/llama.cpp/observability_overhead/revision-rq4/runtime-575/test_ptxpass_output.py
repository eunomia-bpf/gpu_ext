"""Call the actual kretprobe plugin on CPU; retain unfiltered stdout/stderr."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

SENTINEL = b"APPLICATION_STDOUT_SENTINEL\n"
CHILD = r'''
import ctypes, json, sys
plugin = ctypes.CDLL(sys.argv[1])
plugin.process_input.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p]
plugin.process_input.restype = ctypes.c_int
ptx = '.version 7.0\n.target sm_60\n.address_size 64\n.visible .entry test_kernel()\n{\n    ret;\n}\n'
request = {'input': {'full_ptx': ptx, 'to_patch_kernel': 'test_kernel'},
           'ebpf_instructions': [{'upper_32bit': 0, 'lower_32bit': 0xb7},
                                 {'upper_32bit': 0, 'lower_32bit': 0x95}]}
output = ctypes.create_string_buffer(1024 * 1024)
status = plugin.process_input(json.dumps(request).encode(), len(output), output)
assert status == 0, status
response = json.loads(output.value)
assert response['modified'] is True
assert 'call __retprobe_func__test_kernel;' in response['output_ptx']
assert '.entry test_kernel()' in response['output_ptx']
ctypes.CDLL(None).fflush(None)
print('APPLICATION_STDOUT_SENTINEL', flush=True)
print('CPU_PLUGIN_RESULT ' + json.dumps({'status': status, 'modified': response['modified'],
      'output_ptx_bytes': len(response['output_ptx'])}), file=sys.stderr)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    library = args.build_dir.resolve() / 'attach/nv_attach_impl/pass/ptxpass_kretprobe/libptxpass_kretprobe.so'
    if not library.is_file():
        parser.error('build the private ptxpass_kretprobe target first')
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-B', '-c', CHILD, str(library)]
    result = subprocess.run(command, capture_output=True, timeout=45,
                            env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'})
    # Persist the actual streams before checking them. Never strip plugin text.
    (args.output / 'stdout.log').write_bytes(result.stdout)
    (args.output / 'stderr.log').write_bytes(result.stderr)
    diagnostics = result.stderr.decode()
    expected = '[ptxpass] kretprobe: matched=1, in=87, out=230\n'
    passed = (result.returncode == 0 and result.stdout == SENTINEL
              and diagnostics.startswith(expected)
              and diagnostics.count('[ptxpass]') == 1
              and 'CPU_PLUGIN_RESULT ' in diagnostics)
    report = {'passed': passed, 'command': command, 'returncode': result.returncode,
              'plugin': str(library), 'plugin_bytes': library.stat().st_size,
              'plugin_mtime_ns': library.stat().st_mtime_ns,
              'stdout_bytes': len(result.stdout), 'stderr_bytes': len(result.stderr),
              'application_stdout_clean': result.stdout == SENTINEL,
              'diagnostic_on_stderr': diagnostics.startswith(expected),
              'scope': 'actual CPU LLVM/PTX plugin call; no CUDA initialization or GPU execution'}
    (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'command'}, indent=2))
    if not passed:
        raise SystemExit('plugin output-routing regression; original captured streams retained')


if __name__ == '__main__':
    main()
