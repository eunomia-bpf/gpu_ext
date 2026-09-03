"""Small x86 ELF fixtures for the POD host-only link repair; no CUDA."""
from pathlib import Path
import subprocess
import tempfile
import unittest

import host_link as link

FIXTURE = '''.text
.globl pod_device_selector
.type pod_device_selector,@function
pod_device_selector:
endbr64
subq $24,%rsp
movl $1,12(%rsp)
movl 12(%rsp),%edi
call exit
.size pod_device_selector,.-pod_device_selector
.globl numerical_kernel
.type numerical_kernel,@function
numerical_kernel:
ret
.size numerical_kernel,.-numerical_kernel
.data
.quad __cudaRegisterFunction
.quad numerical_kernel
.section .nv_fatbin,"a",@progbits
.byte 1,2,3,4
.section .note.GNU-stack,"",@progbits
'''


class HostLinkTests(unittest.TestCase):
    def compile(self, root, text=FIXTURE):
        source, obj = root / 'fixture.s', root / 'original.o'
        source.write_text(text)
        subprocess.run(['cc', '-c', str(source), '-o', str(obj)], check=True)
        return obj

    def test_only_host_binding_changes_original_and_device_preserved(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            original = self.compile(root)
            output = root / 'localized.o'
            info = link.localize_object(original, output)
            self.assertEqual(link.defined_symbols(original)[link.SYMBOL][0], 'T')
            self.assertEqual(link.defined_symbols(output)[link.SYMBOL][0], 't')
            self.assertEqual(info['unchanged_embedded_gpu_bytes'], 4)
            self.assertEqual(info['cuda_registration_references'], 1)
            self.assertEqual(info['host_references_to_stub'], 0)
            with self.assertRaises(ValueError):
                link.localize_object(original, output)

    def test_referenced_or_not_exact_global_stub_is_rejected(self):
        for text in (FIXTURE.replace('.quad numerical_kernel', '.quad pod_device_selector'),
                     FIXTURE.replace('.globl pod_device_selector', '.local pod_device_selector'),
                     FIXTURE.replace('call exit', 'call other_function')):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                original = self.compile(root, text)
                with self.assertRaises(ValueError):
                    link.localize_object(original, root / 'rejected.o')

    def test_device_section_change_is_rejected_directly(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            original = self.compile(root)
            output = root / 'localized.o'
            link.localize_object(original, output)
            offset, _ = link.fatbin_extent(output)
            with output.open('r+b') as file:
                file.seek(offset)
                file.write(b'\xff')
            with self.assertRaises(ValueError):
                link.equal_fatbin(original, output)

    def test_all_official_templates_are_required(self):
        self.assertEqual(len(link.EXPECTED_TUS), 20)
        with self.assertRaises(ValueError):
            link.localize_for_link([], Path('fused_attn.so'))


if __name__ == '__main__':
    unittest.main()
