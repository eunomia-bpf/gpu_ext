"""CPU-only ABI/transform tests, not GPU engagement or attention correctness."""
import ctypes
import json
from pathlib import Path
import struct
import unittest

ROOT = Path(__file__).resolve().parent


class PTXAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = ctypes.CDLL(str(ROOT / "build/libpod_ptx_adapter.so"))
        cls.lib.process_input.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]
        data = (ROOT / "build/selector.bin").read_bytes()
        words = struct.unpack("<" + "Q" * (len(data) // 8), data)
        cls.words = [{"upper_32bit": v >> 32, "lower_32bit": v & 0xffffffff} for v in words]

    def transform(self, call, capacity=1048576):
        declaration = ".func pod_device_selector(.param .b64 ctx, .param .b64 len);\n"
        declaration += ".func pod_device_selector$3(.param .b64 ctx, .param .b64 len);\n"
        ptx = ".version 8.8\n.target sm_120\n.address_size 64\n" + declaration + ".visible .entry real_attention() {\n" + call + "\nret;\n}\n"
        request = {"input": {"full_ptx": ptx, "to_patch_kernel": "real_attention"},
                   "ebpf_instructions": self.words}
        out = ctypes.create_string_buffer(capacity)
        result = self.lib.process_input(json.dumps(request).encode(), capacity, out)
        return result, out.value

    def test_two_arguments_preserved_and_device_atomic(self):
        result, output = self.transform("call.uni pod_device_selector, (ctx_arg, len_arg);")
        self.assertEqual(result, 0)
        payload = json.loads(output)
        ptx = payload["output_ptx"]
        self.assertTrue(payload["modified"])
        self.assertIn("call.uni pod_device_bpf_selector, (ctx_arg, len_arg);", ptx)
        self.assertIn("pod_device_bpf_selector_param_1", ptx)
        self.assertRegex(ptx, r"atom\.[^\n]*add")
        self.assertLess(ptx.index(".address_size"), ptx.index(".func pod_device_bpf_selector"))
        self.assertEqual(ptx.count(".version"), 1)

    def test_nonuniform_typed_call(self):
        result, _ = self.transform("call pod_device_selector, (arg0, arg1);")
        self.assertEqual(result, 0)

    def test_nvcc_two_argument_clone(self):
        result, output = self.transform("call.uni pod_device_selector$3, (arg0, arg1);")
        self.assertEqual(result, 0)
        self.assertIn("call.uni pod_device_bpf_selector, (arg0, arg1);", json.loads(output)["output_ptx"])

    def test_nvcc_one_argument_clone_rejected(self):
        result, _ = self.transform("call.uni pod_device_selector$3, (arg0);")
        self.assertNotEqual(result, 0)

    def test_missing_hook_rejected(self):
        result, _ = self.transform("call other_function;")
        self.assertNotEqual(result, 0)

    def test_no_argument_probe_rejected(self):
        result, _ = self.transform("call pod_device_selector;")
        self.assertNotEqual(result, 0)

    def test_one_argument_rejected(self):
        result, _ = self.transform("call pod_device_selector, (ctx_arg);")
        self.assertNotEqual(result, 0)

    def test_capacity_requests_existing_transport_retry(self):
        result, output = self.transform("call pod_device_selector, (arg0, arg1);", capacity=32)
        self.assertEqual(result, 0)
        self.assertEqual(output, b"")
        with self.assertRaises(json.JSONDecodeError):
            json.loads(output)

    def test_second_exact_target_does_not_patch_module_twice(self):
        result, output = self.transform("call pod_device_selector, (arg0, arg1);")
        self.assertEqual(result, 0)
        ptx = json.loads(output)["output_ptx"]
        request = {"input": {"full_ptx": ptx, "to_patch_kernel": "real_attention"},
                   "ebpf_instructions": self.words}
        out = ctypes.create_string_buffer(1048576)
        self.assertEqual(self.lib.process_input(json.dumps(request).encode(), len(out), out), 0)
        self.assertFalse(json.loads(out.value)["modified"])
        self.assertEqual(json.loads(out.value)["output_ptx"], ptx)

    def test_unrelated_module_is_not_reported_as_engaged(self):
        request = {"input": {"full_ptx": ".visible .entry other_kernel() { ret; }",
                             "to_patch_kernel": "real_attention"},
                   "ebpf_instructions": self.words}
        out = ctypes.create_string_buffer(4096)
        self.assertEqual(self.lib.process_input(json.dumps(request).encode(), len(out), out), 0)
        self.assertFalse(json.loads(out.value)["modified"])


if __name__ == "__main__":
    unittest.main()
