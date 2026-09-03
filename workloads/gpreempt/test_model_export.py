#!/usr/bin/env python3
"""CPU-only checks for model choice and the upstream parameter serialization."""
import io
import struct
import sys
import unittest
import numpy as np
import export_model


class Tensor:
    def __init__(self, values):
        self.values = values
    def numpy(self):
        return self.values


class ExportTests(unittest.TestCase):
    def test_plan_does_not_import_tvm_or_cuda(self):
        self.assertNotIn("tvm", sys.modules)
        self.assertEqual(export_model.model_spec("vgg", 120)["layers"], 19)
        self.assertEqual(export_model.model_spec("resnet152", 120)["layers"], 152)
        self.assertFalse(export_model.model_spec("vgg", 120)["pretrained_weights"])

    def test_exact_upstream_parameter_layout(self):
        output = io.BytesIO()
        export_model.dump_params({"weight": Tensor(np.array([1, 2], dtype=np.float32))}, output)
        expected = b"TVM_MODEL_PARAMS\0" + struct.pack("<Q", 1) + b"weight\0"
        expected += struct.pack("<Qff", 2, 1, 2)
        self.assertEqual(output.getvalue(), expected)

    def test_wrong_dtype_and_overlong_name_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "float32"):
            export_model.dump_params({"weight": Tensor(np.ones(1, dtype=np.float16))}, io.BytesIO())
        with self.assertRaisesRegex(ValueError, "parameter name"):
            export_model.dump_params({"w" * 256: Tensor(np.ones(1, dtype=np.float32))}, io.BytesIO())


if __name__ == "__main__":
    unittest.main()
