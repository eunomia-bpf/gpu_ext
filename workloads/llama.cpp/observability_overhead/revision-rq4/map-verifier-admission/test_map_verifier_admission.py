#!/usr/bin/env python3
"""CPU-only contract tests for the map-program verifier admission probe."""

import json
import os
from pathlib import Path
import subprocess
import unittest


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "map_verifier_admission.cpp"
EXPECTED_PROGRAMS = [
    "cuda__noop",
    "cuda__device_update",
    "cuda__host_update",
    "cuda__rpc_update",
    "cuda__device_lookup",
    "cuda__host_lookup",
    "cuda__rpc_lookup",
]


class SourceContractTests(unittest.TestCase):
    def test_probe_calls_public_strict_entry_with_map_descriptors(self):
        text = SOURCE.read_text()
        self.assertIn("verify_gpu_program(", text)
        self.assertIn("symbol.name, maps", text)
        self.assertIn("read_elf(path", text)

    def test_probe_requires_real_elf_symbol_boundaries_and_map_relocations(self):
        text = SOURCE.read_text()
        self.assertIn("type == ELFIO::STT_FUNC", text)
        self.assertIn("instruction.src != 1", text)
        self.assertIn("function symbols do not cover the ELF section", text)

    def test_scope_excludes_gpu_execution_and_attach(self):
        text = SOURCE.read_text()
        self.assertIn("not GPU execution or attach safety", text)
        self.assertNotIn("cudaSetDevice", text)
        self.assertNotIn("cuModuleLoad", text)


@unittest.skipUnless(os.environ.get("MAP_VERIFIER_ADMISSION_PROBE"),
                     "set MAP_VERIFIER_ADMISSION_PROBE for real CPU admission")
class BuiltProbeTests(unittest.TestCase):
    def test_real_map_object_and_controls(self):
        probe = Path(os.environ["MAP_VERIFIER_ADMISSION_PROBE"])
        object_path = Path(os.environ["MAP_VERIFIER_ADMISSION_OBJECT"])
        completed = subprocess.run(
            [str(probe), "--object", str(object_path)],
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["schema"], "map-verifier-admission-v1")
        self.assertEqual(result["scope"],
                         "CPU-only direct verify_gpu_program admission; not GPU execution or attach safety")
        self.assertEqual([row["name"] for row in result["programs"]],
                         EXPECTED_PROGRAMS)
        self.assertEqual([row["instructions"] for row in result["programs"]],
                         [2, 20, 20, 20, 26, 26, 26])
        self.assertEqual(result["summary"]["target_programs"], 7)
        self.assertTrue(result["summary"]["control_expectations_met"])
        self.assertEqual(len(result["maps"]), 4)
        self.assertEqual({row["name"] for row in result["maps"]}, {
            "device_values", "host_values", "rpc_values", "observed_values"
        })
        self.assertEqual({row["id"] for program in result["programs"]
                          for row in program["helpers"]}, {1, 2, 511})
        controls = {row["name"]: row for row in result["controls"]}
        self.assertTrue(controls["positive_minimal"]["accepted"])
        self.assertFalse(controls["negative_unknown_helper"]["accepted"])
        self.assertFalse(controls["negative_varying_branch"]["accepted"])
        self.assertFalse(controls["negative_unsupported_gpu_map"]["accepted"])


if __name__ == "__main__":
    unittest.main()
