"""Pure CPU checks of diagnostic interpretation; no torch/CUDA import."""
import unittest
import json
import tempfile
from pathlib import Path
from bench import (CTX_NAMES, UNSET, audit_bridge, audit_decisions, shape_order,
                   half_precision_evidence, save_fp32_failure, recompute_saved_fp64,
                   fp32_diagnostic_name, save_cpu_fp64_report, save_numeric_arrays)


def sparse_fallback_case():
    meta = {"prefill_blocks": 1, "decode_blocks": 3, "factor_p": 1,
            "factor_d": 1, "nsmid": 8, "grid_ctas": 4, "fused_op": 9}
    counters = [1, 0, 1, 0, 1, 0, 1, 0, 4, 3]
    records = []
    for i, sm in enumerate((0, 2, 4, 6)):
        record = dict.fromkeys(CTX_NAMES, 0)
        record.update(counters=4096, abi_version=1, nsmid=8, smid=sm,
                      prefill_slots=1, decode_slots=3, proportional=1,
                      grid_ctas=4, status=1, engine=2, ticket=0,
                      first_op=0, first_claim=i, out_op=int(i > 0),
                      out_cta=max(0, i - 1), fallback_claim=i - 1 if i else UNSET)
        records.append(record)
    return meta, counters, records


class BenchAuditTests(unittest.TestCase):
    def test_exact_fp16_neighbours_and_unchanged_tolerance(self):
        positive = half_precision_evidence(1.0, 1.0001)
        self.assertEqual(positive['adjacent_fp16_lower'], 0.99951171875)
        self.assertEqual(positive['adjacent_fp16_upper'], 1.0009765625)
        self.assertTrue(positive['actual_is_nearest_fp16'])
        self.assertEqual(positive['fixed_allowed_error'], 1e-3 + 1e-5 * 1.0001)
        negative = half_precision_evidence(-1.0, -1.0001)
        self.assertEqual(negative['adjacent_fp16_lower'], -1.0009765625)
        self.assertEqual(negative['adjacent_fp16_upper'], -0.99951171875)
        zero = half_precision_evidence(0.0, 0.0)
        self.assertEqual(zero['adjacent_fp16_lower'], -2 ** -24)
        self.assertEqual(zero['adjacent_fp16_upper'], 2 ** -24)
        unavoidable = half_precision_evidence(4.0, 4.001953125)
        self.assertFalse(unavoidable['nearest_fp16_satisfies_fixed_tolerance'])
        with self.assertRaises(ValueError):
            half_precision_evidence(1.0001, 1.0)

    def saved_fixture(self, path, value0, value1, actual):
        import numpy as np
        metadata = dict(atol=1e-3, rtol=1e-5, head_dim=128, effective_keys=2,
                        query_index=1, query_length=8192, valid_kv=8192, causal=True,
                        scale=128 ** -0.5)
        arrays = dict(q=np.zeros(128, dtype=np.float16), k=np.zeros((2, 128), dtype=np.float16),
                      v=np.array([[value0] * 128, [value1] * 128], dtype=np.float16),
                      actual=np.full(128, actual, dtype=np.float16),
                      fp32_reference=np.full(128, (value0 + value1) / 2, dtype=np.float32))
        save_fp32_failure(path, metadata, arrays)
        return metadata, arrays

    def test_cpu_fp64_can_identify_unavoidable_final_fp16_rounding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'failure'
            metadata, arrays = self.saved_fixture(path, 4.0, 4.00390625, 4.0)
            result = recompute_saved_fp64(path)
            self.assertEqual(result['max_abs_fp32_vs_fp64'], 0)
            self.assertEqual(result['actual_exceeding_fixed_tolerance'], 128)
            self.assertEqual(result['nearest_fp16_exceeding_fixed_tolerance'], 128)
            self.assertEqual(result['max_excess_above_best_final_fp16_rounding'], 0)
            with self.assertRaises(FileExistsError):
                save_fp32_failure(path, metadata, arrays)

    def test_cpu_fp64_does_not_blame_nonfinal_error_on_output_rounding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'failure'
            self.saved_fixture(path, 4.0, 4.0, 4.00390625)
            result = recompute_saved_fp64(path)
            self.assertEqual(result['fp64_reference'], 4.0)
            self.assertEqual(result['nearest_fp16_exceeding_fixed_tolerance'], 0)
            self.assertEqual(result['actual_exceeding_fixed_tolerance'], 128)
            self.assertEqual(result['max_excess_above_best_final_fp16_rounding'], 0.00390625)
            self.assertIn('does not isolate', result['limitation'])

    def test_all_shapes_have_distinct_characterization_paths(self):
        names = {fp32_diagnostic_name(model, bs, phase)
                 for model, bs in shape_order(1, False) for phase in ('prefill', 'decode')}
        self.assertEqual(len(names), 20)

    def test_saved_cpu_source_model_and_report_are_explicit_and_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'failure'
            self.saved_fixture(path, 4.0, 4.00390625, 4.0)
            result = save_cpu_fp64_report(path)
            self.assertEqual(result['two_key_source_model']['half_p_fp32_model_matches'], 128)
            self.assertIn('not exact isolation', result['two_key_source_model']['scope'])
            self.assertTrue((path / 'cpu-fp64-report.json').is_file())
            with self.assertRaises(FileExistsError):
                save_cpu_fp64_report(path)

    def test_pair_failure_preserves_official_fp16_without_calling_it_fp32(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata, arrays = self.saved_fixture(root / 'reference', 4.0, 4.0, 4.00390625)
            arrays['official'] = arrays.pop('fp32_reference').astype(np.float16)
            save_numeric_arrays(root / 'pair', metadata, arrays)
            saved = json.loads((root / 'pair/diagnostic.json').read_text())
            self.assertEqual(saved['arrays']['actual']['dtype'], 'float16')
            self.assertEqual(saved['arrays']['official']['dtype'], 'float16')
            self.assertNotIn('fp32_reference', saved['arrays'])

    def test_bridge_actual_launch_and_limit_accounting(self):
        before = dict(launches=5, runtime_redirects=5)
        after = dict(launches=116, runtime_redirects=116, prepared_functions=2,
                     requested_dynamic_bytes=81920, verified_dynamic_bytes=81920,
                     static_shared_bytes=1024, device_optin_bytes=101376)
        self.assertEqual(audit_bridge(before, after, 111, 81920, "cuda")["expected_launches"], 111)
        self.assertEqual(audit_bridge(before, after, 111, 81920, "bpf")["after"], after)
        for field, value in (("launches", 115), ("runtime_redirects", 115),
                             ("prepared_functions", 0), ("requested_dynamic_bytes", 49152),
                             ("verified_dynamic_bytes", 49152), ("device_optin_bytes", 81920)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit_bridge(before, {**after, field: value}, 111, 81920, "cuda")

    def test_sparse_ids_and_real_exhaustion(self):
        result = audit_decisions(*sparse_fallback_case(), engine=2)
        self.assertEqual(result["fallbacks"], 3)
        self.assertEqual(result["observed_sm_ids"], [0, 2, 4, 6])

    def test_reject_wrong_engine(self):
        with self.assertRaises(ValueError):
            audit_decisions(*sparse_fallback_case(), engine=1)

    def test_reject_duplicate_work_even_when_output_fields_agree(self):
        meta, counters, records = sparse_fallback_case()
        records[2]["out_cta"] = records[2]["fallback_claim"] = 0
        with self.assertRaises(ValueError):
            audit_decisions(meta, counters, records, 2)

    def test_reject_missing_atomic_claim(self):
        meta, counters, records = sparse_fallback_case()
        records[2]["first_claim"] = 10
        with self.assertRaises(ValueError):
            audit_decisions(meta, counters, records, 2)

    def test_reject_rule_and_bounds_corruption(self):
        for key, value in (("ticket", 1), ("smid", 8), ("prefill_slots", 2), ("status", 2)):
            meta, counters, records = sparse_fallback_case()
            records[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit_decisions(meta, counters, records, 2)

    def test_fixed_original_shapes_and_shared_order(self):
        expected = {(model, bs) for model in ("llama-3-8b", "yi-6b")
                    for bs in (32, 64, 96, 128, 192)}
        for block in range(1, 6):
            self.assertEqual(set(shape_order(block, False)), expected)
            self.assertEqual(shape_order(block, False), shape_order(block, False))
        self.assertEqual(shape_order(1, True), [("llama-3-8b", 32)])


if __name__ == "__main__":
    unittest.main()
