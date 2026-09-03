"""Pure CPU checks of diagnostic interpretation; no torch/CUDA import."""
import unittest
from bench import CTX_NAMES, UNSET, audit_bridge, audit_decisions, shape_order


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
