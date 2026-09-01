#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from compile_layout import (
    BLOCK_BYTES,
    CLASS_COLD,
    CLASS_HOT,
    CLASS_SHARED,
    Registration,
    compile_layout,
    load_registrations,
    load_hot_set,
    write_layout,
)


def registration(
    name: str,
    base: int,
    total: int,
    stride: int,
    *,
    layer: int = 0,
    kind: str = "gate",
    bias: bool = False,
    experts: int = 4,
) -> Registration:
    return Registration(
        tgid=17,
        name=name,
        base=base,
        total_bytes=total,
        per_expert_bytes=stride,
        n_experts=experts,
        is_bias=bias,
        layer=layer,
        kind=kind,
    )


class CompileLayoutTests(unittest.TestCase):
    def test_mixed_expert_block_is_hot_if_any_overlap_is_hot(self) -> None:
        stride = 4_406_400
        item = registration(
            "blk.0.ffn_gate_exps.weight",
            8 * BLOCK_BYTES,
            stride * 4,
            stride,
        )
        layout = compile_layout([item], {0: {0}})
        self.assertEqual(layout.classes[0:3], (CLASS_HOT, CLASS_HOT, CLASS_HOT))
        self.assertEqual(layout.classes[3], CLASS_COLD)
        self.assertGreater(layout.classes.count(CLASS_COLD), 0)

    def test_tensor_boundary_is_shared(self) -> None:
        base = 16 * BLOCK_BYTES
        left = registration(
            "blk.0.ffn_gate_exps.weight", base, BLOCK_BYTES + 256, 524_352
        )
        right = registration(
            "blk.0.ffn_up_exps.weight",
            base + BLOCK_BYTES + 256,
            BLOCK_BYTES,
            524_288,
            kind="up",
        )
        layout = compile_layout([left, right], {0: {0}})
        self.assertEqual(layout.classes[1], CLASS_SHARED)

    def test_bias_blocks_are_shared(self) -> None:
        item = registration(
            "blk.0.ffn_gate_exps.bias",
            32 * BLOCK_BYTES,
            BLOCK_BYTES,
            BLOCK_BYTES // 4,
            bias=True,
        )
        layout = compile_layout([item], {0: {0}})
        self.assertEqual(layout.classes, (CLASS_SHARED,))

    def test_hot_set_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hot.txt"
            path.write_text("0 7\n0 7\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_hot_set(path)

    def test_writer_emits_only_classified_blocks(self) -> None:
        item = registration(
            "blk.0.ffn_gate_exps.weight",
            48 * BLOCK_BYTES,
            BLOCK_BYTES,
            BLOCK_BYTES // 4,
        )
        layout = compile_layout([item], {0: {0}})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "classes.txt"
            write_layout(output, layout)
            lines = output.read_text(encoding="utf-8").splitlines()
        self.assertTrue(lines[0].startswith("base "))
        self.assertEqual(lines[1], "0 2")

    def test_runtime_copy_layout_is_ignored(self) -> None:
        records = (
            '{"event":"layout","tgid":17,'
            '"name":"CUDA0#blk.0.ffn_gate_exps.weight#0",'
            '"base":1048576,"total_bytes":4096,"per_expert_bytes":1024,'
            '"n_experts":4,"is_bias":0}\n'
            '{"event":"layout","tgid":17,'
            '"name":"blk.0.ffn_gate_exps.weight",'
            '"base":2097152,"total_bytes":4096,"per_expert_bytes":1024,'
            '"n_experts":4,"is_bias":0}\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(records, encoding="utf-8")
            loaded = load_registrations(path, None)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].name, "blk.0.ffn_gate_exps.weight")


if __name__ == "__main__":
    unittest.main()
