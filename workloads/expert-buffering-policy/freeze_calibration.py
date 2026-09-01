#!/usr/bin/env python3
"""Freeze eight calibration prompts disjoint from the MoE evaluation workload."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from gguf import GGUFReader
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
GPU_EXT = HERE.parents[1]
MOE_DIR = HERE.parent / "moe-infinity"
sys.path.insert(0, str(MOE_DIR))

import freeze_workload as moe_freeze  # noqa: E402

CALIBRATION_SEED = 1796
NUM_PROMPTS = 8
OUTPUT = HERE / "calibration-prompts.json"
EVALUATION_PROMPTS = MOE_DIR / "prompts.json"


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def freeze_calibration() -> dict[str, Any]:
    dataset = json.loads(moe_freeze.DATASET.read_text(encoding="utf-8"))
    evaluation = json.loads(EVALUATION_PROMPTS.read_text(encoding="utf-8"))
    excluded = {
        int(record["source_index"])
        for record in evaluation["records"]
    }
    tokenizer = AutoTokenizer.from_pretrained(
        moe_freeze.HF_MODEL, local_files_only=True
    )
    gguf_reader = GGUFReader(moe_freeze.GGUF_MODEL, "r")
    gguf_tokens = gguf_reader.get_field("tokenizer.ggml.tokens").contents()
    gguf_bos = int(gguf_reader.get_field("tokenizer.ggml.bos_token_id").contents())
    gguf_eos = int(gguf_reader.get_field("tokenizer.ggml.eos_token_id").contents())
    if tokenizer.bos_token_id != gguf_bos or tokenizer.eos_token_id != gguf_eos:
        raise RuntimeError("HF/GGUF BOS or EOS token ID mismatch")

    rng = np.random.default_rng(CALIBRATION_SEED)
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for index_raw in rng.permutation(len(dataset)):
        index = int(index_raw)
        if index in excluded:
            skipped.append({"index": index, "reason": "evaluation_prompt"})
            continue
        row = dataset[index]
        if not isinstance(row, dict):
            skipped.append({"index": index, "reason": "row_not_object"})
            continue
        source_text = moe_freeze.first_human_text(row)
        if source_text is None:
            skipped.append({"index": index, "reason": "no_human_text"})
            continue
        source_ids = tokenizer.encode(source_text, add_special_tokens=False)
        if len(source_ids) < moe_freeze.PROMPT_TOKENS:
            skipped.append(
                {"index": index, "reason": "short", "tokens": len(source_ids)}
            )
            continue

        prompt_ids = [int(token_id) for token_id in source_ids[:moe_freeze.PROMPT_TOKENS]]
        prompt_text = tokenizer.decode(
            prompt_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if tokenizer.encode(prompt_text, add_special_tokens=False) != prompt_ids:
            skipped.append({"index": index, "reason": "hf_roundtrip"})
            continue
        if moe_freeze.llama_tokenize(prompt_text) != prompt_ids:
            skipped.append({"index": index, "reason": "gguf_roundtrip"})
            continue

        used_ids = sorted(set(prompt_ids))
        mismatched_pieces = [
            token_id
            for token_id in used_ids
            if tokenizer.convert_ids_to_tokens(token_id) != gguf_tokens[token_id]
        ]
        if mismatched_pieces:
            skipped.append(
                {
                    "index": index,
                    "reason": "token_piece_mismatch",
                    "token_ids": mismatched_pieces,
                }
            )
            continue

        records.append(
            {
                "role": "calibration",
                "source_index": index,
                "source_id": row.get("id"),
                "source_text": source_text,
                "source_token_count": len(source_ids),
                "prompt_text": prompt_text,
                "prompt_token_ids": prompt_ids,
                "prompt_token_count": len(prompt_ids),
                "unique_token_ids": len(used_ids),
            }
        )
        if len(records) == NUM_PROMPTS:
            break

    if len(records) != NUM_PROMPTS:
        raise RuntimeError(f"found only {len(records)} eligible calibration prompts")

    value = {
        "schema": 1,
        "dataset": {
            "path": str(moe_freeze.DATASET.relative_to(GPU_EXT)),
            "rows": len(dataset),
            "extraction": "first turn whose from field equals human",
        },
        "selection": {
            "rng": "numpy.random.default_rng",
            "seed": CALIBRATION_SEED,
            "scan_candidates": len(skipped) + len(records),
            "excluded_evaluation_source_indices": sorted(excluded),
            "skipped": skipped,
        },
        "tokenizer": {
            "hf_revision": moe_freeze.HF_REVISION,
            "add_special_tokens": False,
            "skip_special_tokens_on_decode": False,
            "clean_up_tokenization_spaces": False,
            "unicode_normalization": "none",
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "gguf_revision": moe_freeze.GGUF_REVISION,
            "llama_tokenize_flags": ["--no-bos", "--no-parse-special"],
        },
        "records": records,
    }
    write_json(OUTPUT, value)
    print(
        json.dumps(
            {
                "event": "calibration_prompts_frozen",
                "records": len(records),
                "scan_candidates": len(skipped) + len(records),
                "output": str(OUTPUT),
            },
            sort_keys=True,
        )
    )
    return value


if __name__ == "__main__":
    freeze_calibration()
