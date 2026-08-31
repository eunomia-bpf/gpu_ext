#!/usr/bin/env python3
"""Freeze canonical prompts, randomized schedules, and bootstrap indices."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from gguf import GGUFReader
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent
GPU_EXT = ROOT.parents[1]
HF_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
GGUF_REVISION = "238abdd290bb874b90a5da1b4549881b7d05c091"
DATASET_SHA256 = "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"
PROMPT_SCAN_SEED = 1797
SCHEDULE_SEED = 1798
BOOTSTRAP_SEED = 1797
PROMPT_TOKENS = 512
NUM_RECORDS = 9
CONFIGS = [
    "llama_ncmoe32",
    "llama_uvm",
    "gpubpf_host_stride_lfu",
    "moe_infinity_075",
]

DATASET = GPU_EXT / "workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
HF_MODEL = (
    ROOT
    / "deps/hf-cache/hub/models--openai--gpt-oss-120b/snapshots"
    / HF_REVISION
)
GGUF_MODEL = (
    ROOT
    / "deps/hf-cache/hub/models--ggml-org--gpt-oss-120b-GGUF/snapshots"
    / GGUF_REVISION
    / "gpt-oss-120b-MXFP4.gguf"
)
LLAMA_TOKENIZE = GPU_EXT / "workloads/llama.cpp/build/bin/llama-tokenize"
PROMPTS_OUT = ROOT / "prompts.json"
SCHEDULE_OUT = ROOT / "schedule.json"
BOOTSTRAP_OUT = ROOT / "bootstrap-indices.npy"
MANIFEST_OUT = ROOT / "workload-manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def first_human_text(row: dict[str, Any]) -> str | None:
    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        return None
    for turn in conversations:
        if (
            isinstance(turn, dict)
            and turn.get("from") == "human"
            and isinstance(turn.get("value"), str)
        ):
            return turn["value"]
    return None


def llama_tokenize(text: str) -> list[int]:
    with tempfile.NamedTemporaryFile(mode="wb", delete=True) as prompt_file:
        prompt_file.write(text.encode("utf-8"))
        prompt_file.flush()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(
            [
                str(LLAMA_TOKENIZE),
                "--model",
                str(GGUF_MODEL),
                "--ids",
                "--no-bos",
                "--no-parse-special",
                "--log-disable",
                "--file",
                prompt_file.name,
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    parsed = json.loads(result.stdout.strip())
    if not isinstance(parsed, list) or not all(isinstance(x, int) for x in parsed):
        raise RuntimeError("llama-tokenize returned a non-integer token array")
    return parsed


def freeze_prompts() -> dict[str, Any]:
    if sha256_file(DATASET) != DATASET_SHA256:
        raise RuntimeError("ShareGPT dataset hash mismatch")
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    if not isinstance(dataset, list):
        raise RuntimeError("ShareGPT dataset root must be a list")

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, local_files_only=True)
    gguf_reader = GGUFReader(GGUF_MODEL, "r")
    gguf_tokens = gguf_reader.get_field("tokenizer.ggml.tokens").contents()
    gguf_bos = int(gguf_reader.get_field("tokenizer.ggml.bos_token_id").contents())
    gguf_eos = int(gguf_reader.get_field("tokenizer.ggml.eos_token_id").contents())
    if tokenizer.bos_token_id != gguf_bos or tokenizer.eos_token_id != gguf_eos:
        raise RuntimeError("HF/GGUF BOS or EOS token ID mismatch")

    rng = np.random.default_rng(PROMPT_SCAN_SEED)
    scan_order = rng.permutation(len(dataset))
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for index_raw in scan_order:
        index = int(index_raw)
        row = dataset[index]
        if not isinstance(row, dict):
            skipped.append({"index": index, "reason": "row_not_object"})
            continue
        source_text = first_human_text(row)
        if source_text is None:
            skipped.append({"index": index, "reason": "no_human_text"})
            continue
        source_ids = tokenizer.encode(source_text, add_special_tokens=False)
        if len(source_ids) < PROMPT_TOKENS:
            skipped.append(
                {"index": index, "reason": "short", "tokens": len(source_ids)}
            )
            continue

        prompt_ids = [int(token_id) for token_id in source_ids[:PROMPT_TOKENS]]
        prompt_text = tokenizer.decode(
            prompt_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        hf_roundtrip = tokenizer.encode(prompt_text, add_special_tokens=False)
        if hf_roundtrip != prompt_ids:
            skipped.append({"index": index, "reason": "hf_roundtrip"})
            continue
        llama_ids = llama_tokenize(prompt_text)
        if llama_ids != prompt_ids:
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

        source_bytes = source_text.encode("utf-8")
        prompt_bytes = prompt_text.encode("utf-8")
        ids_bytes = json.dumps(prompt_ids, separators=(",", ":")).encode("ascii")
        records.append(
            {
                "role": "warmup" if not records else "measured",
                "source_index": index,
                "source_id": row.get("id"),
                "source_text": source_text,
                "source_text_sha256": sha256_bytes(source_bytes),
                "source_token_count": len(source_ids),
                "prompt_text": prompt_text,
                "prompt_text_sha256": sha256_bytes(prompt_bytes),
                "prompt_token_ids": prompt_ids,
                "prompt_token_ids_sha256": sha256_bytes(ids_bytes),
                "prompt_token_count": len(prompt_ids),
                "unique_token_ids": len(used_ids),
            }
        )
        if len(records) == NUM_RECORDS:
            break

    if len(records) != NUM_RECORDS:
        raise RuntimeError(f"found only {len(records)} eligible prompt records")

    value = {
        "schema": 1,
        "dataset": {
            "path": str(DATASET.relative_to(GPU_EXT)),
            "sha256": DATASET_SHA256,
            "rows": len(dataset),
            "extraction": "first turn whose from field equals human",
        },
        "selection": {
            "rng": "numpy.random.default_rng",
            "seed": PROMPT_SCAN_SEED,
            "scan_candidates": len(skipped) + len(records),
            "skipped": skipped,
        },
        "tokenizer": {
            "hf_revision": HF_REVISION,
            "add_special_tokens": False,
            "skip_special_tokens_on_decode": False,
            "clean_up_tokenization_spaces": False,
            "unicode_normalization": "none",
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "gguf_revision": GGUF_REVISION,
            "llama_tokenize_flags": ["--no-bos", "--no-parse-special"],
        },
        "records": records,
    }
    write_json(PROMPTS_OUT, value)
    return value


def freeze_schedule() -> dict[str, Any]:
    rng = np.random.default_rng(SCHEDULE_SEED)
    attempts = []
    measured_prompt_ids = list(range(1, NUM_RECORDS))
    for attempt in range(1, 9):
        attempts.append(
            {
                "attempt": attempt,
                "configuration_order": [
                    CONFIGS[int(index)] for index in rng.permutation(len(CONFIGS))
                ],
                "prompt_order": [
                    int(index) for index in rng.permutation(measured_prompt_ids)
                ],
            }
        )
    value = {
        "schema": 1,
        "rng": "numpy.random.default_rng",
        "seed": SCHEDULE_SEED,
        "stop": "immediately after five valid complete blocks or attempt eight",
        "attempts": attempts,
    }
    write_json(SCHEDULE_OUT, value)
    return value


def freeze_bootstrap() -> np.ndarray:
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0,
        5,
        size=(10000, 5),
        endpoint=False,
        dtype=np.int64,
    )
    np.save(BOOTSTRAP_OUT, indices, allow_pickle=False)
    return indices


def main() -> None:
    prompts = freeze_prompts()
    schedule = freeze_schedule()
    indices = freeze_bootstrap()
    manifest = {
        "schema": 1,
        "generator": {
            "path": Path(__file__).name,
            "sha256": sha256_file(Path(__file__)),
            "numpy": np.__version__,
        },
        "prompts": {
            "path": PROMPTS_OUT.name,
            "sha256": sha256_file(PROMPTS_OUT),
            "records": len(prompts["records"]),
        },
        "schedule": {
            "path": SCHEDULE_OUT.name,
            "sha256": sha256_file(SCHEDULE_OUT),
            "attempts": len(schedule["attempts"]),
        },
        "bootstrap": {
            "path": BOOTSTRAP_OUT.name,
            "sha256": sha256_file(BOOTSTRAP_OUT),
            "shape": list(indices.shape),
            "dtype": str(indices.dtype),
            "seed": BOOTSTRAP_SEED,
            "api": "np.random.default_rng(seed).integers(0,5,(10000,5),endpoint=False,dtype=np.int64)",
        },
        "inputs": {
            "dataset_sha256": sha256_file(DATASET),
            "hf_tokenizer_json_sha256": sha256_file(HF_MODEL / "tokenizer.json"),
            "gguf_sha256": sha256_file(GGUF_MODEL),
            "llama_tokenize_sha256": sha256_file(LLAMA_TOKENIZE),
        },
    }
    write_json(MANIFEST_OUT, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
