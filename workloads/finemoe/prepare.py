"""Freeze real public MT-Bench inputs on CPU; does not instantiate a GPU model."""
import argparse
import json
from pathlib import Path
import random

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
MT_REVISION = "b494d0c6b4e7935f1764f8439e75da3e66beccc7"
MT_SOURCE = HERE / "deps/FastChat-mt-bench/question.jsonl"


def make_dataset(tokenizer, questions, seed=20260903):
    if len(questions) != 80:
        raise ValueError("expected the official 80 MT-Bench questions")
    ids = [row["question_id"] for row in questions]
    if len(ids) != len(list(dict.fromkeys(ids))):
        raise ValueError("duplicate source question ID")
    encoded = tokenizer([row["turns"][0] for row in questions],
                        truncation=True, max_length=16, add_special_tokens=True)["input_ids"]
    unique, token_rows = [], []
    for row, tokens in zip(questions, encoded):
        if not tokens or len(tokens) > 16:
            raise ValueError("unexpected input length")
        if tokens in token_rows:
            continue
        token_rows.append(tokens)
        unique.append({"question_id": row["question_id"], "category": row["category"],
                       "input_ids": tokens})
    if len(unique) < 73:
        raise ValueError("fewer than 64+8+1 unique truncated inputs; do not shrink cohort")
    random.Random(seed).shuffle(unique)
    return {"schema": "finemoe_mtbench_first_turn_v1", "seed": seed,
            "max_input_tokens": 16, "generated_tokens": 16, "batch_size": 1,
            "source_questions": len(questions), "unique_tokenized_inputs": len(unique),
            "history": unique[:64], "evaluation": unique[64:72], "warmup": unique[72:73],
            "unused": unique[73:],
            "deviation": "MT-Bench first-turn prompts replace gated LMSYS-Chat-1M; neither its distribution nor MT-Bench answer quality is reproduced"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    inventory = json.loads((HERE / "source-inventory.json").read_text())
    model = inventory["model"]
    tokenizer = AutoTokenizer.from_pretrained(model["snapshot"], local_files_only=True)
    questions = [json.loads(line) for line in MT_SOURCE.read_text().splitlines()]
    result = make_dataset(tokenizer, questions, args.seed)
    result["dataset"] = {"name": "MT-Bench first turn", "repository": "lm-sys/FastChat",
                         "source_revision": MT_REVISION, "source_file": str(MT_SOURCE),
                         "bytes": MT_SOURCE.stat().st_size,
                         "url": f"https://github.com/lm-sys/FastChat/blob/{MT_REVISION}/fastchat/llm_judge/data/mt_bench/question.jsonl"}
    result["model"] = {key: model[key] for key in ("repository", "source_revision", "snapshot", "dtype")}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "unique_inputs": len(result["history"]) +
                      len(result["evaluation"]) + len(result["warmup"]) + len(result["unused"]),
                      "history": [r["question_id"] for r in result["history"]],
                      "evaluation": [r["question_id"] for r in result["evaluation"]],
                      "warmup": [r["question_id"] for r in result["warmup"]]}), flush=True)


if __name__ == "__main__":
    main()
