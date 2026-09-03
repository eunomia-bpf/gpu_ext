"""Real Qwen worker: original HF golden and official FineMoE MoE.generate.

Called by compare.py under the existing project GPU leases, never a simulation.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "deps/FineMoE-EuroSys26"))
sys.path.insert(0, str(HERE.parent / "moe-infinity"))
from run_moe_head_to_head import atomic_write_json

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM
from transformers.generation.streamers import BaseStreamer

from policy_runtime import ARMS, FineMoePolicy


class TokenRecorder(BaseStreamer):
    def __init__(self):
        self.prompt_seen = False
        self.tokens = []
        self.ready_ns = []

    def put(self, value):
        ids = value.detach().cpu().reshape(-1).tolist()
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        if len(ids) != 1:
            raise RuntimeError("only real batch-one token events are supported")
        self.tokens.extend(ids)
        self.ready_ns.append(time.perf_counter_ns())

    def end(self):
        pass


def all_rows(data):
    return data["history"] + data["evaluation"] + data["warmup"]


def validate_data(data):
    if data.get("schema") != "finemoe_mtbench_first_turn_v1":
        raise ValueError("expected the frozen real MT-Bench dataset")
    if [len(data.get(key, [])) for key in ("history", "evaluation", "warmup")] != [64, 8, 1]:
        raise ValueError("64 history / 8 evaluation / 1 warmup are required")
    if data.get("max_input_tokens") != 16 or data.get("generated_tokens") != 16 or data.get("batch_size") != 1:
        raise ValueError("frozen workload dimensions differ")
    seen = []
    for row in all_rows(data):
        if not row["input_ids"] or len(row["input_ids"]) > 16 or row["input_ids"] in seen:
            raise ValueError("empty, oversized, or duplicate tokenized input")
        seen.append(row["input_ids"])
    if data["model"]["repository"] != "Qwen/Qwen1.5-MoE-A2.7B-Chat":
        raise ValueError("do not replace the original model")
    if data["model"]["source_revision"] != "ec052fda178e241c7c443468d2fa1db6618996be" or data["model"]["dtype"] != "bfloat16":
        raise ValueError("frozen full-model source revision / precision differs")


def generate(model, row, include_logits):
    inputs = torch.tensor([row["input_ids"]], dtype=torch.long, device="cuda:0")
    recorder = TokenRecorder()
    begin = time.perf_counter_ns()
    with torch.inference_mode():
        result = model.generate(inputs, attention_mask=torch.ones_like(inputs),
                                min_new_tokens=16, max_new_tokens=16,
                                do_sample=False, pad_token_id=151643,
                                return_dict_in_generate=True, output_logits=include_logits,
                                streamer=recorder)
    sequences = result.sequences.detach().cpu().tolist()
    tokens = sequences[0][len(row["input_ids"]):]
    if len(tokens) != 16 or tokens != recorder.tokens or len(recorder.ready_ns) != 16:
        raise RuntimeError("real generated output / token event counts differ from 16")
    logits = None
    if include_logits:
        logits = torch.stack(result.logits).detach().float().cpu().numpy()
        if logits.shape[0] != 16 or not np.isfinite(logits).all():
            raise RuntimeError("missing/nonfinite real model logits")
    ready = time.perf_counter_ns()
    return {"question_id": row["question_id"], "input_ids": row["input_ids"],
            "generated_ids": tokens, "begin_ns": begin, "verified_ready_ns": ready,
            "token_ready_ns": recorder.ready_ns,
            "ttft_ns": recorder.ready_ns[0] - begin,
            "tpot_ns": (recorder.ready_ns[-1] - recorder.ready_ns[0]) / 15,
            "generation_ns": ready - begin}, logits


def check_result(result, logits, gold, directory, tolerance):
    expected = gold["requests"][str(result["question_id"])]
    if result["input_ids"] != expected["input_ids"] or result["generated_ids"] != expected["generated_ids"]:
        raise RuntimeError(f"exact real-model token mismatch on question {result['question_id']}")
    check = {"exact_token_match": True, "checked_generated_tokens": 16, "logits_checked": False}
    if logits is not None:
        reference = np.load(directory / expected["logits_file"], allow_pickle=False)
        if logits.shape != reference.shape:
            raise RuntimeError("logit shape differs from original model")
        maximum = float(np.max(np.abs(logits - reference)))
        check.update(logits_checked=True, compared_logits=int(logits.size), max_abs_error=maximum)
        if maximum > tolerance:
            raise RuntimeError(f"numerical gate failed: max_abs_error={maximum} > frozen {tolerance}")
    return check


def create_finemoe(data, offload, online):
    from finemoe import MoE
    if importlib.util.find_spec("flash_attn") is not None:
        raise RuntimeError("unexpected optional flash_attn changes the frozen eager attention path")
    return MoE(data["model"]["snapshot"], {
        "offload_path": str(offload), "device_memory_ratio": .5,
        "prefetch_distance": 6, "store_capacity": 1000, "device": "cuda:0",
        "eval_batch_size": 1, "eval_max_length": 16,
        "eval_mode": "online" if online else "offline"})


def worker(args):
    if os.environ.get("FINEMOE_EXCLUSIVE_LEASE") != "1":
        raise RuntimeError("launch through compare.py, which holds the project GPU leases")
    data = json.loads(args.data.read_text())
    validate_data(data)
    from torch._native.common_utils import check_native_jit_disabled
    native_dsl_disabled = check_native_jit_disabled()
    if not native_dsl_disabled:
        raise RuntimeError("common PyTorch native DSL overrides must be disabled before worker startup")
    versions = {"python": sys.version.split()[0], "torch": str(torch.__version__),
                "torch_cuda": torch.version.cuda, "transformers": transformers.__version__, "numpy": np.__version__,
                "torch_native_dsl_jit_disabled": native_dsl_disabled}
    torch.set_num_threads(4)
    torch.manual_seed(data["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if args.stage == "golden":
        model = AutoModelForCausalLM.from_pretrained(
            data["model"]["snapshot"], torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map="cuda:0", local_files_only=True)
        model.eval()
        result = {"backend": "original Transformers Qwen2MoeForCausalLM eager BF16",
                  "data": str(args.data), "model": data["model"], "requests": {}, "runtime_versions": versions,
                  "same_arm_repeat_max_abs_error": 0.0, "repeat_checks": []}
        checked_ids = [r["question_id"] for r in data["evaluation"] + data["warmup"]]
        for row in all_rows(data):
            record, logits = generate(model, row, row["question_id"] in checked_ids)
            if logits is not None:
                filename = f"question-{row['question_id']}-logits.npy"
                np.save(args.output / filename, logits, allow_pickle=False)
                record["logits_file"] = filename
            result["requests"][str(row["question_id"])] = record
            atomic_write_json(args.output / "golden-progress.json", result)
            print(json.dumps({"stage": "golden", "question_id": row["question_id"], "tokens": 16}), flush=True)
        for row in data["evaluation"] + data["warmup"]:
            record, logits = generate(model, row, True)
            original = result["requests"][str(row["question_id"]) ]
            if record["generated_ids"] != original["generated_ids"]:
                raise RuntimeError("original model repeated greedy tokens are not identical")
            previous = np.load(args.output / original["logits_file"], allow_pickle=False)
            result["repeat_checks"].append({"question_id": row["question_id"],
                "generated_ids": record["generated_ids"], "compared_logits": int(logits.size),
                "max_abs_error": float(np.max(np.abs(logits - previous)))})
            result["same_arm_repeat_max_abs_error"] = max(result["same_arm_repeat_max_abs_error"],
                                                         float(np.max(np.abs(logits - previous))))
        # Freeze the repeat-derived tolerance before either policy arm runs.
        result["absolute_tolerance"] = result["same_arm_repeat_max_abs_error"]
        result["status"] = "passed"
        atomic_write_json(args.output / "golden.json", result)
        return

    golden = json.loads((args.golden / "golden.json").read_text())
    if (golden.get("status") != "passed" or golden["model"] != data["model"] or
            golden["runtime_versions"] != versions):
        raise RuntimeError("missing or incompatible real golden")
    from finemoe.ops.prefetch import prefetch_op
    if getattr(prefetch_op, "finemoe_runtime_revision", None) != "dynamic-set-safety-20260903-v2":
        raise RuntimeError("private extension predates the required budget/CV/lifetime repairs")
    model = create_finemoe(data, args.offload, args.stage == "history")
    policy = FineMoePolicy("demand-only" if args.stage == "history" else args.arm,
                          shadow=args.check_logits, capture=args.check_logits)
    policy.install(model.engine)
    if args.stage == "history":
        records = []
        for row in data["history"]:
            record, _ = generate(model, row, False)
            record["correctness"] = check_result(record, None, golden, args.golden, 0)
            records.append(record)
            atomic_write_json(args.output / "history-progress.json", records)
            print(json.dumps({"stage": "history", "question_id": row["question_id"], "tokens": 16}), flush=True)
        model.engine.expert_map_store.export_store_data(str(args.output / "store"))
        if model.engine.expert_map_store.data_size != 1000:
            raise RuntimeError("64 real history requests did not fill the frozen 1000-entry store")
        prefetch_op.finemoe_copy_snapshot()  # Drain before declaring the stored history complete.
        atomic_write_json(args.output / "history.json", {
            "status": "passed", "model": data["model"], "data": str(args.data), "runtime_versions": versions,
            "question_ids": [r["question_id"] for r in data["history"]],
            "store_capacity": 1000, "store_data_size": model.engine.expert_map_store.data_size,
            "requests": records})
        policy.close()
        model.engine.archer_engine.clean_up_resources()
        return

    history = json.loads((args.history / "history.json").read_text())
    if (history.get("status") != "passed" or history["model"] != data["model"] or history["runtime_versions"] != versions or
            history["question_ids"] != [r["question_id"] for r in data["history"]]):
        raise RuntimeError("history model/cohort does not match frozen evaluation")
    model.engine.expert_map_store.import_store_data(str(args.history / "store"))
    if model.engine.expert_map_store.data_size != 1000:
        raise RuntimeError("frozen history store has an unexpected entry count")
    warmup = []
    for row in data["warmup"]:
        record, logits = generate(model, row, args.check_logits)
        record["correctness"] = check_result(record, logits, golden, args.golden, golden["absolute_tolerance"])
        warmup.append(record)
    before = prefetch_op.finemoe_copy_snapshot()
    prefetch_op.finemoe_begin_measurement()
    policy_before = policy.snapshot()["stats"].copy()
    torch.cuda.reset_peak_memory_stats()
    cpu_before = resource.getrusage(resource.RUSAGE_SELF)
    native_started = prefetch_op.finemoe_clock_ns()
    started = time.perf_counter_ns()
    records = []
    for row in data["evaluation"]:
        record, logits = generate(model, row, args.check_logits)
        record["correctness"] = check_result(record, logits, golden, args.golden, golden["absolute_tolerance"])
        record["verified_ready_ns"] = time.perf_counter_ns()
        records.append(record)
        print(json.dumps({"stage": "cell", "arm": args.arm, "question_id": row["question_id"], "tokens": 16}), flush=True)
    finished = time.perf_counter_ns()
    native_finished = prefetch_op.finemoe_clock_ns()
    cpu_after = resource.getrusage(resource.RUSAGE_SELF)
    drain_started = time.perf_counter_ns()
    after = prefetch_op.finemoe_copy_snapshot()  # Outside application timer, before cache teardown.
    drain_finished = time.perf_counter_ns()
    native_drained = prefetch_op.finemoe_clock_ns()
    cpu_drained = resource.getrusage(resource.RUSAGE_SELF)
    policy_after = policy.snapshot()
    # This clock is exactly the ledger's clock, so copies can be partitioned at
    # the application deadline without treating post-window work as in-window.
    elapsed = (native_finished - native_started) / 1e9
    result = {"status": "passed", "arm": args.arm, "check_logits": args.check_logits,
              "model": data["model"], "data": str(args.data), "begin_ns": started,
              "runtime_versions": versions,
              "end_ns": finished, "clock": "perf_counter_ns", "requests": records, "warmup": warmup,
              "application_native_begin_ns": native_started, "application_native_end_ns": native_finished,
              "native_drained_ns": native_drained, "application_clock": "steady_clock",
              "drain_begin_ns": drain_started, "drain_end_ns": drain_finished,
              "drain_seconds": (drain_finished - drain_started) / 1e9,
              "elapsed_seconds_including_drain": (native_drained - native_started) / 1e9,
              "generated_tokens": 128, "elapsed_seconds": elapsed,
              "tokens_per_second": 128 / elapsed,
              "cpu_seconds": (cpu_after.ru_utime + cpu_after.ru_stime - cpu_before.ru_utime - cpu_before.ru_stime),
              "cpu_seconds_including_drain": (cpu_drained.ru_utime + cpu_drained.ru_stime - cpu_before.ru_utime - cpu_before.ru_stime),
              "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
              "before": before, "after": after, "policy_before": policy_before,
              "policy_after": policy_after, "golden_absolute_tolerance": golden["absolute_tolerance"]}
    atomic_write_json(args.output / "worker-result.json", result)
    policy.close()
    model.engine.archer_engine.clean_up_resources()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("golden", "history", "cell"), required=True)
    parser.add_argument("--arm", choices=ARMS, default="demand-only")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offload", type=Path, required=True)
    parser.add_argument("--golden", type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--check-logits", action="store_true")
    args = parser.parse_args()
    worker(args)


if __name__ == "__main__":
    main()
