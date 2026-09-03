"""Real EB worker; root launches it under the existing GPU leases.

Generation, official requests, retained logits and numerical comparison come
directly from the frozen FineMoE worker. Importing this file alone is CUDA-free.
"""
import argparse
import ctypes as C
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
FINE = HERE.parents[1] / "finemoe"


def frozen_protocol(source):
    sys.path.insert(0, str(FINE))
    spec = importlib.util.spec_from_file_location("finemoe_frozen_inference", FINE / "inference.py")
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    if any(name == "finemoe" or name.startswith("finemoe.") for name in sys.modules):
        raise RuntimeError("FineMoE imported before selecting the private runtime")
    frozen = str(FINE / "deps/FineMoE-EuroSys26")
    sys.path[:] = [entry for entry in sys.path if entry != frozen]
    sys.path.insert(0, str(source))
    return protocol


def shadow_snapshot(library):
    values = [C.c_uint64() for _ in range(3)]
    function = library.eb_shadow_snapshot
    function.argtypes = [C.POINTER(C.c_uint64)] * 3
    function.restype = C.c_int
    if function(*(C.byref(value) for value in values)) != 0:
        raise RuntimeError("untimed shadow snapshot failed")
    return dict(zip(("checks", "mismatches", "jit_calls"), (v.value for v in values)))


def worker(args):
    numerical = args.check_logits
    if (os.environ.get("FINEMOE_EXCLUSIVE_LEASE") != "1" or
            (numerical and os.environ.get("EB_SECTION_VI_CORRECTNESS_ONLY") != "1") or
            (not numerical and "EB_SECTION_VI_CORRECTNESS_ONLY" in os.environ)):
        raise RuntimeError("launch the matching preflight/full mode through correctness.py")
    if (os.environ.get("EB_SECTION_VI_ARM") != args.arm or
            os.environ.get("EB_SECTION_VI_CAPACITY") != str(args.capacity)):
        raise RuntimeError("worker/engine arm or capacity differs")
    p = frozen_protocol(args.source)
    p.validate_data(data := json.loads(args.data.read_text()))
    torch = p.torch
    from torch._native.common_utils import check_native_jit_disabled
    native_disabled = check_native_jit_disabled()
    if not native_disabled:
        raise RuntimeError("frozen native-DSL compatibility setting missing")
    versions = {"python": sys.version.split()[0], "torch": str(torch.__version__),
                "torch_cuda": torch.version.cuda, "transformers": p.transformers.__version__,
                "numpy": p.np.__version__, "torch_native_dsl_jit_disabled": native_disabled}
    golden = json.loads((args.golden / "golden.json").read_text())
    if (golden.get("status") != "passed" or golden["model"] != data["model"] or
            golden["runtime_versions"] != versions or golden["absolute_tolerance"] != 0.0):
        raise RuntimeError("original HF golden, runtime or frozen zero tolerance differs")
    torch.set_num_threads(4)
    torch.manual_seed(data["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    shadow = None
    if numerical and args.arm == "bpf":
        if os.environ.get("EB_SECTION_VI_UNTIMED_SHADOW") != "1":
            raise RuntimeError("BPF correctness requires actual-input native shadow checks")
        shadow = C.CDLL(os.environ["EB_SECTION_VI_LIBRARY"])
    elif ("EB_SECTION_VI_UNTIMED_SHADOW" in os.environ or
          "EB_SECTION_VI_REAL_LIBRARY" in os.environ or
          os.environ.get("EB_SECTION_VI_LIBRARY") != str(HERE / "build/libeb_policy.so")):
        raise RuntimeError("timed/native/FIFO must directly use the selector without shadow")
    from finemoe.ops.prefetch import prefetch_op
    if (not Path(prefetch_op.__file__).resolve().is_relative_to(args.source) or
            getattr(prefetch_op, "expert_buffering_runtime_revision", None) != "section-vi-private-adapter-v1" or
            getattr(prefetch_op, "finemoe_runtime_revision", None) != "dynamic-set-safety-20260903-v2"):
        raise RuntimeError("wrong private offloader or missing common safety repairs")
    model = p.create_finemoe(data, args.offload, False)
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("finemoe.") and path and not Path(path).resolve().is_relative_to(args.source):
            raise RuntimeError(f"mixed frozen/private runtime: {name}")
    if model.engine.expert_map_store.data_size != 0:
        raise RuntimeError("current-batch EB must not load a trajectory history")
    initial = prefetch_op.expert_buffering_snapshot()
    if any(initial[key] != 0 for key in ("decisions", "jit_calls", "admissions", "evictions")):
        raise RuntimeError("new process started with old EB decisions")
    warmup = []
    for row in data["warmup"]:
        record, logits = p.generate(model, row, numerical)
        record["correctness"] = p.retain_and_check_result(
            record, logits, golden, args.golden, golden["absolute_tolerance"],
            args.output if numerical else None)
        warmup.append(record)
    before = prefetch_op.finemoe_copy_snapshot()
    eb_before = prefetch_op.expert_buffering_snapshot()
    prefetch_op.finemoe_begin_measurement()
    cpu_before = resource.getrusage(resource.RUSAGE_SELF)
    begin_ns = prefetch_op.finemoe_clock_ns()
    started = time.perf_counter_ns()
    records = []
    for row in data["evaluation"]:
        record, logits = p.generate(model, row, numerical)
        record["correctness"] = p.retain_and_check_result(
            record, logits, golden, args.golden, golden["absolute_tolerance"],
            args.output if numerical else None)
        records.append(record)
        print(json.dumps({"stage": "untimed-correctness" if numerical else "timed", "arm": args.arm,
                          "question_id": row["question_id"], "tokens": 16}), flush=True)
    finished = time.perf_counter_ns()
    end_ns = prefetch_op.finemoe_clock_ns()
    cpu_after = resource.getrusage(resource.RUSAGE_SELF)
    drain_started = time.perf_counter_ns()
    after = prefetch_op.finemoe_copy_snapshot()
    drain_finished = time.perf_counter_ns()
    eb_after = prefetch_op.expert_buffering_snapshot()
    drained_ns = prefetch_op.finemoe_clock_ns()
    cpu_drained = resource.getrusage(resource.RUSAGE_SELF)
    checked = shadow_snapshot(shadow) if shadow else None
    if shadow and (checked["mismatches"] != 0 or
                   not (checked["checks"] == checked["jit_calls"] == eb_after["jit_calls"] ==
                        eb_after["decisions"] > 0)):
        raise RuntimeError("live JIT/shadow/decision counts disagree")
    result = {"status": "passed", "arm": args.arm, "capacity": args.capacity,
              "correctness_only": numerical, "check_logits": numerical, "model": data["model"],
              "runtime_versions": versions, "golden_absolute_tolerance": golden["absolute_tolerance"],
              "decoding_configuration": p.decoding_configuration(model.model.generation_config),
              "warmup": warmup, "requests": records, "evaluation_generated_tokens": 128,
              "correctness_generated_tokens": 144,
              "begin_ns": started, "end_ns": finished, "clock": "perf_counter_ns",
              "application_native_begin_ns": begin_ns, "application_native_end_ns": end_ns,
              "native_drained_ns": drained_ns, "application_clock": "steady_clock",
              "drain_begin_ns": drain_started, "drain_end_ns": drain_finished,
              "before": before, "after": after, "eb_before": eb_before, "eb_after": eb_after,
              "shadow": checked, "private_source": str(args.source),
              "private_offloader": str(Path(prefetch_op.__file__).resolve()),
              "selector_library": os.environ["EB_SECTION_VI_LIBRARY"],
              "real_selector_library": os.environ.get("EB_SECTION_VI_REAL_LIBRARY"),
              "completed_ns": time.time_ns()}
    if not numerical:
        elapsed = (end_ns - begin_ns) / 1e9
        result.update(elapsed_seconds=elapsed, tokens_per_second=128 / elapsed,
                      elapsed_seconds_including_drain=(drained_ns - begin_ns) / 1e9,
                      drain_seconds=(drain_finished - drain_started) / 1e9,
                      cpu_seconds=cpu_after.ru_utime + cpu_after.ru_stime - cpu_before.ru_utime - cpu_before.ru_stime,
                      cpu_seconds_including_drain=cpu_drained.ru_utime + cpu_drained.ru_stime -
                          cpu_before.ru_utime - cpu_before.ru_stime)
    model.engine.archer_engine.clean_up_resources()
    result["engine_cleanup_returned"] = True
    p.atomic_write_json(args.output / "worker-result.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "data", "golden", "offload", "output"):
        parser.add_argument(f"--{name}", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--arm", choices=("fifo", "native", "bpf"), required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--check-logits", action="store_true")
    worker(parser.parse_args())
