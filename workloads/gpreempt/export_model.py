#!/usr/bin/env python3
"""Export original TVM-testing workloads; actual export requires an idle GPU lease."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
TVM_REVISION = "513c2be0c3b853a3b77de729f0ea75d448ee3c37"


def model_spec(name: str, arch: int) -> dict:
    return {"model": name, "architecture": f"sm_{arch}", "tvm_revision": TVM_REVISION,
            "generator": "tvm.relay.testing", "layers": 19 if name == "vgg" else 152,
            "input_shape": [1, 3, 224, 224], "output_shape": [1, 1000],
            "dtype": "float32", "parameter_seed": 0, "pretrained_weights": False,
            "input_formula": "((element_index % 257) - 128) / 128.0",
            "host_metadata": "original GPreempt recorder after one real CUDA execution",
            "reference_kind": "isolated native TVM CUDA execution; not a model-accuracy claim"}


def dump_params(params, stream) -> None:
    stream.write(b"TVM_MODEL_PARAMS\0")
    stream.write(struct.pack("<Q", len(params)))
    for name, tensor in params.items():
        encoded = name.encode("ascii")
        if len(encoded) >= 256 or b"\0" in encoded:
            raise ValueError("parameter name exceeds the original executor's supported format")
        values = tensor.numpy()
        if str(values.dtype) != "float32":
            raise ValueError("original parameter format supports only float32")
        stream.write(encoded + b"\0")
        stream.write(struct.pack("<Q", values.size))
        stream.write(values.astype("<f4", copy=False).tobytes(order="C"))


def worker(args) -> None:
    # Delayed imports keep --help/--plan and parser tests strictly CPU-only.
    import numpy as np
    import tvm
    from tvm import relay
    from tvm.relay import testing
    from tvm.contrib import graph_executor, nvcc  # Register nvcc, not probing NVRTC fallback.

    if not tvm.runtime.enabled("cuda"):
        raise RuntimeError("the pinned, recorder-patched TVM must be built with CUDA")
    spec = model_spec(args.model, args.arch)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "model-spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    factory = testing.vgg.get_workload if args.model == "vgg" else testing.resnet.get_workload
    mod, params = factory(num_layers=spec["layers"], batch_size=1, image_shape=(3, 224, 224))
    target = tvm.target.Target(f"cuda -arch=sm_{args.arch}", host="llvm")
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    cuda_module = lib.get_lib().imported_modules[0]
    source = cuda_module.get_source("cu")
    if "__global__" not in source:
        raise RuntimeError("TVM did not preserve CUDA source; refusing to treat PTX as .cu")
    (args.output / "mod.cu").write_text(source)
    (args.output / "mod.json").write_text(lib.get_graph_json())
    with (args.output / "mod.params").open("xb") as stream:
        dump_params(lib.get_params(), stream)
    subprocess.run(["/usr/local/cuda-12.9/bin/nvcc", f"-arch=sm_{args.arch}",
                    "-ccbin", "/usr/bin/g++-13", "-O3", "--cubin",
                    str(args.output / "mod.cu"), "-o", str(args.output / "mod.cubin")], check=True)
    device = tvm.cuda(0)
    module = graph_executor.GraphModule(lib["default"](device))
    elements = np.arange(np.prod(spec["input_shape"]), dtype=np.int64)
    data = ((elements % 257 - 128).astype(np.float32) / np.float32(128)).reshape(spec["input_shape"])
    module.set_input("data", data)
    module.run()
    device.sync()
    host = json.loads(module.module["get_host_json"]())
    kernels = [kernel for function in host.get("funcs", []) for kernel in function.get("kernels", [])]
    if not kernels or any(len(kernel.get("launch_params", [])) != 6 for kernel in kernels):
        raise RuntimeError("missing real kernel launch metadata from original TVM recorder")
    (args.output / "host.json").write_text(json.dumps(host, indent=2) + "\n")
    reference = module.get_output(0).numpy()
    if (list(reference.shape) != spec["output_shape"] or str(reference.dtype) != "float32"
            or not np.isfinite(reference).all()):
        raise RuntimeError("isolated native output is nonfinite or has an unexpected shape")
    with (args.output / "reference.f32").open("xb") as stream:
        stream.write(reference.astype("<f4", copy=False).tobytes(order="C"))
    spec["recorded_kernels"] = len(kernels)
    spec["reference_elements"] = int(reference.size)
    spec["files"] = {path.name: path.stat().st_size for path in args.output.iterdir()
                     if path.is_file() and path.name != "model-spec.json"}
    (args.output / "model-spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(json.dumps({"status": "exported", "model": args.model, "kernels": len(kernels),
                      "performance_reproduction": False}), flush=True)


def run(args) -> None:
    import run_smoke
    safety = run_smoke.safety
    raw = args.evidence_output or args.output.parent / (args.output.name + "-export-evidence")
    raw.mkdir(parents=True, exist_ok=False)
    lease = None
    child = None
    before = None
    result = {"status": "failed", "model": args.model, "timeout_seconds": args.timeout,
              "performance_reproduction": False}
    try:
        lease = safety.LeaseSet.acquire()
        before = safety.safety_snapshot()
        safety.validate_pre_server_safety(before)
        env = {"PATH": "/usr/local/cuda-12.9/bin:/usr/bin:/bin", "LANG": "C.UTF-8",
               "CUDA_VISIBLE_DEVICES": "0", "CUDA_PATH": "/usr/local/cuda-12.9",
               "PYTHONPATH": str(HERE / "deps/tvm/python"), "TVM_LIBRARY_PATH": str(HERE / "build/tvm"),
               "LD_LIBRARY_PATH": f"{HERE / 'build/tvm'}:/usr/local/cuda-12.9/lib64",
               "TVM_NUM_THREADS": "4", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
               "PYTHONDONTWRITEBYTECODE": "1"}
        command = ["taskset", "-c", "8-15", str(HERE / "deps/tvm-venv/bin/python"),
                   str(Path(__file__).resolve()), "--worker", "--model", args.model,
                   "--arch", str(args.arch), "--output", str(args.output)]
        with (raw / "export.log").open("x") as stream:
            child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            result["returncode"] = child.wait(timeout=args.timeout)
        if result["returncode"]:
            raise RuntimeError("model export failed; preserve partial assets and export.log")
        result["status"] = "passed"
    except BaseException as exc:
        result["error"] = str(exc)
        raise
    finally:
        try:
            if child is not None:
                safety.stop_owned_process_group(child)
            if before is not None:
                result["safety_after"] = safety.wait_for_post_server_safety(before)
        except BaseException as exc:
            result.update(status="failed", cleanup_error=str(exc))
            raise
        finally:
            safety.atomic_write_json(raw / "result.json", result)
            if lease is not None:
                lease.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["vgg", "resnet152"], required=True)
    parser.add_argument("--arch", type=int, default=120)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evidence-output", type=Path,
                        help="new directory for retry logs; never overwrites an earlier attempt")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--plan", action="store_true", help="print explicit model choices without loading TVM/CUDA")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.arch <= 0 or not 1 <= args.timeout <= 3600:
        parser.error("architecture must be positive and timeout must be 1–3600 seconds")
    if args.plan:
        print(json.dumps(model_spec(args.model, args.arch), indent=2))
        return
    if args.output is None:
        parser.error("--output is required for actual export")
    args.output = args.output.resolve()
    if args.evidence_output:
        args.evidence_output = args.evidence_output.resolve()
    def interrupted(signum, _frame):
        raise InterruptedError(f"signal {signum}; cleaning up owned model export")
    signal.signal(signal.SIGTERM, interrupted)
    (worker if args.worker else run)(args)


if __name__ == "__main__":
    main()
