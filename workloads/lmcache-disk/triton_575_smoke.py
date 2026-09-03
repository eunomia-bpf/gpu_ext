#!/usr/bin/env python3
"""One finite Triton load/launch diagnostic; not a model or performance test.

Run this file unchanged with the default compiler and then with the explicit
TRITON_PTXAS_BLACKWELL_PATH override. The coordinator owns GPU leases, safety
checks, process timeout, log capture, and a fresh cache directory for each run.
"""

import importlib.metadata
import os
import sys


def report(message: str) -> None:
    print(message, flush=True)


report(f"python={sys.executable}")
report(
    f"engine=torch {importlib.metadata.version('torch')}; "
    f"triton {importlib.metadata.version('triton')}"
)
for variable in ("TRITON_PTXAS_PATH", "TRITON_PTXAS_BLACKWELL_PATH"):
    report(f"requested_{variable}={os.environ.get(variable, '<unset>')}")

import torch
import triton
import triton.language as tl
from triton.backends.nvidia.compiler import get_ptxas


@triton.jit
def vector_add(x, y, output, count: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < count
    result = tl.load(x + offsets, mask=mask) + tl.load(y + offsets, mask=mask)
    tl.store(output + offsets, result, mask=mask)


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report(f"imports_complete; torch_cuda_build={torch.version.cuda}")
    device = torch.device("cuda:0")
    major, minor = torch.cuda.get_device_capability(device)
    arch = major * 10 + minor
    compiler = get_ptxas(arch)
    report(f"device={torch.cuda.get_device_name(device)}; arch=sm_{arch}")
    report(f"selected_compiler={compiler.path}; compiler_release={compiler.version}")

    count = 4096
    # Integer-valued FP32 inputs and sums are exact throughout this range.
    x_cpu = torch.arange(count, dtype=torch.float32)
    y_cpu = 2 * x_cpu + 1
    expected = x_cpu + y_cpu
    x = x_cpu.to(device)
    y = y_cpu.to(device)
    output = torch.empty(count, device=device, dtype=torch.float32)
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    report(
        f"inputs_ready; elements={count}; tensor_bytes={3 * count * 4}; "
        f"device_free_bytes={free_bytes}; device_total_bytes={total_bytes}"
    )

    report("begin_one_triton_compile_load_launch")
    vector_add[(triton.cdiv(count, 256),)](
        x, y, output, count, 256, num_warps=4, num_stages=1
    )
    torch.cuda.synchronize(device)
    actual = output.cpu()
    if not torch.equal(actual, expected):
        mismatches = torch.count_nonzero(actual != expected).item()
        raise AssertionError(f"exact output mismatch: {mismatches}/{count}")
    report(f"PASS: one Triton launch; all {count} FP32 outputs exactly match CPU")


if __name__ == "__main__":
    main()
