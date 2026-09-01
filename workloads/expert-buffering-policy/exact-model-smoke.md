# Exact-model marker smoke

Date: 2026-08-31

This is an implementation smoke, not calibration and not a performance sample.
It uses the exact GPT-OSS-120B MXFP4 GGUF with `--n-cpu-moe 36`, a 104-token
prompt, and one generated token to exercise both device-offloaded and CPU expert
operations.

## Result

The marker-enabled llama.cpp build completed the request on the RTX 5090. The
trace contained:

- 216 source-model expert layouts: 108 weights, 108 biases, all 36 layers, and
  128 experts per tensor;
- 105 runtime CUDA expert-copy layouts, which have `CUDA0#...` names and are
  intentionally outside the strict source-layout parser grammar;
- one graph ordinal;
- 2,916 route events assigned to that graph; and
- 108 distinct routed source weight bases covering all three expert weight
  tensors in all 36 layers.

The initial short-prompt diagnostic produced no routes because its batch was
below the CUDA backend's 32-row operation-offload threshold. A subsequent
104-token run exposed that layers 0--34 used the selected-expert device-copy
path while layer 35 executed `MUL_MAT_ID` on the CPU. The CPU operator already
iterates the selected IDs, so the marker was added at that existing loop. It
adds no ID copy or synchronization and emits each distinct selected expert once
per expert-weight operation. The rerun then covered 36/36 layers.

Disposition: **PASS** for exact-model automatic source-layout registration,
graph assignment, route observation across device and CPU paths, request
completion, and clean teardown. The GPU returned to 15 MiB used memory and 0%
utilization.

The frozen eight-prompt calibration, top-ten hot-set construction, exact-output
correctness, PMM engagement on the exact model, and all performance cells remain
future stages.

No file/content hashes, checksums, digests, or fingerprints were generated or
used.
