# Table 1: recovered llama.cpp source version

2026-09-09: the llama.cpp version is present in the original benchmark JSON,
embedded in each cell's `stdout`. It was not absent from the run records;
the earlier runtime map overlooked this nested field.

| Published cohort | Benchmark records | Reported `build_commit` | `build_number` |
| --- | ---: | --- | ---: |
| [Original seven-arm campaign](../../workloads/llama.cpp/observability_overhead/revision-rq4/results-table1-warp-plt-575-06/cells.json) | 70 | `26836b27` | 7102 |
| [GPU-array extension](../../workloads/llama.cpp/observability_overhead/revision-rq4/results-onevalue-array-bootstrap-575-20260907/cells.json) | 20 | `26836b27` | 7102 |

Root parsed every record in both cohorts; each reports the version shown.
No GPU run or new timing sample was needed. The first five GPU-array pairs
selected by the paper are part of the retained 20-record extension.

From the repository root, this query reproduces the version inventory for
either linked `cells.json` (replace the quoted path):

```sh
jq '[.[] | (.stdout|fromjson)[] | {build_commit,build_number}] |
    group_by([.build_commit,.build_number]) |
    map({build_commit:.[0].build_commit,
         build_number:.[0].build_number,records:length})' "CELLS_JSON"
```

The local llama.cpp repository resolves the abbreviated Git commit to
`26836b27ae1ec9d6e94c6b56306cca75c7e86814`, titled
“Fix UVM warmup-then-migrate strategy for driver 575+”, dated 2026-02-17.
Its origin is `https://github.com/eunomia-bpf/llama.cpp`.

## Existing build configuration: separate supporting information

The current `workloads/llama.cpp/build-ptx-1b/common/build-info.cpp` reports
the same commit/build number and gcc-12. Its current CMake cache specifies:

- Release; C/C++ compilers `/usr/bin/gcc-12` and `/usr/bin/g++-12`.
- CUDA architecture list `120-real;120-virtual`, retaining native code and PTX.
- `GGML_CUDA=ON`, `GGML_CUDA_NO_VMM=ON`, `GGML_CUDA_FA=ON`.
- `GGML_CUDA_FORCE_CUBLAS=OFF`, `GGML_CUDA_FORCE_MMQ=OFF`.
- `GGML_CUDA_GRAPHS=ON` at build time; the recorded Table-1 invocation disables
  graphs through `GGML_CUDA_DISABLE_GRAPHS=1` at runtime.

These are observations of the existing build, not a new clean rebuild or
proof that every historical build flag is recoverable from `build_commit`.
The version string does not describe uncommitted source edits or establish
binary identity. The current llama.cpp source checkout was clean when read.
The remaining task is to build this source/configuration in a fresh location
and document model preparation and execution; source-version discovery alone
does not complete that workflow. See the [runtime map](table1-runtime.md).
