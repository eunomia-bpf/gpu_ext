# SASS-only device-hook admission audit: STOP

Date: 2026-09-04

## Decision

Do **not** run a standalone NVBit-on-SASS experiment as a new gpubpf
experiment. It is technically easy and would likely pass, but it has no
independent decision value for Reviewer A's question: success would prove only
that NVBit can instrument a cubin, while failure would identify an NVBit or
target compatibility issue. Neither outcome establishes or refutes a gpubpf
eBPF-to-SASS attach backend because that backend does not exist in the audited
product path.

This is a paper-value STOP, not an infrastructure blocker. No GPU command was
run and no SASS preflight runner was created.

## Reviewer question and current answer

Reviewer A asked whether device hooks work for workloads shipped as SASS or a
binary without PTX. The submitted response said there was a working
NVBit-based SASS patching prototype. The current source and retained evidence
do not support that statement.

The defensible answer is:

- host-side memory and scheduling policies do not inspect application PTX;
- current gpubpf device hooks require PTX in the target fatbin;
- NVBit demonstrates a plausible NVIDIA SASS instrumentation substrate, but
  the repository has no path from a gpubpf eBPF program to NVBit-inserted SASS;
  and
- therefore the artifact does not support SASS-only device hooks today.

The revised implementation text already adopts this boundary. A standalone
NVBit smoke would not justify restoring the stronger response.

## CPU-only source evidence

### Current gpubpf attach path

The product implementation in the sibling `bpftime-table1-575` checkout:

1. intercepts or discovers CUDA fatbins;
2. extracts embedded PTX with `cuobjdump` or accepts an external PTX directory;
3. lowers eBPF to PTX;
4. patches the target PTX and compiles it with the NVIDIA PTX compiler; and
5. loads the resulting module and binds the patched symbol.

The relevant implementation is in:

- `attach/nv_attach_impl/nv_attach_impl.cpp`;
- `attach/nv_attach_impl/nv_attach_fatbin_record.cpp`; and
- `attach/nv_attach_impl/pass/ptxpass_core/`.

No NVBit or SASS instrumentation implementation is present under the product
attach/runtime source boundary. When a target contains no PTX, the extraction
map is empty; there is no alternate cubin patcher that consumes the eBPF
program.

### Existing NVBit path

The functional NVBit 1.8 adapter under
`workloads/llama.cpp/observability_overhead/revision-rq4/nvbit_adapters/`
uses `nvbit_get_instrs` and `nvbit_insert_call` to add precompiled device
functions at selected SASS instructions. It does not accept eBPF bytecode, run
the device verifier, provide gpubpf helpers/maps, or connect to
`attach/nv_attach_impl`.

The accepted RTX 5090 Table 1 subset confirms that this adapter can collect
exit records and histograms on the PTX-enabled llama.cpp build. That target was
selected so both gpubpf and NVBit could run the same binary; it does not test a
SASS-only gpubpf path.

### A real SASS-only target is already available

A new toy target is unnecessary. CPU-only `cuobjdump` inspection found:

- `workloads/llama.cpp/build-ptx-1b/bin/libggml-cuda.so`: embedded
  `sm_120` PTX and cubins; and
- `workloads/llama.cpp/build/bin/libggml-cuda.so`: `sm_120` cubins and no
  embedded PTX.

Thus the ordinary llama.cpp build is already a realistic SASS-only candidate
for a future preflight. Creating another vector-add binary would add a weaker
fixture, not stronger evidence.

## Paper-value admission test

- **Largest credible story from the proposed NVBit-only run:** NVBit 1.8 can
  observe one SASS-only llama.cpp kernel on RTX 5090.
- **Load-bearing uncertainty:** none for gpubpf. NVBit's binary
  instrumentation capability is its documented purpose and is not the missing
  mechanism.
- **Positive result:** paper must still say gpubpf device hooks require PTX.
- **Negative result:** paper must still say gpubpf device hooks require PTX;
  only the proposed implementation route becomes less attractive.
- **Independent evidence beyond current results:** a target-format control for
  NVBit, not an eBPF mechanism result.
- **Role:** dependency-only, and redundant with the external tool's established
  scope for the paper's current claim.

Because every outcome produces the same paper decision, this proposal fails
admission as a standalone experiment.

## What would make a real experiment admissible

Resume only after a product-side adapter provides all of the following:

1. the same verified eBPF program accepted by the PTX path is consumed by the
   SASS path;
2. an explicit lowering or interpreter ABI connects that program to an
   NVBit-inserted device call;
3. the device call uses the intended gpubpf helper/map semantics rather than a
   hard-coded NVBit counter;
4. attachment records program identity and a device-produced value that cannot
   arise from loader activity alone; and
5. failure to find PTX selects this backend intentionally rather than silently
   running an unrelated NVBit tool.

At that point, the smallest real preflight should reuse the ordinary
SASS-only llama.cpp build and the deterministic TinyLlama correctness input.
It should compare no attachment, PTX gpubpf on the PTX-enabled build, and SASS
gpubpf on the SASS-only build. Required gates are: no PTX in the SASS target;
same eBPF input and observable; explicit attachment and device-event evidence;
exact application-output equality; expected event geometry; no malformed or
dropped records; timeout; and owned cleanup. An independent analyzer should
reject a hard-coded NVBit event, a missing program identity, or a target that
contains PTX.

Only that experiment could support the narrow claim that the named gpubpf
device policy attaches to a SASS-only NVIDIA binary. It would still not be a
cross-vendor portability result.

## Safe paper wording

> Host-side gpubpf policies do not depend on application GPU code. The current
> device-hook implementation extracts and rewrites PTX and therefore does not
> attach to SASS-only binaries. NVBit shows that SASS instrumentation is a
> plausible future backend on NVIDIA GPUs, but our NVBit adapter is an
> observability baseline rather than an eBPF-to-SASS gpubpf backend.

