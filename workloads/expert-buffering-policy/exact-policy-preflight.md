# Exact-model policy preflight

Date: 2026-08-31

Status: **the original hot-LIFO attempt failed; the repaired protection mode
passed the exact-model request**. This is not a performance or correctness
sample.

## Repaired protection-mode smoke

The repair preserves native UVM ordering for cold blocks and requests the used
list tail only for hot/shared blocks, both at activation and on access. This
removes the aggressive cold-head pressure while retaining the mechanism needed
to protect the frozen hot set. The earlier `page` and `hot` modes remain
available unchanged so that the failure is reproducible rather than hidden.

The repaired `protect` mode first passed a live 1 GiB CUDA smoke: all 512
activations matched the class table, 128 hot blocks requested the tail, 384
cold blocks retained native ordering, all 128 hot accesses refreshed the tail,
and the typed setter reported zero failures.

The exact GPT-OSS-120B MXFP4 server was then loaded under the same custom UVM
module and command used below. Its 216 live source layouts compiled into 29,619
spanned two-MiB blocks: 25,741 cold, 3,234 hot, 183 shared, and 461 gaps. The
hot-overlap footprint was 6,817,841,152 bytes. The same 512-input, one-output
request completed successfully in 6.31 seconds and returned HTTP 200, 512
prompt tokens, one completion token, valid text, and
`finish_reason=length`. The final policy counters were:

```text
activate=34545 mapped=33193 hot_tail=4657 cold_head=0
shared_tail=276 default=1352 setter_failure=0 access=288440
cold_native=28260 hot_access_tail=39930 shared_access_tail=1015
```

Disposition: **PASS** for the repaired shortest exact-model execution path.
The server created its context checkpoint normally and reported no CUDA error.
This validates the code repair, not steady-state correctness or performance.
`protect` changes the frozen action table in `plan.md`, so it must be admitted
as a reviewed experimental configuration before the full repeated matrix can
use it.

## Original hot-LIFO failure

The exact GPT-OSS-120B MXFP4 server loaded under the custom UVM module. While
that process remained alive, 216 source layouts were compiled with the frozen
top-ten hot set into 29,619 spanned two-MiB blocks. The resulting live table
classified 25,741 blocks cold, 3,234 hot, and 183 shared; its hot-overlap
footprint was 6,817,841,152 bytes.

The hot policy attached successfully and a 512-input, one-output request drove:

```text
activate=531110 mapped=530667 hot_tail=1958 cold_head=528597
shared_tail=112 default=443 setter_failure=0 access=142698
```

The server then failed at `cudaStreamSynchronize` with `cudaErrorIllegalAddress`
while creating its context checkpoint, and the client received an empty reply.
The request is invalid despite the positive policy counters and zero typed
setter failures.

The owned policy link was detached. A matched control then reloaded the same
model under the same custom UVM module without any struct_ops policy and ran the
same 512-input, one-output request successfully in 6.29 seconds, returning one
token with `finish_reason=length`. This isolates the failure to the policy's
heavy reorder behavior rather than the model or plain custom-UVM path. The
observed 528,597 cold-head decisions indicate immediate LIFO pressure and
severe reactivation churn; this is a diagnosis, not yet proof of the precise
driver fault.

After both attempts, the trace loader and server shut down, no struct_ops map
remained, the custom UVM module was unloaded, the distribution `nvidia_uvm`
module was restored, and the RTX 5090 returned to 15 MiB used memory and 0%
utilization.

No file/content hashes, checksums, digests, or fingerprints were generated or
used.
