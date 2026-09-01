# Exact-model policy preflight

Date: 2026-08-31

Status: **policy engagement passed; request correctness failed**. This is not a
performance or correctness sample.

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

The trace loader and server shut down, no struct_ops map remained, the custom
UVM module was unloaded, the distribution `nvidia_uvm` module was restored, and
the RTX 5090 returned to 15 MiB used memory and 0% utilization.

No file/content hashes, checksums, digests, or fingerprints were generated or
used.
