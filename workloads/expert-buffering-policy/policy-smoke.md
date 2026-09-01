# Expert hot-residency policy smoke

Date: 2026-08-31

This is an implementation smoke, not a model experiment or performance sample.
It checks the shortest live path from a semantic tensor layout to typed NVIDIA
UVM PMM decisions.

## Live result

The CUDA caller allocated 1 GiB of managed memory, registered it as a synthetic
four-expert weight tensor, and explicitly prefetched the allocation to the RTX
5090 after the policy reported ready. The compiled class table contained 512
two-MiB blocks: 128 hot and 384 cold. The attached policy reported:

```text
activate=512 mapped=512 hot_tail=128 cold_head=384
shared_tail=0 default=0 setter_failure=0 access=512
```

Disposition: **PASS**. Every activation was mapped to the semantic class table;
all hot blocks requested the used-list tail, all cold blocks requested the
used-list head, and the typed reorder helper returned no failures. The policy
link detached cleanly afterward.

Earlier 8 MiB and 1 GiB diagnostic attempts that only touched managed pages did
not produce PMM activation callbacks. Those attempts are non-results: explicit
GPU prefetch was needed to force the allocation path in this smoke.

The custom UVM module was then unloaded and the distribution `nvidia_uvm`
module restored. The GPU reported 15 MiB used memory and 0% utilization after
restoration.

This smoke does not validate the exact GPT-OSS-120B layout, hot-expert profile,
correctness, throughput, latency, or the Expert Buffering comparison. Those are
the next experiment stage.

No file/content hashes, checksums, digests, or fingerprints were generated or
used.
