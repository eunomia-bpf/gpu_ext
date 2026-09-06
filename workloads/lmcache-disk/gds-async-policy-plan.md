# gpubpf-Controlled Asynchronous GPU Storage

## Decision boundary

Place the policy decision immediately before a trusted LMCache/cuFile executor
submits an asynchronous transfer. A new host-side `gpu_storage_decide`
struct_ops callback receives only scalar policy inputs: read/write operation,
byte count, opaque object/request identity, caller identity, priority,
deadline/slack, estimated transfer and recompute cost, queue depth, and
GPU-memory pressure. It never receives a file descriptor, file offset, GPU
pointer, CUDA stream, or another DMA capability.

The callback may request `SUBMIT_NOW`, bounded `DEFER`, or `RECOMPUTE` for a
read, together with bounded priority and batch hints. The UVM bridge validates
the request. An absent policy, unknown action, excessive delay or batch, or a
write-side recompute request becomes `SUBMIT_NOW`. LMCache/cuFile retains file,
buffer, stream, completion, and error ownership. Deferred requests stay in a
trusted per-GPU deadline queue and are released through cuFile stream or batch
submission. Completion updates queue and transfer estimates used by later
decisions.

Use LMCache MP mode's `GDSContext.transfer_async()` as the first real adapter
point. It already reaches `cuFileReadAsync` and `cuFileWriteAsync` and retains
each submission until its CUDA-stream completion event. Call
`gpu_storage_decide` once per logical KV chunk, before `transfer_async()`
splits the chunk into at most 16 MiB registered-buffer regions. A decision per
region would be incorrect because a 24 MiB KV chunk could otherwise be only
partially submitted.

The trusted adapter marks whether a request is safe to defer. Background
writes and speculative prefetches run on owned storage streams and may enter a
deadline queue. A demand read on an application stream is never deferred: the
bridge reduces `DEFER` to `SUBMIT_NOW`, unless the caller also marked the read
as recomputable, in which case `RECOMPUTE` returns a cache miss to LMCache's
request layer. This preserves CUDA stream ordering while still allowing the
policy to schedule asynchronous storage work.

```
SSD_ONLY -> READ_QUEUED -> READING -> GPU_READY
GPU_DIRTY -> WRITE_QUEUED -> WRITING -> SSD_DURABLE
READ_QUEUED -> RECOMPUTE_SELECTED -> GPU_READY
```

## Policies

The first policy port is slack-aware read/write decoupling: urgent reads bypass
background writes, while slack-rich writes are coalesced. The gpubpf-specific
extension combines that policy with live HBM pressure and fetch-versus-
recompute cost. Under high pressure it submits enough dirty writes to make KV
objects durable and evictable, suppresses speculative reads, and still submits
or recomputes demand reads before their deadlines. This is a cross-layer
storage-placement policy; it does not claim to reproduce a GPU-native I/O
transport.

Use a separate `gpu_storage_ops` struct_ops type rather than adding the hook to
`gpu_mem_ops`. Storage scheduling and UVM residency policies can then be
attached simultaneously. The ioctl bridge fills the caller process identity;
application-provided tenant, cost, deadline, and pressure values remain policy
hints and never become DMA capabilities. The BPF callback records its requested
action through a kfunc, after which trusted code bounds delay, priority, and
batch size. The adapter alone retains the file handle, slab offset, GPU buffer,
stream, and completion object.

The matched native implementation evaluates the same deadline/pressure/cost
formula in the adapter. The BPF arm evaluates it through the ioctl and
`gpu_storage_ops`; both feed the same trusted deadline queue and cuFile
executor. This makes their difference the mechanism cost rather than a change
in transport or policy.

## Concrete control flow

Keep the kernel boundary synchronous and the storage work asynchronous. The
LMCache adapter constructs one versioned scalar request for a logical KV chunk
and invokes the UVM decision ioctl. The ioctl calls the attached
`gpu_storage_ops.decide` callback and immediately returns a validated reply.
It never submits storage I/O from kernel context. The trusted adapter then
places the request in one of two execution lanes:

1. **Ordered demand lane.** Demand reads use `cuFileReadAsync` on the owned
   CUDA storage stream. They preserve stream ordering and may be changed to
   recomputation, but an unsafe `DEFER` reply is reduced to `SUBMIT_NOW`.
2. **Independent background lane.** Durable writes and speculative reads are
   held in a per-GPU deadline queue. The adapter releases them as cuFile batch
   submissions when the requested batch is full, the oldest deadline expires,
   or an urgent read needs the device. This lane is where BPF can reorder,
   defer, and coalesce transfers without violating application-stream order.

The first ABI needs four operations, not storage capabilities:

- `DECIDE(request) -> reply`: choose action, bounded delay, priority, and batch
  target for one logical chunk;
- `COMPLETE(request_id, bytes, latency, status)`: update scalar service-time
  and queue estimates after the trusted executor observes completion;
- `CANCEL(request_id)`: remove a request that LMCache no longer needs; and
- `QUERY_CAPS`: report ABI version and bounded limits to the adapter.

The trusted side validates the request size/version and reply bounds. Missing
policy, malformed reply, timeout, unsupported recomputation, and write-side
`RECOMPUTE` all fall back to `SUBMIT_NOW`. The BPF program sees opaque object
and request IDs but never the file handle, file offset, GPU address, CUDA
stream, completion event, or batch handle.

Completion telemetry is deliberately aggregate: exponentially weighted read
and write service time, current queue bytes/depth, deadline misses, and recent
HBM pressure. This is enough to make scheduling decisions without turning BPF
maps into a second request queue. LMCache remains the sole owner of request
lifetime and buffer reuse.

After the all-or-none path runs, add a bounded hybrid-restore policy inspired
by KVPR rather than another transport. For each read, evaluate only five legal
splits (fetch 0%, 25%, 50%, 75%, or 100% of aligned KV subchunks). For split
`p`, estimate completion as
`max(queue_delay + p * full_fetch_time, (1 - p) * full_recompute_time)` and
choose the minimum that meets the request deadline. Return aligned
`fetch_bytes` and `recompute_tokens`; the trusted LMCache runtime performs both
operations and their synchronization. Native and BPF implementations use the
same five-candidate calculation. Adding HBM pressure to suppress speculative
fetches and accelerate dirty writes then tests a gpubpf-specific cross-layer
extension without claiming that BPF implements the storage transport.

## Comparison

Use three matched storage-control arms over the same LMCache/cuFile executor:

1. plain FIFO LMCache/cuFile submission;
2. the deadline/pressure/recompute policy implemented natively; and
3. the identical policy executed through `gpu_storage_decide`.

Measure representative 24 MiB KV objects in mixed urgent-read/background-write
traffic, then the end-to-end LMCache workload. Report read tail latency, TTFT,
request and token throughput, storage bandwidth, queue delay, batch size,
defer/recompute counts, and native-versus-BPF policy cost. Record direct
P2PDMA, cuFile compatibility, and POSIX execution as separate transport labels;
none suppresses performance collection.

Run the transport microbenchmark as a 3-by-2 comparison: FIFO, matched native,
and BPF control, each with the ordered-stream executor and the mixed
stream-plus-batch executor. The first publishable result is not raw peak NVMe
bandwidth; it is whether scheduling background writes away from urgent reads
reduces read p99/TTFT while preserving aggregate bandwidth, followed by the
native-versus-BPF delta under the identical policy. The end-to-end five-arm
LMCache experiment then adds recompute, CPU tier, and ordinary disk baselines.

## Current RTX 5090 path

The host uses Linux 6.15.11, NVIDIA OpenRM 575.57.08, CUDA 12.9, an RTX 5090,
and local ext4 NVMe storage. cuFile and its tools are installed, and `gdscheck`
reports that the GPU supports GDS. The current platform report nevertheless
marks NVMe and NVMe P2PDMA unsupported and enables compatibility mode. Initial
plain/native/BPF measurements therefore run and are labelled compatibility;
direct P2PDMA is a second transport-labelled campaign after the platform path
is enabled.

The host kernel exports `p2pdma_pgmap_ops`, and the target NVMe controller and
GPU sit below the same host PCIe root complex. The remaining direct-path
conditions are not met: NVMe multipathing is enabled, no `nvidia-fs` module is
installed, the NVIDIA driver registry has no static-BAR/P2P overrides, and
NVIDIA's documented NVMe-P2PDMA GPU list does not include GeForce RTX 5090.
Consequently, a successful cuFile call on this host is not evidence of direct
NVMe-to-GPU DMA. The experiment records the cuFile compatibility path as the
transport while still measuring real O_DIRECT local-NVMe transfers and the
storage scheduling policy above them.

This distinction does not change the policy experiment. `gpu_storage_decide`
does not expose or depend on a DMA implementation: LMCache/cuFile owns the
file, registered GPU buffer, CUDA stream, and completion in either mode. A
future supported P2PDMA campaign changes only the executor's transport label,
not the BPF program, native control, request sequence, or decision ABI.

Platform references: NVIDIA's
[GDS troubleshooting guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/)
describes the NVMe-P2PDMA kernel, multipath, registry and GPU requirements;
the [GDS overview](https://docs.nvidia.com/gpudirect-storage/overview-guide/)
defines compatibility mode as the staged fallback path.
