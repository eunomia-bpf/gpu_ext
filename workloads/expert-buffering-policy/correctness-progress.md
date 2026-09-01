# Expert-policy correctness progress

Date: 2026-08-31

Status: **PASS**. Attempt 5 completed the four-cell correctness and engagement
gate under independently approved proposal 6. It is not a performance result.

## Preserved runner attempts

1. Attempt 1 stopped before any request because the initial runner required
   exactly one `/dev/nvidia-uvm` descriptor, while the server owned descriptors
   9 and 10.
2. Attempt 2 admitted descriptor 9 and began real requests. It was interrupted
   after five requests when its zero warm-up event count was incorrectly
   treated as evidence that the other descriptor should be selected.
3. Attempt 3 tried both descriptors. Descriptor 10 rejected the event-tracker
   ioctl with an I/O error; descriptor 9 admitted the tracker. The runner then
   incorrectly required a positive eviction immediately after one warm-up and
   failed with zero events.
4. Attempt 4 retained the sole admitted descriptor and completed the plain-UVM
   warm-up plus sixteen 512-input/64-output requests. All requests completed,
   but repeated greedy text diverged late in the first compared prompt. The
   UVM Tools tracker remained at zero completed events with zero drops. This
   closed the old byte-identity and positive-event gates rather than being
   hidden or relabeled.

Proposal 5 replaces those unreachable gates with API/token/UTF-8/server-error
correctness and complete per-hot-block activation snapshots. The new snapshot
path passed a live loader smoke over a 512-block admitted table: it emitted all
128 hot indices with zero counts, matching base and block metadata and class 2.
The policy detached cleanly and the distribution UVM module was restored.

Raw attempt directories remain local under `raw/correctness/` and are not
overwritten or reused.

## Completed attempt 5

All four configurations completed one 512+64 warm-up and two passes over the
eight frozen 512+64 prompts. Every response passed API status, token accounting,
length termination, UTF-8, and server/CUDA-error gates. Repeated text matched
for 3/8 prompts in each UVM configuration and 5/8 in `llama_ncmoe32`; the first
pass of both gpubpf configurations matched plain UVM for 8/8 prompts. These are
recorded observations, not a bitwise-equivalence claim.

The observation-only cell classified 604,571 activations and 3,762,053 accesses
with zero reorder requests and zero setter failures. Its 3,234 complete hot-
block snapshots yielded 163,919,691,776 repeated hot-activation bytes. The
protection cell completed with 81,394 hot-tail activations, 518,143 cold-native
activations, 470,659 hot-access refreshes, zero cold-head requests, and zero
setter failures. Its repeated hot-activation value was 163,913,400,320 bytes,
only 6,291,456 bytes (three 2 MiB blocks, 0.00384%) below observation-only.
This near-null setup result is not promoted to a performance conclusion.

The framework context recorded 216 layouts, 641,958 route events, and 1,105
graphs with zero drops. Exactly the 32 CPU-streamed layers were covered; every
one appeared in all 1,105 graphs with zero incomplete graphs. The UVM Tools
tracker reported zero completed events and zero drops in every cell; it remains
diagnostic only.

The tracked semantic result is `correctness-result.json`. All owned processes
and struct_ops state were removed, the distribution `nvidia_uvm` was restored,
and the GPU returned to 15 MiB used memory and 0% utilization.

No file/content hashes, checksums, digests, or fingerprints were generated or
used.
