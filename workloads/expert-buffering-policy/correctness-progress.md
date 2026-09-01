# Expert-policy correctness progress

Date: 2026-08-31

Status: the proposal-5 runner repair is approved; a complete four-cell run is
still pending. None of the attempts below is a correctness or performance
result.

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

No file/content hashes, checksums, digests, or fingerprints were generated or
used.
