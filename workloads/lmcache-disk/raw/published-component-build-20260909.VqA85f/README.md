# LMCache storage components: published-source build

2026-09-09, 11:26:30–11:26:36 PDT: all four requested storage components
built with exit zero in an independent GitHub checkout. Source revision:
gpu_ext `fcb77b7d`; pinned libbpf submodule: `02bdeb7`. This reused the host
compiler/CUDA installation, not the main workspace's binaries or libbpf build.

The existing independent sparse clone was fast-forwarded, then populated
with `workloads/lmcache-disk/gds-control/`, `vmlinux/` and `libbpf/`.
`git submodule update --init --depth 1 -- libbpf` fetched the pinned source.
The four targets had not been built in that checkout before this invocation.

[build.sh](build.sh) records the exact build command and source directory;
[build.log](build.log) contains compiler output. Root held both shared leases.
No policy was loaded, driver changed, storage request issued or GPU workload
launched. No completed performance cell was repeated.

| Component | Size, bytes |
| --- | ---: |
| `gds_policy.bpf.o` | 13,320 |
| `gds_policy` | 1,517,256 |
| `ioctl_probe` | 25,264 |
| `gds_executor` | 1,057,656 |

The CUDA executor linked against the host CUDA 12.9 cuFile library. This is
build evidence only: it does not demonstrate GPU-direct P2P, a fresh Python
serving environment or availability of the modified driver interface.
The manuscript submodule remained unchanged. Two generated executables
appeared as untracked files; the artifact update adds their exact names to
the workload ignore list. Binaries, libbpf outputs and caches are not published.

See the [runtime guide](../../../../docs/artifact/lmcache-runtime.md) for
portable command shapes and the distinction between the completed storage
campaign and the separate physical-reclaim serving experiment.
