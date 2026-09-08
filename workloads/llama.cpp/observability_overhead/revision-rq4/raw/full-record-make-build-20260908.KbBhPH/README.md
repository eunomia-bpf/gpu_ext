# Reusable full-record Makefile build

Local Qwen supplied the Makefile, README and build-output ignore rules.
Root corrected the shared-header dependency glob from `%` to `*` in the
BPF/host rules so edits to the separately named layout header trigger
recompilation. The measured BPF/collector logic is unchanged.

The five source/build files were staged in the required three-level
bpftime example layout at:
`/home/yunwei37/workspace/gpu/bpftime-auto-warp/example/gpu/full-record-make-20260908.ppxziY/`.
This is a build directory, not another Git worktree or driver dependency.

The actual command, under both shared experiment leases, was:

```sh
make -C /home/yunwei37/workspace/gpu/bpftime-auto-warp/example/gpu/full-record-make-20260908.ppxziY -j2 CUDA_HOME=/usr/local/cuda-12.9
```

It exits zero after building libbpf, bootstrap bpftool, the BPF object,
skeleton and collector. `build.log` retains the harmless inherited clang
warning about an unused linker argument. No GPU workload or completed
performance cell was repeated. The prior five-block results remain tied
to their original built executable; this verifies the reusable build entry.
Compiled binaries and the dependency build cache are not added to Git.
