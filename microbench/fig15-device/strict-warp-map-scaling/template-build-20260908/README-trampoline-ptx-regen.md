# Embedded trampoline PTX regeneration recipe (bpftime-auto-warp)

Root integration: applied to bpftime `fa96b77`; CMake configuration succeeds
with the opt-in enabled and this host's explicit GCC 13 flag override.
`raw/observed-counts-20260908.7LoqI4/configure-template.log` (in the parent
benchmark directory) records configuration. The upcoming count-runtime build
will use this dependency; no new timing measurement is claimed by this patch.

Problem: `attach/nv_attach_impl/trampoline_ptx.h` embeds the device-side map
ABI; a stale embedded header used to silently mismatch the 48-byte host
`MapBasicInfo` (40 bytes before). bpftime-auto-warp e5e52e5 regenerated it
manually; the normal CMake build had no dependency that would do this.

Patch (plain source diff, no index lines):
`bpftime-trampoline-ptx-regen.patch`, against
`attach/nv_attach_impl/CMakeLists.txt` at e5e52e5. It adds an opt-in
`BPFTIME_REGENERATE_TRAMPOLINE_PTX` (default OFF): when ON, a custom command
runs the existing `attach/nv_attach_impl/trampoline/Makefile` target
`../trampoline_ptx.h` (same generator, no new framework), writing the
checked-in header before `bpftime_nv_attach_impl` (nv_attach_utils.cpp)
compiles; consumer recompiles then flow from normal header dependency
scanning. DEPENDS: `default_trampoline.cu` (`MapBasicInfo` and the helper
enums are defined in the .cu; host mirror lives in
`runtime/include/bpf_attach_ctx.hpp`) plus the Makefile itself. The nested
make is found via `find_program(make, gmake)` (never `CMAKE_MAKE_PROGRAM`,
which is ninja under the Ninja generator), with a hard configure error when
neither exists.

Enable (root integration later; nothing was run here):

    cmake -DBPFTIME_ENABLE_CUDA_ATTACH=ON -DBPFTIME_REGENERATE_TRAMPOLINE_PTX=ON [...]

On this host only, pass an explicit cache override (never a hardcoded
default; includes the portable -U pair plus the local GCC 13 install dir):

    cmake ... -DBPFTIME_TRAMPOLINE_CLANG_FLAGS="-U__FLOAT128__ -U__SIZEOF_FLOAT128__ --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"

Equivalent manual regeneration (known-good on this host):

    cd attach/nv_attach_impl/trampoline && make ../trampoline_ptx.h CLANG_FLAGS="-U__FLOAT128__ -U__SIZEOF_FLOAT128__ --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"

Notes:
- Off by default: ordinary builds (also CUDA-off, and CUDA-on without the
  clang-based generator) are unchanged; no CUDA requirement is added.
- Bounded forced regeneration: CMake runs the command only when the OUTPUT
  header is stale vs DEPENDS (no rebuild when up to date); GNU make `-B
  ../trampoline_ptx.h` then forces the nested .cu -> .s -> header chain, so
  a Makefile-only DEPENDS trigger cannot be skipped as up-to-date inside.
- The recipe writes the source-tree header by design (matches the manual
  regeneration workflow); `PTX_TARGET_ARCH` and tool selection stay with the
  Makefile. `BPFTIME_TRAMPOLINE_CLANG_FLAGS` defaults to empty, keeping the
  Makefile's portable `CLANG_FLAGS ?=` defaults; a set override is one
  quoted token passed through make intact.
- Verified here: `git apply --check` plus a demo apply to a throwaway copy
  (result byte-identical to the intended edit), and a `make -n` dry run of
  the generator target (executes nothing). No builds, no GPU, live runtime
  checkout untouched.
