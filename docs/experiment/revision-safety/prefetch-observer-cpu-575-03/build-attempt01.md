# Observer CPU build attempt01

The first build after replacing the unsupported range observer exited 1 during
BPF compilation. `BPF_PROG` internally names its raw tracing argument `ctx`,
while `diagnostic_enter` also named the copied 88-byte local value `ctx`.
Clang reported a conflicting redefinition at `fixture.bpf.c:203`, followed by
member-access errors on the macro's pointer variable.

This was a source-level naming collision. No object was loaded, no driver or
service changed, and no GPU work ran. The copied value was renamed
`diagnostic`; no type, gate, or runtime behavior changed. The successful retry
is retained as [`build-02.log`](build-02.log). The original terminal stream was
not redirected to a file, so this contemporaneous record does not label itself
as raw compiler output.
