# Built-module interface check attempt01

A CPU-only check passed the freshly built split-module BTF text to an
experimental member-type resolver. It rejected the first member with
`diagnostic member is not a scalar integer` because a split module refers to
integer types in its base kernel BTF while the plain module dump does not emit
those base definitions. This was a checker limitation, not a module or ABI
failure; no module was loaded and no GPU work ran.

The live gate retains checks that the tagged structure has exactly 88 bytes,
14 named members at the expected offsets, no extra address field, the exact two
phase values, and a void function taking one pointer to the const structure.
Field widths and signedness remain compile-time properties of the driver types,
the fixture's static ABI assertions, and the already retained source/build
review; the runner does not pretend that a base-less split-BTF dump can resolve
them. A second check against the built module passes with this honest boundary.
