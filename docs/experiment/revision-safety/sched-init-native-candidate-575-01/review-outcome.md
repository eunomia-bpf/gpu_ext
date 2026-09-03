# OpenCode read-only design review outcome

OpenCode 1.18.27 reviewed only direct file attachments in one session on CPU
18. Both invocations used the configured default model, `snapshot:false`,
`share:disabled`, and `permission:{"*":"deny"}`. It did not call a tool. The
raw JSON event streams and empty stderr captures are retained beside this file.

The first review returned `REQUIRED FIXES` with one blocker: the lifecycle
described an unconditional four-module restore rather than restoring the exact
pre-mutation loaded NVIDIA module subset and device-node set. It also requested
three non-blocking clarifications: filter both scheduling RPC commands before
ring output, name only the tested transition statuses in the claim, and keep
semantic constructor-patch review plus a real native numerical preflight as
hard gates.

The plan was revised only in those places. The follow-up returned `READY` and
confirmed that the matrix, native/validator/setter/GSP evidence separation,
claim scope, and CPU-only status remained intact. This is a design verdict, not
evidence that a driver was built or an experiment ran.
