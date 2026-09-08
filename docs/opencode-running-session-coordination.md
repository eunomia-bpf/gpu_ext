# Coordinating existing local OpenCode sessions

On 2026-09-07 UTC, the root used the installed OpenCode API to append
implementation pointers to running sessions without restarting them or
starting another model task. The local SDK and server OpenAPI both expose
`POST /session/{sessionID}/message` with `noReply: true`.

A headless server was started with `opencode serve --pure --hostname
127.0.0.1 --port 0`; it selected `http://127.0.0.1:4096`. This is a loopback
coordination endpoint, not a fourth inference session. Requests specify the
session's worktree using the `directory` query parameter. Supply the existing
model and agent explicitly, especially `agent: explore` for a read-only child.
The endpoint returned a stored **user** message without generating an answer.
Appending a message is not proof the ongoing model call has consumed it yet.
The server's in-memory status must not replace checking the actual CLI PID
or its existing tool handle for sessions running in other processes.

The root supplied the GPU_ARRAY constructor and whole-value lookup locations
from `device-array-runtime-notes.md`, asked the existing investigation to
return those findings, and directed the implementation tasks to concrete
source files. It also supplied the completed final-only collector result:
3493.318 token/s and 90.812% overhead in five pairs, preserving 720896 events
per run. This tells the implementation tasks that changing polling alone did
not resolve the overhead; it does not claim an array implementation is done.

Recipients were `ses_f86824339ffeqWuw36tuQdVu74` (Qwen 27B candidate),
`ses_f86de4662ffe3FBnh2esbuZJkw` (GLM candidate), and its already-running
read-only child `ses_f86dba1b8ffeG3lEjod1UqGxV1`. The third root model task
remains `ses_f866bda8dffeljd81GZlstX8rx` (Qwen Next storage feedback).
No new nested agent was created. No process was stopped for silence,
and no short model timeout was introduced.

## Cluster-direct continuation preparation — 2026-09-08

XSched's terminal generation error is HTTP 524 with a Cloudflare HTML
timeout response, not an OpenCode session timeout imposed by root.
Authenticated gateway `/v1/models` and `/health/readiness` return HTTP 200;
that proves gateway reachability, not successful model generation. Both
Qwen tasks and later a GLM request have encountered generation retries.
Current live sessions are allowed to continue; none was aborted for silence.

The gateway's read-only model information identifies the existing cluster
backends. Using the node's configured cluster DNS, root resolved the Qwen
27B service and obtained HTTP 200 from its direct `/v1/models` and `/health`
endpoints. It advertises `qwen3.8-27b-nvfp4`. The Qwen Next and GLM backends
require different direct credentials; their HTTP 401 responses are not
evidence that those backends are offline. No credentials or infrastructure
configuration were changed.

A private, temporary `OPENCODE_CONFIG` override adds only the direct Qwen
27B provider, preserving its 200K context and 16384 output-token settings
and disabling the client's request timeout. The existing global config
and server remain unchanged. A separate loopback coordinator accepts the
override and can see the same saved Fig.13 session. Starting that coordinator
does not start an inference task or create a fourth active model session.
The direct path is prepared for continuation only after an existing task
actually ends; a successful generation and implementation are still unproven.

OpenCode documents the scoped override mechanism in its
[custom configuration path instructions](https://dev.opencode.ai/docs/config/#custom-path).
The private service address is resolved at use time rather than treated as
a portable artifact address. Do not restart running sessions just to adopt it.
