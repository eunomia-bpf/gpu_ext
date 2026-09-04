# OpenCode endpoint child-PATH review outcome

OpenCode session `ses_f92b5ec33ffeufYHTDlr9tWTex` received the deny-all
request, lifecycle wrapper, and CPU test with
`spark-gateway/qwen3.8-27b-nvfp4-200k`. The bounded 180-second attempt produced
no model text and OpenCode logged `AI_APICallError`. Therefore there is **no
Qwen verdict**, and this record must not be presented as a model PASS.

The local CPU-only validation nevertheless exercised the concrete regression:

- the environment overwrites a hostile inherited PATH with the exact fixed
  allowlist;
- `cuobjdump` and `nvcc` resolve to CUDA 12.9, while the five required system
  commands resolve to their exact `/usr/bin` executables;
- removal of the CUDA directory, arbitrary PATH reordering, and a prefixed fake
  `cuobjdump` all fail before a child or module mutation;
- the fixed environment successfully runs `cuobjdump --list-ptx` against the
  selected llama.cpp CUDA library and lists its `sm_120` PTX images.

This is an honest failed model-review attempt plus executable CPU evidence, not
a substitute for lifecycle attempt 07.
