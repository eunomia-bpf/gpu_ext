# OpenCode phase-analysis review outcome

OpenCode 1.18.27 reviewed the two analyzer files under model
`opencode/ling-3.0-flash-fin-free`, session
`ses_f955d2278ffe4p8kUM8EoRhQMf`. The invocation used `snapshot:false`,
`share:"disabled"`, `permission:{"*":"deny"}`, and disabled write, edit,
bash, webfetch, and task tools.

The initial review and a same-session review of the operator-timing delta both
returned `READY`. The latter used the same session and configuration. It found
that the analyzer recomputes the
preflight, fixed 15-cell order, matched-work metadata, saved phase summaries,
operator/client records, runtime inventory, safety state, and telemetry. It
also confirmed that the estimator uses five paired block log-ratios with
whole-block bootstrap resampling.

The reviewer checked the then-active campaign and observed that it was still
12/15 with `complete:false`; the analyzer rejected it as intended. This was a
read-only source and partial-record review, not a GPU experiment or evidence
that the formal campaign completed.

After the campaign completed, the same analyzer passed all 15 cells and five
matched blocks. The follow-up checked the retained result and confirmed that
CUDA-event and host-wall operator means are recomputed from all 100 raw samples
per cell. It specifically rejected interpreting the 2.372x complete
measurement-and-audit loop as operator overhead; the paired BPF/CUDA operator
ratios are 1.01777x by CUDA event and 1.01809x by synchronized host wall.
