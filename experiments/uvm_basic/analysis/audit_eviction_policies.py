#!/usr/bin/env python3
"""Conservative source audit for Stage 3 eviction-policy candidates."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def audit(path: Path) -> dict[str, object]:
    text = path.read_text(errors="replace")
    hooks = sorted(set(re.findall(r'SEC\("struct_ops/([^"/]+)', text)))
    kfuncs = sorted(set(re.findall(r"\b(bpf_gpu_[a-zA-Z0-9_]+)\s*\(", text)))
    move_head = "bpf_gpu_block_move_head" in text
    move_tail = "bpf_gpu_block_move_tail" in text
    direct_list = bool(re.search(r"\blist_(?:move|add|del|splice)(?:_tail|_init)?\s*\(", text))
    sleepable = "bpf_gpu_migrate_range" in text or "bpf_wq" in text
    printk = "bpf_printk" in text and not all(
        line.lstrip().startswith("//") for line in text.splitlines() if "bpf_printk" in line
    )
    bounded_state = "BPF_MAP_TYPE_PERCPU_ARRAY" in text or "BPF_MAP_TYPE_ARRAY" in text
    suitable = not move_head and not direct_list and not sleepable and not printk
    return {
        "policy": path.stem.removesuffix(".bpf"), "source": str(path), "hooks": hooks,
        "kfuncs": kfuncs, "move_head": move_head, "move_tail": move_tail,
        "direct_list_operation": direct_list, "sleepable_kfunc_or_wq": sleepable,
        "high_frequency_printk": printk, "bounded_state": bounded_state,
        "initial_oversub_suitable": suitable,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-ext-root", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    extension = args.gpu_ext_root.resolve() / "extension"
    names = ["eviction_fifo", "eviction_cycle_moe", "prefetch_always_max_cycle_moe",
             "prefetch_cooperative", "eviction_lfu", "eviction_mru"]
    rows = [audit(extension / f"{name}.bpf.c") for name in names
            if (extension / f"{name}.bpf.c").exists()]
    result = args.experiment_dir.resolve() / "results" / "stage3" / "eviction_policy_audit.json"
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    lines = ["# Eviction Policy Safety Audit", "",
             "Evidence class: `STATIC_SOURCE_AUDIT`. No policy was attached or executed.", "",
             "| Policy | Hooks | move_head | move_tail | Direct list | Sleepable/WQ | printk | Bounded state | Initial suitable |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        display = dict(row)
        display["hooks"] = ", ".join(row["hooks"]) or "none"
        lines.append("| {policy} | {hooks} | {move_head} | {move_tail} | {direct_list_operation} | "
                     "{sleepable_kfunc_or_wq} | {high_frequency_printk} | {bounded_state} | "
                     "{initial_oversub_suitable} |".format(**display))
    lines += ["", "## Gate", "",
              "- Any `move_head`, direct linked-list operation, or sleepable migration/WQ use is excluded from the first oversubscription run.",
              "- `eviction_fifo` is also excluded until its high-frequency `bpf_printk` calls and implementation/comment mismatch are fixed.",
              "- `prefetch_always_max_cycle_moe` uses only `move_tail` plus bounded per-CPU state and is the sole current joint-policy candidate, but it remains gated on stable Stage 3C results.",
              "- `prefetch_cooperative` is excluded from the initial matrix because it schedules sleepable cross-block migration with `bpf_wq`.",
              ""]
    (args.experiment_dir.resolve() / "docs" / "EVICTION_POLICY_SAFETY_AUDIT.md").write_text("\n".join(lines))
    print(f"wrote {result}")


if __name__ == "__main__":
    main()
