#!/usr/bin/env python3
"""Compile semantic expert tensor observations into a PMM block-class table."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

BLOCK_BYTES = 2 * 1024 * 1024
MAX_LAYOUT_BLOCKS = 65536
MAX_HOT_BYTES = 8 * 1024 * 1024 * 1024

CLASS_COLD = 1
CLASS_HOT = 2
CLASS_SHARED = 3

FLAG_COLD = 1
FLAG_HOT = 2
FLAG_SHARED = 4

NAME_PATTERN = re.compile(
    r"^blk\.(?P<layer>\d+)\.ffn_(?P<kind>gate|down|up)_exps\."
    r"(?P<suffix>weight|bias)$"
)


@dataclass(frozen=True)
class Registration:
    tgid: int
    name: str
    base: int
    total_bytes: int
    per_expert_bytes: int
    n_experts: int
    is_bias: bool
    layer: int
    kind: str

    @property
    def end(self) -> int:
        return self.base + self.total_bytes


@dataclass(frozen=True)
class CompiledLayout:
    base: int
    blocks: int
    classes: tuple[int, ...]
    registrations: int
    hot_bytes: int


def parse_registration(record: dict) -> Registration:
    match = NAME_PATTERN.fullmatch(str(record.get("name", "")))
    if not match:
        raise ValueError(f"unsupported expert tensor name: {record.get('name')!r}")
    values = {
        "tgid": int(record["tgid"]),
        "base": int(record["base"]),
        "total_bytes": int(record["total_bytes"]),
        "per_expert_bytes": int(record["per_expert_bytes"]),
        "n_experts": int(record["n_experts"]),
        "is_bias": bool(int(record["is_bias"])),
    }
    if values["base"] <= 0 or values["total_bytes"] <= 0:
        raise ValueError("layout base and size must be positive")
    if values["per_expert_bytes"] <= 0 or values["n_experts"] <= 1:
        raise ValueError("expert stride and count must be positive")
    suffix_bias = match.group("suffix") == "bias"
    if values["is_bias"] != suffix_bias:
        raise ValueError(f"bias flag disagrees with tensor name {record['name']}")
    return Registration(
        name=record["name"],
        layer=int(match.group("layer")),
        kind=match.group("kind"),
        **values,
    )


def load_registrations(path: Path, requested_tgid: int | None) -> list[Registration]:
    registrations: dict[tuple, Registration] = {}
    seen_tgids: set[int] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at input line {line_number}: {error}") from error
            if record.get("event") != "layout":
                continue
            registration = parse_registration(record)
            seen_tgids.add(registration.tgid)
            if requested_tgid is not None and registration.tgid != requested_tgid:
                continue
            key = (
                registration.tgid,
                registration.name,
                registration.base,
                registration.total_bytes,
                registration.per_expert_bytes,
                registration.n_experts,
                registration.is_bias,
            )
            registrations[key] = registration
    if requested_tgid is None:
        if len(seen_tgids) != 1:
            raise ValueError(f"expected one layout-producing TGID, found {sorted(seen_tgids)}")
    if not registrations:
        raise ValueError("no matching layout registrations")
    return sorted(registrations.values(), key=lambda item: (item.base, item.name))


def load_hot_set(path: Path) -> dict[int, set[int]]:
    hot: dict[int, set[int]] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 2:
                raise ValueError(f"invalid hot-set line {line_number}")
            layer, expert = map(int, fields)
            if layer < 0 or expert < 0 or expert >= 128:
                raise ValueError(f"out-of-range hot-set entry at line {line_number}")
            layer_set = hot.setdefault(layer, set())
            if expert in layer_set:
                raise ValueError(f"duplicate hot-set entry at line {line_number}")
            layer_set.add(expert)
    if not hot:
        raise ValueError("hot set is empty")
    return hot


def validate_strict(
    registrations: list[Registration], hot: dict[int, set[int]], expected: int
) -> None:
    if len(registrations) != expected:
        raise ValueError(f"expected {expected} registrations, found {len(registrations)}")
    weight = [item for item in registrations if not item.is_bias]
    bias = [item for item in registrations if item.is_bias]
    if len(weight) != 108 or len(bias) != 108:
        raise ValueError(f"expected 108 weight and 108 bias tensors, found {len(weight)}/{len(bias)}")
    layers = {item.layer for item in registrations}
    if layers != set(range(36)):
        raise ValueError(f"expected layers 0..35, found {sorted(layers)}")
    for layer in range(36):
        if len(hot.get(layer, set())) != 10:
            raise ValueError(f"layer {layer} does not have exactly ten hot experts")
    for item in registrations:
        if item.n_experts != 128:
            raise ValueError(f"{item.name} has {item.n_experts} experts, expected 128")


def compile_layout(
    registrations: list[Registration], hot: dict[int, set[int]]
) -> CompiledLayout:
    base = min(item.base for item in registrations) // BLOCK_BYTES * BLOCK_BYTES
    end = max(item.end for item in registrations)
    aligned_end = (end + BLOCK_BYTES - 1) // BLOCK_BYTES * BLOCK_BYTES
    blocks = (aligned_end - base) // BLOCK_BYTES
    if blocks <= 0 or blocks > MAX_LAYOUT_BLOCKS:
        raise ValueError(f"layout span contains {blocks} blocks, limit is {MAX_LAYOUT_BLOCKS}")

    flags = [0] * blocks
    owner = [-1] * blocks
    boundary = [False] * blocks
    hot_overlap = [False] * blocks

    for registration_index, item in enumerate(registrations):
        first_block = (item.base - base) // BLOCK_BYTES
        last_block = (item.end - 1 - base) // BLOCK_BYTES
        for block_index in range(first_block, last_block + 1):
            if owner[block_index] == -1:
                owner[block_index] = registration_index
            elif owner[block_index] != registration_index:
                boundary[block_index] = True

            if item.is_bias:
                flags[block_index] |= FLAG_SHARED
                continue

            block_start = base + block_index * BLOCK_BYTES
            overlap_start = max(block_start, item.base)
            overlap_end = min(block_start + BLOCK_BYTES, item.end)
            first_expert = (overlap_start - item.base) // item.per_expert_bytes
            last_expert = (overlap_end - 1 - item.base) // item.per_expert_bytes
            first_expert = max(0, min(first_expert, item.n_experts - 1))
            last_expert = max(0, min(last_expert, item.n_experts - 1))
            overlaps_hot = any(
                expert in hot.get(item.layer, set())
                for expert in range(first_expert, last_expert + 1)
            )
            if overlaps_hot:
                flags[block_index] |= FLAG_HOT
                hot_overlap[block_index] = True
            else:
                flags[block_index] |= FLAG_COLD

    classes: list[int] = []
    for block_index, block_flags in enumerate(flags):
        if boundary[block_index] or block_flags & FLAG_SHARED:
            classes.append(CLASS_SHARED)
        elif block_flags & FLAG_HOT:
            classes.append(CLASS_HOT)
        elif block_flags & FLAG_COLD:
            classes.append(CLASS_COLD)
        else:
            classes.append(0)

    return CompiledLayout(
        base=base,
        blocks=blocks,
        classes=tuple(classes),
        registrations=len(registrations),
        hot_bytes=sum(hot_overlap) * BLOCK_BYTES,
    )


def write_layout(path: Path, layout: CompiledLayout) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = path.parent if path.parent != Path("") else Path(".")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=descriptor, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(
                f"base {layout.base} blocks {layout.blocks} "
                f"hot_bytes {layout.hot_bytes} registrations {layout.registrations}\n"
            )
            for index, block_class in enumerate(layout.classes):
                if block_class:
                    output.write(f"{index} {block_class}\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--hot-set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tgid", type=int)
    parser.add_argument("--expected-layouts", type=int, default=216)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    registrations = load_registrations(args.input, args.tgid)
    hot = load_hot_set(args.hot_set)
    if args.strict:
        validate_strict(registrations, hot, args.expected_layouts)
    elif len(registrations) != args.expected_layouts:
        raise ValueError(
            f"expected {args.expected_layouts} registrations, found {len(registrations)}"
        )
    layout = compile_layout(registrations, hot)
    if args.strict and layout.hot_bytes > MAX_HOT_BYTES:
        raise ValueError(
            f"hot protection is {layout.hot_bytes} bytes, above {MAX_HOT_BYTES}"
        )
    write_layout(args.output, layout)
    summary = {
        "event": "layout_compiled",
        "registrations": layout.registrations,
        "base": layout.base,
        "blocks": layout.blocks,
        "classified_blocks": sum(value != 0 for value in layout.classes),
        "cold_blocks": layout.classes.count(CLASS_COLD),
        "hot_blocks": layout.classes.count(CLASS_HOT),
        "shared_blocks": layout.classes.count(CLASS_SHARED),
        "hot_bytes": layout.hot_bytes,
        "output": str(args.output),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
