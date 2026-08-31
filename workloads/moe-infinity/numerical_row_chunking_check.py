#!/usr/bin/env python3
"""Run the repaired MoEMLP row path against a same-parameter GPU reference."""

from __future__ import annotations

import json

import torch

from moe_infinity import _store


ROWS = (1, 256, 257, 353)
RTOL = 1.0e-2
ATOL = 1.0e-2


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the MoEMLP numerical gate")

    results = []
    for rows in ROWS:
        max_abs, max_rel, close = _store.row_chunking_numerical_check(
            rows, RTOL, ATOL
        )
        torch.cuda.synchronize()
        record = {
            "rows": rows,
            "max_abs": max_abs,
            "max_rel": max_rel,
            "within_tolerance": bool(close),
        }
        results.append(record)
        if not record["within_tolerance"]:
            raise RuntimeError(f"MoEMLP numerical mismatch: {record}")

    max_abs, max_rel, exact = _store.deterministic_accumulation_check(
        353, RTOL, ATOL
    )
    torch.cuda.synchronize()
    accumulation = {
        "rows": 353,
        "arrival_orders": 4,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "exact": bool(exact),
    }
    if not accumulation["exact"]:
        raise RuntimeError(
            f"deterministic expert accumulation mismatch: {accumulation}"
        )

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "dtype": "bfloat16",
                "rtol": RTOL,
                "atol": ATOL,
                "results": results,
                "accumulation": accumulation,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
