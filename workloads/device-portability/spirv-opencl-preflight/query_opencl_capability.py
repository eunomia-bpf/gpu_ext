#!/usr/bin/env python3
"""Emit the selected OpenCL GPU's IL capability from an isolated process."""

from __future__ import annotations

import json

from run_spirv_opencl_preflight import query_opencl_capability


if __name__ == "__main__":
    print(json.dumps(query_opencl_capability(), sort_keys=True))
