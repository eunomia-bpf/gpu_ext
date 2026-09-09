#!/usr/bin/env bash
set -euo pipefail
date -Is
make -C /home/yunwei37/workspace/gpu/gpu_ext-kernel-575-gds/kernel-open -j2 modules KERNEL_UNAME=6.15.11-061511-generic
date -Is
