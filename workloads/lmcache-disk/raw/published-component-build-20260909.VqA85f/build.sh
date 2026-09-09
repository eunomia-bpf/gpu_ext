#!/usr/bin/env bash
set -euo pipefail
# Root invoked this script under the two shared build/GPU leases.
# The source checkout was fetched from GitHub at fcb77b7d.
cd /tmp/gpubpf-artifact-clean-20260909.hHEHPu/repo
date --iso-8601=seconds
git log -1 --format='%h %s'
git -C libbpf log -1 --format='%h %s'
make -C workloads/lmcache-disk/gds-control -j2 \
  CUDA_HOME=/usr/local/cuda-12.9 \
  gds_policy.bpf.o gds_policy ioctl_probe gds_executor
date --iso-8601=seconds
