#!/usr/bin/env bash
set -euo pipefail
date -Is
nvidia-smi --query-gpu=utilization.gpu,memory.used,pstate,power.draw --format=csv,noheader
sudo -n fuser -v /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm || true
sudo -n systemctl restart nvidia-persistenced
nvidia-smi --query-gpu=utilization.gpu,memory.used,pstate,power.draw --format=csv,noheader
systemctl is-active nvidia-persistenced gdm
date -Is
