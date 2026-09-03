/* SPDX-License-Identifier: GPL-2.0 */
/* Compile the same policy for BPF to measure execution-mechanism cost rather
 * than an accidental algorithm difference. Semantic tests also assert actions
 * independently of C/BPF parity. No helpers or CUDA-side effects are linked. */
#include "idle_policy.c"
