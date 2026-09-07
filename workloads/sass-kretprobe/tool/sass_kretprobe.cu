// sass-kretprobe NVBit tool.
//
// Scope: one CUDA context, one target kernel symbol, one in-flight launch
// record. Instrumentation is inserted into the target kernel's own SASS at
// every EXIT; the injected call invokes sk_bpf_trampoline (this tool's own
// device code, embedded in the tool fatbin), which in turn calls bpf_exit,
// the SASS body of the ptxpass-compiled BPF program. Each logical thread
// writes the BPF result into its own 8-byte slot so concurrent threads never
// race on one shared marker.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <unordered_set>
#include <vector>

#include "nvbit.h"
#include "nvbit_tool.h"

static constexpr uint64_t kSlotBytes = 8;
static constexpr uint64_t kCapacityThreads = 1u << 20;

static std::string target_symbol;
static uint64_t slots_dev = 0;
static uint32_t target_launches = 0;
static std::unordered_set<CUfunction> instrumented;

static bool is_target_launch_cbid(nvbit_api_cuda_t cbid)
{
	return cbid == API_CUDA_cuLaunchKernel ||
	       cbid == API_CUDA_cuLaunchKernel_ptsz ||
	       cbid == API_CUDA_cuLaunchKernelEx ||
	       cbid == API_CUDA_cuLaunchKernelEx_ptsz;
}

static CUfunction launch_cbid_function(nvbit_api_cuda_t cbid,
				       const void *params)
{
	if (cbid == API_CUDA_cuLaunchKernelEx ||
	    cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
		return reinterpret_cast<const cuLaunchKernelEx_params *>(params)
			->f;
	}
	if (cbid == API_CUDA_cuLaunchKernel_ptsz) {
		return reinterpret_cast<const cuLaunchKernel_ptsz_params *>(
				   params)
			->f;
	}
	return reinterpret_cast<const cuLaunchKernel_params *>(params)->f;
}

static uint64_t launch_cbid_threads(nvbit_api_cuda_t cbid,
				    const void *params)
{
	if (cbid == API_CUDA_cuLaunchKernelEx ||
	    cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
		const auto *ex =
			reinterpret_cast<const cuLaunchKernelEx_params *>(params);
		if (ex->config == nullptr) {
			return 0;
		}
		const CUlaunchConfig *c = ex->config;
		return (uint64_t)c->blockDimX * c->blockDimY * c->blockDimZ *
		       (uint64_t)c->gridDimX * c->gridDimY * c->gridDimZ;
	}
	if (cbid == API_CUDA_cuLaunchKernel_ptsz) {
		const auto *p =
			reinterpret_cast<const cuLaunchKernel_ptsz_params *>(params);
		return (uint64_t)p->blockDimX * p->blockDimY * p->blockDimZ *
		       (uint64_t)p->gridDimX * p->gridDimY * p->gridDimZ;
	}
	const auto *p =
		reinterpret_cast<const cuLaunchKernel_params *>(params);
	return (uint64_t)p->blockDimX * p->blockDimY * p->blockDimZ *
	       (uint64_t)p->gridDimX * p->gridDimY * p->gridDimZ;
}

void nvbit_at_init()
{
	const char *sym = getenv("SK_TARGET_SYMBOL");
	if (sym == nullptr || sym[0] == '\0') {
		fprintf(stderr,
			"SKRET error: SK_TARGET_SYMBOL (target kernel mangled symbol) is required\n");
		exit(1);
	}
	target_symbol = sym;
	fprintf(stderr, "SKRET tool loaded target=%s\n",
		target_symbol.c_str());
}

void nvbit_tool_init(CUcontext ctx)
{
	void *pool = nullptr;
	if (cudaMalloc(&pool, kCapacityThreads * kSlotBytes) != cudaSuccess ||
	    pool == nullptr) {
		fprintf(stderr, "SKRET error: cannot allocate slot pool\n");
		exit(1);
	}
	slots_dev = reinterpret_cast<uint64_t>(pool);
	(void)ctx;
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
			 const char *event_name, void *params,
			 CUresult *pStatus)
{
	if (!is_target_launch_cbid(cbid)) {
		return;
	}
	CUfunction f = launch_cbid_function(cbid, params);
	if (f == nullptr || !nvbit_is_func_kernel(ctx, f)) {
		return;
	}
	const char *mangled = nvbit_get_func_name(ctx, f, true);
	const char *demangled = nvbit_get_func_name(ctx, f, false);
	if (target_symbol != mangled && target_symbol != demangled) {
		return;
	}

	if (is_exit) {
		const uint64_t threads = launch_cbid_threads(cbid, params);
		cuCtxSynchronize();
		const uint64_t checked =
			threads < kCapacityThreads ? threads : kCapacityThreads;
		std::vector<uint8_t> host(checked * kSlotBytes, 0);
		if (cudaMemcpy(host.data(), (void *)slots_dev,
			       host.size(), cudaMemcpyDeviceToHost) ==
		    cudaSuccess) {
			uint64_t wrote = 0;
			for (uint64_t i = 0; i < checked; i++) {
				uint64_t v;
				memcpy(&v, host.data() + i * kSlotBytes, 8);
				if (v == 42) {
					wrote++;
				}
			}
			fprintf(stderr,
				"SKRET launch=%u threads=%lu expected_slots=%lu wrote_42=%lu\n",
				target_launches, (unsigned long)threads,
				(unsigned long)checked, (unsigned long)wrote);
		} else {
			fprintf(stderr, "SKRET error: slot readback failed\n");
		}
		return;
	}

	target_launches++;
	// Reset the slots this launch will use so the post-launch readback
	// distinguishes BPF-written markers from stale data.
	{
		const uint64_t threads = launch_cbid_threads(cbid, params);
		const uint64_t checked =
			threads < kCapacityThreads ? threads : kCapacityThreads;
		if (cudaMemset((void *)slots_dev, 0, checked * kSlotBytes) !=
		    cudaSuccess) {
			fprintf(stderr, "SKRET error: slot reset failed\n");
		}
	}
	if (instrumented.insert(f).second) {
		const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
		uint32_t exits = 0;
		for (Instr *in : instrs) {
			if (strcmp(in->getOpcode(), "EXIT") != 0) {
				continue;
			}
			nvbit_insert_call(in, "sk_bpf_trampoline",
					  IPOINT_BEFORE);
			nvbit_add_call_arg_guard_pred_val(in);
			nvbit_add_call_arg_const_val64(in, slots_dev);
			nvbit_add_call_arg_const_val64(in, kCapacityThreads);
			exits++;
		}
		if (exits == 0) {
			fprintf(stderr, "SKRET error: no EXIT instruction in %s\n",
				mangled);
		} else {
			fprintf(stderr, "SKRET instrumented_exits=%u target=%s\n",
				exits, mangled);
		}
	}
	nvbit_enable_instrumented(ctx, f, true, false);
}

void nvbit_at_term()
{
	fprintf(stderr, "SKRET done target_launches=%u\n", target_launches);
}
