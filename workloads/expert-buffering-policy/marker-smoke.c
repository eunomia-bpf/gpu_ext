/* SPDX-License-Identifier: MIT */

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef void (*layout_marker_fn)(const char *, const void *, uint64_t, uint64_t,
				 uint32_t, uint32_t);
typedef void (*route_marker_fn)(const void *, uint32_t);

int main(int argc, char **argv)
{
	void *library;
	layout_marker_fn layout;
	route_marker_fn route;
	const void *base = (const void *)(uintptr_t)0x40000000ULL;

	if (argc != 2) {
		fprintf(stderr, "usage: %s LIBGGML_BASE\n", argv[0]);
		return 2;
	}
	library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
	if (!library) {
		fprintf(stderr, "dlopen failed: %s\n", dlerror());
		return 1;
	}
	layout = (layout_marker_fn)dlsym(library, "gpubpf_expert_tensor_layout");
	route = (route_marker_fn)dlsym(library, "gpubpf_expert_route");
	if (!layout || !route) {
		fprintf(stderr, "required marker symbol is missing\n");
		dlclose(library);
		return 1;
	}

	layout("blk.7.ffn_gate_exps.weight", base, 564019200ULL,
	       4406400ULL, 128, 0);
	route(base, 17);
	printf("marker calls completed\n");
	dlclose(library);
	return 0;
}
