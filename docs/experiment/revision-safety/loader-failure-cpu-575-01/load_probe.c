#define _GNU_SOURCE

#include <errno.h>
#include <linux/bpf.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

static long load_program(const struct bpf_insn *instructions, size_t count,
			 const char *name)
{
	static const char license[] = "GPL";
	union bpf_attr attr;

	memset(&attr, 0, sizeof(attr));
	attr.prog_type = BPF_PROG_TYPE_SOCKET_FILTER;
	attr.insn_cnt = (uint32_t)count;
	attr.insns = (uint64_t)(uintptr_t)instructions;
	attr.license = (uint64_t)(uintptr_t)license;
	strncpy((char *)attr.prog_name, name, sizeof(attr.prog_name) - 1);

	errno = 0;
	return syscall(__NR_bpf, BPF_PROG_LOAD, &attr, sizeof(attr));
}

static void print_result(const char *label, long result)
{
	const int saved_errno = errno;

	printf("%s_rc=%ld %s_errno=%d\n", label, result, label, saved_errno);
}

int main(int argc, char **argv)
{
	const struct bpf_insn invalid[] = {
		{ .code = BPF_ST | BPF_DW | BPF_MEM,
		  .dst_reg = BPF_REG_10,
		  .off = -520,
		  .imm = 0 },
		{ .code = BPF_ALU64 | BPF_MOV | BPF_K,
		  .dst_reg = BPF_REG_0,
		  .imm = 0 },
		{ .code = BPF_JMP | BPF_EXIT },
	};
	const struct bpf_insn valid[] = {
		{ .code = BPF_ST | BPF_DW | BPF_MEM,
		  .dst_reg = BPF_REG_10,
		  .off = -8,
		  .imm = 0 },
		{ .code = BPF_ALU64 | BPF_MOV | BPF_K,
		  .dst_reg = BPF_REG_0,
		  .imm = 0 },
		{ .code = BPF_JMP | BPF_EXIT },
	};
	long result;

	if (argc != 2 || (strcmp(argv[1], "invalid-then-valid") != 0 &&
			  strcmp(argv[1], "valid-only") != 0)) {
		fprintf(stderr, "usage: %s {invalid-then-valid|valid-only}\n",
			argv[0]);
		return 64;
	}

	if (strcmp(argv[1], "invalid-then-valid") == 0) {
		result = load_program(invalid,
				      sizeof(invalid) / sizeof(invalid[0]),
				      "unsafe_stack");
		print_result("invalid", result);
	}

	result = load_program(valid, sizeof(valid) / sizeof(valid[0]),
			      "safe_stack");
	print_result("valid", result);
	return 0;
}
