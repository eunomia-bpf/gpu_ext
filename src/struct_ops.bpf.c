/* SPDX-License-Identifier: GPL-2.0 */
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "module/bpf_testmod.h"

char _license[] SEC("license") = "GPL";

/* Declare the external kfunc from nvidia-uvm module */
extern int bpf_uvm_strstr(const char *str, u32 str__sz, const char *substr, u32 substr__sz) __ksym;

/* Implement the struct_ops callbacks */
SEC("struct_ops/test_1")
int BPF_PROG(bpf_testmod_test_1)
{
	char str[] = "Hello, GPU world!";
	char substr[] = "GPU";
	int result;

	bpf_printk("BPF test_1 called!\n");

	/* Test the kfunc */
	result = bpf_uvm_strstr(str, sizeof(str) - 1, substr, sizeof(substr) - 1);
	if (result != -1) {
		bpf_printk("'%s' found in '%s' at index %d\n", substr, str, result);
	} else {
		bpf_printk("'%s' not found in '%s'\n", substr, str);
	}

	return 42;
}

SEC("struct_ops/test_2")
int BPF_PROG(bpf_testmod_test_2, int a, int b)
{
	int result = a + b;
	bpf_printk("BPF test_2 called: %d + %d = %d\n", a, b, result);
	return result;
}

SEC("struct_ops/test_3")
int BPF_PROG(bpf_testmod_test_3, const char *buf, int len)
{
	char read_buf[64] = {0};
	int read_len = len < sizeof(read_buf) ? len : sizeof(read_buf) - 1;

	bpf_printk("BPF test_3 called with buffer length %d\n", len);

	/* Safely read from kernel buffer using bpf_probe_read_kernel */
	if (buf && read_len > 0) {
		long ret = bpf_probe_read_kernel(read_buf, read_len, buf);
		if (ret == 0) {
			/* Successfully read buffer - print first few characters */
			bpf_printk("Buffer content: '%c%c%c%c'\n",
				   read_buf[0], read_buf[1], read_buf[2], read_buf[3]);
			bpf_printk("Full buffer: %s\n", read_buf);
		} else {
			bpf_printk("Failed to read buffer, ret=%ld\n", ret);
		}
	}

	return len;
}

/* Define the struct_ops map */
SEC(".struct_ops")
struct bpf_testmod_ops testmod_ops = {
	.test_1 = (void *)bpf_testmod_test_1,
	.test_2 = (void *)bpf_testmod_test_2,
	.test_3 = (void *)bpf_testmod_test_3,
};
