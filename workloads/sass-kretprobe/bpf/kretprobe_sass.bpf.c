#ifndef SEC
#define SEC(name) __attribute__((section(name), used))
#endif

SEC("cuda__/kretprobe_sass")
int cuda__kretprobe_sass(unsigned long long *ctx)
{
	*ctx = 42;
	return 0;
}
