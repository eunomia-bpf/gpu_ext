// Thin libbpf loader, run only under the existing bpftime syscall server.
// It performs no host policy decision. The parent owns its private SHM name
// and must finish/join the CUDA client before closing this loader's stdin.
#include <bpf/libbpf.h>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <poll.h>
#include <string>
#include <unistd.h>

static volatile sig_atomic_t stopped = 0;
static void stop(int) { stopped = 1; }

int main(int argc, char **argv) {
    const char *preload = std::getenv("LD_PRELOAD");
    const char *shm = std::getenv("BPFTIME_GLOBAL_SHM_NAME");
    if (argc != 3 || !preload || !std::strstr(preload, "libbpftime-syscall-server.so") ||
        !shm || std::strncmp(shm, "pod_attention_", 14) != 0 || std::strchr(shm, '/')) {
        std::cerr << "usage: private bpftime env loader selector.bpf.o exact-kernels.txt\n";
        return 2;
    }
    if (std::getenv("BPFTIME_RUN_WITH_KERNEL")) {
        std::cerr << "POD device loader refuses kernel-BPF passthrough mode\n";
        return 2;
    }
    std::ifstream input(argv[2]);
    std::string line, kernels;
    unsigned count = 0;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        if (line.find("true_fused_tb_fwd_kernel") == std::string::npos ||
            line.find_first_of(" ,\t\r") != std::string::npos) {
            std::cerr << "invalid exact POD kernel name\n";
            return 2;
        }
        if (count++) kernels += ',';
        kernels += line;
    }
    if (!input.eof() || !count) { std::cerr << "empty/invalid kernel inventory\n"; return 2; }
    auto *object = bpf_object__open_file(argv[1], nullptr);
    if (!object) { std::cerr << "cannot open selector BPF object\n"; return 1; }
    auto *program = bpf_object__find_program_by_name(object, "cuda__podsel");
    if (!program || bpf_program__set_type(program, BPF_PROG_TYPE_KPROBE) || bpf_object__load(object)) {
        std::cerr << "cannot load CUDA selector BPF\n";
        bpf_object__close(object);
        return 1;
    }
    auto *link = bpf_program__attach_kprobe(program, false, kernels.c_str());
    if (libbpf_get_error(link)) {
        std::cerr << "cannot attach CUDA selector BPF\n";
        bpf_object__close(object);
        return 1;
    }
    std::signal(SIGINT, stop);
    std::signal(SIGTERM, stop);
    std::cout << "POD_LOADER_READY kernels=" << count << "\n" << std::flush;
    while (!stopped) {
        pollfd fd{STDIN_FILENO, POLLIN | POLLHUP, 0};
        int ret = poll(&fd, 1, 500);
        if (ret < 0 && errno != EINTR) { stopped = 1; break; }
        if (ret > 0 && fd.revents) {
            char byte;
            if (read(STDIN_FILENO, &byte, 1) <= 0) break;
        }
    }
    bpf_link__destroy(link);
    bpf_object__close(object);
    std::cout << "POD_LOADER_CLOSED\n";
    return 0;
}
