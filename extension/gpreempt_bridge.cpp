// SPDX-License-Identifier: GPL-2.0
#include "gpreempt_bridge.h"
#include "ebpf-vm.h"
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <linux/bpf.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

static_assert(sizeof(gp_hint_input) == 32 && sizeof(gp_record) == 48 &&
              sizeof(gp_scope) == 16 && sizeof(gp_handle_key) == 8, "bridge ABI");

namespace {
struct State {
    std::unique_ptr<ebpf_vm, decltype(&ebpf_destroy)> vm{nullptr, ebpf_destroy};
    ebpf_jit_fn execute = nullptr;
    std::once_flag vm_once, maps_once;
    int scope_fd = -1, record_fd = -1;
    std::atomic<gp_u64> events[4]{}, actions[4]{}, scopes{0}, registrations{0}, ends{0}, errors{0};
    ~State() {
        if (scope_fd >= 0) close(scope_fd);
        if (record_fd >= 0) close(record_fd);
        std::printf("gpreempt_bridge_stats: backend=%s preprocess=%llu due=%llu infer=%llu "
                    "reset=%llu hint=%llu block=%llu release=%llu scopes=%llu "
                    "registered=%llu ended=%llu errors=%llu\n",
                    execute ? "ubpf-jit" : "original-c", events[1].load(), events[2].load(),
                    events[3].load(), actions[0].load(), actions[1].load(), actions[2].load(),
                    actions[3].load(), scopes.load(), registrations.load(), ends.load(), errors.load());
        std::fflush(stdout);
    }
};
State &state() { static State s; return s; }

int map_open(const std::string &path, gp_u32 key_size, gp_u32 value_size)
{
    union bpf_attr attr{};
    attr.pathname = reinterpret_cast<gp_u64>(path.c_str());
    int fd = syscall(SYS_bpf, BPF_OBJ_GET, &attr, sizeof(attr));
    if (fd < 0) throw std::runtime_error("open pinned map " + path + ": " + std::strerror(errno));
    struct bpf_map_info info{};
    attr = {};
    attr.info.bpf_fd = fd;
    attr.info.info_len = sizeof(info);
    attr.info.info = reinterpret_cast<gp_u64>(&info);
    if (syscall(SYS_bpf, BPF_OBJ_GET_INFO_BY_FD, &attr, sizeof(attr)) < 0 ||
        info.type != BPF_MAP_TYPE_HASH || info.key_size != key_size || info.value_size != value_size) {
        close(fd);
        throw std::runtime_error("pinned map ABI mismatch: " + path);
    }
    return fd;
}

void init_maps()
{
    State &s = state();
    std::call_once(s.maps_once, [&] {
        const char *directory = std::getenv("GPREEMPT_BPF_MAPS");
        if (!directory || directory[0] != '/')
            throw std::runtime_error("GPREEMPT_BPF_MAPS must name the loader's absolute pin directory");
        int scope = map_open(std::string(directory) + "/scopes", 8, sizeof(gp_scope));
        int record;
        try { record = map_open(std::string(directory) + "/records", sizeof(gp_handle_key), sizeof(gp_record)); }
        catch (...) { close(scope); throw; }
        s.scope_fd = scope;
        s.record_fd = record;
    });
}

template<typename K, typename V> void lookup(int fd, const K &key, V &value)
{
    union bpf_attr attr{};
    attr.map_fd = fd;
    attr.key = reinterpret_cast<gp_u64>(&key);
    attr.value = reinterpret_cast<gp_u64>(&value);
    if (syscall(SYS_bpf, BPF_MAP_LOOKUP_ELEM, &attr, sizeof(attr)) < 0)
        throw std::runtime_error("required BPF engagement record missing");
}

gp_u64 current_id() { return (static_cast<gp_u64>(getpid()) << 32) | syscall(SYS_gettid); }
int fail(const std::exception &error)
{
    ++state().errors;
    std::fprintf(stderr, "gpreempt_bridge_error: %s\n", error.what());
    return -1;
}

void init_vm()
{
    State &s = state();
    std::call_once(s.vm_once, [&] {
        const char *path = std::getenv("GPREEMPT_HINT_CODE");
        if (!path) throw std::runtime_error("GPREEMPT_HINT_CODE must name hint bytecode");
        std::ifstream input(path, std::ios::binary);
        std::vector<char> code{std::istreambuf_iterator<char>(input), {}};
        if (!input.is_open() || code.empty() || code.size() > 65536)
            throw std::runtime_error("invalid hint bytecode file");
        s.vm.reset(ebpf_create("ubpf"));
        if (!s.vm) throw std::runtime_error("cannot create ubpf VM");
        char *error = nullptr;
        if (ebpf_load(s.vm.get(), code.data(), code.size(), &error) != 0) {
            std::string message = error ? error : "hint BPF load failed";
            std::free(error);
            throw std::runtime_error(message);
        }
        s.execute = ebpf_compile(s.vm.get(), &error);
        if (!s.execute) {
            std::string message = error ? error : "hint BPF JIT failed";
            std::free(error);
            throw std::runtime_error(message);
        }
        std::printf("gpreempt_hint_ready: backend=ubpf-jit clock=system_clock comparison=strict_gt\n");
        std::fflush(stdout);
    });
}
}

// These distinct exported no-inline marker sites execute only in the BPF arm.
// Their uprobes synchronously update the maps before the caller verifies them.
extern "C" __attribute__((noinline, visibility("default")))
void gpreempt_bpf_scope_enter(gp_u32 role) { asm volatile("" : : "r"(role) : "memory"); }
extern "C" __attribute__((noinline, visibility("default")))
void gpreempt_bpf_register(gp_u64 context, gp_u32 client, gp_u32 tsg, gp_u32 role)
{ asm volatile("" : : "r"(context), "r"(client), "r"(tsg), "r"(role) : "memory"); }
extern "C" __attribute__((noinline, visibility("default")))
void gpreempt_bpf_scope_leave(void) { asm volatile("" : : : "memory"); }

extern "C" int gpreempt_bpf_enabled(void)
{
    static int enabled = [] {
        const char *policy = std::getenv("GPREEMPT_POLICY");
        if (!policy || !std::strcmp(policy, "original")) return 0;
        if (!std::strcmp(policy, "bpf")) return 1;
        std::fprintf(stderr, "gpreempt_bridge_error: invalid GPREEMPT_POLICY (original or bpf)\n");
        std::abort();
    }();
    return enabled;
}

extern "C" int gpreempt_ctx_begin(gp_u32 role)
{
    if (!gpreempt_bpf_enabled()) return role <= GP_BE ? 0 : -1;
    try {
        if (role > GP_BE) throw std::runtime_error("invalid context role");
        init_maps();
        init_vm(); // Fail before any CUDA context creation if JIT is unavailable.
        gpreempt_bpf_scope_enter(role);
        gp_scope scope{};
        lookup(state().scope_fd, current_id(), scope);
        if (scope.role != role || scope.gr_inits || scope.registered || scope.errors)
            throw std::runtime_error("BPF context scope entry did not engage cleanly");
        ++state().scopes;
        return 0;
    } catch (const std::exception &error) { return fail(error); }
}

extern "C" int gpreempt_ctx_register(gp_u64 context, gp_u32 client, gp_u32 tsg, gp_u32 role)
{
    if (!gpreempt_bpf_enabled()) return 0;
    try {
        if (!context || !client || !tsg || role > GP_BE)
            throw std::runtime_error("invalid context registration");
        init_maps();
        gpreempt_bpf_register(context, client, tsg, role);
        gp_record record{};
        gp_scope scope{};
        const gp_handle_key key{client, tsg};
        lookup(state().record_fd, key, record);
        lookup(state().scope_fd, current_id(), scope);
        if (scope.errors || scope.gr_inits != 1 || scope.registered != 1 ||
            record.registered != 1 || record.cuda_context != context ||
            record.pid_tgid != current_id() || record.role != role ||
            record.engine < 1 || record.engine > 8 ||
            record.timeslice_us != (role == GP_LC ? 1000000ULL : 1ULL))
            throw std::runtime_error("queried context does not match the BPF-controlled GR TSG");
        ++state().registrations;
        std::printf("gpreempt_context_registered: role=%u hclient=%u htsg=%u tsg_id=%llu "
                    "engine=%u runlist=%u timeslice_us=%llu cuda_context=%llu\n", role, client, tsg,
                    record.tsg_id, record.engine, record.runlist_id, record.timeslice_us, context);
        std::fflush(stdout);
        return 0;
    } catch (const std::exception &error) { return fail(error); }
}

extern "C" int gpreempt_ctx_end(void)
{
    if (!gpreempt_bpf_enabled()) return 0;
    try {
        init_maps();
        gp_scope scope{};
        lookup(state().scope_fd, current_id(), scope);
        if (scope.errors || scope.gr_inits != 1 || scope.registered != 1)
            throw std::runtime_error("incomplete BPF context scope");
        gpreempt_bpf_scope_leave();
        union bpf_attr attr{};
        gp_u64 key = current_id();
        attr.map_fd = state().scope_fd;
        attr.key = reinterpret_cast<gp_u64>(&key);
        attr.value = reinterpret_cast<gp_u64>(&scope);
        if (syscall(SYS_bpf, BPF_MAP_LOOKUP_ELEM, &attr, sizeof(attr)) == 0 || errno != ENOENT)
            throw std::runtime_error("BPF context scope did not detach");
        ++state().ends;
        return 0;
    } catch (const std::exception &error) { return fail(error); }
}

extern "C" int gpreempt_hint_decide(gp_u32 event, gp_u32 role, gp_u64 now,
                                     gp_u64 deadline, gp_u32 initialized, gp_u32 reserve)
{
    try {
        if (role > GP_BE || initialized > 1 || reserve > 1 || event < GP_PREPROCESS || event > GP_INFER)
            throw std::runtime_error("invalid hint input");
        const gp_hint_input input{now, deadline, event, role, initialized, reserve};
        gp_u64 result = 0;
        if (gpreempt_bpf_enabled()) {
            init_vm();
            result = state().execute(const_cast<gp_hint_input *>(&input), sizeof(input));
        } else if (role == GP_LC && initialized) {
            if (event == GP_PREPROCESS) result = GP_RESET | (reserve ? GP_HINT : GP_BLOCK);
            else if (event == GP_DUE) result = now > deadline ? GP_BLOCK : 0;
            else result = GP_RELEASE;
        }
        if (result > 15) throw std::runtime_error("hint BPF rejected input or returned invalid action");
        ++state().events[event];
        for (unsigned int bit = 0; bit < 4; ++bit)
            if (result & (1U << bit)) ++state().actions[bit];
        return static_cast<int>(result);
    } catch (const std::exception &error) { return fail(error); }
}
