/* CPU-only frontend to the existing bpftime GPU compiler. This does not edit
 * or rebuild bpftime. Also exports its existing JSON PTX-pass ABI. */
#include "ptxpass/core.hpp"
#include <cstring>
#include <iterator>
#include <stdexcept>

static constexpr const char *callee = "pod_device_bpf_selector";

static std::string compile(const std::vector<uint64_t> &words) {
    if (words.empty()) throw std::runtime_error("empty POD eBPF program");
    // Installed LLVM 15 knows sm_80, not sm_120. This emits ordinary compatible
    // PTX instructions; the containing attention module retains sm_120 target.
    auto ptx = ptxpass::compile_ebpf_to_ptx_from_words(words, "sm_80", callee,
                                                   false, true);
    if (ptx.find(std::string(".func ") + callee + "(") == std::string::npos ||
        ptx.find(std::string(callee) + "_param_1") == std::string::npos)
        throw std::runtime_error("GPU compiler did not preserve the typed context ABI");
    return ptx;
}

static std::pair<std::string, bool> patch(const std::string &ptx,
                                         const std::vector<uint64_t> &words) {
    // The call is void with exactly two argument symbols. Never insert a
    // generic entry probe or erase arguments if the named hook is missing.
    // --keep-device-functions retains the real ABI but NVCC may name its
    // address-space-specialized implementation pod_device_selector$<number>.
    const std::regex call(R"(\b(call(?:\.uni)?)\s+(pod_device_selector(?:\$[0-9]+)?)\s*,\s*\(\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*,\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*\)\s*;)");
    if (!std::regex_search(ptx, call)) {
        // The first exact kernel target patches every typed POD call in its
        // containing module. Later exact targets in that same module are no-ops.
        if (ptx.find(std::string(".func ") + callee + "(") != std::string::npos &&
            std::regex_search(ptx, std::regex(R"(\bcall(?:\.uni)?\s+pod_device_bpf_selector\s*,\s*\(\s*[A-Za-z0-9_$]+\s*,\s*[A-Za-z0-9_$]+\s*\)\s*;)")))
            return {ptx, false};
        throw std::runtime_error("no two-argument POD selector call in PTX");
    }
    if (ptx.find(std::string(".func ") + callee + "(") != std::string::npos)
        throw std::runtime_error("POD selector already patched");
    for (std::sregex_iterator i(ptx.begin(), ptx.end(), call), end; i != end; ++i) {
        const auto name = std::regex_replace((*i)[2].str(), std::regex(R"(\$)"), R"(\$)");
        const std::regex declaration("\\.func\\s+" + name +
            R"(\s*\(\s*\.param\s+\.b64\s+[A-Za-z0-9_$]+\s*,\s*\.param\s+\.b64\s+[A-Za-z0-9_$]+\s*\))");
        if (!std::regex_search(ptx, declaration))
            throw std::runtime_error("POD selector declaration does not have the two-argument ABI");
    }
    auto out = std::regex_replace(ptx, call,
                                 std::string("$1 ") + callee + ", ($3, $4);");
    // Insert after the module header, not before .version/.target. PTX allows
    // function definitions before the attention kernels that call them.
    auto header_end = out.find(".address_size");
    if (header_end == std::string::npos) throw std::runtime_error("missing PTX header");
    header_end = out.find('\n', header_end);
    if (header_end == std::string::npos) throw std::runtime_error("truncated PTX header");
    out.insert(header_end + 1, "\n" + compile(words) + "\n");
    return {out, true};
}

extern "C" void print_config(int length, char *out) {
    ptxpass::pass_config::PassConfig cfg;
    cfg.name = "pod_typed_selector";
    cfg.description = "Replace POD's typed device selector; never entry-probe fallback";
    cfg.attach_points.includes = {"^kprobe/.*true_fused_tb_fwd_kernel.*$"};
    cfg.attach_type = 8;
    cfg.parameters = nlohmann::json::object();
    cfg.validation = nlohmann::json::object();
    auto result = nlohmann::json(cfg).dump();
    if (length > 0) snprintf(out, length, "%s", result.c_str());
}

extern "C" int process_input(const char *input, int length, char *output) {
    try {
        const auto req = ptxpass::pass_runtime_request_from_string(input);
        if (ptxpass::find_kernel_body(req.input.full_ptx, req.input.to_patch_kernel).first == std::string::npos) {
            // The existing runtime presents every fatbin to each exact target.
            // An unrelated module is not a failed/missing POD attachment.
            auto unchanged = ptxpass::emit_runtime_response_and_return(req.input.full_ptx, false);
            if (length <= 0) return ptxpass::ExitCode::TransformFailed;
            if (unchanged.size() >= static_cast<size_t>(length)) { output[0] = '\0'; return 0; }
            std::memcpy(output, unchanged.c_str(), unchanged.size() + 1);
            return 0;
        }
        auto [out, modified] = patch(req.input.full_ptx, req.get_uint64_ebpf_instructions());
        auto result = ptxpass::emit_runtime_response_and_return(out, modified);
        if (length <= 0) throw std::runtime_error("invalid POD PTX response capacity");
        if (result.size() >= static_cast<size_t>(length)) {
            // Existing bpftime doubles its buffer only when JSON decoding
            // fails. Empty transport output requests that retry; it is never
            // a valid transformed response and contains no partial PTX.
            output[0] = '\0';
            return ptxpass::ExitCode::Success;
        }
        std::memcpy(output, result.c_str(), result.size() + 1);
        return ptxpass::ExitCode::Success;
    } catch (const std::exception &e) {
        std::cerr << "POD PTX adapter: " << e.what() << '\n';
        return ptxpass::ExitCode::TransformFailed;
    }
}

#ifdef POD_PTX_ADAPTER_CLI
int main(int argc, char **argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: pod-ptx-compile selector.bin");
        std::ifstream in(argv[1], std::ios::binary);
        if (!in) throw std::runtime_error("cannot open eBPF bytecode");
        std::vector<char> bytes((std::istreambuf_iterator<char>(in)), {});
        if (bytes.empty() || bytes.size() % 8) throw std::runtime_error("invalid bytecode length");
        std::vector<uint64_t> words(bytes.size() / 8);
        std::memcpy(words.data(), bytes.data(), bytes.size());
        std::cout << compile(words);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
#endif
