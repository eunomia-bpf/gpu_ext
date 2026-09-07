// bpf_to_ptx: bounded compiler exporter for the SASS application-injection
// task. Loads a real clang-built BPF ELF section, runs the existing strict
// GPU verifier (8-byte PREVAIL context + uniformity + SIMT checks), and then
// emits a standalone, device-callable PTX file through the existing
// ptxpass eBPF-to-PTX compiler.
//
// CLI:
//   bpf_to_ptx INPUT_BPF_OBJECT SECTION OUTPUT_PTX [SYMBOL=bpf_exit] [SM=sm_120]
//
// Output ABI (untouched compiler output, with_arguments=true):
//   .visible .func SYMBOL(.param .b64 context_ptr,
//                         .param .b64 context_length)
// Parameter 0 is the verify-time 8-byte read/write context pointer (R1),
// parameter 1 is its length in bytes. This exporter never promotes the
// function to .entry and never edits the generated body.
//
// PTX target split (do not "fix" this):
//   - LLVM 15 NVPTX codegen sees sm_86 (LLVM15-supported; sm_120 is not).
//   - The standalone header carries the real ptxas target: .version 8.7 and
//     .target SM (default sm_120, CUDA 12.9 ptxas). PTX is
//     forward-compatible, so sm_86 codegen is valid input for a sm_120 ptxas.
//
// Exit codes: 0 ok, 1 usage, 2 BPF ELF load error, 3 verifier rejected,
// 4 eBPF-to-PTX compile failure, 5 output failure.

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <elf.h>
#include <fcntl.h>
#include <fstream>
#include <gelf.h>
#include <iostream>
#include <libelf.h>
#include <optional>
#include <string>
#include <unistd.h>
#include <vector>

#include "gpu_verifier.hpp"
#include "ptxpass/core.hpp"

namespace {

constexpr const char *DEFAULT_SYMBOL = "bpf_exit";
constexpr const char *DEFAULT_SM = "sm_120";
// LLVM 15 cannot emit sm_120; keep the proven LLVM-side NVPTX target and let
// the standalone header below carry the actual ptxas SM.
constexpr const char *LLVM_PTX_CODEGEN_TARGET = "sm_86";
constexpr const char *PTX_ISA_VERSION = "8.7";
// Exact read/write context proven by PREVAIL and handed to the generated
// function as its two ABI parameters (pointer, length).
constexpr size_t PREVAIL_GPU_CONTEXT_SIZE = sizeof(uint64_t);

// Semantically identical to the fd976ea sass_aot reference loader: extract
// the eBPF instruction words of the named section from a real BPF ELF
// object. Returns an error string, or std::nullopt on success.
std::optional<std::string> load_bpf_program_words(
	const std::string &object_path, const std::string &section_name,
	std::vector<uint64_t> &words, std::string &matched_section)
{
	words.clear();
	matched_section.clear();
	if (::elf_version(EV_CURRENT) == EV_NONE)
		return "libelf initialization failed";
	const int fd = ::open(object_path.c_str(), O_RDONLY);
	if (fd < 0)
		return "cannot open BPF object: " + object_path;
	Elf *elf = elf_begin(fd, ELF_C_READ, nullptr);
	if (!elf) {
		std::string error =
			std::string("elf_begin failed: ") + elf_errmsg(-1);
		::close(fd);
		return error;
	}
	auto finish = [&](std::optional<std::string> error) {
		elf_end(elf);
		::close(fd);
		return error;
	};
	GElf_Ehdr ehdr{};
	if (elf_kind(elf) != ELF_K_ELF || gelf_getclass(elf) != ELFCLASS64 ||
	    !gelf_getehdr(elf, &ehdr) || ehdr.e_machine != EM_BPF) {
		return finish("not a 64-bit ELF BPF object: " + object_path);
	}
	size_t shstrndx = 0;
	if (elf_getshdrstrndx(elf, &shstrndx) != 0)
		return finish("cannot read ELF section-name table: " +
			      object_path);
	Elf_Scn *scn = nullptr;
	while ((scn = elf_nextscn(elf, scn)) != nullptr) {
		GElf_Shdr shdr{};
		if (!gelf_getshdr(scn, &shdr))
			continue;
		const char *name = elf_strptr(elf, shstrndx, shdr.sh_name);
		if (!name || section_name != name)
			continue;
		if (shdr.sh_type != SHT_PROGBITS ||
		    (shdr.sh_flags & (SHF_ALLOC | SHF_EXECINSTR)) !=
			    (SHF_ALLOC | SHF_EXECINSTR) ||
		    shdr.sh_size == 0 || shdr.sh_size % sizeof(uint64_t) != 0) {
			return finish("section " + std::string(name) +
				      " is not BPF instruction data");
		}
		Elf_Data *data = elf_getdata(scn, nullptr);
		if (!data || !data->d_buf || data->d_size != shdr.sh_size) {
			return finish("elf_getdata failed for section " +
				      std::string(name));
		}
		words.resize(shdr.sh_size / sizeof(uint64_t));
		std::memcpy(words.data(), data->d_buf, data->d_size);
		matched_section = name;
		return finish(std::nullopt);
	}
	return finish("no BPF program section named " + section_name + " in " +
		      object_path);
}

bool write_text_file(const std::string &path, const std::string &content)
{
	std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
	if (!ofs)
		return false;
	ofs << content;
	return ofs.good();
}

} // namespace

int main(int argc, char **argv)
{
	if (argc < 4 || argc > 6) {
		std::cerr << "usage: " << (argc > 0 ? argv[0] : "bpf_to_ptx")
			  << " INPUT_BPF_OBJECT SECTION OUTPUT_PTX"
			  << " [SYMBOL=" << DEFAULT_SYMBOL << "]"
			  << " [SM=" << DEFAULT_SM << "]" << std::endl;
		return 1;
	}
	const std::string object_path = argv[1];
	const std::string section_name = argv[2];
	const std::string output_path = argv[3];
	const std::string symbol = argc > 4 ? argv[4] : DEFAULT_SYMBOL;
	const std::string sm = argc > 5 ? argv[5] : DEFAULT_SM;

	// Stage 1: load the actual clang-built BPF ELF section.
	std::vector<uint64_t> words;
	std::string matched_section;
	if (auto error = load_bpf_program_words(object_path, section_name,
						words, matched_section)) {
		std::cerr << "[bpf_to_ptx] BPF ELF load failed: " << *error
			  << std::endl;
		return 2;
	}
	std::cerr << "[bpf_to_ptx] loaded " << words.size()
		  << " eBPF instruction words from section "
		  << matched_section << " of " << object_path << std::endl;

	// Stage 2: existing strict GPU verifier. context_size=8 gives PREVAIL
	// an 8-byte read/write context descriptor; the same call chain also
	// runs the uniformity and SIMT safety checks. Rejected programs never
	// reach the PTX compiler.
	if (auto verifier_error =
		    bpftime::verifier::gpu::verify_gpu_program_with_context(
			    words.data(), words.size(), section_name,
			    PREVAIL_GPU_CONTEXT_SIZE, {})) {
		std::cerr << "[bpf_to_ptx] GPU verifier rejected program: "
			  << *verifier_error << std::endl;
		return 3;
	}
	std::cerr << "[bpf_to_ptx] GPU verifier accepted program"
		  << " (8-byte PREVAIL context, uniformity + SIMT checks)"
		  << std::endl;

	// Stage 3: existing ptxpass compiler. The function body and ABI come
	// straight from the compiler; add_register_guard_... stays false and
	// with_arguments=true keeps the two-parameter device-callable form.
	std::string func_ptx;
	try {
		func_ptx = ptxpass::compile_ebpf_to_ptx_from_words(
			words, LLVM_PTX_CODEGEN_TARGET, symbol,
			/*add_register_guard_and_filter_version_headers=*/
			false,
			/*with_arguments=*/true);
	} catch (const std::exception &ex) {
		std::cerr << "[bpf_to_ptx] eBPF-to-PTX compilation failed: "
			  << ex.what() << std::endl;
		return 4;
	} catch (...) {
		std::cerr << "[bpf_to_ptx] eBPF-to-PTX compilation failed: "
			  << "unknown error" << std::endl;
		return 4;
	}
	if (func_ptx.empty()) {
		std::cerr << "[bpf_to_ptx] eBPF-to-PTX compilation produced"
			  << " no PTX" << std::endl;
		return 4;
	}

	// The compiler emits a plain, device-callable function; require its
	// exact marker instead of promoting anything to .entry.
	if (func_ptx.find(".visible .func " + symbol + "(") ==
	    std::string::npos) {
		std::cerr << "[bpf_to_ptx] generated PTX does not contain .visible .func "
			  << symbol << " with the expected two-parameter ABI"
			  << std::endl;
		return 4;
	}

	// Stage 4: standalone PTX translation unit. Only headers are added;
	// the function body stays byte-identical compiler output.
	const std::string ptx = std::string(".version ") + PTX_ISA_VERSION +
				"\n.target " + sm + "\n.address_size 64\n" +
				func_ptx;
	if (!write_text_file(output_path, ptx)) {
		std::cerr << "[bpf_to_ptx] cannot write PTX file: "
			  << output_path << std::endl;
		return 5;
	}

	std::cerr << "[bpf_to_ptx] wrote standalone PTX: " << output_path
		  << " (.visible .func " << symbol
		  << "(context_ptr, context_length); ptxas .target " << sm
		  << "; LLVM codegen target " << LLVM_PTX_CODEGEN_TARGET << ")"
		  << std::endl;
	return 0;
}
