// SPDX-License-Identifier: MIT
// CPU-only tests for SIMT checks not covered by gpu_revision_safety_test.cpp.

#include <bpftime-verifier.hpp>
#include <ebpf_vm_isa.hpp>
#include <gpu_verifier.hpp>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <optional>
#include <string>

using namespace bpftime;
using namespace bpftime::verifier;
using namespace bpftime::verifier::gpu;

namespace
{

ebpf_inst insn(uint8_t opcode, uint8_t dst = 0, uint8_t src = 0,
	       int16_t offset = 0, int32_t immediate = 0)
{
	ebpf_inst instruction{};
	instruction.opcode = opcode;
	instruction.dst = dst;
	instruction.src = src;
	instruction.offset = offset;
	instruction.imm = immediate;
	return instruction;
}

ebpf_inst mov_imm(uint8_t dst, int32_t immediate)
{
	return insn(INST_CLS_ALU64 | INST_SRC_IMM | INST_ALU_OP_MOV, dst, 0,
		    0, immediate);
}

ebpf_inst mov_reg(uint8_t dst, uint8_t src)
{
	return insn(INST_CLS_ALU64 | INST_SRC_REG | INST_ALU_OP_MOV, dst, src);
}

ebpf_inst add_imm(uint8_t dst, int32_t immediate)
{
	return insn(INST_CLS_ALU64 | INST_SRC_IMM | INST_ALU_OP_ADD, dst, 0,
		    0, immediate);
}

ebpf_inst call(int32_t helper)
{
	return insn(INST_OP_CALL, 0, 0, 0, helper);
}

ebpf_inst store_imm(uint8_t dst, int16_t offset, int32_t immediate)
{
	return insn(INST_CLS_ST | INST_SIZE_DW | (INST_MEM << 5), dst, 0,
		    offset, immediate);
}

ebpf_inst store_reg(uint8_t dst, uint8_t src, int16_t offset = 0)
{
	return insn(INST_CLS_STX | INST_SIZE_DW | (INST_MEM << 5), dst, src,
		    offset);
}

ebpf_inst load_map(uint8_t dst, int32_t fd)
{
	return insn(INST_OP_LDDW_IMM, dst, 1, 0, fd);
}

ebpf_inst jump_equal_imm(uint8_t dst, int32_t immediate, int16_t offset)
{
	return insn(INST_CLS_JMP | INST_SRC_IMM | 0x10, dst, 0, offset,
		    immediate);
}

ebpf_inst exit_insn()
{
	return insn(INST_OP_EXIT);
}

BpftimeMapDescriptor shared_array_map(int fd, uint32_t value_size = 8)
{
	return BpftimeMapDescriptor{
		.original_fd = fd,
		.type = 1503,
		.key_size = 4,
		.value_size = value_size,
		.max_entries = 16,
		.inner_map_fd = static_cast<unsigned int>(-1),
	};
}

template <size_t N>
std::optional<std::string>
verify(const std::array<ebpf_inst, N> &program,
       const std::map<int, BpftimeMapDescriptor> &maps = {})
{
	return verify_gpu_program(program.data(), program.size(),
				  "cuda__additional_safety_matrix", maps);
}

[[noreturn]] void fail(const std::string &name, const std::string &message)
{
	std::cerr << "FAIL layer=simt case=" << name << " reason=" << message
		  << '\n';
	std::exit(EXIT_FAILURE);
}

void expect_pair(const std::string &name,
		 const std::optional<std::string> &unsafe_result,
		 const std::string &diagnostic,
		 const std::optional<std::string> &control_result)
{
	if (!unsafe_result) {
		fail(name, "unsafe program was accepted");
	}
	if (unsafe_result->find(diagnostic) == std::string::npos) {
		fail(name, "missing diagnostic: " + diagnostic + "; got: " +
				   *unsafe_result);
	}
	if (control_result) {
		fail(name, "matched control was rejected: " + *control_result);
	}
	std::cout << "PASS layer=simt case=" << name
		  << " unsafe=rejected control=accepted diagnostic=\""
		  << diagnostic << "\"\n";
}

template <int Helper>
std::array<ebpf_inst, 12> direct_shared_store_program()
{
	return {
		store_imm(10, -8, 0), load_map(1, 1), {}, mov_reg(2, 10),
		add_imm(2, -8),       call(1),         jump_equal_imm(0, 0, 3),
		mov_reg(6, 0),        call(Helper),     store_reg(6, 0),
		mov_imm(0, 0),        exit_insn(),
	};
}

template <int Helper>
std::array<ebpf_inst, 13> shared_helper_output_program()
{
	return {
		store_imm(10, -8, 0), load_map(1, 1), {}, mov_reg(2, 10),
		add_imm(2, -8),       call(1),         jump_equal_imm(0, 0, 4),
		mov_reg(1, 0),        mov_reg(2, 0),    mov_reg(3, 0),
		call(Helper),          mov_imm(0, 0),    exit_insn(),
	};
}

std::array<ebpf_inst, 14> map_update_flags_program(bool lane_varying)
{
	return {
		lane_varying ? call(511) : mov_imm(0, 0),
		mov_reg(6, 0),         store_imm(10, -8, 0),
		store_imm(10, -16, 7), load_map(1, 1),
		{},                    mov_reg(2, 10),
		add_imm(2, -8),        mov_reg(3, 10),
		add_imm(3, -16),       mov_reg(4, 6),
		call(2),               mov_imm(0, 0),
		exit_insn(),
	};
}

template <int Helper>
std::array<ebpf_inst, 11> trace_payload_program()
{
	return {
		call(Helper),       mov_reg(6, 0),      store_imm(10, -8, 0x6425),
		mov_reg(1, 10),     add_imm(1, -8),     mov_imm(2, 3),
		mov_reg(3, 6),      mov_imm(4, 0),      mov_imm(5, 0),
		call(6),            exit_insn(),
	};
}

} // namespace

int main()
{
	const std::map<int, BpftimeMapDescriptor> maps = {
		{ 1, shared_array_map(1) },
	};
	const std::map<int, BpftimeMapDescriptor> wide_maps = {
		{ 1, shared_array_map(1, 24) },
	};

	expect_pair("direct-shared-store",
		    verify(direct_shared_store_program<511>(), maps),
		    "Shared Map Value Uniformity",
		    verify(direct_shared_store_program<510>(), maps));
	expect_pair("varying-helper-output-to-shared-map",
		    verify(shared_helper_output_program<505>(), wide_maps),
		    "Shared Map Value Uniformity",
		    verify(shared_helper_output_program<503>(), wide_maps));
	expect_pair("map-update-flags",
		    verify(map_update_flags_program(true), maps),
		    "Map Helper Key Uniformity",
		    verify(map_update_flags_program(false), maps));
	expect_pair("host-bridge-payload", verify(trace_payload_program<511>()),
		    "Host Bridge Payload Uniformity",
		    verify(trace_payload_program<510>()));

	std::cout << "PASS all: 4 additional SIMT pairs\n";
	return EXIT_SUCCESS;
}
