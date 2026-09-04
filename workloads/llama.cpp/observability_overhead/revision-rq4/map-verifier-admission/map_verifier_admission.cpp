#include <gpu_verifier.hpp>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include "asm_files.hpp"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#include "build_metadata.hpp"
#include "gpu_platform.hpp"

#include <elfio/elfio.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view kObjectSection = "kprobe/fig15_map_kernel";
constexpr std::array<std::string_view, 7> kExpectedPrograms = {
    "cuda__noop",          "cuda__device_update", "cuda__host_update",
    "cuda__rpc_update",    "cuda__device_lookup", "cuda__host_lookup",
    "cuda__rpc_lookup",
};

struct FunctionSymbol {
  std::string name;
  size_t offset_bytes = 0;
  size_t size_bytes = 0;
};

struct NamedMap {
  std::string name;
  int fd = 0;
  bpftime::verifier::BpftimeMapDescriptor descriptor{};
};

struct ProgramResult {
  FunctionSymbol symbol;
  std::vector<ebpf_inst> instructions;
  std::set<int32_t> helpers;
  std::set<int> map_fds;
  bool accepted = false;
  std::string error;
};

struct ControlResult {
  std::string name;
  bool expected_accept = false;
  bool accepted = false;
  std::string error;
};

EbpfProgramType elf_gpu_program_type(const std::string &,
                                     const std::string &path) {
  return bpftime::gpu_platform_spec.get_program_type(
      "cuda__map_verifier_elf", path);
}

ebpf_platform_t make_elf_platform() {
  auto platform = bpftime::gpu_platform_spec;
  platform.get_program_type = &elf_gpu_program_type;
  return platform;
}

std::string json_escape(std::string_view input) {
  std::string output;
  for (const unsigned char value : input) {
    switch (value) {
    case '"':
      output += "\\\"";
      break;
    case '\\':
      output += "\\\\";
      break;
    case '\n':
      output += "\\n";
      break;
    case '\r':
      output += "\\r";
      break;
    case '\t':
      output += "\\t";
      break;
    default:
      if (value < 0x20) {
        const char digits[] = "0123456789abcdef";
        output += "\\u00";
        output += digits[value >> 4];
        output += digits[value & 0xf];
      } else {
        output += static_cast<char>(value);
      }
    }
  }
  return output;
}

bool expected_program(std::string_view name) {
  return std::find(kExpectedPrograms.begin(), kExpectedPrograms.end(), name) !=
         kExpectedPrograms.end();
}

std::vector<FunctionSymbol> read_function_symbols(const std::string &path) {
  ELFIO::elfio reader;
  if (!reader.load(path)) {
    throw std::runtime_error("cannot parse ELF object");
  }
  const auto *program_section = reader.sections[std::string(kObjectSection)];
  const auto *symbol_section = reader.sections[".symtab"];
  if (program_section == nullptr || symbol_section == nullptr) {
    throw std::runtime_error("ELF lacks the expected program or symbol section");
  }

  ELFIO::const_symbol_section_accessor symbols(reader, symbol_section);
  std::vector<FunctionSymbol> result;
  for (ELFIO::Elf_Xword index = 0; index < symbols.get_symbols_num(); ++index) {
    std::string name;
    ELFIO::Elf64_Addr value{};
    ELFIO::Elf_Xword size{};
    unsigned char bind{};
    unsigned char type{};
    ELFIO::Elf_Half section_index{};
    unsigned char other{};
    if (!symbols.get_symbol(index, name, value, size, bind, type,
                            section_index, other)) {
      continue;
    }
    if (section_index == program_section->get_index() &&
        type == ELFIO::STT_FUNC && expected_program(name)) {
      if (size == 0 || value % sizeof(ebpf_inst) != 0 ||
          size % sizeof(ebpf_inst) != 0 ||
          value + size > program_section->get_size()) {
        throw std::runtime_error("invalid function bounds for " + name);
      }
      result.push_back(FunctionSymbol{name, static_cast<size_t>(value),
                                      static_cast<size_t>(size)});
    }
  }
  std::sort(result.begin(), result.end(), [](const auto &left, const auto &right) {
    return left.offset_bytes < right.offset_bytes;
  });
  if (result.size() != kExpectedPrograms.size()) {
    throw std::runtime_error("ELF does not contain exactly seven target functions");
  }
  for (size_t index = 0; index < result.size(); ++index) {
    if (result[index].name != kExpectedPrograms[index]) {
      throw std::runtime_error("ELF target function order changed");
    }
    if (index != 0 &&
        result[index - 1].offset_bytes + result[index - 1].size_bytes !=
            result[index].offset_bytes) {
      throw std::runtime_error("ELF target functions are not contiguous");
    }
  }
  return result;
}

std::map<int, bpftime::verifier::BpftimeMapDescriptor>
convert_maps(const raw_program &program) {
  std::map<int, bpftime::verifier::BpftimeMapDescriptor> result;
  for (const auto &map : program.info.map_descriptors) {
    const auto [unused, inserted] = result.emplace(
        map.original_fd,
        bpftime::verifier::BpftimeMapDescriptor{
            .original_fd = map.original_fd,
            .type = map.type,
            .key_size = map.key_size,
            .value_size = map.value_size,
            .max_entries = map.max_entries,
            .inner_map_fd = map.inner_map_fd,
        });
    (void)unused;
    if (!inserted) {
      throw std::runtime_error("duplicate pseudo-fd in ELF map descriptors");
    }
  }
  return result;
}

std::map<int, std::string> read_map_names(const std::string &path,
                                          const raw_program &program) {
  ELFIO::elfio reader;
  if (!reader.load(path)) {
    throw std::runtime_error("cannot reopen ELF object for relocations");
  }
  const auto *program_section = reader.sections[std::string(kObjectSection)];
  const auto *maps_section = reader.sections[".maps"];
  const auto *symbol_section = reader.sections[".symtab"];
  const auto *relocation_section =
      reader.sections[std::string(".rel") + std::string(kObjectSection)];
  if (program_section == nullptr || maps_section == nullptr ||
      symbol_section == nullptr || relocation_section == nullptr) {
    throw std::runtime_error("ELF lacks map-relocation metadata");
  }

  ELFIO::const_symbol_section_accessor symbols(reader, symbol_section);
  ELFIO::const_relocation_section_accessor relocations(reader,
                                                        relocation_section);
  std::map<int, std::string> result;
  for (ELFIO::Elf_Xword index = 0; index < relocations.get_entries_num();
       ++index) {
    ELFIO::Elf64_Addr offset{};
    ELFIO::Elf_Word symbol_index{};
    unsigned type{};
    ELFIO::Elf_Sxword addend{};
    if (!relocations.get_entry(index, offset, symbol_index, type, addend)) {
      throw std::runtime_error("cannot read ELF relocation");
    }
    std::string name;
    ELFIO::Elf64_Addr value{};
    ELFIO::Elf_Xword size{};
    unsigned char bind{};
    unsigned char symbol_type{};
    ELFIO::Elf_Half section_index{};
    unsigned char other{};
    if (!symbols.get_symbol(symbol_index, name, value, size, bind, symbol_type,
                            section_index, other)) {
      throw std::runtime_error("cannot read relocation symbol");
    }
    if (section_index != maps_section->get_index()) {
      throw std::runtime_error("unexpected non-map relocation in BPF section");
    }
    if (offset % sizeof(ebpf_inst) != 0 ||
        offset / sizeof(ebpf_inst) >= program.prog.size()) {
      throw std::runtime_error("map relocation lies outside program section");
    }
    const auto &instruction = program.prog[offset / sizeof(ebpf_inst)];
    if (instruction.opcode != INST_OP_LDDW_IMM || instruction.src != 1) {
      throw std::runtime_error("map relocation was not applied as pseudo-fd");
    }
    const int fd = instruction.imm;
    const auto existing = result.find(fd);
    if (existing != result.end() && existing->second != name) {
      throw std::runtime_error("one pseudo-fd names multiple maps");
    }
    result[fd] = name;
  }
  if (result.size() != program.info.map_descriptors.size()) {
    throw std::runtime_error("not every ELF map descriptor is named by relocation");
  }
  return result;
}

std::vector<NamedMap>
name_maps(const std::map<int, bpftime::verifier::BpftimeMapDescriptor> &maps,
          const std::map<int, std::string> &names) {
  std::vector<NamedMap> result;
  for (const auto &[fd, descriptor] : maps) {
    const auto name = names.find(fd);
    if (name == names.end()) {
      throw std::runtime_error("missing name for map pseudo-fd");
    }
    result.push_back(NamedMap{name->second, fd, descriptor});
  }
  std::sort(result.begin(), result.end(), [](const auto &left, const auto &right) {
    return left.name < right.name;
  });
  return result;
}

ProgramResult verify_function(
    const FunctionSymbol &symbol, const raw_program &whole_program,
    const std::map<int, bpftime::verifier::BpftimeMapDescriptor> &maps) {
  const size_t first = symbol.offset_bytes / sizeof(ebpf_inst);
  const size_t count = symbol.size_bytes / sizeof(ebpf_inst);
  ProgramResult result;
  result.symbol = symbol;
  result.instructions.assign(whole_program.prog.begin() + first,
                             whole_program.prog.begin() + first + count);
  for (const auto &instruction : result.instructions) {
    if (instruction.opcode == INST_OP_CALL) {
      result.helpers.insert(instruction.imm);
    }
    if (instruction.opcode == INST_OP_LDDW_IMM && instruction.src == 1) {
      if (!maps.contains(instruction.imm)) {
        throw std::runtime_error("program references an undescribed map pseudo-fd");
      }
      result.map_fds.insert(instruction.imm);
    }
  }
  const auto error = bpftime::verifier::gpu::verify_gpu_program(
      result.instructions.data(), result.instructions.size(), symbol.name, maps);
  result.accepted = !error.has_value();
  result.error = error.value_or("");
  return result;
}

ebpf_inst make_instruction(uint8_t opcode, uint8_t destination = 0,
                           uint8_t source = 0, int16_t offset = 0,
                           int32_t immediate = 0) {
  ebpf_inst instruction{};
  instruction.opcode = opcode;
  instruction.dst = destination;
  instruction.src = source;
  instruction.offset = offset;
  instruction.imm = immediate;
  return instruction;
}

ControlResult run_control(
    std::string name, bool expected_accept,
    const std::vector<ebpf_inst> &instructions,
    const std::map<int, bpftime::verifier::BpftimeMapDescriptor> &maps = {}) {
  const auto error = bpftime::verifier::gpu::verify_gpu_program(
      instructions.data(), instructions.size(), "cuda__" + name, maps);
  return ControlResult{std::move(name), expected_accept, !error.has_value(),
                       error.value_or("")};
}

std::string helper_uniformity(bpftime::GpuHelperUniformity value) {
  return value == bpftime::GpuHelperUniformity::UNIFORM ? "uniform" :
                                                          "varying";
}

std::string helper_effect(bpftime::GpuHelperEffectClass value) {
  return value == bpftime::GpuHelperEffectClass::PROHIBITED ? "prohibited" :
                                                              "none";
}

std::string helper_behavior(bpftime::GpuHelperBehavior value) {
  switch (value) {
  case bpftime::GpuHelperBehavior::GENERIC:
    return "generic";
  case bpftime::GpuHelperBehavior::MAP_LOOKUP:
    return "map_lookup";
  case bpftime::GpuHelperBehavior::MAP_UPDATE:
    return "map_update";
  case bpftime::GpuHelperBehavior::MAP_DELETE:
    return "map_delete";
  }
  throw std::logic_error("unknown helper behavior");
}

void write_helper(int32_t id) {
  const auto *helper = bpftime::find_gpu_helper_prototype(id);
  std::cout << "{\"id\":" << id << ",\"known\":"
            << (helper != nullptr ? "true" : "false");
  if (helper != nullptr) {
    std::cout << ",\"name\":\"" << json_escape(helper->name)
              << "\",\"return_uniformity\":\""
              << helper_uniformity(helper->return_uniformity)
              << "\",\"effect\":\"" << helper_effect(helper->effect_class)
              << "\",\"behavior\":\""
              << helper_behavior(helper->behavior) << "\"";
  }
  std::cout << "}";
}

void write_json(const std::string &path, size_t object_size,
                size_t section_instructions, size_t relocation_count,
                const std::vector<NamedMap> &maps,
                const std::map<int, std::string> &map_names,
                const std::vector<ProgramResult> &programs,
                const std::vector<ControlResult> &controls) {
  size_t accepted_programs = 0;
  for (const auto &program : programs) {
    accepted_programs += program.accepted ? 1 : 0;
  }
  bool controls_passed = true;
  for (const auto &control : controls) {
    controls_passed =
        controls_passed && control.accepted == control.expected_accept;
  }

  std::cout << "{\"schema\":\"map-verifier-admission-v1\""
            << ",\"scope\":\"CPU-only direct verify_gpu_program admission; "
               "not GPU execution or attach safety\""
            << ",\"bpftime_source_revision\":\""
            << kBpftimeSourceRevision << "\",\"build_type\":\""
            << kAdmissionBuildType << "\",\"elf\":{\"path\":\""
            << json_escape(path) << "\",\"size_bytes\":" << object_size
            << ",\"section\":\"" << kObjectSection
            << "\",\"section_instructions\":" << section_instructions
            << ",\"map_relocations_applied\":" << relocation_count << "}";

  std::cout << ",\"maps\":[";
  for (size_t index = 0; index < maps.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    const auto &map = maps[index];
    std::cout << "{\"name\":\"" << json_escape(map.name)
              << "\",\"pseudo_fd\":" << map.fd << ",\"type\":"
              << map.descriptor.type << ",\"key_size\":"
              << map.descriptor.key_size << ",\"value_size\":"
              << map.descriptor.value_size << ",\"max_entries\":"
              << map.descriptor.max_entries << "}";
  }
  std::cout << "]";

  std::cout << ",\"programs\":[";
  for (size_t index = 0; index < programs.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    const auto &program = programs[index];
    std::cout << "{\"name\":\"" << json_escape(program.symbol.name)
              << "\",\"elf_offset_bytes\":" << program.symbol.offset_bytes
              << ",\"elf_size_bytes\":" << program.symbol.size_bytes
              << ",\"instructions\":" << program.instructions.size()
              << ",\"helpers\":[";
    size_t helper_index = 0;
    for (const int32_t helper : program.helpers) {
      if (helper_index++ != 0) {
        std::cout << ',';
      }
      write_helper(helper);
    }
    std::cout << "],\"maps\":[";
    size_t map_index = 0;
    for (const int fd : program.map_fds) {
      if (map_index++ != 0) {
        std::cout << ',';
      }
      std::cout << "{\"name\":\"" << json_escape(map_names.at(fd))
                << "\",\"pseudo_fd\":" << fd << "}";
    }
    std::cout << "],\"accepted\":"
              << (program.accepted ? "true" : "false")
              << ",\"error\":\"" << json_escape(program.error) << "\"}";
  }
  std::cout << "]";

  std::cout << ",\"controls\":[";
  for (size_t index = 0; index < controls.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    const auto &control = controls[index];
    std::cout << "{\"name\":\"" << json_escape(control.name)
              << "\",\"expected\":\""
              << (control.expected_accept ? "accept" : "reject")
              << "\",\"accepted\":"
              << (control.accepted ? "true" : "false")
              << ",\"error\":\"" << json_escape(control.error) << "\"}";
  }
  std::cout << "]";

  std::cout << ",\"summary\":{\"target_programs\":" << programs.size()
            << ",\"accepted\":" << accepted_programs
            << ",\"rejected\":" << programs.size() - accepted_programs
            << ",\"control_expectations_met\":"
            << (controls_passed ? "true" : "false") << "}}\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3 || std::string_view(argv[1]) != "--object") {
      std::cerr << "usage: map_verifier_admission --object BPF_ELF\n";
      return 64;
    }
    const std::string path = std::filesystem::canonical(argv[2]).string();
    if (!std::filesystem::is_regular_file(path)) {
      throw std::runtime_error("BPF ELF is not a regular file");
    }

    auto elf_platform = make_elf_platform();
    const auto elf_programs =
        read_elf(path, std::string(kObjectSection), nullptr, &elf_platform);
    if (elf_programs.size() != 1 ||
        elf_programs.front().section != kObjectSection) {
      throw std::runtime_error("ELF reader did not return exactly one section");
    }
    const auto &whole_program = elf_programs.front();
    const auto symbols = read_function_symbols(path);
    if (symbols.front().offset_bytes != 0 ||
        symbols.back().offset_bytes + symbols.back().size_bytes !=
            whole_program.prog.size() * sizeof(ebpf_inst)) {
      throw std::runtime_error("function symbols do not cover the ELF section");
    }

    const auto map_descriptors = convert_maps(whole_program);
    const auto map_names = read_map_names(path, whole_program);
    const auto named_maps = name_maps(map_descriptors, map_names);
    std::vector<ProgramResult> results;
    for (const auto &symbol : symbols) {
      results.push_back(
          verify_function(symbol, whole_program, map_descriptors));
    }

    const std::vector<ebpf_inst> positive = {
        make_instruction(INST_CLS_ALU64 | INST_SRC_IMM | INST_ALU_OP_MOV, 0,
                         0, 0, 0),
        make_instruction(INST_OP_EXIT),
    };
    const std::vector<ebpf_inst> unknown_helper = {
        make_instruction(INST_OP_CALL, 0, 0, 0, 512),
        make_instruction(INST_CLS_ALU64 | INST_SRC_IMM | INST_ALU_OP_MOV, 0,
                         0, 0, 0),
        make_instruction(INST_OP_EXIT),
    };
    const std::vector<ebpf_inst> varying_branch = {
        make_instruction(INST_OP_CALL, 0, 0, 0, 511),
        make_instruction(INST_CLS_JMP | INST_SRC_IMM | 0x10, 0, 0, 1, 0),
        make_instruction(INST_CLS_ALU64 | INST_SRC_IMM | INST_ALU_OP_MOV, 0,
                         0, 0, 0),
        make_instruction(INST_OP_EXIT),
    };
    auto unsupported_maps = map_descriptors;
    unsupported_maps.emplace(
        4095, bpftime::verifier::BpftimeMapDescriptor{
                  .original_fd = 4095,
                  .type = 1599,
                  .key_size = 4,
                  .value_size = 8,
                  .max_entries = 1,
                  .inner_map_fd = static_cast<unsigned int>(-1),
              });
    const std::vector<ControlResult> controls = {
        run_control("positive_minimal", true, positive),
        run_control("negative_unknown_helper", false, unknown_helper),
        run_control("negative_varying_branch", false, varying_branch),
        run_control("negative_unsupported_gpu_map", false, positive,
                    unsupported_maps),
    };

    size_t relocation_count = 0;
    for (const auto &result : results) {
      relocation_count += result.map_fds.size();
    }
    write_json(path, std::filesystem::file_size(path),
               whole_program.prog.size(), relocation_count, named_maps,
               map_names, results, controls);
    const bool controls_passed = std::all_of(
        controls.begin(), controls.end(), [](const auto &control) {
          return control.accepted == control.expected_accept;
        });
    return controls_passed ? 0 : 66;
  } catch (const std::exception &error) {
    std::cerr << "map verifier admission error: " << error.what() << '\n';
    return 70;
  }
}
