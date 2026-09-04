#include <gpu_verifier.hpp>

#include "build_metadata.hpp"

#include <ebpf_vm_isa.hpp>

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr std::array<size_t, 5> kAllowedSizes = {16, 64, 256, 1024, 4096};
constexpr int32_t kWarpIdHelper = 510;
constexpr std::string_view kSection = "cuda__verifier_scaling";

enum class Family { Linear, Diamonds };
enum class Mode { Describe, AcceptOnly, Timed };

struct Options {
  Family family = Family::Linear;
  size_t instructions = 0;
  Mode mode = Mode::Timed;
  std::optional<int> required_cpu;
};

struct Shape {
  size_t instruction_count = 0;
  size_t conditional_branches = 0;
  size_t helper_calls = 0;
  size_t exits = 0;
  std::optional<int16_t> minimum_branch_offset;
  std::optional<int16_t> maximum_branch_offset;
};

ebpf_inst make_instruction(uint8_t opcode, uint8_t dst = 0, uint8_t src = 0,
                           int16_t offset = 0, int32_t immediate = 0) {
  ebpf_inst instruction{};
  instruction.opcode = opcode;
  instruction.dst = dst;
  instruction.src = src;
  instruction.offset = offset;
  instruction.imm = immediate;
  return instruction;
}

ebpf_inst make_call(int32_t helper) {
  return make_instruction(INST_OP_CALL, 0, 0, 0, helper);
}

ebpf_inst make_mov64_register(uint8_t destination, uint8_t source) {
  return make_instruction(INST_CLS_ALU64 | INST_SRC_REG | INST_ALU_OP_MOV,
                          destination, source);
}

ebpf_inst make_mov64_immediate(uint8_t destination, int32_t immediate) {
  return make_instruction(INST_CLS_ALU64 | INST_SRC_IMM | INST_ALU_OP_MOV,
                          destination, 0, 0, immediate);
}

ebpf_inst make_add64_immediate(uint8_t destination, int32_t immediate) {
  return make_instruction(INST_CLS_ALU64 | INST_SRC_IMM | INST_ALU_OP_ADD,
                          destination, 0, 0, immediate);
}

ebpf_inst make_jeq_immediate(uint8_t destination, int32_t immediate,
                             int16_t offset) {
  return make_instruction(INST_CLS_JMP | INST_SRC_IMM | 0x10, destination, 0,
                          offset, immediate);
}

ebpf_inst make_exit() { return make_instruction(INST_OP_EXIT); }

bool same_instruction(const ebpf_inst &left, const ebpf_inst &right) {
  return left.opcode == right.opcode && left.dst == right.dst &&
         left.src == right.src && left.offset == right.offset &&
         left.imm == right.imm;
}

bool is_allowed_size(size_t size) {
  for (const auto allowed : kAllowedSizes) {
    if (size == allowed) {
      return true;
    }
  }
  return false;
}

std::vector<ebpf_inst> build_program(Family family, size_t size) {
  if (!is_allowed_size(size)) {
    throw std::invalid_argument("instruction count is outside frozen set");
  }

  std::vector<ebpf_inst> program;
  program.reserve(size);
  program.push_back(make_call(kWarpIdHelper));
  program.push_back(make_mov64_register(1, 0));
  program.push_back(make_mov64_immediate(0, 0));

  const size_t body_size = size - 4;
  if (family == Family::Linear) {
    for (size_t index = 0; index < body_size; ++index) {
      program.push_back(make_add64_immediate(0, 1));
    }
  } else {
    if (body_size % 2 != 0) {
      throw std::logic_error("diamond body must contain whole pairs");
    }
    for (size_t index = 0; index < body_size / 2; ++index) {
      program.push_back(make_jeq_immediate(1, static_cast<int32_t>(index), 1));
      program.push_back(make_add64_immediate(0, 1));
    }
  }
  program.push_back(make_exit());
  return program;
}

bool is_conditional_jump(const ebpf_inst &instruction) {
  const uint8_t cls = instruction.opcode & INST_CLS_MASK;
  return (cls == INST_CLS_JMP || cls == INST_CLS_JMP32) &&
         instruction.opcode != INST_OP_CALL &&
         instruction.opcode != INST_OP_EXIT && instruction.opcode != INST_OP_JA;
}

Shape validate_shape(const std::vector<ebpf_inst> &program, Family family,
                     size_t requested_size) {
  if (program.size() != requested_size) {
    throw std::logic_error("constructed instruction count mismatch");
  }
  if (!same_instruction(program.at(0), make_call(kWarpIdHelper)) ||
      !same_instruction(program.at(1), make_mov64_register(1, 0)) ||
      !same_instruction(program.at(2), make_mov64_immediate(0, 0)) ||
      !same_instruction(program.back(), make_exit())) {
    throw std::logic_error("frozen prefix or exit mismatch");
  }

  Shape shape;
  shape.instruction_count = program.size();
  for (size_t pc = 0; pc < program.size(); ++pc) {
    const auto &instruction = program[pc];
    if (instruction.opcode == INST_OP_CALL) {
      ++shape.helper_calls;
    }
    if (instruction.opcode == INST_OP_EXIT) {
      ++shape.exits;
    }
    if (is_conditional_jump(instruction)) {
      ++shape.conditional_branches;
      const int64_t target = static_cast<int64_t>(pc) + 1 + instruction.offset;
      if (target <= static_cast<int64_t>(pc) || target < 0 ||
          target >= static_cast<int64_t>(program.size())) {
        throw std::logic_error("branch target is not forward/in-range");
      }
      if (!shape.minimum_branch_offset ||
          instruction.offset < *shape.minimum_branch_offset) {
        shape.minimum_branch_offset = instruction.offset;
      }
      if (!shape.maximum_branch_offset ||
          instruction.offset > *shape.maximum_branch_offset) {
        shape.maximum_branch_offset = instruction.offset;
      }
    }
  }

  const size_t expected_branches =
      family == Family::Diamonds ? (requested_size - 4) / 2 : 0;
  if (shape.helper_calls != 1 || shape.exits != 1 ||
      shape.conditional_branches != expected_branches) {
    throw std::logic_error("structural count mismatch");
  }

  for (size_t pc = 3; pc + 1 < program.size(); ++pc) {
    const size_t body_index = pc - 3;
    if (family == Family::Linear) {
      if (!same_instruction(program[pc], make_add64_immediate(0, 1))) {
        throw std::logic_error("linear body mismatch");
      }
    } else if (body_index % 2 == 0) {
      const auto expected =
          make_jeq_immediate(1, static_cast<int32_t>(body_index / 2), 1);
      if (!same_instruction(program[pc], expected)) {
        throw std::logic_error("diamond branch mismatch");
      }
    } else if (!same_instruction(program[pc], make_add64_immediate(0, 1))) {
      throw std::logic_error("diamond ALU body mismatch");
    }
  }

  if (family == Family::Linear &&
      (shape.minimum_branch_offset || shape.maximum_branch_offset)) {
    throw std::logic_error("linear family unexpectedly contains a branch");
  }
  if (family == Family::Diamonds &&
      (shape.minimum_branch_offset != 1 || shape.maximum_branch_offset != 1)) {
    throw std::logic_error("diamond branch displacement mismatch");
  }
  return shape;
}

std::string family_name(Family family) {
  return family == Family::Linear ? "linear" : "diamonds";
}

std::string mode_name(Mode mode) {
  switch (mode) {
  case Mode::Describe:
    return "describe";
  case Mode::AcceptOnly:
    return "accept_only";
  case Mode::Timed:
    return "timed";
  }
  throw std::logic_error("unknown mode");
}

size_t parse_size(const char *text) {
  char *end = nullptr;
  errno = 0;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') {
    throw std::invalid_argument("invalid instruction count");
  }
  return static_cast<size_t>(value);
}

int parse_cpu(const char *text) {
  char *end = nullptr;
  errno = 0;
  const long value = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value < 0 ||
      value >= CPU_SETSIZE) {
    throw std::invalid_argument("invalid CPU");
  }
  return static_cast<int>(value);
}

Options parse_options(int argc, char **argv) {
  Options options;
  bool have_family = false;
  bool have_instructions = false;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--family" && index + 1 < argc) {
      const std::string_view value(argv[++index]);
      if (value == "linear") {
        options.family = Family::Linear;
      } else if (value == "diamonds") {
        options.family = Family::Diamonds;
      } else {
        throw std::invalid_argument("unknown family");
      }
      have_family = true;
    } else if (argument == "--instructions" && index + 1 < argc) {
      options.instructions = parse_size(argv[++index]);
      have_instructions = true;
    } else if (argument == "--describe") {
      options.mode = Mode::Describe;
    } else if (argument == "--accept-only") {
      options.mode = Mode::AcceptOnly;
    } else if (argument == "--require-cpu" && index + 1 < argc) {
      options.required_cpu = parse_cpu(argv[++index]);
    } else {
      throw std::invalid_argument("unknown or incomplete argument");
    }
  }
  if (!have_family || !have_instructions) {
    throw std::invalid_argument("--family and --instructions are required");
  }
  return options;
}

void require_affinity(int cpu) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) != 0) {
    throw std::runtime_error("sched_getaffinity failed");
  }
  if (CPU_COUNT(&mask) != 1 || !CPU_ISSET(cpu, &mask)) {
    throw std::runtime_error("process affinity differs from required CPU");
  }
}

int64_t timespec_delta_ns(const timespec &before, const timespec &after) {
  return (static_cast<int64_t>(after.tv_sec) - before.tv_sec) * 1000000000LL +
         (static_cast<int64_t>(after.tv_nsec) - before.tv_nsec);
}

std::string json_escape(const std::string &input) {
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

void write_optional_integer(std::optional<int64_t> value) {
  if (value) {
    std::cout << *value;
  } else {
    std::cout << "null";
  }
}

void write_result(const Options &options, const Shape &shape,
                  std::optional<bool> accepted, const std::string &error,
                  std::optional<int64_t> elapsed_ns,
                  std::optional<int64_t> process_cpu_ns,
                  std::optional<int> cpu_before, std::optional<int> cpu_after,
                  std::optional<long> minor_faults,
                  std::optional<long> major_faults,
                  std::optional<long> voluntary_switches,
                  std::optional<long> involuntary_switches) {
  std::cout << "{\"schema\":\"device-verifier-scaling-probe-v1\""
            << ",\"bpftime_source_revision\":\"" << kBpftimeSourceRevision
            << "\""
            << ",\"build_type\":\"" << kVerifierScalingBuildType << "\""
            << ",\"mode\":\"" << mode_name(options.mode) << "\""
            << ",\"family\":\"" << family_name(options.family) << "\""
            << ",\"requested_instructions\":" << options.instructions
            << ",\"instruction_count\":" << shape.instruction_count
            << ",\"conditional_branches\":" << shape.conditional_branches
            << ",\"helper_calls\":" << shape.helper_calls
            << ",\"exits\":" << shape.exits << ",\"minimum_branch_offset\":";
  write_optional_integer(
      shape.minimum_branch_offset
          ? std::optional<int64_t>(*shape.minimum_branch_offset)
          : std::nullopt);
  std::cout << ",\"maximum_branch_offset\":";
  write_optional_integer(
      shape.maximum_branch_offset
          ? std::optional<int64_t>(*shape.maximum_branch_offset)
          : std::nullopt);
  std::cout << ",\"section\":\"" << kSection << "\",\"accepted\":";
  if (accepted) {
    std::cout << (*accepted ? "true" : "false");
  } else {
    std::cout << "null";
  }
  std::cout << ",\"error\":\"" << json_escape(error) << "\",\"elapsed_ns\":";
  write_optional_integer(elapsed_ns);
  std::cout << ",\"process_cpu_ns\":";
  write_optional_integer(process_cpu_ns);
  std::cout << ",\"cpu_before\":";
  write_optional_integer(cpu_before ? std::optional<int64_t>(*cpu_before)
                                    : std::nullopt);
  std::cout << ",\"cpu_after\":";
  write_optional_integer(cpu_after ? std::optional<int64_t>(*cpu_after)
                                   : std::nullopt);
  std::cout << ",\"minor_faults\":";
  write_optional_integer(minor_faults ? std::optional<int64_t>(*minor_faults)
                                      : std::nullopt);
  std::cout << ",\"major_faults\":";
  write_optional_integer(major_faults ? std::optional<int64_t>(*major_faults)
                                      : std::nullopt);
  std::cout << ",\"voluntary_context_switches\":";
  write_optional_integer(voluntary_switches
                             ? std::optional<int64_t>(*voluntary_switches)
                             : std::nullopt);
  std::cout << ",\"involuntary_context_switches\":";
  write_optional_integer(involuntary_switches
                             ? std::optional<int64_t>(*involuntary_switches)
                             : std::nullopt);
  std::cout << "}\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto program = build_program(options.family, options.instructions);
    const Shape shape =
        validate_shape(program, options.family, options.instructions);
    if (options.required_cpu) {
      require_affinity(*options.required_cpu);
    }

    if (options.mode == Mode::Describe) {
      write_result(options, shape, std::nullopt, "", std::nullopt, std::nullopt,
                   std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                   std::nullopt, std::nullopt);
      return 0;
    }

    if (options.mode == Mode::AcceptOnly) {
      const auto error = bpftime::verifier::gpu::verify_gpu_program(
          program.data(), program.size(), std::string(kSection));
      write_result(options, shape, !error.has_value(), error.value_or(""),
                   std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                   std::nullopt, std::nullopt, std::nullopt, std::nullopt);
      return error ? 66 : 0;
    }

    const int cpu_before = sched_getcpu();
    if (cpu_before < 0) {
      throw std::runtime_error("sched_getcpu before failed");
    }
    rusage usage_before{};
    rusage usage_after{};
    timespec raw_before{};
    timespec raw_after{};
    timespec cpu_time_before{};
    timespec cpu_time_after{};
    if (getrusage(RUSAGE_SELF, &usage_before) != 0 ||
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_time_before) != 0 ||
        clock_gettime(CLOCK_MONOTONIC_RAW, &raw_before) != 0) {
      throw std::runtime_error("failed to read pre-verification clocks");
    }

    const auto error = bpftime::verifier::gpu::verify_gpu_program(
        program.data(), program.size(), std::string(kSection));

    if (clock_gettime(CLOCK_MONOTONIC_RAW, &raw_after) != 0 ||
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_time_after) != 0 ||
        getrusage(RUSAGE_SELF, &usage_after) != 0) {
      throw std::runtime_error("failed to read post-verification clocks");
    }
    const int cpu_after = sched_getcpu();
    if (cpu_after < 0) {
      throw std::runtime_error("sched_getcpu after failed");
    }

    const int64_t elapsed_ns = timespec_delta_ns(raw_before, raw_after);
    const int64_t process_cpu_ns =
        timespec_delta_ns(cpu_time_before, cpu_time_after);
    if (elapsed_ns <= 0 || process_cpu_ns <= 0) {
      throw std::runtime_error("non-positive verification interval");
    }
    if (options.required_cpu && (cpu_before != *options.required_cpu ||
                                 cpu_after != *options.required_cpu)) {
      throw std::runtime_error("CPU changed during verification");
    }

    write_result(options, shape, !error.has_value(), error.value_or(""),
                 elapsed_ns, process_cpu_ns, cpu_before, cpu_after,
                 usage_after.ru_minflt - usage_before.ru_minflt,
                 usage_after.ru_majflt - usage_before.ru_majflt,
                 usage_after.ru_nvcsw - usage_before.ru_nvcsw,
                 usage_after.ru_nivcsw - usage_before.ru_nivcsw);
    return error ? 66 : 0;
  } catch (const std::invalid_argument &error) {
    std::cerr << "argument error: " << error.what() << '\n';
    return 64;
  } catch (const std::exception &error) {
    std::cerr << "probe error: " << error.what() << '\n';
    return 70;
  }
}
