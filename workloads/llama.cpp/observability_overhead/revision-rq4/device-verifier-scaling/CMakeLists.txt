cmake_minimum_required(VERSION 3.20)

project(device_verifier_scaling LANGUAGES C CXX)

set(BPFTIME_ROOT "" CACHE PATH "Path to the bpftime source tree")
if(NOT IS_DIRECTORY "${BPFTIME_ROOT}/bpftime-verifier")
  message(FATAL_ERROR "BPFTIME_ROOT must name a bpftime source tree")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# bpftime-verifier's standalone CMake expects a target named libbpf and the
# legacy LIBBPF_* variables.  Use the host package in this isolated verifier
# build; no runtime, attach implementation, CUDA component, or existing bpftime
# build tree is involved.
find_package(PkgConfig REQUIRED)
pkg_check_modules(SYSTEM_LIBBPF REQUIRED IMPORTED_TARGET libbpf)
add_library(libbpf INTERFACE)
target_link_libraries(libbpf INTERFACE PkgConfig::SYSTEM_LIBBPF)
set(LIBBPF_INCLUDE_DIRS ${SYSTEM_LIBBPF_INCLUDE_DIRS})
set(LIBBPF_LIBRARIES PkgConfig::SYSTEM_LIBBPF)

add_subdirectory(
  "${BPFTIME_ROOT}/bpftime-verifier"
  "${CMAKE_CURRENT_BINARY_DIR}/bpftime-verifier"
  EXCLUDE_FROM_ALL
)

add_executable(verifier_scaling_probe verifier_scaling_probe.cpp)
execute_process(
  COMMAND git -C "${BPFTIME_ROOT}" rev-parse HEAD
  OUTPUT_VARIABLE BPFTIME_SOURCE_REVISION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE BPFTIME_GIT_RESULT
)
if(NOT BPFTIME_GIT_RESULT EQUAL 0 OR BPFTIME_SOURCE_REVISION STREQUAL "")
  message(FATAL_ERROR "could not record the bpftime source Git revision")
endif()
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/build_metadata.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/build_metadata.hpp"
  @ONLY
)
target_include_directories(verifier_scaling_probe PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${BPFTIME_ROOT}/bpftime-verifier/ebpf-verifier/src"
)
target_link_libraries(verifier_scaling_probe PRIVATE bpftime-verifier)
target_compile_options(verifier_scaling_probe PRIVATE
  -Wall -Wextra -Wpedantic -Werror
)
if(UNIX AND NOT APPLE)
  target_link_options(verifier_scaling_probe PRIVATE -Wl,--build-id=none)
endif()
