cmake_minimum_required(VERSION 3.20)
project(uvm_basic LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(uvm_vector_add src/uvm_vector_add.cu)
add_executable(uvm_phase_scan src/uvm_phase_scan.cu)
target_compile_features(uvm_vector_add PRIVATE cxx_std_17 cuda_std_17)
target_compile_features(uvm_phase_scan PRIVATE cxx_std_17 cuda_std_17)
target_compile_options(uvm_vector_add PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>)
target_link_libraries(uvm_vector_add PRIVATE CUDA::cudart)
target_compile_options(uvm_phase_scan PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>)
target_link_libraries(uvm_phase_scan PRIVATE CUDA::cudart)
set(UVM_BASIC_CUDA_ARCH "80" CACHE STRING "CUDA architecture for this experiment")
set_target_properties(uvm_vector_add PROPERTIES CUDA_ARCHITECTURES "${UVM_BASIC_CUDA_ARCH}")
set_target_properties(uvm_phase_scan PROPERTIES CUDA_ARCHITECTURES "${UVM_BASIC_CUDA_ARCH}")

find_path(NVTX_INCLUDE_DIR nvtx3/nvToolsExt.h HINTS ${CUDAToolkit_INCLUDE_DIRS})
find_library(NVTX_LIBRARY nvToolsExt HINTS
  ${CUDAToolkit_LIBRARY_DIR}
  ${CUDAToolkit_LIBRARY_ROOT}/lib64)
if(NVTX_INCLUDE_DIR AND NVTX_LIBRARY)
  target_compile_definitions(uvm_vector_add PRIVATE UVM_BASIC_HAVE_NVTX=1)
  target_compile_definitions(uvm_phase_scan PRIVATE UVM_BASIC_HAVE_NVTX=1)
  target_include_directories(uvm_vector_add PRIVATE ${NVTX_INCLUDE_DIR})
  target_include_directories(uvm_phase_scan PRIVATE ${NVTX_INCLUDE_DIR})
  target_link_libraries(uvm_vector_add PRIVATE ${NVTX_LIBRARY})
  target_link_libraries(uvm_phase_scan PRIVATE ${NVTX_LIBRARY})
endif()
