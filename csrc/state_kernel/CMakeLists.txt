cmake_minimum_required(VERSION 3.20)
project(state_kernel)

enable_language(CUDA)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/attention)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../cutlass/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../cutlass/tools/util/include)

add_executable(smem_calculator src/smem_calculator.cu)
target_compile_options(smem_calculator PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)

