cmake_minimum_required(VERSION 3.20)
project(cutlass_experiment CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
# Enable CUDA support
enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

# Include CUDA
include_directories(${CUDA_INCLUDE_DIRS})

# Set CUTLASS directory
set(CUTLASS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../cutlass CACHE PATH "Path to CUTLASS")
set(CSRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Include CUTLASS
include_directories(${CUTLASS_DIR}/include)
include_directories(${CUTLASS_DIR}/tools/util/include)

# Include state_kernel
include_directories(${CSRC_DIR}/state_kernel/src)

set(CMAKE_CUDA_ARCHITECTURES 80)
message ("CUDA ARCHS: ${CMAKE_CUDA_ARCHITECTURES}")


# Add subdirectory
add_subdirectory(state_expand)