cmake_minimum_required(VERSION 3.18)
project(gpu_matmul LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Query nanobind's CMake dir from the active Python
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
)

# Load nanobind's CMake macros
include(${NANOBIND_CMAKE_DIR}/nanobind-config.cmake)

# --- THIS IS THE CRITICAL PART ---
# Create the "_cpp_kernels" target from your source files.
# This was missing from your file.
nanobind_add_module(_cuda_gemm
    STABLE_ABI
    gemm.cu
    gemm.cpp
)

# Enable CUDA separable compilation for the target you just created
set_target_properties(_cuda_gemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# Install the final compiled library into the "kernels" python package
install(TARGETS _cuda_gemm LIBRARY DESTINATION kernels/gemm)