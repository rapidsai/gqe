# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
  set(CMAKE_C_COMPILER_LAUNCHER ccache)
  set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
  set(env{CCACHE_BASEDIR} ${CMAKE_CURRENT_BINARY_DIR})
endif()

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "70-real" "80-real" "90-real" "100-real" "120")
endif()

# Setup rapids-cmake
if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/GQE_RAPIDS.cmake)
  file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-24.12/RAPIDS.cmake
       ${CMAKE_CURRENT_BINARY_DIR}/GQE_RAPIDS.cmake
  )
endif()
include(${CMAKE_CURRENT_BINARY_DIR}/GQE_RAPIDS.cmake)
include(rapids-cpm)
include(rapids-find)
rapids_cpm_init()

project(gqe LANGUAGES CXX CUDA)

# ##################################################################################################
# * compiler options ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall>")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Werror>")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wimplicit-fallthrough>")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wsuggest-override>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Werror>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>")

# needed for clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ##################################################################################################
# * build options ---------------------------------------------------------------------------------

option(GQE_ENABLE_CUSTOMIZED_PARQUET "Enable the customized Parquet reader" OFF)
option(GQE_ENABLE_QUERY_COMPILER "Enable the query compiler" OFF)

# ##################################################################################################
# * dependencies ----------------------------------------------------------------------------------

include(${rapids-cmake-dir}/cpm/gtest.cmake)
include(${rapids-cmake-dir}/cpm/spdlog.cmake)
rapids_cpm_gtest()
rapids_cpm_spdlog()

CPMAddPackage(
  NAME cuco
  GITHUB_REPOSITORY NVIDIA/cuCollections
  GIT_TAG d4b2e2a303ef0064acdc6169d2ca926ec1fb5d2e
  OPTIONS
     "BUILD_TESTS OFF"
     "BUILD_BENCHMARKS OFF"
     "BUILD_EXAMPLES OFF"
)

CPMAddPackage(
    NAME NVTX
    GITHUB_REPOSITORY NVIDIA/NVTX
    GIT_TAG v3.1.0-c-cpp
    GIT_SHALLOW TRUE)

CPMAddPackage(
    NAME substrait
    GITHUB_REPOSITORY substrait-io/substrait
    GIT_TAG v0.66.1)

# Setup CTest
if(PROJECT_IS_TOP_LEVEL)
  include(CTest)
endif()

rapids_find_package(cudf CONFIG REQUIRED COMPONENTS testing)
rapids_find_package(MPI REQUIRED) 
rapids_find_package(nvcomp CONFIG REQUIRED)
rapids_find_package(NVSHMEM REQUIRED)
rapids_find_package(Threads REQUIRED)

# Prefer to use the config mode for Protobuf, but if the config file is not present, we fallback to
# use the module mode.
rapids_find_package(Protobuf QUIET CONFIG)
if (NOT Protobuf_FOUND)
  rapids_find_package(Protobuf REQUIRED MODULE)
endif()

if(GQE_ENABLE_CUSTOMIZED_PARQUET)
  rapids_find_package(Thrift CONFIG REQUIRED)
endif()

if(GQE_ENABLE_QUERY_COMPILER)
  find_package(MLIR REQUIRED CONFIG)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  include_directories(${LLVM_INCLUDE_DIRS})
  include_directories(${MLIR_INCLUDE_DIRS})

  link_directories(${LLVM_BUILD_LIBRARY_DIR})
  add_definitions(${LLVM_DEFINITIONS})
endif()

# ##################################################################################################
# * targets ---------------------------------------------------------------------------------------

add_subdirectory(src)
if(PROJECT_IS_TOP_LEVEL)
  add_subdirectory(test)
endif()
if(PROJECT_IS_TOP_LEVEL)
  add_subdirectory(benchmark)
endif()
if(PROJECT_IS_TOP_LEVEL)
  add_subdirectory(rust/gqe-sys)
endif()
