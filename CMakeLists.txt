# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
  set(CMAKE_C_COMPILER_LAUNCHER ccache)
  set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
  set(env{CCACHE_BASEDIR} ${CMAKE_CURRENT_LIST_DIR})
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80-real" "90-real" "100-real" "120")
endif()

set(CMAKE_CUDA_ARCHITECTURES
    "${CMAKE_CUDA_ARCHITECTURES}"
    CACHE STRING "CUDA architectures")

# Setup rapids-cmake
set(rapids-cmake-version "25.10")
if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/GQE_RAPIDS.cmake)
  file(
    DOWNLOAD
    https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-25.10/RAPIDS.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/GQE_RAPIDS.cmake)
endif()
include(${CMAKE_CURRENT_BINARY_DIR}/GQE_RAPIDS.cmake)
include(rapids-cpm)
include(rapids-find)
rapids_cpm_init()

project(gqe LANGUAGES CXX CUDA)

# ##############################################################################
# * build options
# ---------------------------------------------------------------------------------

option(GQE_ENABLE_CUSTOMIZED_PARQUET "Enable the customized Parquet reader" OFF)
option(GQE_ENABLE_QUERY_COMPILER "Enable the query compiler" OFF)
option(GQE_ENABLE_SUBSTRAIT_PRODUCER "Enable the Substrait producer" OFF)

# ##############################################################################
# * shared dependencies
# ---------------------------------------------------------------------------
# NOTE: Fetch shared dependencies FIRST so nvcomp uses the same versions RMM
# must be fetched BEFORE cudf is found and BEFORE nvcomp is built
find_package(spdlog REQUIRED)
include(${rapids-cmake-dir}/cpm/gtest.cmake)
include(${rapids-cmake-dir}/cpm/spdlog.cmake)
include(${rapids-cmake-dir}/cpm/rmm.cmake)
include(${rapids-cmake-dir}/cpm/cccl.cmake)

rapids_cpm_gtest()
rapids_cpm_spdlog()
rapids_cpm_rmm()
rapids_cpm_cccl()

cpmaddpackage(
  NAME
  NVTX
  GITHUB_REPOSITORY
  NVIDIA/NVTX
  GIT_TAG
  v3.1.0-c-cpp
  GIT_SHALLOW
  TRUE)

# ##############################################################################
# * nvcomp setup
# ----------------------------------------------------------------------------------

# CMake variable for gqe-nvcomp tag
set(GQE_NVCOMP_TAG
    "4b0330cb579cc3f0dd868916139044fcc7aca4aa"
    CACHE STRING "Git tag for gqe-nvcomp")
set(GQE_NVCOMP_SOURCE_DIR
    ""
    CACHE PATH
          "Local source directory for gqe-nvcomp (overrides GIT_TAG if set)")

# Fetch nvcomp from source using CPM
if(GQE_NVCOMP_SOURCE_DIR)
  message(
    STATUS "Using nvcomp from local source directory: ${GQE_NVCOMP_SOURCE_DIR}")
  cpmaddpackage(
    NAME
    nvcomp
    SOURCE_DIR
    ${GQE_NVCOMP_SOURCE_DIR}
    OPTIONS
    "BUILD_TESTS OFF"
    "BUILD_BENCHMARKS OFF"
    "BUILD_EXAMPLES OFF"
    "BUILD_NVCOMP ON"
    "BUILD_NVCOMPDX OFF"
    "CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "Using nvcomp from git with tag: ${GQE_NVCOMP_TAG}")
  cpmaddpackage(
    NAME
    nvcomp
    GIT_REPOSITORY
    https://gitlab-master.nvidia.com/Devtech-Compute/gqe-nvcomp.git
    GIT_TAG
    ${GQE_NVCOMP_TAG}
    GIT_SUBMODULES_RECURSE
    TRUE
    OPTIONS
    "BUILD_TESTS OFF"
    "BUILD_BENCHMARKS OFF"
    "BUILD_EXAMPLES OFF"
    "BUILD_NVCOMP ON"
    "BUILD_NVCOMPDX OFF"
    "CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}")
endif()

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
  set(CMAKE_C_COMPILER_LAUNCHER ccache)
  set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
  set(env{CCACHE_BASEDIR} ${CMAKE_CURRENT_LIST_DIR})
endif()

# ##############################################################################
# * compiler options
# ------------------------------------------------------------------------------
# NOTE: These are set AFTER nvcomp to avoid propagating strict flags to the
# nvcomp subproject

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-U__ARM_NEON") # Disables
                                                                    # ARM NEON
                                                                    # intrinsics
                                                                    # that
                                                                    # cauases
                                                                    # issues
                                                                    # with nvcc.

add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall>")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Werror>")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wimplicit-fallthrough>")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wsuggest-override>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall>")
add_compile_options(
  "$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wno-unused-local-typedefs>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Werror>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>")

# needed for clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ##############################################################################
# * dependencies
# ----------------------------------------------------------------------------------

find_package(CUDAToolkit REQUIRED COMPONENTS cupti)

find_package(spdlog REQUIRED)
include(${rapids-cmake-dir}/cpm/gtest.cmake)
include(${rapids-cmake-dir}/cpm/spdlog.cmake)
rapids_cpm_gtest()
rapids_cpm_spdlog()

cpmaddpackage(
  NAME
  cuco
  GITHUB_REPOSITORY
  NVIDIA/cuCollections
  GIT_TAG
  d4b2e2a303ef0064acdc6169d2ca926ec1fb5d2e
  OPTIONS
  "BUILD_TESTS OFF"
  "BUILD_BENCHMARKS OFF"
  "BUILD_EXAMPLES OFF")

cpmaddpackage(NAME substrait GITHUB_REPOSITORY substrait-io/substrait GIT_TAG
              v0.66.1)

# Setup CTest
if(PROJECT_IS_TOP_LEVEL)
  include(CTest)
endif()

rapids_find_package(cudf CONFIG REQUIRED COMPONENTS testing)
rapids_find_package(MPI REQUIRED)
rapids_find_package(NVSHMEM REQUIRED)
rapids_find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)

# Prefer to use the config mode for Protobuf, but if the config file is not
# present, we fallback to use the module mode.
rapids_find_package(Protobuf QUIET CONFIG)
if(NOT Protobuf_FOUND)
  rapids_find_package(Protobuf REQUIRED MODULE)
endif()
rapids_find_package(gRPC CONFIG REQUIRED)

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

  add_subdirectory(include)
endif()

# ##############################################################################
# * targets
# ---------------------------------------------------------------------------------------

add_subdirectory(proto)
rapids_cmake_write_git_revision_file(gqe-version ${CMAKE_CURRENT_BINARY_DIR}
                                     PREFIX GQE)

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

if(GQE_ENABLE_SUBSTRAIT_PRODUCER)
  set(CONDA_PREFIX "$ENV{CONDA_PREFIX}")
  include(ExternalProject)
  ExternalProject_Add(
    producer
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/rust/producer
    CONFIGURE_COMMAND ""
    BUILD_COMMAND cargo install --root ${CONDA_PREFIX} --path
                  ${CMAKE_CURRENT_LIST_DIR}/rust/producer
    INSTALL_COMMAND "")
endif()
