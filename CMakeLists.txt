# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80-real" "90-real" "100-real" "120")
endif()

set(CMAKE_CUDA_ARCHITECTURES
    "${CMAKE_CUDA_ARCHITECTURES}"
    CACHE STRING "CUDA architectures")

set(rapids-cmake-version "26.06")
set(rapids-cmake-branch "release/26.06")
set(rapids-cmake-checkout-tag "release/26.06")
# SHA256 of RAPIDS.cmake at:
# https://raw.githubusercontent.com/rapidsai/rapids-cmake/v26.06.00/RAPIDS.cmake
set(rapids-cmake-expected-hash
    "f21479e8ba5af680d9302a8d1f5cba4e0505072a1d8b4c5482a15ae7ecdaefd3")
if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/GQE_RAPIDS.cmake)
  file(
    DOWNLOAD
    https://raw.githubusercontent.com/rapidsai/rapids-cmake/v26.06.00/RAPIDS.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/GQE_RAPIDS.cmake
    EXPECTED_HASH SHA256=${rapids-cmake-expected-hash}
    STATUS rapids-cmake-download-status)
  list(GET rapids-cmake-download-status 0 rapids-cmake-download-code)
  if(NOT rapids-cmake-download-code EQUAL 0)
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/GQE_RAPIDS.cmake)
    message(
      FATAL_ERROR
        "Failed to download/verify RAPIDS.cmake (status: ${rapids-cmake-download-status}). "
        "This may be a checksum mismatch; expected SHA256=${rapids-cmake-expected-hash}."
    )
  endif()
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
option(GQE_ENABLE_RUST_SERVER "Build the Rust gqe-server binary" OFF)
option(GQE_ENABLE_RUST_CLI "Build the Rust gqe-cli binary" ON)
option(GQE_ENABLE_FABRIC_SHARED_POOL_TESTS
       "Compile shared NUMA pool fabric-handle tests" OFF)
option(GQE_ENABLE_PUBLIC_BUILD "Enable building for external user" ON)
set(GQE_NVCOMP_LIB_DIR
    ""
    CACHE
      PATH
      "Optional path to prebuilt nvCOMP installation (must contain cmake config files)."
)

# ##############################################################################
# * shared dependencies
# ---------------------------------------------------------------------------
# NOTE: Fetch shared dependencies FIRST so nvcomp uses the same versions RMM
# must be fetched BEFORE cudf is found and BEFORE nvcomp is built
include(${rapids-cmake-dir}/cpm/gtest.cmake)
include(${rapids-cmake-dir}/cpm/rapids_logger.cmake)
include(${rapids-cmake-dir}/cpm/rmm.cmake)
include(${rapids-cmake-dir}/cpm/cccl.cmake)

rapids_cpm_gtest()
# Temporary workaround: GQE still links spdlog directly in some paths, so we
# disable rapids_logger's default spdlog symbol hiding to avoid link failures.
# Follow-up to migrate GQE logging to rapids_logger and re-enable symbol hiding:
# https://gitlab-master.nvidia.com/Devtech-Compute/gqe/-/issues/383
set(RAPIDS_LOGGER_HIDE_ALL_SPDLOG_SYMBOLS
    OFF
    CACHE BOOL "" FORCE)
rapids_cpm_rapids_logger()
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

include(${CMAKE_CURRENT_LIST_DIR}/cmake/nvcomp.cmake)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
  set(CMAKE_C_COMPILER_LAUNCHER ccache)
  set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
endif()

find_package(spdlog REQUIRED)

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
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fopenmp>")

# needed for clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ##############################################################################
# * dependencies
# ----------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${gqe_SOURCE_DIR}/cmake/modules")

find_package(CUDAToolkit REQUIRED COMPONENTS cupti)

include(${rapids-cmake-dir}/cpm/gtest.cmake)
include(${rapids-cmake-dir}/cpm/rapids_logger.cmake)
rapids_cpm_gtest()
rapids_cpm_rapids_logger()

# Resolve cuco via rapids-cmake so it stays aligned with the same dependency set
# cuDF expects for this RAPIDS release.
include(${rapids-cmake-dir}/cpm/cuco.cmake)
rapids_cpm_cuco()

cpmaddpackage(NAME substrait GITHUB_REPOSITORY substrait-io/substrait GIT_TAG
              v0.66.1)

# Setup CTest
if(PROJECT_IS_TOP_LEVEL)
  include(CTest)
endif()

rapids_find_package(cudf CONFIG REQUIRED COMPONENTS testing)
rapids_find_package(NVSHMEM REQUIRED)
rapids_find_package(Threads REQUIRED)
rapids_find_package(TBB CONFIG REQUIRED)
# GLOBAL is required so arrow-flight-sql's add_subdirectory can find
# BOOST::headers
find_package(Boost REQUIRED GLOBAL)
find_package(NCCL REQUIRED)

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

add_subdirectory(src/node_manager)
add_subdirectory(src/task_manager)

if(PROJECT_IS_TOP_LEVEL)
  add_subdirectory(test)
endif()
if(PROJECT_IS_TOP_LEVEL)
  add_subdirectory(benchmark)
endif()

# Rust crates. cargo's target-dir is anchored to ${CMAKE_BINARY_DIR} so build
# artifacts live together. Exposed via GQE_RUST_TARGET_DIR for consumers.
set(GQE_RUST_TARGET_DIR
    ${CMAKE_BINARY_DIR}/cargo-target
    CACHE PATH "Cargo target directory for gqe Rust crates")
find_program(CARGO_EXECUTABLE cargo REQUIRED)

if(GQE_ENABLE_RUST_CLI)
  add_custom_target(
    gqe-cli ALL
    COMMAND
      ${CARGO_EXECUTABLE} build --release --bin gqe-cli --manifest-path
      ${CMAKE_CURRENT_LIST_DIR}/rust/Cargo.toml --target-dir
      ${GQE_RUST_TARGET_DIR}
    COMMENT "Building Rust crate gqe-cli"
    USES_TERMINAL VERBATIM)
endif()

if(GQE_ENABLE_SUBSTRAIT_PRODUCER)
  add_custom_target(
    producer ALL
    COMMAND
      ${CARGO_EXECUTABLE} install --root $ENV{CONDA_PREFIX} --path
      ${CMAKE_CURRENT_LIST_DIR}/rust/producer --target-dir
      ${GQE_RUST_TARGET_DIR}
    COMMENT "Installing Rust crate producer to $CONDA_PREFIX/bin"
    USES_TERMINAL VERBATIM)
endif()

if(GQE_ENABLE_RUST_SERVER)
  add_custom_target(
    gqe-server ALL
    COMMAND
      ${CARGO_EXECUTABLE} build --release --bin gqe-server --manifest-path
      ${CMAKE_CURRENT_LIST_DIR}/rust/Cargo.toml --target-dir
      ${GQE_RUST_TARGET_DIR}
    COMMENT "Building Rust crate gqe-server"
    USES_TERMINAL VERBATIM)
endif()
