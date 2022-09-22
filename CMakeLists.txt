# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

# Setup rapids-cmake
if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/GQE_RAPIDS.cmake)
  file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-22.08/RAPIDS.cmake
       ${CMAKE_CURRENT_BINARY_DIR}/GQE_RAPIDS.cmake
  )
endif()
include(${CMAKE_CURRENT_BINARY_DIR}/GQE_RAPIDS.cmake)
include(rapids-cpm)
include(rapids-find)
rapids_cpm_init()

project(gqe LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall>")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Werror>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Werror>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>")

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "70" "80")
endif()

# needed for clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Setup gtest
include(${rapids-cmake-dir}/cpm/gtest.cmake)
rapids_cpm_gtest()

# Setup CTest
if(PROJECT_IS_TOP_LEVEL)
  include(CTest)
endif()

rapids_find_package(cudf CONFIG REQUIRED COMPONENTS testing)

add_subdirectory(src)
if(PROJECT_IS_TOP_LEVEL)
  add_subdirectory(test)
endif()
