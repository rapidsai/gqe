# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

option(DEBUG_MLIR_MODULE "Enable MLIR module debug output in tests" OFF)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

# See: https://discourse.llvm.org/t/undefined-reference-to-typeinfo-for-llvm-genericoptionvalue/71526/4
set(NO_RTTI "-fno-rtti")
add_definitions(${NO_RTTI})

add_executable(mlir_cpu_engine mlir_cpu_engine.cpp)
target_link_libraries(
  mlir_cpu_engine
  PRIVATE
  GTest::gtest_main
  ${dialect_libs}
  MLIRBuiltinToLLVMIRTranslation
  MLIRExecutionEngine
  )
add_test(NAME mlir_cpu_engine_test COMMAND mlir_cpu_engine)

add_executable(mlir_gpu_engine mlir_gpu_engine.cpp kernel_launch.cpp)
target_include_directories(mlir_gpu_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(mlir_gpu_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_link_libraries(
  mlir_gpu_engine
  PRIVATE
  GTest::gtest_main
  cuda
  ${dialect_libs}
  MLIRBuiltinToLLVMIRTranslation
  MLIRExecutionEngine
  MLIRUBToLLVM
  MLIRGPUToLLVMIRTranslation
  )
add_test(NAME mlir_gpu_engine_test COMMAND mlir_gpu_engine)

add_executable(mlir_gpu_standalone mlir_gpu_standalone.cpp kernel_launch.cpp)
target_include_directories(mlir_gpu_standalone PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(mlir_gpu_standalone PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_link_libraries(
  mlir_gpu_standalone
  PRIVATE
  GTest::gtest_main
  cuda
  ${dialect_libs}
  MLIRBuiltinToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  )
add_test(NAME mlir_gpu_standalone_test COMMAND mlir_gpu_standalone)

