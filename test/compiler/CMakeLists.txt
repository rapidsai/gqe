# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

find_package(CUDAToolkit)

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

add_executable(mlir_gpu_engine mlir_gpu_engine.cpp ${CMAKE_SOURCE_DIR}/src/utility/cuda_driver.cpp)
target_include_directories(mlir_gpu_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(mlir_gpu_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_link_libraries(
  mlir_gpu_engine
  PRIVATE
  GTest::gtest_main
  CUDA::cudart
  CUDA::cuda_driver
  ${dialect_libs}
  MLIRBuiltinToLLVMIRTranslation
  MLIRExecutionEngine
  MLIRUBToLLVM
  MLIRGPUToLLVMIRTranslation
  )
add_test(NAME mlir_gpu_engine_test COMMAND mlir_gpu_engine)

add_executable(mlir_gpu_standalone mlir_gpu_standalone.cpp ${CMAKE_SOURCE_DIR}/src/utility/cuda_driver.cpp)
target_include_directories(mlir_gpu_standalone PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(mlir_gpu_standalone PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_link_libraries(
  mlir_gpu_standalone
  PRIVATE
  GTest::gtest_main
  CUDA::cudart
  cudf::cudf
  gqe
  )
add_test(NAME mlir_gpu_standalone_test COMMAND mlir_gpu_standalone)

add_executable(mlir_tpch_q6 mlir_tpch_q6.cpp ${CMAKE_SOURCE_DIR}/src/utility/cuda_driver.cpp)
target_include_directories(mlir_tpch_q6 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(mlir_tpch_q6 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_link_libraries(
  mlir_tpch_q6
  PRIVATE
  GTest::gtest_main
  GTest::gmock
  CUDA::cudart
  cudf::cudf
  gqe
)
add_test(NAME mlir_tpch_q6_test COMMAND mlir_tpch_q6)

set_tests_properties(
  mlir_tpch_q6_test
  PROPERTIES
  ENVIRONMENT
  TPCH_DATA_DIR=/scratch/local/sf1_id32
)
