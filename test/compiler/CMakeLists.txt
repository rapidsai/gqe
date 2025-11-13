# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
target_compile_definitions(mlir_tpch_q6 PRIVATE GQE_ENABLE_QUERY_COMPILER)
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

add_executable(mlir_cudf_table_materialize_test mlir_cudf_table_materialize_test.cpp ${CMAKE_SOURCE_DIR}/src/utility/cuda_driver.cpp)
target_include_directories(mlir_cudf_table_materialize_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(mlir_cudf_table_materialize_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_compile_definitions(mlir_cudf_table_materialize_test PRIVATE GQE_ENABLE_QUERY_COMPILER)
target_link_libraries(
  mlir_cudf_table_materialize_test
  PRIVATE
  GTest::gtest_main
  GTest::gmock
  CUDA::cudart
  cudf::cudf
  cudf::cudftestutil
  cudf::cudftestutil_impl
  gqe
)
add_test(NAME mlir_cudf_table_materialize_test COMMAND mlir_cudf_table_materialize_test)

set_tests_properties(
  mlir_tpch_q6_test
  mlir_cudf_table_materialize_test
  PROPERTIES
  ENVIRONMENT
  TPCH_DATA_DIR=/scratch/local/sf1_id32
)
