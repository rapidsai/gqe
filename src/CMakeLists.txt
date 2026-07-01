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

# We cannot use add_custom_command() since the file names need to be known at
# configuration time so depenencies can be known during compile time
execute_process(
  COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generate_substrait.sh
          ${substrait_SOURCE_DIR} ${CMAKE_BINARY_DIR})

file(GLOB substrait_sources ${CMAKE_BINARY_DIR}/substrait/*.cc
     ${CMAKE_BINARY_DIR}/substrait/extensions/*.cc)
add_library(substrait STATIC ${substrait_sources})
target_include_directories(substrait PUBLIC ${CMAKE_BINARY_DIR})
target_link_libraries(substrait PRIVATE protobuf::libprotobuf)
# Setting `-DNDEBUG` to match the ABI of libprotobuf.so See this issue for
# details: https://github.com/protocolbuffers/protobuf/issues/9947
target_compile_options(substrait PRIVATE -DNDEBUG)

get_target_property(INCLUDES proto INCLUDE_DIRECTORIES)
message(STATUS "Include dirs for proto: ${INCLUDES}")
add_subdirectory(libperfect)

add_library(
  gqe STATIC
  logical/aggregate.cpp
  logical/fetch.cpp
  logical/filter.cpp
  logical/join.cpp
  logical/project.cpp
  logical/read.cpp
  logical/relation.cpp
  logical/set.cpp
  logical/sort.cpp
  logical/window.cpp
  logical/user_defined.cpp
  logical/write.cpp
  logical/relation_visitor.cpp
  logical/from_substrait.cpp
  memory_resource/memory_utilities.cpp
  memory_resource/numa_memory_resource.cpp
  memory_resource/pinned_memory_resource.cpp
  memory_resource/pgas_memory_resource.cpp
  memory_resource/boost_shared_memory_resource.cpp
  memory_resource/numa_pool_memory_resource.cpp
  memory_resource/shared_numa_pool_memory_resource.cpp
  executor/groupby/groupby.cu
  executor/aggregate.cpp
  executor/concatenate.cpp
  executor/eval.cpp
  executor/like.cu
  executor/filter.cpp
  executor/fetch.cpp
  executor/gen_ident_col.cu
  executor/mark_join.cu
  executor/unique_key_inner_join.cu
  executor/join.cpp
  executor/join.cu
  executor/optimization_parameters.cpp
  executor/project.cpp
  executor/read.cpp
  executor/sort.cpp
  executor/qep_adapter.cpp
  executor/qep_task_graph_transformation.cpp
  executor/task.cpp
  executor/window.cpp
  executor/write.cpp
  executor/task_graph.cpp
  executor/partition.cpp
  executor/partition_merge.cpp
  executor_next/utilities/thread_event.cpp
  expression/binary_op.cpp
  expression/cast.cpp
  expression/if_then_else.cpp
  expression/scalar_function.cpp
  expression/subquery.cpp
  expression/unary_op.cpp
  expression/is_null.cpp
  expression/json_formatter.cpp
  optimizer/estimator.cpp
  optimizer/logical_optimization.cpp
  optimizer/optimization_configuration.cpp
  optimizer/physical_transformation.cpp
  optimizer/physical_plan/plan_join.cpp
  optimizer/physical_plan/plan_window.cpp
  optimizer/relation_properties.cpp
  optimizer/rules/column_name_assignment.cpp
  optimizer/rules/constant_folding.cpp
  optimizer/rules/extract_complex_expressions_into_project.cpp
  optimizer/rules/projection_pushdown.cpp
  optimizer/rules/join_children_swap.cpp
  optimizer/rules/aggregate_perfect_hash.cpp
  optimizer/rules/join_unique_keys.cpp
  optimizer/rules/mean_decomposition.cpp
  optimizer/rules/not_not.cpp
  optimizer/rules/uniqueness_propagation.cpp
  optimizer/rules/string_to_int_literal.cpp
  optimizer/rules/fix_partial_filter_column_references.cpp
  optimizer/statistics.cpp
  physical/aggregate.cpp
  physical/fetch.cpp
  physical/filter.cpp
  physical/gen_ident_col.cpp
  physical/join.cpp
  physical/project.cpp
  physical/read.cpp
  physical/set.cpp
  physical/sort.cpp
  physical/window.cpp
  physical/user_defined.cpp
  physical/write.cpp
  physical/relation_visitor.cpp
  physical/shuffle.cpp
  qep/algorithms.cpp
  qep/query_execution_plan.cpp
  qep/shapes/masked_table.cpp
  qep/shapes/row_count.cpp
  qep/state.cpp
  qep/task.cpp
  rpc/serialization/data_type.cpp
  rpc/serialization/expression_deserialization.cpp
  rpc/serialization/expression_serialization.cpp
  rpc/serialization/relation_deserialization.cpp
  rpc/serialization/relation_serialization.cpp
  rpc/serialization/optimization_parameters.cpp
  rpc/serialization/statistics.cpp
  rpc/serialization/storage.cpp
  rpc/serialization/uuid.cpp
  rpc/task_migration.cpp
  storage/descriptor.cpp
  storage/compression.cpp
  storage/in_memory.cpp
  storage/nvcomp_adapter.cpp
  storage/sliced_columns.cpp
  storage/sliced_columns.cu
  storage/parquet.cpp
  storage/parquet.cu
  storage/parquet_reader.cpp
  storage/parquet_reader.cu
  utility/cuda.cpp
  utility/cupti_range.cpp
  utility/cupti_activity.cpp
  utility/hashing.cpp
  utility/linux.cpp
  utility/timer.cpp
  utility/tpcds.cpp
  utility/tpch.cpp
  utility/uuid.cpp
  utility/multi_process_helpers.cpp
  utility/cudf_to_arrow.cpp
  utility/kernel_fusion_helpers.cpp
  utility/serialization.cpp
  catalog.cpp
  query_context.cpp
  task_manager_context.cpp
  device_properties.cpp
  scheduler.cpp
  types.cpp
  communicator.cpp
  storage/zone_map.cpp)

target_include_directories(gqe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(gqe PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../proto)
target_include_directories(
  gqe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/CUPTI/include)

target_link_libraries(gqe PUBLIC proto)
target_link_libraries(gqe PUBLIC gqe-version)
target_link_libraries(gqe PUBLIC cuco::cuco)
target_link_libraries(gqe PUBLIC CUDA::cuda_driver)
target_link_libraries(gqe PUBLIC CUDA::cupti)
target_link_libraries(gqe PUBLIC cudf::cudf)
target_link_libraries(gqe PUBLIC arrow_shared)
target_link_libraries(gqe PUBLIC parquet_shared)
target_link_libraries(gqe PUBLIC numa)
target_link_libraries(gqe PRIVATE nvtx3-cpp)
target_link_libraries(gqe PUBLIC spdlog::spdlog)
target_link_libraries(gqe PUBLIC substrait)
target_link_libraries(gqe PRIVATE Boost::headers)
target_link_libraries(gqe PRIVATE Threads::Threads)
# Dynamic link nvcomp: the .so is installed alongside the executables (see
# cmake/nvcomp.cmake) and resolved via $ORIGIN/../${GQE_INSTALL_LIBDIR} RPATH.
target_link_libraries(gqe PUBLIC nvcomp::nvcomp)
target_link_libraries(gqe PUBLIC nvcomp::nvcomp_cpu)
target_link_libraries(gqe PUBLIC lz4)
target_link_libraries(gqe PUBLIC uring)
target_link_libraries(gqe PUBLIC nvshmem::nvshmem_host)
target_link_libraries(gqe PRIVATE libperfect)
target_link_libraries(gqe PUBLIC CUDA::cuda_driver)
target_link_libraries(gqe PUBLIC CUDA::nvml)

if(NOT (nvcomp_VERSION_MAJOR EQUAL 5 AND nvcomp_VERSION_MINOR EQUAL 2))
  message(
    FATAL_ERROR
      "Unsupported nvcomp version: ${nvcomp_VERSION_MAJOR}.${nvcomp_VERSION_MINOR}. GQE requires nvcomp 5.2."
  )
endif()

# Download and configure Parquet format
if(GQE_ENABLE_CUSTOMIZED_PARQUET)
  file(
    DOWNLOAD
    https://raw.githubusercontent.com/apache/parquet-format/c185faf0c4fc0c7d3075d1abd4ed0985cdbe5d87/src/main/thrift/parquet.thrift
    ${CMAKE_BINARY_DIR}/parquet/parquet.thrift)
  execute_process(
    COMMAND
      thrift -r --gen cpp -out ${CMAKE_BINARY_DIR}/parquet
      ${CMAKE_BINARY_DIR}/parquet/parquet.thrift COMMAND_ERROR_IS_FATAL ANY)
  add_library(gqe_parquet STATIC ${CMAKE_BINARY_DIR}/parquet/parquet_types.cpp)
  target_include_directories(gqe_parquet PUBLIC ${CMAKE_BINARY_DIR}/parquet)
  target_link_libraries(gqe_parquet PUBLIC thrift::thrift)

  target_link_libraries(gqe PRIVATE gqe_parquet uring)
  add_compile_definitions(ENABLE_CUSTOMIZED_PARQUET)
endif()

if(GQE_ENABLE_QUERY_COMPILER)
  include_directories(${PROJECT_BINARY_DIR}/include
                      ${CMAKE_CURRENT_SOURCE_DIR}/../include)
  add_subdirectory(compiler)

  get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
  get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

  set(COMPILER_LIBS
      ${dialect_libs}
      ${conversion_libs}
      # GQE compiler libraries
      GQECompilerRelAlg
      GQECompilerRelAlgPipelines
      GQECompilerGPUPipelines
      # Upstream MLIR libraries
      MLIRBuiltinToLLVMIRTranslation
      MLIRGPUToLLVMIRTranslation
      MLIRLLVMToLLVMIRTranslation)

  target_include_directories(gqe PUBLIC ${CMAKE_BINARY_DIR}/include)
  target_sources(gqe PUBLIC utility/cuda_driver.cpp
                            utility/kernel_arguments_builder.cpp)
  target_link_libraries(gqe PUBLIC ${COMPILER_LIBS})
endif()
