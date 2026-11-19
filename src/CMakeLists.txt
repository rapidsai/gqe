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

add_library(gqe_device_properties STATIC cpu_set.cpp device_properties.cpp)
target_include_directories(gqe_device_properties
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(
  gqe_device_properties
  PUBLIC CUDA::cuda_driver CUDA::cudart_static CUDA::nvml rmm::rmm
  PRIVATE fmt::fmt)

add_library(
  gqe STATIC
  aggregation.cpp
  logical/aggregate.cpp
  logical/fetch.cpp
  logical/filter.cpp
  logical/join.cpp
  logical/project.cpp
  logical/read.cpp
  logical/relation.cpp
  logical/set.cpp
  logical/sort.cpp
  logical/top_k.cpp
  logical/window.cpp
  logical/user_defined.cpp
  logical/write.cpp
  logical/relation_visitor.cpp
  logical/from_substrait.cpp
  memory_resource/memory_utilities.cpp
  memory_resource/numa_memory_resource.cpp
  memory_resource/pinned_memory_resource.cpp
  memory_resource/pgas_memory_resource.cpp
  memory_resource/shared_memory_resource.cpp
  memory_resource/numa_pool_memory_resource.cpp
  memory_resource/shared_numa_pool_memory_resource.cpp
  executor/groupby/groupby.cpp
  executor/groupby/hash_groupby.cu
  executor/perfect_hash/masked_join.cu
  executor/perfect_hash/unique_indices.cu
  executor/perfect_hash/scatter_aggregate.cpp
  executor/perfect_hash/scatter_aggregate_float32.cu
  executor/perfect_hash/scatter_aggregate_float64.cu
  executor/perfect_hash/scatter_aggregate_int32.cu
  executor/perfect_hash/scatter_aggregate_int64.cu
  executor/aggregate.cpp
  executor/concatenate.cpp
  executor/eval.cpp
  executor/masked_ast.cu
  executor/like.cu
  executor/filter.cpp
  executor/fetch.cpp
  executor/gen_ident_col.cu
  executor/mark_join.cu
  executor/unique_key_inner_join.cu
  executor/join.cpp
  executor/join_predicate.cpp
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
  expression/is_null.cpp
  expression/json_formatter.cpp
  expression/scalar_function.cpp
  expression/subquery.cpp
  expression/unary_op.cpp
  expression/utility.cpp
  optimizer/calibration/b200.cpp
  optimizer/calibration/gb200.cpp
  optimizer/calibration/gb300.cpp
  optimizer/decode_throughput.cpp
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
  optimizer/rules/extract_join_predicate_into_project.cpp
  optimizer/rules/join_extraction_utils.cpp
  optimizer/rules/normalize_decimal_widths.cpp
  optimizer/rules/identity_project_elimination.cpp
  optimizer/rules/projection_pushdown.cpp
  optimizer/rules/join_children_swap.cpp
  optimizer/rules/aggregate_perfect_hash.cpp
  optimizer/rules/join_unique_keys.cpp
  optimizer/rules/mean_decomposition.cpp
  optimizer/rules/join_perfect_hash.cpp
  optimizer/rules/not_not.cpp
  optimizer/rules/uniqueness_propagation.cpp
  optimizer/rules/string_to_int_literal.cpp
  optimizer/rules/fix_partial_filter_column_references.cpp
  optimizer/rules/sort_fetch_to_top_k.cpp
  optimizer/statistics.cpp
  optimizer/utility.cpp
  physical/aggregate.cpp
  physical/bounded_hash_aggregate.cpp
  physical/fetch.cpp
  physical/filter.cpp
  physical/gen_ident_col.cpp
  physical/join.cpp
  physical/plan.cpp
  physical/project.cpp
  physical/read.cpp
  physical/set.cpp
  physical/sort.cpp
  physical/top_k.cpp
  physical/window.cpp
  physical/user_defined.cpp
  physical/write.cpp
  physical/relation_visitor.cpp
  physical/shuffle.cpp
  qep/algorithms.cpp
  qep/column_materialization.cpp
  qep/indirect_gather.cpp
  qep/operator/bounded_hash_aggregate/build_task.cpp
  qep/operator/bounded_hash_aggregate/insert_kernels_int32.cu
  qep/operator/bounded_hash_aggregate/insert_kernels_int64.cu
  qep/operator/bounded_hash_aggregate/insert_kernels_not_null_int32.cu
  qep/operator/bounded_hash_aggregate/insert_kernels_not_null_int64.cu
  qep/operator/bounded_hash_aggregate/not_null_check.cu
  qep/operator/bounded_hash_aggregate/scan_kernels_int32.cu
  qep/operator/bounded_hash_aggregate/scan_kernels_int64.cu
  qep/operator/bounded_hash_aggregate/scan_task.cpp
  qep/operator/equality_semi_anti_join.cpp
  qep/operator/fetch.cpp
  qep/operator/filter.cpp
  qep/operator/filter_mask.cu
  qep/operator/filter_join_indices.cpp
  qep/operator/gather.cpp
  qep/operator/hash_aggregate.cpp
  qep/operator/hash_join.cpp
  qep/operator/conditional_inner_join.cpp
  qep/operator/conditional_semi_anti_join.cpp
  qep/operator/mixed_semi_anti_join.cpp
  qep/operator/unique_key_inner_join.cpp
  qep/operator/intermediate_state_read.cpp
  qep/operator/intermediate_state_write.cpp
  qep/operator/join_materialized_column_layout.cpp
  qep/operator/join_materialization.cpp
  qep/operator/join_output.cpp
  qep/operator/mark_join/launchers.cu
  qep/operator/mark_join/tasks.cpp
  qep/operator/materialized_chunks.cpp
  qep/operator/merge_sort.cpp
  qep/operator/perfect_hash_aggregate.cpp
  qep/operator/perfect_hash_join.cpp
  qep/operator/reduce_aggregate.cpp
  qep/operator/top_k.cpp
  qep/operator/project.cpp
  qep/pipeline_normalization.cpp
  qep/physical_qep_transformation.cpp
  qep/query_execution_plan.cpp
  qep/shapes/deferred_column.cpp
  qep/shapes/indirect_table.cpp
  qep/shapes/keyed_accumulator.cpp
  qep/shapes/logical_table.cpp
  qep/shapes/masked_table.cpp
  qep/shapes/row_count.cpp
  qep/shapes/write_statistics.cpp
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
  scheduling/sm_idle_monitor.cpp
  storage/buffer.cpp
  storage/chunked_in_memory_reader.cpp
  storage/chunked_in_memory_writer.cpp
  storage/chunked_parquet_reader.cpp
  storage/chunked_parquet_writer.cpp
  storage/compression.cpp
  storage/descriptor.cpp
  storage/hive_partitioning.cpp
  storage/in_memory.cpp
  storage/in_memory_resolve.cpp
  storage/in_memory_readable_view.cpp
  storage/in_memory_read_task.cpp
  storage/in_memory_reference_iterate.cpp
  storage/pruning.cpp
  storage/source_references.cpp
  storage/shared_segment.cpp
  storage/segment_layout.cpp
  storage/nvcomp_adapter.cpp
  storage/oneshot_in_memory_reader.cpp
  storage/oneshot_parquet_reader.cpp
  storage/parquet.cpp
  storage/parquet.cu
  storage/parquet_common.cpp
  storage/parquet_resolve.cpp
  storage/parquet_footer_cache.cpp
  storage/parquet_reader.cpp
  storage/parquet_reader.cu
  storage/parquet_reference_iterate.cpp
  storage/shared_memory.cpp
  storage/sliced_columns.cpp
  storage/sliced_columns.cu
  utility/cuda.cpp
  utility/async_logging.cpp
  utility/cupti_range.cpp
  utility/cupti_activity.cpp
  utility/cupti_pm_sampler.cpp
  utility/hashing.cpp
  utility/helpers.cpp
  utility/linux.cpp
  utility/thread_entry.cpp
  utility/logger.cpp
  utility/grpc_log_sink.cpp
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
  query_stream_manager.cpp
  task_manager_context.cpp
  scheduler.cpp
  types.cpp
  communicator.cpp
  storage/zone_map.cpp)

target_include_directories(gqe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(gqe PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../proto)
target_include_directories(
  gqe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/CUPTI/include)

# Arrow publishes its headers through directory-scope include_directories() in
# its own build tree rather than through the arrow_shared interface, so linking
# the library below does not bring them along. SYSTEM because nvcc runs with
# --Werror=all-warnings, which fails on the discarded nodiscard Status in
# Arrow's result.h.
target_include_directories(gqe SYSTEM BEFORE PUBLIC ${ARROW_INCLUDE_DIRS})

target_link_libraries(gqe PUBLIC proto)
# PUBLIC, so that the library and everything that includes its headers agree on
# whether the test-only code is present. gqe is not installed or exported.
target_link_libraries(gqe PUBLIC gqe_fault_injection)
target_link_libraries(gqe PUBLIC gqe-version)
target_link_libraries(gqe PUBLIC gqe_device_properties)
target_link_libraries(gqe PUBLIC cuco::cuco)
target_link_libraries(gqe PUBLIC CUDA::cuda_driver)
target_link_libraries(gqe PUBLIC CUDA::cupti)
target_link_libraries(gqe PUBLIC cudf::cudf)
target_link_libraries(gqe PUBLIC arrow_shared)
target_link_libraries(gqe PUBLIC parquet_shared)
target_link_libraries(gqe PUBLIC numa)
# PUBLIC because utility/cuda.hpp is a public header that includes
# nvtx3/nvtx3.hpp and defines the GQE domain tag, so consumers must compile it
# against this same nvtx3.
target_link_libraries(gqe PUBLIC nvtx3-cpp)
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
target_link_libraries(gqe PUBLIC CUDA::cuda_driver)
target_link_libraries(gqe PUBLIC CUDA::nvml)

if(NOT (nvcomp_VERSION_MAJOR EQUAL 5 AND nvcomp_VERSION_MINOR EQUAL 3))
  message(
    FATAL_ERROR
      "Unsupported nvcomp version: ${nvcomp_VERSION_MAJOR}.${nvcomp_VERSION_MINOR}. GQE requires nvcomp 5.3."
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
