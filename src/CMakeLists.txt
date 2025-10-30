# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# We cannot use add_custom_command() since the file names need to be known at
#  configuration time so depenencies can be known during compile time
execute_process(
    COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generate_substrait.sh ${substrait_SOURCE_DIR} ${CMAKE_BINARY_DIR}
)

file(GLOB substrait_sources ${CMAKE_BINARY_DIR}/substrait/*.cc ${CMAKE_BINARY_DIR}/substrait/extensions/*.cc)
add_library(substrait ${substrait_sources})
target_include_directories(substrait PUBLIC ${CMAKE_BINARY_DIR})
target_link_libraries(substrait PRIVATE protobuf::libprotobuf)
# Setting `-DNDEBUG` to match the ABI of libprotobuf.so
# See this issue for details: https://github.com/protocolbuffers/protobuf/issues/9947
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
    logical/from_substrait.cpp
    memory_resource/memory_utilities.cpp
    memory_resource/numa_memory_resource.cpp
    memory_resource/pinned_memory_resource.cpp
    memory_resource/system_memory_resource.cpp
    memory_resource/pgas_memory_resource.cpp
    memory_resource/boost_shared_memory_resource.cpp
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
    executor/task.cpp
    executor/window.cpp
    executor/write.cpp
    executor/task_graph.cpp
    executor/partition.cpp
    executor/partition_merge.cpp
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
    optimizer/relation_properties.cpp
    optimizer/rules/column_name_assignment.cpp
    optimizer/rules/projection_pushdown.cpp
    optimizer/rules/join_children_swap.cpp
    optimizer/rules/join_unique_keys.cpp
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
    physical/shuffle.cpp
    rpc/task_migration.cpp
    storage/compression.cpp
    storage/in_memory.cpp
    storage/sliced_columns.cpp
    storage/sliced_columns.cu
    storage/parquet.cpp
    storage/parquet.cu
    storage/parquet_reader.cpp
    storage/parquet_reader.cu
    utility/cuda.cpp
    utility/cupti.cpp
    utility/hashing.cpp
    utility/linux.cpp
    utility/timer.cpp
    utility/tpcds.cpp
    utility/tpch.cpp
    utility/mpi_helpers.cpp
    utility/kernel_fusion_helpers.cpp
    catalog.cpp
    query_context.cpp
    task_manager_context.cpp
    device_properties.cpp
    scheduler.cpp
    types.cpp
    communicator.cpp
    storage/zone_map.cpp
)

target_include_directories(gqe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(gqe PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../proto)
target_include_directories(gqe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/CUPTI/include)

target_link_libraries(gqe PUBLIC proto)
target_link_libraries(gqe PUBLIC gqe-version)
target_link_libraries(gqe PUBLIC cuco::cuco)
target_link_libraries(gqe PUBLIC CUDA::cuda_driver)
target_link_libraries(gqe PUBLIC CUDA::cupti)
target_link_libraries(gqe PUBLIC cudf::cudf)
target_link_libraries(gqe PUBLIC numa)
target_link_libraries(gqe PRIVATE nvtx3-cpp)
target_link_libraries(gqe PUBLIC spdlog::spdlog_header_only)
target_link_libraries(gqe PUBLIC substrait)
target_link_libraries(gqe PRIVATE Threads::Threads)
target_link_libraries(gqe PRIVATE nvcomp::nvcomp)
target_link_libraries(gqe PUBLIC uring)
target_link_libraries(gqe PUBLIC MPI::MPI_CXX)
target_link_libraries(gqe PUBLIC nvshmem::nvshmem_host)
target_link_libraries(gqe PRIVATE libperfect)
target_link_libraries(gqe PUBLIC CUDA::cuda_driver)

# Download and configure Parquet format
if(GQE_ENABLE_CUSTOMIZED_PARQUET)
    file(DOWNLOAD
        https://raw.githubusercontent.com/apache/parquet-format/c185faf0c4fc0c7d3075d1abd4ed0985cdbe5d87/src/main/thrift/parquet.thrift
        ${CMAKE_BINARY_DIR}/parquet/parquet.thrift)
    execute_process(
        COMMAND thrift -r --gen cpp -out ${CMAKE_BINARY_DIR}/parquet ${CMAKE_BINARY_DIR}/parquet/parquet.thrift
        COMMAND_ERROR_IS_FATAL ANY)
    add_library(parquet ${CMAKE_BINARY_DIR}/parquet/parquet_types.cpp)
    target_include_directories(parquet PUBLIC ${CMAKE_BINARY_DIR}/parquet)
    target_link_libraries(parquet PUBLIC thrift::thrift)

    target_link_libraries(gqe PRIVATE parquet uring)
    add_compile_definitions(ENABLE_CUSTOMIZED_PARQUET)
endif()

if(GQE_ENABLE_QUERY_COMPILER)
  include_directories(${PROJECT_BINARY_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../include)
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
    MLIRLLVMToLLVMIRTranslation
  )

  target_include_directories(gqe PUBLIC ${CMAKE_BINARY_DIR}/include)
  target_sources(
    gqe
    PUBLIC
    utility/cuda_driver.cpp
    utility/kernel_arguments_builder.cpp
  )
  target_link_libraries(
    gqe
    PUBLIC
    ${COMPILER_LIBS}
  )
endif()
