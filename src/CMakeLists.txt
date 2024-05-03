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
    COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generate_substrait.sh ${CMAKE_CURRENT_SOURCE_DIR}/../substrait ${CMAKE_BINARY_DIR}
)

file(GLOB substrait_sources ${CMAKE_BINARY_DIR}/substrait/*.cc ${CMAKE_BINARY_DIR}/substrait/extensions/*.cc)
add_library(substrait ${substrait_sources})
target_include_directories(substrait PUBLIC ${CMAKE_BINARY_DIR})
target_link_libraries(substrait PRIVATE protobuf::libprotobuf)
# Setting `-DNDEBUG` to match the ABI of libprotobuf.so
# See this issue for details: https://github.com/protocolbuffers/protobuf/issues/9947
target_compile_options(substrait PRIVATE -DNDEBUG)

add_library(
    gqe STATIC
    logical/aggregate.cpp
    logical/fetch.cpp
    logical/filter.cpp
    logical/join.cpp
    logical/project.cpp
    logical/read.cpp
    logical/set.cpp
    logical/sort.cpp
    logical/window.cpp
    logical/user_defined.cpp
    logical/write.cpp
    logical/from_substrait.cpp
    memory_resource/numa_memory_resource.cpp
    memory_resource/pinned_memory_resource.cpp
    memory_resource/system_memory_resource.cpp
    executor/groupby/groupby.cu
    executor/aggregate.cpp
    executor/concatenate.cpp
    executor/eval.cpp
    executor/filter.cpp
    executor/fetch.cpp
    executor/gen_ident_col.cu
    executor/unique_key_inner_join.cu
    executor/join.cpp
    executor/optimization_parameters.cpp
    executor/project.cpp
    executor/read.cpp
    executor/sort.cpp
    executor/task.cpp
    executor/window.cpp
    executor/write.cpp
    executor/task_graph.cpp
    expression/binary_op.cpp
    expression/cast.cpp
    expression/if_then_else.cpp
    expression/scalar_function.cpp
    expression/subquery.cpp
    expression/unary_op.cpp
    expression/is_null.cpp
    optimizer/estimator.cpp
    optimizer/logical_optimization.cpp
    optimizer/optimization_configuration.cpp
    optimizer/physical_transformation.cpp
    optimizer/rules/join_children_swap.cpp
    optimizer/rules/not_not.cpp
    storage/in_memory.cpp
    storage/parquet.cpp
    storage/parquet.cu
    storage/parquet_reader.cpp
    storage/parquet_reader.cu
    utility/cuda.cpp
    utility/linux.cpp
    utility/tpcds.cpp
    utility/tpch.cpp
    catalog.cpp
    query_context.cpp
    types.cpp
)

target_include_directories(gqe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(gqe PUBLIC cuco::cuco)
target_link_libraries(gqe PUBLIC cudf::cudf)
target_link_libraries(gqe PUBLIC numa)
target_link_libraries(gqe PRIVATE nvtx3-cpp)
target_link_libraries(gqe PUBLIC spdlog::spdlog)
target_link_libraries(gqe PUBLIC substrait)
target_link_libraries(gqe PRIVATE Threads::Threads)

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

    target_link_libraries(gqe PRIVATE parquet)
    target_link_libraries(gqe PRIVATE nvcomp::nvcomp)
    add_compile_definitions(ENABLE_CUSTOMIZED_PARQUET)
endif()
