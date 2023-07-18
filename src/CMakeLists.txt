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
    COMMAND sh ${CMAKE_SOURCE_DIR}/scripts/generate_substrait.sh ${CMAKE_SOURCE_DIR}/substrait ${CMAKE_BINARY_DIR}
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
    executor/aggregate.cpp
    executor/concatenate.cpp
    executor/eval.cpp
    executor/filter.cpp
    executor/fetch.cpp
    executor/gen_ident_col.cu
    executor/join.cpp
    executor/project.cpp
    executor/read.cpp
    executor/sort.cpp
    executor/task.cpp
    executor/window.cpp
    executor/write.cpp
    executor/task_graph.cpp
    expression/binary_op.cpp
    expression/scalar_function.cpp
    optimizer/physical_transformation.cpp
    optimizer/estimator.cpp
    storage/in_memory.cpp
    storage/parquet.cpp
    utility/cuda.cpp
    utility/helpers.cpp
    utility/linux.cpp
    utility/tpcds.cpp
    utility/tpch.cpp
    catalog.cpp
    types.cpp
)

target_include_directories(gqe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(gqe PUBLIC cudf::cudf)
target_link_libraries(gqe PUBLIC numa)
target_link_libraries(gqe PUBLIC spdlog::spdlog)
target_link_libraries(gqe PRIVATE Threads::Threads)
target_link_libraries(gqe PUBLIC substrait)
