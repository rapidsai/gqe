# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

if(GQE_ENABLE_QUERY_COMPILER)
  add_subdirectory(compiler)
endif()

add_library(logical_test_common logical/common.cpp)
set_property(TARGET logical_test_common PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
target_include_directories(logical_test_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(logical_test_common PRIVATE ${CMAKE_BINARY_DIR})
target_link_libraries(logical_test_common PUBLIC cudf::cudf)

add_executable(run_substrait_consumer logical/run_substrait_consumer.cpp)
target_link_libraries(run_substrait_consumer PUBLIC gqe)
target_link_libraries(run_substrait_consumer PRIVATE logical_test_common)

add_executable(join_task_test executor/join_task_test.cpp)
target_include_directories(join_task_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(join_task_test PUBLIC gqe)
target_link_libraries(join_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(join_task_test PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME join_task COMMAND join_task_test)

add_executable(unique_key_inner_join_test executor/unique_key_inner_join_test.cpp)
target_include_directories(unique_key_inner_join_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(unique_key_inner_join_test PUBLIC gqe)
target_link_libraries(unique_key_inner_join_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(unique_key_inner_join_test PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME unique_key_inner_join COMMAND unique_key_inner_join_test)

add_executable(read_task_test executor/read_task_test.cpp)
target_include_directories(read_task_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(read_task_test PUBLIC gqe)
target_link_libraries(read_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(read_task_test PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME read_task COMMAND read_task_test)

add_executable(concatenate_task_test executor/concatenate_task_test.cpp)
target_include_directories(concatenate_task_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(concatenate_task_test PUBLIC gqe)
target_link_libraries(concatenate_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(concatenate_task_test PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME concatenate_task COMMAND concatenate_task_test)

add_executable(project_task_test executor/project_task_test.cpp)
target_include_directories(project_task_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(project_task_test PUBLIC gqe)
target_link_libraries(project_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(project_task_test PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME project_task COMMAND project_task_test)

add_executable(sort_task_test executor/sort_task_test.cpp)
target_include_directories(sort_task_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(sort_task_test PUBLIC gqe)
target_link_libraries(sort_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(sort_task_test PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME sort_task COMMAND sort_task_test)

add_executable(filter_task_test executor/filter_task_test.cpp)
target_include_directories(filter_task_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(filter_task_test PUBLIC gqe)
target_link_libraries(filter_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(filter_task_test PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME filter_task COMMAND filter_task_test)

add_executable(fetch_task_test executor/fetch_task_test.cpp)
target_include_directories(fetch_task_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(fetch_task_test PUBLIC gqe)
target_link_libraries(fetch_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(fetch_task_test PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME fetch_task COMMAND fetch_task_test)

add_executable(aggregate_task_test executor/aggregate_task_test.cpp)
target_include_directories(aggregate_task_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(aggregate_task_test PUBLIC gqe)
target_link_libraries(aggregate_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(aggregate_task_test PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME aggregate_task COMMAND aggregate_task_test)

add_executable(in_memory_storage integration/in_memory_storage.cpp)
target_include_directories(in_memory_storage PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(in_memory_storage PUBLIC gqe)
target_link_libraries(in_memory_storage PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(in_memory_storage PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME in_memory_storage COMMAND in_memory_storage)

add_executable(logical_to_execution integration/logical_to_execution.cpp)
target_include_directories(logical_to_execution PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(logical_to_execution PUBLIC gqe)
target_link_libraries(logical_to_execution PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(logical_to_execution PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME logical_to_execution COMMAND logical_to_execution)

add_executable(substrait_to_physical integration/substrait_to_physical.cpp)
target_include_directories(substrait_to_physical PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(substrait_to_physical PUBLIC gqe)
target_link_libraries(substrait_to_physical PUBLIC GTest::gtest_main)
target_compile_definitions(substrait_to_physical PUBLIC TEST_RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resource/")
add_test(NAME substrait_to_physical COMMAND substrait_to_physical)

add_executable(memory_resource memory_resource/memory_resource.cpp)
target_include_directories(memory_resource PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(memory_resource PUBLIC gqe)
target_link_libraries(memory_resource PUBLIC GTest::gtest_main GTest::gmock)
add_test(NAME memory_resource COMMAND memory_resource)

add_executable(parquet_storage integration/parquet_storage.cpp)
target_include_directories(parquet_storage PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(parquet_storage PUBLIC gqe)
target_link_libraries(parquet_storage PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(parquet_storage PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME parquet_storage COMMAND parquet_storage)

add_executable(eval_expressions_test executor/eval_expressions_test.cpp)
target_include_directories(eval_expressions_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(eval_expressions_test PUBLIC gqe)
target_link_libraries(eval_expressions_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(eval_expressions_test PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME eval_expressions COMMAND eval_expressions_test)

add_executable(window_task_test executor/window_task_test.cpp)
target_include_directories(window_task_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(window_task_test PUBLIC gqe)
target_link_libraries(window_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(window_task_test PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME window_task COMMAND window_task_test)

add_executable(in_memory_read_task_test storage/in_memory_read_task_test.cpp)
target_include_directories(in_memory_read_task_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(in_memory_read_task_test PUBLIC gqe)
target_link_libraries(in_memory_read_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(in_memory_read_task_test PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME in_memory_read_task COMMAND in_memory_read_task_test)

add_executable(expression_equal_test logical/expression_equal_test.cpp)
target_include_directories(expression_equal_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(expression_equal_test PUBLIC gqe)
target_link_libraries(expression_equal_test PUBLIC GTest::gtest_main GTest::gmock)
add_test(NAME expression_equal_test COMMAND expression_equal_test)

add_executable(relation_equal_test logical/relation_equal_test.cpp)
target_include_directories(relation_equal_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(relation_equal_test PUBLIC gqe)
target_link_libraries(relation_equal_test PUBLIC GTest::gtest_main GTest::gmock)
add_test(NAME relation_equal_test COMMAND relation_equal_test)

add_executable(groupby_test executor/groupby/argmax_tests.cpp 
                            executor/groupby/argmin_tests.cpp 
                            executor/groupby/count_tests.cpp 
                            executor/groupby/groupby_test_util.cpp
                            executor/groupby/keys_tests.cpp
                            executor/groupby/max_tests.cpp 
                            executor/groupby/min_tests.cpp 
                            executor/groupby/product_tests.cpp
                            executor/groupby/stress_test.cpp
                            executor/groupby/sum_of_squares_tests.cpp
                            executor/groupby/sum_tests.cpp
                            )
target_link_libraries(groupby_test PUBLIC gqe)
target_link_libraries(groupby_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(groupby_test PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME groupby_test COMMAND groupby_test)

if(GQE_ENABLE_CUSTOMIZED_PARQUET)
    add_executable(parquet_reader_test storage/parquet_reader.cpp)
    target_include_directories(parquet_reader_test PUBLIC ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(parquet_reader_test PUBLIC gqe)
    target_link_libraries(parquet_reader_test PUBLIC GTest::gtest_main GTest::gmock)
    target_link_libraries(parquet_reader_test PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
    add_test(NAME parquet_reader COMMAND parquet_reader_test)
endif()

add_executable(rewrite_rule_test optimizer/rewrite_rule_test.cpp)
target_include_directories(rewrite_rule_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(rewrite_rule_test PUBLIC gqe)
target_link_libraries(rewrite_rule_test PUBLIC GTest::gtest_main GTest::gmock)
add_test(NAME rewrite_rule_test COMMAND rewrite_rule_test)

add_executable(string_to_int_literal_test optimizer/string_to_int_literal_test.cpp)
target_include_directories(string_to_int_literal_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(string_to_int_literal_test PUBLIC gqe)
target_link_libraries(string_to_int_literal_test PUBLIC GTest::gtest_main GTest::gmock)
add_test(NAME string_to_int_literal_test COMMAND string_to_int_literal_test)

add_executable(join_children_swap_test optimizer/join_children_swap_test.cpp)
target_include_directories(join_children_swap_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(join_children_swap_test PUBLIC gqe)
target_link_libraries(join_children_swap_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(join_children_swap_test PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME join_children_swap_test COMMAND join_children_swap_test)

add_executable(projection_pushdown_test optimizer/projection_pushdown_test.cpp)
target_include_directories(projection_pushdown_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(projection_pushdown_test PUBLIC gqe)
target_link_libraries(projection_pushdown_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(projection_pushdown_test PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME projection_pushdown_test COMMAND projection_pushdown_test)

add_executable(uniqueness_prop_test optimizer/uniqueness_propagation_test.cpp)
target_include_directories(uniqueness_prop_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(uniqueness_prop_test PUBLIC gqe)
target_link_libraries(uniqueness_prop_test PUBLIC GTest::gtest_main GTest::gmock)
add_test(NAME uniqueness_prop_test COMMAND uniqueness_prop_test)

add_executable(task_graph_test task_graph/task_graph_generation_test.cpp)
target_include_directories(task_graph_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(task_graph_test PUBLIC gqe)
target_link_libraries(task_graph_test PUBLIC GTest::gtest_main GTest::gmock)
add_test(NAME task_graph_test COMMAND task_graph_test)

add_executable(multi_process multiprocess/execution.cpp)
target_include_directories(multi_process PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(multi_process PUBLIC gqe)
target_link_libraries(multi_process PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(multi_process PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME multi_process COMMAND multi_process)

add_executable(partial_filter_projection_test optimizer/fix_partial_filter_column_references_test.cpp)
target_include_directories(partial_filter_projection_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(partial_filter_projection_test PUBLIC gqe)
target_link_libraries(partial_filter_projection_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(partial_filter_projection_test PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME partial_filter_projection_test COMMAND partial_filter_projection_test)

