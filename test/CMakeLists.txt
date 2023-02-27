# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
target_link_libraries(join_task_test PUBLIC cudf::cudftestutil)
add_test(NAME join_task COMMAND join_task_test)

add_executable(read_task_test executor/read_task_test.cpp)
target_include_directories(read_task_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(read_task_test PUBLIC gqe)
target_link_libraries(read_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(read_task_test PUBLIC cudf::cudftestutil)
add_test(NAME read_task COMMAND read_task_test)

add_executable(concatenate_task_test executor/concatenate_task_test.cpp)
target_include_directories(concatenate_task_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(concatenate_task_test PUBLIC gqe)
target_link_libraries(concatenate_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(concatenate_task_test PUBLIC cudf::cudftestutil)
add_test(NAME concatenate_task COMMAND concatenate_task_test)

add_executable(project_task_test executor/project_task_test.cpp)
target_include_directories(project_task_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(project_task_test PUBLIC gqe)
target_link_libraries(project_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(project_task_test PUBLIC cudf::cudftestutil)
add_test(NAME project_task COMMAND project_task_test)

add_executable(sort_task_test executor/sort_task_test.cpp)
target_include_directories(sort_task_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(sort_task_test PUBLIC gqe)
target_link_libraries(sort_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(sort_task_test PUBLIC cudf::cudftestutil)
add_test(NAME sort_task COMMAND sort_task_test)

add_executable(filter_task_test executor/filter_task_test.cpp)
target_include_directories(filter_task_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(filter_task_test PUBLIC gqe)
target_link_libraries(filter_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(filter_task_test PUBLIC cudf::cudftestutil)
add_test(NAME filter_task COMMAND filter_task_test)

add_executable(fetch_task_test executor/fetch_task_test.cpp)
target_include_directories(fetch_task_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(fetch_task_test PUBLIC gqe)
target_link_libraries(fetch_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(fetch_task_test PUBLIC cudf::cudftestutil)
add_test(NAME fetch_task COMMAND fetch_task_test)

add_executable(aggregate_task_test executor/aggregate_task_test.cpp)
target_include_directories(aggregate_task_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(aggregate_task_test PUBLIC gqe)
target_link_libraries(aggregate_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(aggregate_task_test PUBLIC cudf::cudftestutil)
add_test(NAME aggregate_task COMMAND aggregate_task_test)

add_executable(logical_to_execution integration/logical_to_execution.cpp)
target_include_directories(logical_to_execution PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(logical_to_execution PUBLIC gqe)
target_link_libraries(logical_to_execution PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(logical_to_execution PUBLIC cudf::cudftestutil)
add_test(NAME logical_to_execution COMMAND logical_to_execution)

add_executable(eval_expressions_test executor/eval_expressions_test.cpp)
target_include_directories(eval_expressions_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(eval_expressions_test PUBLIC gqe)
target_link_libraries(eval_expressions_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(eval_expressions_test PUBLIC cudf::cudftestutil)
add_test(NAME eval_expressions COMMAND eval_expressions_test)
