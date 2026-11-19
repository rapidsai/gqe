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

if(GQE_ENABLE_QUERY_COMPILER)
  add_subdirectory(compiler)
endif()

add_executable(cupti_test utility/cupti_test.cpp)
target_include_directories(cupti_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(cupti_test PUBLIC gqe)
target_link_libraries(cupti_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(cupti_test PUBLIC cudf::cudftestutil
                                        cudf::cudftestutil_impl)
add_test(NAME cupti_test COMMAND cupti_test)

add_executable(sm_idle_monitor_test scheduling/sm_idle_monitor_test.cu)
target_include_directories(sm_idle_monitor_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(sm_idle_monitor_test PUBLIC gqe)
target_link_libraries(sm_idle_monitor_test PUBLIC GTest::gtest_main
                                                  GTest::gmock)
target_link_libraries(sm_idle_monitor_test PUBLIC cudf::cudftestutil
                                                  cudf::cudftestutil_impl)
add_test(NAME sm_idle_monitor_test COMMAND sm_idle_monitor_test)

add_executable(cuda_runtime_smoke_test utility/cuda_runtime_smoke_test.cpp)
target_link_libraries(cuda_runtime_smoke_test PRIVATE CUDA::cudart)
add_test(NAME cuda_runtime_smoke_test COMMAND cuda_runtime_smoke_test)

add_executable(device_properties_test utility/device_properties_test.cpp)
target_link_libraries(device_properties_test PUBLIC gqe GTest::gtest_main)
add_test(NAME device_properties_test COMMAND device_properties_test)

add_executable(decode_throughput_test optimizer/decode_throughput_test.cpp)
target_link_libraries(decode_throughput_test PUBLIC gqe GTest::gtest_main)
add_test(NAME decode_throughput_test COMMAND decode_throughput_test)

# The calibration generator's tests need a Python whose SQLite has
# percentile_cont, such as the GQE build container's.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(
    NAME calibration_generator_test
    COMMAND ${Python3_EXECUTABLE} -m unittest discover -s
            ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/calibration -p "test_*.py")
endif()

add_executable(uuid_test utility/uuid_test.cpp)
target_link_libraries(uuid_test PRIVATE gqe GTest::gtest_main)
add_test(NAME uuid_test COMMAND uuid_test)

add_executable(query_context_test query_context_test.cpp)
target_link_libraries(query_context_test PRIVATE gqe GTest::gtest_main)
add_test(NAME query_context_test COMMAND query_context_test)

add_executable(logger_test utility/logger_test.cpp)
target_link_libraries(logger_test PRIVATE gqe GTest::gtest_main)
add_test(NAME logger_test COMMAND logger_test)

add_executable(async_logging_test utility/async_logging_test.cpp)
target_link_libraries(async_logging_test PRIVATE gqe GTest::gtest_main)
add_test(NAME async_logging_test COMMAND async_logging_test)
set_tests_properties(async_logging_test PROPERTIES TIMEOUT 60)

add_executable(types_test utility/types_test.cpp)
target_link_libraries(types_test PRIVATE gqe GTest::gtest_main)
add_test(NAME types_test COMMAND types_test)

add_executable(catalog_test catalog/catalog_test.cpp)
target_link_libraries(catalog_test PRIVATE gqe GTest::gtest_main)
add_test(NAME catalog_test COMMAND catalog_test)

add_executable(optimization_parameters_test
               rpc/optimization_parameters_test.cpp)
target_link_libraries(optimization_parameters_test
                      PRIVATE gqe GTest::gtest_main GTest::gmock)
add_test(NAME optimization_parameters_test COMMAND optimization_parameters_test)

add_executable(bounded_hash_aggregate_relation_test
               physical/bounded_hash_aggregate_relation_test.cpp)
target_link_libraries(bounded_hash_aggregate_relation_test
                      PRIVATE gqe GTest::gtest_main)
add_test(NAME bounded_hash_aggregate_relation_test
         COMMAND bounded_hash_aggregate_relation_test)

add_executable(physical_plan_serdes_test rpc/physical_plan_serdes_test.cpp)
target_link_libraries(physical_plan_serdes_test PRIVATE gqe GTest::gtest_main)
add_test(NAME physical_plan_serdes_test COMMAND physical_plan_serdes_test)

add_executable(statistics_serdes_test rpc/statistics_serdes_test.cpp)
target_link_libraries(statistics_serdes_test PRIVATE gqe GTest::gtest_main)
add_test(NAME statistics_serdes_test COMMAND statistics_serdes_test)

add_executable(tbb_test utility/tbb_test.cpp)
target_link_libraries(tbb_test PRIVATE GTest::gtest_main TBB::tbb)
add_test(NAME tbb_test COMMAND tbb_test)

add_executable(nccl_test utility/nccl_test.cpp)
target_include_directories(nccl_test PRIVATE ${NCCL_INCLUDE_DIRS})
target_link_libraries(nccl_test PRIVATE CUDA::cudart GTest::gtest_main
                                        ${NCCL_LIBRARIES})
add_test(NAME nccl_test COMMAND nccl_test)

add_executable(helpers_test utility/helpers.cpp)
target_link_libraries(helpers_test PUBLIC gqe GTest::gtest_main)
add_test(NAME helpers_test COMMAND helpers_test)

add_executable(linux_test utility/linux_test.cpp)
target_link_libraries(linux_test PUBLIC gqe GTest::gtest_main)
add_test(NAME linux_test COMMAND linux_test)

add_executable(thread_entry_test utility/thread_entry_test.cpp)
target_link_libraries(thread_entry_test PUBLIC gqe GTest::gtest_main)
add_test(NAME thread_entry_test COMMAND thread_entry_test)

add_executable(fault_injection_test utility/fault_injection_test.cpp)
target_link_libraries(fault_injection_test PUBLIC gqe GTest::gtest_main)
add_test(NAME fault_injection_test COMMAND fault_injection_test)

add_library(logical_test_common logical/common.cpp)
set_property(TARGET logical_test_common PROPERTY CUDA_ARCHITECTURES
                                                 ${GPU_ARCHS})
target_include_directories(logical_test_common
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(logical_test_common PRIVATE ${CMAKE_BINARY_DIR})
target_link_libraries(logical_test_common PUBLIC cudf::cudf)
target_link_libraries(logical_test_common PUBLIC nvcomp::nvcomp)

add_executable(run_substrait_consumer logical/run_substrait_consumer.cpp)
target_link_libraries(run_substrait_consumer PUBLIC gqe)
target_link_libraries(run_substrait_consumer PRIVATE logical_test_common)

add_executable(join_predicate_test executor/join_predicate_test.cpp)
target_include_directories(join_predicate_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(
  join_predicate_test PUBLIC gqe GTest::gtest_main GTest::gmock
                             cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME join_predicate COMMAND join_predicate_test)

add_executable(join_task_test executor/join_task_test.cpp)
target_include_directories(join_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(join_task_test PUBLIC gqe)
target_link_libraries(join_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(join_task_test PUBLIC cudf::cudftestutil
                                            cudf::cudftestutil_impl)
add_test(NAME join_task COMMAND join_task_test)

add_executable(mark_join_test executor/mark_join_test.cpp)
target_include_directories(mark_join_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(mark_join_test PUBLIC gqe)
target_link_libraries(mark_join_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(mark_join_test PUBLIC cudf::cudftestutil
                                            cudf::cudftestutil_impl)
add_test(NAME mark_join COMMAND mark_join_test)

add_executable(unique_key_inner_join_test
               executor/unique_key_inner_join_test.cpp)
target_include_directories(unique_key_inner_join_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(unique_key_inner_join_test PUBLIC gqe)
target_link_libraries(unique_key_inner_join_test PUBLIC GTest::gtest_main
                                                        GTest::gmock)
target_link_libraries(unique_key_inner_join_test PUBLIC cudf::cudftestutil
                                                        cudf::cudftestutil_impl)
add_test(NAME unique_key_inner_join COMMAND unique_key_inner_join_test)

add_executable(read_task_test executor/read_task_test.cpp)
target_include_directories(read_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(read_task_test PUBLIC gqe)
target_link_libraries(read_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(read_task_test PUBLIC cudf::cudftestutil
                                            cudf::cudftestutil_impl)
add_test(NAME read_task COMMAND read_task_test)

add_executable(concatenate_task_test executor/concatenate_task_test.cpp)
target_include_directories(concatenate_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(concatenate_task_test PUBLIC gqe)
target_link_libraries(concatenate_task_test PUBLIC GTest::gtest_main
                                                   GTest::gmock)
target_link_libraries(concatenate_task_test PUBLIC cudf::cudftestutil
                                                   cudf::cudftestutil_impl)
add_test(NAME concatenate_task COMMAND concatenate_task_test)

add_executable(project_task_test executor/project_task_test.cpp)
target_include_directories(project_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(project_task_test PUBLIC gqe)
target_link_libraries(project_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(project_task_test PUBLIC cudf::cudftestutil
                                               cudf::cudftestutil_impl)
add_test(NAME project_task COMMAND project_task_test)

add_executable(qep_adapter_test executor/qep_adapter_test.cpp)
target_include_directories(qep_adapter_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_adapter_test PUBLIC gqe)
target_link_libraries(qep_adapter_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(qep_adapter_test PUBLIC cudf::cudftestutil
                                              cudf::cudftestutil_impl)
add_test(NAME qep_adapter COMMAND qep_adapter_test)

add_executable(qep_algorithms_test qep/algorithms_test.cpp)
target_include_directories(qep_algorithms_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_algorithms_test PUBLIC gqe)
target_link_libraries(qep_algorithms_test PUBLIC GTest::gtest_main GTest::gmock)
add_test(NAME qep_algorithms_test COMMAND qep_algorithms_test)

add_executable(qep_physical_qep_transformation_test
               qep/physical_qep_transformation_test.cpp)
target_include_directories(qep_physical_qep_transformation_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_physical_qep_transformation_test PUBLIC gqe)
target_link_libraries(qep_physical_qep_transformation_test
                      PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(qep_physical_qep_transformation_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME qep_physical_qep_transformation
         COMMAND qep_physical_qep_transformation_test)

add_executable(qep_pipeline_normalization_test
               qep/pipeline_normalization_test.cpp)
target_include_directories(qep_pipeline_normalization_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_pipeline_normalization_test PUBLIC gqe)
target_link_libraries(qep_pipeline_normalization_test PUBLIC GTest::gtest_main
                                                             GTest::gmock)
add_test(NAME qep_pipeline_normalization
         COMMAND qep_pipeline_normalization_test)

add_executable(qep_masked_table_test qep/shapes/masked_table_test.cpp)
target_include_directories(qep_masked_table_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_masked_table_test PUBLIC gqe)
target_link_libraries(qep_masked_table_test PUBLIC GTest::gtest_main
                                                   GTest::gmock)
target_link_libraries(qep_masked_table_test PUBLIC cudf::cudftestutil
                                                   cudf::cudftestutil_impl)
add_test(NAME qep_masked_table COMMAND qep_masked_table_test)

add_executable(qep_indirect_table_test qep/shapes/indirect_table_test.cpp)
target_include_directories(qep_indirect_table_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_indirect_table_test PUBLIC gqe)
target_link_libraries(qep_indirect_table_test PUBLIC GTest::gtest_main
                                                     GTest::gmock)
target_link_libraries(qep_indirect_table_test PUBLIC cudf::cudftestutil
                                                     cudf::cudftestutil_impl)
add_test(NAME qep_indirect_table COMMAND qep_indirect_table_test)

add_executable(qep_state_test qep/state_test.cpp)
target_link_libraries(qep_state_test PUBLIC gqe)
target_link_libraries(qep_state_test PUBLIC GTest::gtest_main
                                            cudf::cudftestutil)
add_test(NAME qep_state COMMAND qep_state_test)

add_executable(qep_state_release_ordering_test
               qep/state_release_ordering_test.cu)
target_link_libraries(qep_state_release_ordering_test PUBLIC gqe)
target_link_libraries(qep_state_release_ordering_test PUBLIC GTest::gtest_main
                                                             cudf::cudftestutil)
add_test(NAME qep_state_release_ordering
         COMMAND qep_state_release_ordering_test)

add_executable(qep_logical_table_test qep/shapes/logical_table_test.cpp)
target_include_directories(qep_logical_table_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_logical_table_test PUBLIC gqe)
target_link_libraries(qep_logical_table_test PUBLIC GTest::gtest_main
                                                    GTest::gmock)
target_link_libraries(qep_logical_table_test PUBLIC cudf::cudftestutil
                                                    cudf::cudftestutil_impl)
add_test(NAME qep_logical_table COMMAND qep_logical_table_test)

add_executable(qep_indirect_gather_test qep/indirect_gather_test.cpp)
target_include_directories(qep_indirect_gather_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_indirect_gather_test PUBLIC gqe)
target_link_libraries(qep_indirect_gather_test PUBLIC GTest::gtest_main
                                                      GTest::gmock)
target_link_libraries(qep_indirect_gather_test PUBLIC cudf::cudftestutil
                                                      cudf::cudftestutil_impl)
add_test(NAME qep_indirect_gather COMMAND qep_indirect_gather_test)

add_executable(
  qep_column_materialization_test qep/column_materialization_test.cpp
                                  qep/column_materialization_lifetime_test.cu)
target_include_directories(qep_column_materialization_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_column_materialization_test PUBLIC gqe)
target_link_libraries(qep_column_materialization_test PUBLIC GTest::gtest_main
                                                             GTest::gmock)
target_link_libraries(qep_column_materialization_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME qep_column_materialization
         COMMAND qep_column_materialization_test)

add_executable(qep_fetch_task_test qep/operator/fetch_task_test.cpp)
target_include_directories(qep_fetch_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_fetch_task_test PUBLIC gqe)
target_link_libraries(qep_fetch_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(qep_fetch_task_test PUBLIC cudf::cudftestutil
                                                 cudf::cudftestutil_impl)
add_test(NAME qep_fetch COMMAND qep_fetch_task_test)

add_executable(qep_filter_task_test qep/operator/filter_task_test.cpp)
target_include_directories(qep_filter_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_filter_task_test PUBLIC gqe)
target_link_libraries(qep_filter_task_test PUBLIC GTest::gtest_main
                                                  GTest::gmock)
target_link_libraries(qep_filter_task_test PUBLIC cudf::cudftestutil
                                                  cudf::cudftestutil_impl)
add_test(NAME qep_filter COMMAND qep_filter_task_test)

add_executable(qep_reduce_aggregate_task_test
               qep/operator/reduce_aggregate_task_test.cpp)
target_include_directories(qep_reduce_aggregate_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_reduce_aggregate_task_test PUBLIC gqe)
target_link_libraries(qep_reduce_aggregate_task_test PUBLIC GTest::gtest_main
                                                            GTest::gmock)
target_link_libraries(qep_reduce_aggregate_task_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME qep_reduce_aggregate COMMAND qep_reduce_aggregate_task_test)

add_executable(qep_perfect_hash_aggregate_task_test
               qep/operator/perfect_hash_aggregate_task_test.cpp)
target_include_directories(qep_perfect_hash_aggregate_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_perfect_hash_aggregate_task_test PUBLIC gqe)
target_link_libraries(qep_perfect_hash_aggregate_task_test
                      PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(qep_perfect_hash_aggregate_task_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME qep_perfect_hash_aggregate
         COMMAND qep_perfect_hash_aggregate_task_test)

add_executable(qep_bounded_hash_aggregate_task_test
               qep/operator/bounded_hash_aggregate_task_test.cpp)
target_include_directories(qep_bounded_hash_aggregate_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_bounded_hash_aggregate_task_test PUBLIC gqe)
target_link_libraries(qep_bounded_hash_aggregate_task_test
                      PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(qep_bounded_hash_aggregate_task_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME qep_bounded_hash_aggregate
         COMMAND qep_bounded_hash_aggregate_task_test)

add_executable(qep_hash_aggregate_task_test
               qep/operator/hash_aggregate_task_test.cpp)
target_include_directories(qep_hash_aggregate_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_hash_aggregate_task_test PUBLIC gqe)
target_link_libraries(qep_hash_aggregate_task_test PUBLIC GTest::gtest_main
                                                          GTest::gmock)
target_link_libraries(qep_hash_aggregate_task_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME qep_hash_aggregate COMMAND qep_hash_aggregate_task_test)

add_executable(qep_top_k_task_test qep/operator/top_k_task_test.cpp)
target_include_directories(qep_top_k_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_top_k_task_test PUBLIC gqe)
target_link_libraries(qep_top_k_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(qep_top_k_task_test PUBLIC cudf::cudftestutil
                                                 cudf::cudftestutil_impl)
add_test(NAME qep_top_k COMMAND qep_top_k_task_test)

add_executable(qep_gather_task_test qep/operator/gather_task_test.cpp)
target_include_directories(qep_gather_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_gather_task_test PUBLIC gqe)
target_link_libraries(qep_gather_task_test PUBLIC GTest::gtest_main
                                                  GTest::gmock)
target_link_libraries(qep_gather_task_test PUBLIC cudf::cudftestutil
                                                  cudf::cudftestutil_impl)
add_test(NAME qep_gather COMMAND qep_gather_task_test)

add_executable(qep_merge_sort_task_test qep/operator/merge_sort_task_test.cpp)
target_include_directories(qep_merge_sort_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_merge_sort_task_test PUBLIC gqe)
target_link_libraries(qep_merge_sort_task_test PUBLIC GTest::gtest_main
                                                      GTest::gmock)
target_link_libraries(qep_merge_sort_task_test PUBLIC cudf::cudftestutil
                                                      cudf::cudftestutil_impl)
add_test(NAME qep_merge_sort COMMAND qep_merge_sort_task_test)

add_executable(qep_join_materialization_test
               qep/operator/join_materialization_test.cpp)
target_include_directories(qep_join_materialization_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(
  qep_join_materialization_test
  PUBLIC gqe GTest::gtest_main GTest::gmock cudf::cudftestutil
         cudf::cudftestutil_impl)
add_test(NAME qep_join_materialization COMMAND qep_join_materialization_test)

add_executable(qep_perfect_hash_join_task_test
               qep/operator/perfect_hash_join_task_test.cpp)
target_include_directories(qep_perfect_hash_join_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_perfect_hash_join_task_test PUBLIC gqe)
target_link_libraries(qep_perfect_hash_join_task_test PUBLIC GTest::gtest_main
                                                             GTest::gmock)
target_link_libraries(qep_perfect_hash_join_task_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME qep_perfect_hash_join COMMAND qep_perfect_hash_join_task_test)

add_executable(qep_hash_join_task_test qep/operator/hash_join_task_test.cpp)
target_include_directories(qep_hash_join_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_hash_join_task_test PUBLIC gqe)
target_link_libraries(qep_hash_join_task_test PUBLIC GTest::gtest_main
                                                     GTest::gmock)
target_link_libraries(qep_hash_join_task_test PUBLIC cudf::cudftestutil
                                                     cudf::cudftestutil_impl)
add_test(NAME qep_hash_join COMMAND qep_hash_join_task_test)

add_executable(qep_equality_semi_anti_join_task_test
               qep/operator/equality_semi_anti_join_task_test.cpp)
target_include_directories(qep_equality_semi_anti_join_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_equality_semi_anti_join_task_test PUBLIC gqe)
target_link_libraries(qep_equality_semi_anti_join_task_test
                      PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(qep_equality_semi_anti_join_task_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME qep_equality_semi_anti_join
         COMMAND qep_equality_semi_anti_join_task_test)

add_executable(qep_mixed_semi_anti_join_task_test
               qep/operator/mixed_semi_anti_join_task_test.cpp)
target_include_directories(qep_mixed_semi_anti_join_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_mixed_semi_anti_join_task_test PUBLIC gqe)
target_link_libraries(qep_mixed_semi_anti_join_task_test
                      PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(qep_mixed_semi_anti_join_task_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME qep_mixed_semi_anti_join
         COMMAND qep_mixed_semi_anti_join_task_test)

add_executable(qep_conditional_semi_anti_join_task_test
               qep/operator/conditional_semi_anti_join_task_test.cpp)
target_include_directories(qep_conditional_semi_anti_join_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_conditional_semi_anti_join_task_test PUBLIC gqe)
target_link_libraries(qep_conditional_semi_anti_join_task_test
                      PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(qep_conditional_semi_anti_join_task_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME qep_conditional_semi_anti_join
         COMMAND qep_conditional_semi_anti_join_task_test)

add_executable(qep_conditional_inner_join_task_test
               qep/operator/conditional_inner_join_task_test.cpp)
target_include_directories(qep_conditional_inner_join_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_conditional_inner_join_task_test PUBLIC gqe)
target_link_libraries(qep_conditional_inner_join_task_test
                      PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(qep_conditional_inner_join_task_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME qep_conditional_inner_join
         COMMAND qep_conditional_inner_join_task_test)

add_executable(qep_mark_join_task_test qep/operator/mark_join_task_test.cpp)
target_include_directories(qep_mark_join_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_mark_join_task_test PUBLIC gqe)
target_link_libraries(qep_mark_join_task_test PUBLIC GTest::gtest_main
                                                     GTest::gmock)
target_link_libraries(qep_mark_join_task_test PUBLIC cudf::cudftestutil
                                                     cudf::cudftestutil_impl)
add_test(NAME qep_mark_join COMMAND qep_mark_join_task_test)

add_executable(qep_gather_integration_test
               executor/qep_gather_integration_test.cpp)
target_include_directories(qep_gather_integration_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_gather_integration_test PUBLIC gqe)
target_link_libraries(qep_gather_integration_test PUBLIC GTest::gtest_main
                                                         GTest::gmock)
target_link_libraries(qep_gather_integration_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME qep_gather_integration COMMAND qep_gather_integration_test)

add_executable(qep_fetch_integration_test
               executor/qep_fetch_integration_test.cpp)
target_include_directories(qep_fetch_integration_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_fetch_integration_test PUBLIC gqe)
target_link_libraries(qep_fetch_integration_test PUBLIC GTest::gtest_main
                                                        GTest::gmock)
target_link_libraries(qep_fetch_integration_test PUBLIC cudf::cudftestutil
                                                        cudf::cudftestutil_impl)
add_test(NAME qep_fetch_integration COMMAND qep_fetch_integration_test)

add_executable(qep_sort_integration_test executor/qep_sort_integration_test.cpp)
target_include_directories(qep_sort_integration_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_sort_integration_test PUBLIC gqe)
target_link_libraries(qep_sort_integration_test PUBLIC GTest::gtest_main
                                                       GTest::gmock)
target_link_libraries(qep_sort_integration_test PUBLIC cudf::cudftestutil
                                                       cudf::cudftestutil_impl)
add_test(NAME qep_sort_integration COMMAND qep_sort_integration_test)

add_executable(qep_top_k_integration_test
               executor/qep_top_k_integration_test.cpp)
target_include_directories(qep_top_k_integration_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_top_k_integration_test PUBLIC gqe)
target_link_libraries(qep_top_k_integration_test PUBLIC GTest::gtest_main
                                                        GTest::gmock)
target_link_libraries(qep_top_k_integration_test PUBLIC cudf::cudftestutil
                                                        cudf::cudftestutil_impl)
add_test(NAME qep_top_k_integration COMMAND qep_top_k_integration_test)

add_executable(qep_intermediate_state_read_task_test
               qep/operator/intermediate_state_read_task_test.cpp)
target_include_directories(qep_intermediate_state_read_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_intermediate_state_read_task_test PUBLIC gqe)
target_link_libraries(qep_intermediate_state_read_task_test
                      PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(qep_intermediate_state_read_task_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME qep_intermediate_state_read
         COMMAND qep_intermediate_state_read_task_test)

add_executable(qep_intermediate_state_read_integration_test
               executor/qep_intermediate_state_read_integration_test.cpp)
target_include_directories(qep_intermediate_state_read_integration_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_intermediate_state_read_integration_test PUBLIC gqe)
target_link_libraries(qep_intermediate_state_read_integration_test
                      PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(qep_intermediate_state_read_integration_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME qep_intermediate_state_read_integration
         COMMAND qep_intermediate_state_read_integration_test)

add_executable(qep_intermediate_state_write_task_test
               qep/operator/intermediate_state_write_task_test.cpp)
target_include_directories(qep_intermediate_state_write_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_intermediate_state_write_task_test PUBLIC gqe)
target_link_libraries(qep_intermediate_state_write_task_test
                      PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(qep_intermediate_state_write_task_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME qep_intermediate_state_write
         COMMAND qep_intermediate_state_write_task_test)

add_executable(qep_intermediate_state_write_integration_test
               executor/qep_intermediate_state_write_integration_test.cpp)
target_include_directories(qep_intermediate_state_write_integration_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_intermediate_state_write_integration_test PUBLIC gqe)
target_link_libraries(qep_intermediate_state_write_integration_test
                      PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(qep_intermediate_state_write_integration_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME qep_intermediate_state_write_integration
         COMMAND qep_intermediate_state_write_integration_test)

add_executable(qep_project_task_test qep/operator/project_task_test.cpp)
target_include_directories(qep_project_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_project_task_test PUBLIC gqe)
target_link_libraries(qep_project_task_test PUBLIC GTest::gtest_main
                                                   GTest::gmock)
target_link_libraries(qep_project_task_test PUBLIC cudf::cudftestutil
                                                   cudf::cudftestutil_impl)
add_test(NAME qep_project COMMAND qep_project_task_test)

add_executable(qep_join_integration_test executor/qep_join_integration_test.cpp)
target_include_directories(qep_join_integration_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_join_integration_test PUBLIC gqe)
target_link_libraries(qep_join_integration_test PUBLIC GTest::gtest_main
                                                       GTest::gmock)
target_link_libraries(qep_join_integration_test PUBLIC cudf::cudftestutil
                                                       cudf::cudftestutil_impl)
add_test(NAME qep_join_integration COMMAND qep_join_integration_test)

add_executable(qep_deferred_column_test qep/shapes/deferred_column_test.cpp)
target_include_directories(qep_deferred_column_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_deferred_column_test PUBLIC gqe)
target_link_libraries(qep_deferred_column_test PUBLIC GTest::gtest_main
                                                      GTest::gmock)
target_link_libraries(qep_deferred_column_test PUBLIC cudf::cudftestutil
                                                      cudf::cudftestutil_impl)
add_test(NAME qep_deferred_column COMMAND qep_deferred_column_test)

add_executable(qep_equality_semi_anti_join_integration_test
               executor/qep_equality_semi_anti_join_integration_test.cpp)
target_include_directories(qep_equality_semi_anti_join_integration_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_equality_semi_anti_join_integration_test PUBLIC gqe)
target_link_libraries(qep_equality_semi_anti_join_integration_test
                      PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(qep_equality_semi_anti_join_integration_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME qep_equality_semi_anti_join_integration
         COMMAND qep_equality_semi_anti_join_integration_test)

add_executable(qep_conditional_semi_anti_join_integration_test
               executor/qep_conditional_semi_anti_join_integration_test.cpp)
target_include_directories(qep_conditional_semi_anti_join_integration_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_conditional_semi_anti_join_integration_test
                      PUBLIC gqe)
target_link_libraries(qep_conditional_semi_anti_join_integration_test
                      PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(qep_conditional_semi_anti_join_integration_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME qep_conditional_semi_anti_join_integration
         COMMAND qep_conditional_semi_anti_join_integration_test)

add_executable(qep_conditional_inner_join_integration_test
               executor/qep_conditional_inner_join_integration_test.cpp)
target_include_directories(qep_conditional_inner_join_integration_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_conditional_inner_join_integration_test PUBLIC gqe)
target_link_libraries(qep_conditional_inner_join_integration_test
                      PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(qep_conditional_inner_join_integration_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME qep_conditional_inner_join_integration
         COMMAND qep_conditional_inner_join_integration_test)

add_executable(qep_mark_join_integration_test
               executor/qep_mark_join_integration_test.cpp)
target_include_directories(qep_mark_join_integration_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_mark_join_integration_test PUBLIC gqe)
target_link_libraries(qep_mark_join_integration_test PUBLIC GTest::gtest_main
                                                            GTest::gmock)
target_link_libraries(qep_mark_join_integration_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME qep_mark_join_integration COMMAND qep_mark_join_integration_test)

add_executable(qep_row_count_test qep/shapes/row_count_test.cpp)
target_include_directories(qep_row_count_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_row_count_test PUBLIC gqe)
target_link_libraries(qep_row_count_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(qep_row_count_test PUBLIC cudf::cudftestutil
                                                cudf::cudftestutil_impl)
add_test(NAME qep_row_count COMMAND qep_row_count_test)

add_executable(qep_write_statistics_test qep/shapes/write_statistics_test.cpp)
target_include_directories(qep_write_statistics_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_write_statistics_test PUBLIC gqe)
target_link_libraries(qep_write_statistics_test PUBLIC GTest::gtest_main
                                                       GTest::gmock)
target_link_libraries(qep_write_statistics_test PUBLIC cudf::cudftestutil
                                                       cudf::cudftestutil_impl)
add_test(NAME qep_write_statistics COMMAND qep_write_statistics_test)

add_executable(qep_task_graph_transformation_test
               executor/qep_task_graph_transformation_test.cpp)
target_include_directories(qep_task_graph_transformation_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(qep_task_graph_transformation_test PUBLIC gqe)
target_link_libraries(qep_task_graph_transformation_test
                      PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(qep_task_graph_transformation_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME qep_task_graph_transformation
         COMMAND qep_task_graph_transformation_test)

add_executable(sort_task_test executor/sort_task_test.cpp)
target_include_directories(sort_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(sort_task_test PUBLIC gqe)
target_link_libraries(sort_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(sort_task_test PUBLIC cudf::cudftestutil
                                            cudf::cudftestutil_impl)
add_test(NAME sort_task COMMAND sort_task_test)

add_executable(filter_task_test executor/filter_task_test.cpp)
target_include_directories(filter_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(filter_task_test PUBLIC gqe)
target_link_libraries(filter_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(filter_task_test PUBLIC cudf::cudftestutil
                                              cudf::cudftestutil_impl)
add_test(NAME filter_task COMMAND filter_task_test)

add_executable(fetch_task_test executor/fetch_task_test.cpp)
target_include_directories(fetch_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(fetch_task_test PUBLIC gqe)
target_link_libraries(fetch_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(fetch_task_test PUBLIC cudf::cudftestutil
                                             cudf::cudftestutil_impl)
add_test(NAME fetch_task COMMAND fetch_task_test)

add_executable(aggregate_task_test executor/aggregate_task_test.cpp)
target_include_directories(aggregate_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(aggregate_task_test PUBLIC gqe)
target_link_libraries(aggregate_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(aggregate_task_test PUBLIC cudf::cudftestutil
                                                 cudf::cudftestutil_impl)
add_test(NAME aggregate_task COMMAND aggregate_task_test)

add_executable(shuffle_task_test executor/shuffle_task_test.cpp)
target_include_directories(shuffle_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(shuffle_task_test PUBLIC gqe)
target_link_libraries(shuffle_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(shuffle_task_test PUBLIC cudf::cudftestutil
                                               cudf::cudftestutil_impl)
add_test(NAME shuffle_task COMMAND shuffle_task_test)

add_executable(shuffle_join_test executor/shuffle_join_test.cpp)
target_include_directories(shuffle_join_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(shuffle_join_test PUBLIC gqe)
target_link_libraries(shuffle_join_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(shuffle_join_test PUBLIC cudf::cudftestutil
                                               cudf::cudftestutil_impl)
add_test(NAME shuffle_join COMMAND shuffle_join_test)

add_executable(in_memory_storage integration/in_memory_storage.cpp)
target_include_directories(in_memory_storage
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(in_memory_storage PUBLIC gqe)
target_link_libraries(in_memory_storage PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(in_memory_storage PUBLIC cudf::cudftestutil
                                               cudf::cudftestutil_impl)
add_test(NAME in_memory_storage COMMAND in_memory_storage)

add_executable(logical_to_execution integration/logical_to_execution.cpp)
target_include_directories(logical_to_execution
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(logical_to_execution PUBLIC gqe)
target_link_libraries(logical_to_execution PUBLIC GTest::gtest_main
                                                  GTest::gmock)
target_link_libraries(logical_to_execution PUBLIC cudf::cudftestutil
                                                  cudf::cudftestutil_impl)
add_test(NAME logical_to_execution COMMAND logical_to_execution)

add_executable(substrait_to_physical integration/substrait_to_physical.cpp)
target_include_directories(substrait_to_physical
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(substrait_to_physical PUBLIC gqe)
target_link_libraries(substrait_to_physical PUBLIC GTest::gtest_main)
target_compile_definitions(
  substrait_to_physical
  PUBLIC TEST_RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resource/")
add_test(NAME substrait_to_physical COMMAND substrait_to_physical)

add_executable(memory_resource memory_resource/memory_resource.cpp)
target_include_directories(memory_resource
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(memory_resource PUBLIC gqe)
target_link_libraries(memory_resource PUBLIC GTest::gtest_main GTest::gmock)
if(GQE_ENABLE_FABRIC_SHARED_POOL_TESTS)
  target_compile_definitions(memory_resource
                             PRIVATE GQE_ENABLE_FABRIC_SHARED_POOL_TESTS)
endif()
add_test(NAME memory_resource COMMAND memory_resource)

add_executable(parquet_storage integration/parquet_storage.cpp)
target_include_directories(parquet_storage
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(parquet_storage PUBLIC gqe)
target_link_libraries(parquet_storage PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(parquet_storage PUBLIC cudf::cudftestutil
                                             cudf::cudftestutil_impl)
add_test(NAME parquet_storage COMMAND parquet_storage)

add_executable(eval_expressions_test executor/eval_expressions_test.cpp)
target_include_directories(eval_expressions_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(eval_expressions_test PUBLIC gqe)
target_link_libraries(eval_expressions_test PUBLIC GTest::gtest_main
                                                   GTest::gmock)
target_link_libraries(eval_expressions_test PUBLIC cudf::cudftestutil
                                                   cudf::cudftestutil_impl)
add_test(NAME eval_expressions COMMAND eval_expressions_test)

add_executable(like_expression_test executor/like_expression_test.cpp)
target_include_directories(like_expression_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(like_expression_test PUBLIC gqe)
target_link_libraries(like_expression_test PUBLIC GTest::gtest_main
                                                  GTest::gmock)
target_link_libraries(like_expression_test PUBLIC cudf::cudftestutil
                                                  cudf::cudftestutil_impl)
add_test(NAME like_expression COMMAND like_expression_test)

add_executable(window_task_test executor/window_task_test.cpp)
target_include_directories(window_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(window_task_test PUBLIC gqe)
target_link_libraries(window_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(window_task_test PUBLIC cudf::cudftestutil
                                              cudf::cudftestutil_impl)
add_test(NAME window_task COMMAND window_task_test)

add_executable(in_memory_read_task_test storage/in_memory_read_task_test.cpp)
target_include_directories(in_memory_read_task_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(in_memory_read_task_test PUBLIC gqe)
target_link_libraries(in_memory_read_task_test PUBLIC GTest::gtest_main
                                                      GTest::gmock)
target_link_libraries(in_memory_read_task_test PUBLIC cudf::cudftestutil
                                                      cudf::cudftestutil_impl)
add_test(NAME in_memory_read_task COMMAND in_memory_read_task_test)

# The test poisons a segment deliberately, and clears the poisoned flag between
# cases, which only a build with fault injection provides.
if(GQE_ENABLE_FAULT_INJECTION)
  add_executable(shared_memory_poisoning_test
                 storage/shared_memory_poisoning_test.cpp)
  target_link_libraries(
    shared_memory_poisoning_test
    PUBLIC gqe GTest::gtest_main GTest::gmock cudf::cudftestutil
           cudf::cudftestutil_impl)
  add_test(NAME shared_memory_poisoning_test
           COMMAND shared_memory_poisoning_test)
endif()

add_executable(segment_layout_test storage/segment_layout_test.cpp)
target_link_libraries(segment_layout_test PRIVATE gqe GTest::gtest_main)
add_test(NAME segment_layout_test COMMAND segment_layout_test)

add_executable(shared_segment_test storage/shared_segment_test.cpp)
target_link_libraries(shared_segment_test PRIVATE gqe GTest::gtest_main
                                                  GTest::gmock)
add_test(NAME shared_segment_test
         COMMAND shared_segment_test --gtest_filter=-SharedSegmentNumaTest.*)
# Hardware cases report CTest skips separately from the mandatory allocator
# tests.
add_test(
  NAME shared_segment_numa
  COMMAND shared_segment_test
          --gtest_filter=SharedSegmentNumaTest.BoundPolicyCreatesAndAttaches)
add_test(
  NAME shared_segment_numa_multinode
  COMMAND shared_segment_test
          --gtest_filter=SharedSegmentNumaTest.PagesResideOnEachArenasBoundNode)
add_test(NAME shared_segment_numa_membind
         COMMAND shared_segment_test
                 --gtest_filter=SharedSegmentNumaTest.MembindRestrictsArenas)
add_test(
  NAME shared_segment_numa_cpu_affinity
  COMMAND
    shared_segment_test
    --gtest_filter=SharedSegmentNumaTest.CpuAffinityDoesNotRestrictMemoryNodes)
set_tests_properties(
  shared_segment_numa shared_segment_numa_multinode shared_segment_numa_membind
  shared_segment_numa_cpu_affinity PROPERTIES LABELS "numa;hardware"
                                              SKIP_REGULAR_EXPRESSION "SKIPPED")
set_property(
  TEST shared_segment_numa_multinode shared_segment_numa_membind
       shared_segment_numa_cpu_affinity
  APPEND
  PROPERTY LABELS multi_node)

add_executable(output_table_builder_test storage/output_table_builder_test.cpp)
target_include_directories(output_table_builder_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(output_table_builder_test PUBLIC gqe)
target_link_libraries(output_table_builder_test PUBLIC GTest::gtest_main
                                                       GTest::gmock)
target_link_libraries(output_table_builder_test PUBLIC cudf::cudftestutil
                                                       cudf::cudftestutil_impl)
add_test(NAME output_table_builder COMMAND output_table_builder_test)

add_executable(chunked_parquet_reader_test
               storage/chunked_parquet_reader_test.cpp)
target_include_directories(chunked_parquet_reader_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(chunked_parquet_reader_test PUBLIC gqe)
target_link_libraries(chunked_parquet_reader_test PUBLIC GTest::gtest_main
                                                         GTest::gmock)
target_link_libraries(chunked_parquet_reader_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME chunked_parquet_reader COMMAND chunked_parquet_reader_test)

add_executable(chunked_in_memory_reader_test
               storage/chunked_in_memory_reader_test.cpp)
target_include_directories(chunked_in_memory_reader_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(chunked_in_memory_reader_test PUBLIC gqe)
target_link_libraries(chunked_in_memory_reader_test PUBLIC GTest::gtest_main
                                                           GTest::gmock)
target_link_libraries(chunked_in_memory_reader_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME chunked_in_memory_reader COMMAND chunked_in_memory_reader_test)

add_executable(in_memory_reference_read_test
               storage/in_memory_reference_read_test.cpp)
target_include_directories(in_memory_reference_read_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(in_memory_reference_read_test PUBLIC gqe)
target_link_libraries(in_memory_reference_read_test PUBLIC GTest::gtest_main
                                                           GTest::gmock)
target_link_libraries(in_memory_reference_read_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME in_memory_reference_read COMMAND in_memory_reference_read_test)

add_executable(parquet_reference_read_test
               storage/parquet_reference_read_test.cpp)
target_include_directories(parquet_reference_read_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(parquet_reference_read_test PUBLIC gqe)
target_link_libraries(parquet_reference_read_test PUBLIC GTest::gtest_main
                                                         GTest::gmock)
target_link_libraries(parquet_reference_read_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME parquet_reference_read COMMAND parquet_reference_read_test)

add_executable(oneshot_parquet_reader_test
               storage/oneshot_parquet_reader_test.cpp)
target_include_directories(oneshot_parquet_reader_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(oneshot_parquet_reader_test PUBLIC gqe)
target_link_libraries(oneshot_parquet_reader_test PUBLIC GTest::gtest_main
                                                         GTest::gmock)
target_link_libraries(oneshot_parquet_reader_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME oneshot_parquet_reader COMMAND oneshot_parquet_reader_test)

add_executable(oneshot_in_memory_reader_test
               storage/oneshot_in_memory_reader_test.cpp)
target_include_directories(oneshot_in_memory_reader_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(oneshot_in_memory_reader_test PUBLIC gqe)
target_link_libraries(oneshot_in_memory_reader_test PUBLIC GTest::gtest_main
                                                           GTest::gmock)
target_link_libraries(oneshot_in_memory_reader_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME oneshot_in_memory_reader COMMAND oneshot_in_memory_reader_test)

add_executable(hive_partitioning_test storage/hive_partitioning_test.cpp)
target_include_directories(hive_partitioning_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(hive_partitioning_test PUBLIC gqe)
target_link_libraries(hive_partitioning_test PUBLIC GTest::gtest_main
                                                    GTest::gmock)
target_link_libraries(hive_partitioning_test PUBLIC cudf::cudftestutil
                                                    cudf::cudftestutil_impl)
add_test(NAME hive_partitioning COMMAND hive_partitioning_test)

add_executable(chunked_parquet_writer_test
               storage/chunked_parquet_writer_test.cpp)
target_include_directories(chunked_parquet_writer_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(chunked_parquet_writer_test PUBLIC gqe)
target_link_libraries(chunked_parquet_writer_test PUBLIC GTest::gtest_main
                                                         GTest::gmock)
target_link_libraries(chunked_parquet_writer_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME chunked_parquet_writer COMMAND chunked_parquet_writer_test)

add_executable(chunked_in_memory_writer_test
               storage/chunked_in_memory_writer_test.cpp)
target_include_directories(chunked_in_memory_writer_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(chunked_in_memory_writer_test PUBLIC gqe)
target_link_libraries(chunked_in_memory_writer_test PUBLIC GTest::gtest_main
                                                           GTest::gmock)
target_link_libraries(chunked_in_memory_writer_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME chunked_in_memory_writer COMMAND chunked_in_memory_writer_test)

add_executable(expression_equal_test logical/expression_equal_test.cpp)
target_include_directories(expression_equal_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(expression_equal_test PUBLIC gqe)
target_link_libraries(expression_equal_test PUBLIC GTest::gtest_main
                                                   GTest::gmock)
add_test(NAME expression_equal_test COMMAND expression_equal_test)

add_executable(substrait_if_then_test logical/substrait_if_then_test.cpp)
target_include_directories(substrait_if_then_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(substrait_if_then_test PUBLIC gqe)
target_link_libraries(substrait_if_then_test PUBLIC GTest::gtest_main
                                                    GTest::gmock)
target_link_libraries(substrait_if_then_test PRIVATE logical_test_common)
add_test(NAME substrait_if_then_test COMMAND substrait_if_then_test)

add_executable(relation_equal_test logical/relation_equal_test.cpp)
target_include_directories(relation_equal_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(relation_equal_test PUBLIC gqe)
target_link_libraries(relation_equal_test PUBLIC GTest::gtest_main GTest::gmock)
add_test(NAME relation_equal_test COMMAND relation_equal_test)

add_executable(project_relation_test logical/project_relation_test.cpp
                                     physical/project_relation_test.cpp)
target_link_libraries(project_relation_test PRIVATE gqe GTest::gtest_main)
add_test(NAME project_relation_test COMMAND project_relation_test)

add_executable(
  groupby_test
  executor/groupby/argmax_tests.cpp
  executor/groupby/argmin_tests.cpp
  executor/groupby/count_tests.cpp
  executor/groupby/fallback_tests.cpp
  executor/groupby/groupby_test_util.cpp
  executor/groupby/keys_tests.cpp
  executor/groupby/max_tests.cpp
  executor/groupby/min_tests.cpp
  executor/groupby/product_tests.cpp
  executor/groupby/stress_test.cpp
  executor/groupby/sum_of_squares_tests.cpp
  executor/groupby/sum_tests.cpp)

target_include_directories(groupby_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(groupby_test PUBLIC gqe)
target_link_libraries(groupby_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(groupby_test PUBLIC cudf::cudftestutil
                                          cudf::cudftestutil_impl)
add_test(NAME groupby_test COMMAND groupby_test)

add_executable(
  perfect_hash_test
  executor/perfect_hash/unique_indices_tests.cpp
  executor/perfect_hash/scatter_aggregate_tests.cpp
  executor/perfect_hash/table_plan_tests.cu
  executor/perfect_hash/present_readback_tests.cu
  executor/perfect_hash/rank_layout_tests.cu
  executor/perfect_hash/empty_input_tests.cu)
target_include_directories(perfect_hash_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(perfect_hash_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(perfect_hash_test PUBLIC cudf::cudftestutil
                                               cudf::cudftestutil_impl)
target_link_libraries(perfect_hash_test PUBLIC gqe)
add_test(NAME perfect_hash_test COMMAND perfect_hash_test)

if(GQE_ENABLE_CUSTOMIZED_PARQUET)
  add_executable(parquet_reader_test storage/parquet_reader.cpp)
  target_include_directories(parquet_reader_test
                             PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(parquet_reader_test PUBLIC gqe)
  target_link_libraries(parquet_reader_test PUBLIC GTest::gtest_main
                                                   GTest::gmock)
  target_link_libraries(parquet_reader_test PUBLIC cudf::cudftestutil
                                                   cudf::cudftestutil_impl)
  add_test(NAME parquet_reader COMMAND parquet_reader_test)
endif()

add_executable(rewrite_rule_test optimizer/rewrite_rule_test.cpp)
target_include_directories(rewrite_rule_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(rewrite_rule_test PUBLIC gqe)
target_link_libraries(rewrite_rule_test PUBLIC GTest::gtest_main GTest::gmock)
add_test(NAME rewrite_rule_test COMMAND rewrite_rule_test)

add_executable(string_to_int_literal_test
               optimizer/string_to_int_literal_test.cpp)
target_include_directories(string_to_int_literal_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(string_to_int_literal_test PUBLIC gqe)
target_link_libraries(string_to_int_literal_test PUBLIC GTest::gtest_main
                                                        GTest::gmock)
add_test(NAME string_to_int_literal_test COMMAND string_to_int_literal_test)

add_executable(constant_folding_test optimizer/constant_folding_test.cpp)
target_include_directories(constant_folding_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(constant_folding_test PUBLIC gqe)
target_link_libraries(constant_folding_test PUBLIC GTest::gtest_main
                                                   GTest::gmock)
add_test(NAME constant_folding_test COMMAND constant_folding_test)

add_executable(extract_complex_expressions_into_project_test
               optimizer/extract_complex_expressions_into_project_test.cpp)
target_include_directories(extract_complex_expressions_into_project_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(extract_complex_expressions_into_project_test PUBLIC gqe)
target_link_libraries(extract_complex_expressions_into_project_test
                      PUBLIC GTest::gtest_main GTest::gmock)

add_test(NAME extract_complex_expressions_into_project_test
         COMMAND extract_complex_expressions_into_project_test)

add_executable(join_predicate_extraction_into_project_test
               optimizer/join_predicate_extraction_into_project_test.cpp)
target_include_directories(join_predicate_extraction_into_project_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(join_predicate_extraction_into_project_test PUBLIC gqe)
target_link_libraries(join_predicate_extraction_into_project_test
                      PUBLIC GTest::gtest_main GTest::gmock)
add_test(NAME join_predicate_extraction_into_project_test
         COMMAND join_predicate_extraction_into_project_test)

add_executable(mean_decomposition_test optimizer/mean_decomposition_test.cpp)
target_include_directories(mean_decomposition_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(mean_decomposition_test PUBLIC gqe)
target_link_libraries(mean_decomposition_test PUBLIC GTest::gtest_main
                                                     GTest::gmock)
add_test(NAME mean_decomposition_test COMMAND mean_decomposition_test)

add_executable(normalize_decimal_widths_test
               optimizer/normalize_decimal_widths_test.cpp)
target_include_directories(normalize_decimal_widths_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(normalize_decimal_widths_test PUBLIC gqe)
target_link_libraries(normalize_decimal_widths_test PUBLIC GTest::gtest_main
                                                           GTest::gmock)
add_test(NAME normalize_decimal_widths_test
         COMMAND normalize_decimal_widths_test)

add_executable(sort_fetch_to_top_k_test optimizer/sort_fetch_to_top_k_test.cpp)
target_include_directories(sort_fetch_to_top_k_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(sort_fetch_to_top_k_test PUBLIC gqe)
target_link_libraries(sort_fetch_to_top_k_test PUBLIC GTest::gtest_main
                                                      GTest::gmock)
add_test(NAME sort_fetch_to_top_k_test COMMAND sort_fetch_to_top_k_test)

add_executable(join_children_swap_test optimizer/join_children_swap_test.cpp)
target_include_directories(join_children_swap_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(join_children_swap_test PUBLIC gqe)
target_link_libraries(join_children_swap_test PUBLIC GTest::gtest_main
                                                     GTest::gmock)
target_link_libraries(join_children_swap_test PUBLIC cudf::cudftestutil
                                                     cudf::cudftestutil_impl)
add_test(NAME join_children_swap_test COMMAND join_children_swap_test)

add_executable(identity_project_elimination_test
               optimizer/identity_project_elimination_test.cpp)
target_include_directories(identity_project_elimination_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(identity_project_elimination_test PUBLIC gqe)
target_link_libraries(identity_project_elimination_test PUBLIC GTest::gtest_main
                                                               GTest::gmock)
target_link_libraries(identity_project_elimination_test
                      PUBLIC cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME identity_project_elimination_test
         COMMAND identity_project_elimination_test)

add_executable(projection_pushdown_test optimizer/projection_pushdown_test.cpp)
target_include_directories(projection_pushdown_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(projection_pushdown_test PUBLIC gqe)
target_link_libraries(projection_pushdown_test PUBLIC GTest::gtest_main
                                                      GTest::gmock)
target_link_libraries(projection_pushdown_test PUBLIC cudf::cudftestutil
                                                      cudf::cudftestutil_impl)
add_test(NAME projection_pushdown_test COMMAND projection_pushdown_test)

add_executable(fix_partial_filter_column_references_test
               optimizer/fix_partial_filter_column_references_test.cpp)
target_include_directories(fix_partial_filter_column_references_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(fix_partial_filter_column_references_test PUBLIC gqe)
target_link_libraries(fix_partial_filter_column_references_test
                      PUBLIC GTest::gtest_main GTest::gmock)
add_test(NAME fix_partial_filter_column_references_test
         COMMAND fix_partial_filter_column_references_test)

add_executable(uniqueness_prop_test optimizer/uniqueness_propagation_test.cpp)
target_include_directories(uniqueness_prop_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(uniqueness_prop_test PUBLIC gqe)
target_link_libraries(uniqueness_prop_test PUBLIC GTest::gtest_main
                                                  GTest::gmock)
add_test(NAME uniqueness_prop_test COMMAND uniqueness_prop_test)

add_executable(join_unique_keys_test optimizer/join_unique_keys_test.cpp)
target_include_directories(join_unique_keys_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(join_unique_keys_test PUBLIC gqe)
target_link_libraries(join_unique_keys_test PUBLIC GTest::gtest_main
                                                   GTest::gmock)
add_test(NAME join_unique_keys_test COMMAND join_unique_keys_test)

add_executable(join_perfect_hash_test optimizer/join_perfect_hash_test.cpp)
target_include_directories(join_perfect_hash_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(join_perfect_hash_test PUBLIC gqe)
target_link_libraries(join_perfect_hash_test PUBLIC GTest::gtest_main
                                                    GTest::gmock)
add_test(NAME join_perfect_hash_test COMMAND join_perfect_hash_test)

add_executable(plan_join_test optimizer/plan_join_test.cpp)
target_include_directories(plan_join_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(plan_join_test PUBLIC gqe)
target_link_libraries(plan_join_test PUBLIC GTest::gtest_main GTest::gmock)
add_test(NAME plan_join_test COMMAND plan_join_test)

add_executable(aggregate_perfect_hash_test
               optimizer/aggregate_perfect_hash_test.cpp)
target_include_directories(aggregate_perfect_hash_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(aggregate_perfect_hash_test PUBLIC gqe)
target_link_libraries(aggregate_perfect_hash_test PUBLIC GTest::gtest_main
                                                         GTest::gmock)
add_test(NAME aggregate_perfect_hash_test COMMAND aggregate_perfect_hash_test)

add_executable(relation_properties_test optimizer/relation_properties_test.cpp)
target_include_directories(relation_properties_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(relation_properties_test PUBLIC gqe)
target_link_libraries(relation_properties_test PUBLIC GTest::gtest_main
                                                      GTest::gmock)
add_test(NAME relation_properties_test COMMAND relation_properties_test)

add_executable(substrait_ddl_test logical/substrait_ddl_test.cpp)
target_include_directories(substrait_ddl_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(substrait_ddl_test PUBLIC gqe)
target_link_libraries(substrait_ddl_test PRIVATE proto)
target_link_libraries(substrait_ddl_test PUBLIC GTest::gtest_main GTest::gmock)
add_test(NAME substrait_ddl_test COMMAND substrait_ddl_test)

add_executable(substrait_output_names_test
               logical/substrait_output_names_test.cpp)
target_include_directories(substrait_output_names_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(substrait_output_names_test PUBLIC gqe)
target_link_libraries(substrait_output_names_test PUBLIC GTest::gtest_main
                                                         GTest::gmock)
target_compile_definitions(
  substrait_output_names_test
  PUBLIC TEST_RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resource/")
add_test(NAME substrait_output_names_test COMMAND substrait_output_names_test)

add_executable(substrait_json_test logical/substrait_json_test.cpp)
target_include_directories(substrait_json_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(substrait_json_test PUBLIC gqe)
target_link_libraries(substrait_json_test PRIVATE protobuf::libprotobuf)
target_link_libraries(substrait_json_test PUBLIC GTest::gtest_main GTest::gmock)
target_compile_definitions(
  substrait_json_test
  PUBLIC TEST_RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resource/")
add_test(NAME substrait_json_test COMMAND substrait_json_test)

add_executable(column_name_assignment_test
               optimizer/column_name_assignment_test.cpp)
target_include_directories(column_name_assignment_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(column_name_assignment_test PUBLIC gqe)
target_link_libraries(column_name_assignment_test PUBLIC GTest::gtest_main
                                                         GTest::gmock)
target_compile_definitions(
  column_name_assignment_test
  PUBLIC TEST_RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resource/")
add_test(NAME column_name_assignment_test COMMAND column_name_assignment_test)

add_executable(task_graph_test executor/task_graph_generation_test.cpp)
target_include_directories(task_graph_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(task_graph_test PUBLIC gqe)
target_link_libraries(task_graph_test PUBLIC GTest::gtest_main GTest::gmock)
add_test(NAME task_graph_test COMMAND task_graph_test)

add_executable(serdes_test executor/task_serdes_test.cpp)
target_include_directories(serdes_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(serdes_test PUBLIC gqe)
target_link_libraries(serdes_test PRIVATE proto)
target_link_libraries(serdes_test PUBLIC GTest::gmock)
target_link_libraries(serdes_test PUBLIC cudf::cudftestutil
                                         cudf::cudftestutil_impl)
add_test(NAME serdes_test COMMAND serdes_test)

add_executable(multi_process_test executor/multi_process_test.cpp)
target_include_directories(multi_process_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(multi_process_test PUBLIC gqe)
target_link_libraries(multi_process_test PRIVATE proto)
target_link_libraries(multi_process_test PUBLIC GTest::gmock)
target_link_libraries(multi_process_test PUBLIC cudf::cudftestutil
                                                cudf::cudftestutil_impl)
add_test(NAME multi_process_test COMMAND multi_process_test)

add_executable(zone_map_test storage/zone_map_test.cpp)
target_include_directories(zone_map_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(zone_map_test PUBLIC gqe)
target_link_libraries(zone_map_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(zone_map_test PUBLIC cudf::cudftestutil
                                           cudf::cudftestutil_impl)
add_test(NAME zone_map_test COMMAND zone_map_test)

add_executable(
  executor_next_test
  executor_next/utilities/signal_barrier_test.cpp
  executor_next/utilities/thread_event_test.cpp
  executor_next/utilities/object_registry_test.cpp)
target_link_libraries(executor_next_test PRIVATE gqe GTest::gtest_main
                                                 GTest::gmock)
add_test(NAME executor_next_test COMMAND executor_next_test)

add_executable(
  remote_table_cache_test
  task_manager/remote_table_cache_test.cpp
  ${CMAKE_SOURCE_DIR}/src/task_manager/remote_table_cache.cpp)
target_compile_features(remote_table_cache_test PRIVATE cxx_std_20)
target_link_libraries(remote_table_cache_test PRIVATE gqe GTest::gtest_main)
add_test(NAME remote_table_cache_test COMMAND remote_table_cache_test)

include(${PROJECT_SOURCE_DIR}/cmake/arrow_flight_sql.cmake)
add_executable(
  plan_executor_test
  task_manager/plan_executor_test.cpp
  ${CMAKE_SOURCE_DIR}/src/task_manager/plan_executor.cpp
  ${CMAKE_SOURCE_DIR}/src/task_manager/remote_table_cache.cpp)
target_include_directories(plan_executor_test BEFORE
                           PRIVATE ${ARROW_FLIGHT_SQL_INCLUDE_DIRS})
target_link_libraries(
  plan_executor_test
  PRIVATE gqe ${ARROW_FLIGHT_SQL_LIBS} GTest::gtest_main GTest::gmock
          cudf::cudftestutil cudf::cudftestutil_impl)
add_test(NAME plan_executor_test COMMAND plan_executor_test)

add_executable(
  owner_thread_test task_manager/owner_thread_test.cpp
                    ${CMAKE_SOURCE_DIR}/src/task_manager/owner_thread.cpp)
target_link_libraries(owner_thread_test PRIVATE gqe GTest::gtest_main)
add_test(NAME owner_thread_test COMMAND owner_thread_test)

add_executable(
  task_manager_service_test
  task_manager/service_test.cpp
  ${CMAKE_SOURCE_DIR}/src/task_manager/owner_thread.cpp
  ${CMAKE_SOURCE_DIR}/src/task_manager/service.cpp
  ${CMAKE_SOURCE_DIR}/src/node_manager/fault_boundary.cpp)
target_include_directories(task_manager_service_test BEFORE
                           PRIVATE ${ARROW_FLIGHT_SQL_INCLUDE_DIRS})
target_link_libraries(task_manager_service_test
                      PRIVATE gqe ${ARROW_FLIGHT_SQL_LIBS} GTest::gtest_main)
add_test(NAME task_manager_service_test COMMAND task_manager_service_test)

add_executable(
  fault_boundary_test node_manager/fault_boundary_test.cpp
                      ${CMAKE_SOURCE_DIR}/src/node_manager/fault_boundary.cpp)
target_include_directories(fault_boundary_test BEFORE
                           PRIVATE ${ARROW_FLIGHT_SQL_INCLUDE_DIRS})
target_link_libraries(fault_boundary_test PRIVATE gqe ${ARROW_FLIGHT_SQL_LIBS}
                                                  GTest::gtest_main)
add_test(NAME fault_boundary_test COMMAND fault_boundary_test)

add_executable(
  process_group_test node_manager/process_group_test.cpp
                     ${CMAKE_SOURCE_DIR}/src/node_manager/spawn.cpp)
target_include_directories(process_group_test BEFORE
                           PRIVATE ${ARROW_FLIGHT_SQL_INCLUDE_DIRS})
target_link_libraries(process_group_test PRIVATE gqe ${ARROW_FLIGHT_SQL_LIBS}
                                                 GTest::gtest_main)
add_test(NAME process_group_test COMMAND process_group_test)

add_executable(
  task_manager_group_test
  node_manager/task_manager_group_test.cpp
  ${CMAKE_SOURCE_DIR}/src/node_manager/task_manager_group.cpp
  ${CMAKE_SOURCE_DIR}/src/node_manager/spawn.cpp)
target_include_directories(task_manager_group_test BEFORE
                           PRIVATE ${ARROW_FLIGHT_SQL_INCLUDE_DIRS})
target_link_libraries(task_manager_group_test
                      PRIVATE gqe ${ARROW_FLIGHT_SQL_LIBS} GTest::gtest_main)
add_test(NAME task_manager_group_test COMMAND task_manager_group_test)

add_executable(
  nvshmem_bootstrap_test
  node_manager/nvshmem_bootstrap_test.cpp
  ${CMAKE_SOURCE_DIR}/src/node_manager/nvshmem_bootstrap.cpp)
target_include_directories(nvshmem_bootstrap_test BEFORE
                           PRIVATE ${ARROW_FLIGHT_SQL_INCLUDE_DIRS})
target_link_libraries(nvshmem_bootstrap_test
                      PRIVATE gqe ${ARROW_FLIGHT_SQL_LIBS} GTest::gtest_main)
add_test(NAME nvshmem_bootstrap_test COMMAND nvshmem_bootstrap_test)

# stats_tables.cpp lives in the gqe_node_manager executable; compile it directly
# into the test (as process_group_test does with spawn.cpp) so the projection
# helpers can be exercised. Arrow Flight SQL include dirs must come first
# (BEFORE) so stats_tables.cpp compiles against the same Arrow headers it is
# linked against — matching gqe_node_manager — otherwise Arrow symbols (e.g.
# RecordBatch::Make) fail to resolve against the wrong Arrow copy.
add_executable(
  stats_tables_test node_manager/stats_tables_test.cpp
                    ${CMAKE_SOURCE_DIR}/src/node_manager/stats_tables.cpp)
target_include_directories(stats_tables_test BEFORE
                           PRIVATE ${ARROW_FLIGHT_SQL_INCLUDE_DIRS})
target_include_directories(stats_tables_test
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(stats_tables_test PUBLIC gqe ${ARROW_FLIGHT_SQL_LIBS})
target_link_libraries(stats_tables_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(stats_tables_test PUBLIC cudf::cudftestutil
                                               cudf::cudftestutil_impl)
add_test(NAME stats_tables_test COMMAND stats_tables_test)

# context.cpp lives in the gqe_node_manager executable; compile it directly into
# the test, as stats_tables_test does with stats_tables.cpp.
add_executable(context_test node_manager/context_test.cpp
                            ${CMAKE_SOURCE_DIR}/src/node_manager/context.cpp)
target_link_libraries(context_test PRIVATE gqe GTest::gtest_main GTest::gmock)
add_test(NAME context_test COMMAND context_test)

# The root find_package(Boost) requests no components, so the compiled `process`
# library target (Boost::process) is never defined; request it explicitly here.
find_package(Boost REQUIRED CONFIG COMPONENTS process)

set(FLIGHT_SQL_COMMON_SOURCES integration/flight_sql_test_harness.cpp
                              integration/flight_sql_test_utils.cpp)
set(FLIGHT_SQL_COMPILE_DEFS "GQE_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\""
                            "GQE_PROJECT_BINARY_DIR=\"${CMAKE_BINARY_DIR}\"")

add_executable(flight_sql_ddl_test integration/flight_sql_ddl_test.cpp
                                   ${FLIGHT_SQL_COMMON_SOURCES})
target_link_libraries(flight_sql_ddl_test PRIVATE ${ARROW_FLIGHT_SQL_LIBS}
                                                  GTest::gtest Boost::process)
target_compile_definitions(flight_sql_ddl_test
                           PRIVATE ${FLIGHT_SQL_COMPILE_DEFS})
add_test(NAME flight_sql_ddl_test COMMAND flight_sql_ddl_test)

add_executable(flight_sql_stats_test integration/flight_sql_stats_test.cpp
                                     ${FLIGHT_SQL_COMMON_SOURCES})
target_link_libraries(flight_sql_stats_test PRIVATE ${ARROW_FLIGHT_SQL_LIBS}
                                                    GTest::gtest Boost::process)
target_compile_definitions(flight_sql_stats_test
                           PRIVATE ${FLIGHT_SQL_COMPILE_DEFS})
add_test(NAME flight_sql_stats_test COMMAND flight_sql_stats_test)

add_executable(flight_sql_tpch_test integration/flight_sql_tpch_test.cpp
                                    ${FLIGHT_SQL_COMMON_SOURCES})
target_link_libraries(flight_sql_tpch_test PRIVATE ${ARROW_FLIGHT_SQL_LIBS}
                                                   GTest::gtest Boost::process)
target_compile_definitions(flight_sql_tpch_test
                           PRIVATE ${FLIGHT_SQL_COMPILE_DEFS})
add_test(NAME flight_sql_tpch_test COMMAND flight_sql_tpch_test)

add_executable(
  flight_sql_physical_plan_tpch_test
  integration/flight_sql_physical_plan_tpch_test.cpp
  ${FLIGHT_SQL_COMMON_SOURCES})
target_link_libraries(
  flight_sql_physical_plan_tpch_test PRIVATE ${ARROW_FLIGHT_SQL_LIBS}
                                             GTest::gtest proto Boost::process)
target_compile_definitions(flight_sql_physical_plan_tpch_test
                           PRIVATE ${FLIGHT_SQL_COMPILE_DEFS})
add_test(NAME flight_sql_physical_plan_tpch_test
         COMMAND flight_sql_physical_plan_tpch_test)

add_executable(
  flight_sql_lifecycle_test integration/flight_sql_lifecycle_test.cpp
                            integration/flight_sql_test_utils.cpp)
# gqe provides gqe::shared_segment, used to poison the node manager's segment.
target_link_libraries(flight_sql_lifecycle_test PRIVATE gqe GTest::gtest_main
                                                        Boost::process)
target_compile_definitions(flight_sql_lifecycle_test
                           PRIVATE ${FLIGHT_SQL_COMPILE_DEFS})
add_test(NAME flight_sql_lifecycle_test COMMAND flight_sql_lifecycle_test)
add_dependencies(flight_sql_lifecycle_test gqe_node_manager gqe_task_manager)

add_executable(
  task_manager_process_test task_manager/task_manager_process_test.cpp
                            integration/flight_sql_test_utils.cpp)
target_link_libraries(task_manager_process_test PRIVATE gqe GTest::gtest_main
                                                        Boost::process)
target_compile_definitions(task_manager_process_test
                           PRIVATE ${FLIGHT_SQL_COMPILE_DEFS})
add_test(NAME task_manager_process_test COMMAND task_manager_process_test)
add_dependencies(task_manager_process_test gqe_task_manager)
