# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

add_executable(join_task_test executor/join_task_test.cpp)
target_include_directories(join_task_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(join_task_test PUBLIC gqe)
target_link_libraries(join_task_test PUBLIC GTest::gtest_main GTest::gmock)
target_link_libraries(join_task_test PUBLIC cudf::cudftestutil)
add_test(NAME join_task COMMAND join_task_test)

add_library(logical_test_common logical/common.cpp)
set_property(TARGET logical_test_common PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
target_include_directories(logical_test_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(logical_test_common PUBLIC cudf::cudf)

add_executable(run_substrait_consumer logical/run_substrait_consumer.cpp)
target_link_libraries(run_substrait_consumer PUBLIC gqe)
target_link_libraries(run_substrait_consumer PRIVATE logical_test_common)