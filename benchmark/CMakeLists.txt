# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

add_executable(tpcds tpcds.cpp)
target_include_directories(tpcds PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(tpcds PUBLIC gqe)

add_executable(q3 hardcoded/q3.cpp)
target_include_directories(q3 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(q3 PUBLIC gqe)

add_executable(q38 hardcoded/q38.cpp)
target_include_directories(q38 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(q38 PUBLIC gqe)
