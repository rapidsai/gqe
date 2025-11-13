# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_executable(tpc tpc.cpp)
target_include_directories(tpc PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(tpc PRIVATE nvtx3-cpp)
target_link_libraries(tpc PUBLIC gqe)

add_executable(q3 hardcoded/q3.cpp)
target_include_directories(q3 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(q3 PUBLIC gqe)

add_executable(q3_udr hardcoded/q3_udr.cu)
target_include_directories(q3_udr PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(q3_udr PUBLIC gqe)
target_link_libraries(q3_udr PUBLIC cuco::cuco)

add_executable(q6 hardcoded/q6.cpp)
target_include_directories(q6 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(q6 PUBLIC gqe)

add_executable(q7 hardcoded/q7.cpp)
target_include_directories(q7 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(q7 PUBLIC gqe)

add_executable(q22 hardcoded/q22.cpp)
target_include_directories(q22 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(q22 PUBLIC gqe)

add_executable(q38 hardcoded/q38.cpp)
target_include_directories(q38 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(q38 PUBLIC gqe)

add_executable(q43 hardcoded/q43.cpp)
target_include_directories(q43 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(q43 PUBLIC gqe)

add_executable(q48 hardcoded/q48.cpp)
target_include_directories(q48 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(q48 PUBLIC gqe)

add_executable(load_tpcds load_tpcds.cpp)
target_include_directories(load_tpcds PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(load_tpcds PUBLIC gqe)
