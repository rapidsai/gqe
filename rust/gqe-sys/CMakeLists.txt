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

set(CXX_GQE_BINDINGS_DIR "" CACHE PATH "Path to the bindings directory")
message(STATUS "Using bindings at ${CXX_GQE_BINDINGS_DIR}")

add_library(gqe-cxx STATIC EXCLUDE_FROM_ALL)

target_include_directories(gqe-cxx
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CXX_GQE_BINDINGS_DIR}/crate
        ${CXX_GQE_BINDINGS_DIR}/include
        ${CXX_GQE_BINDINGS_DIR}/sources
)

target_sources(gqe-cxx
    PRIVATE
        cpp/api.cpp
        cpp/cxx_bindings.cpp
        cpp/executor.cpp
        cpp/logical.cpp
        cpp/physical.cpp
        cpp/storage.cpp
)

target_link_libraries(gqe-cxx PUBLIC gqe)
