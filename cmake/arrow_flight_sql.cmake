# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Build Arrow Flight SQL from source via CPM.

include_guard(GLOBAL)
include(${rapids-cmake-dir}/cpm/find.cmake)

set(ARROW_VERSION "18.1.0")
string(APPEND CMAKE_CXX_FLAGS " -Wno-error=deprecated-declarations"
       " -Wno-error=maybe-uninitialized" " -Wno-error=suggest-override")
cpmaddpackage(
  NAME
  ArrowFlightSql
  GITHUB_REPOSITORY
  apache/arrow
  GIT_TAG
  apache-arrow-${ARROW_VERSION}
  GIT_SHALLOW
  TRUE
  SOURCE_SUBDIR
  cpp
  OPTIONS
  "ARROW_FLIGHT ON"
  "ARROW_FLIGHT_SQL ON"
  "ARROW_COMPUTE ON"
  "ARROW_CSV OFF"
  "ARROW_DATASET OFF"
  "ARROW_FILESYSTEM OFF"
  "ARROW_JSON OFF"
  "ARROW_PARQUET ON"
  "ARROW_IPC ON"
  "ARROW_BUILD_SHARED ON"
  "ARROW_BUILD_STATIC OFF"
  "ARROW_BUILD_TESTS OFF"
  "ARROW_BUILD_BENCHMARKS OFF"
  "ARROW_BUILD_EXAMPLES OFF"
  "ARROW_BUILD_UTILITIES OFF"
  "ARROW_PROTOBUF_USE_SHARED ON"
  "ARROW_DEPENDENCY_SOURCE AUTO"
  "ARROW_SIMD_LEVEL NONE"
  "ARROW_WITH_RE2 OFF"
  "ARROW_WITH_UTF8PROC OFF"
  "ARROW_ENABLE_THREADING ON")

set(ARROW_FLIGHT_SQL_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include ${ArrowFlightSql_SOURCE_DIR}/cpp/src
    ${ArrowFlightSql_BINARY_DIR}/src
    CACHE INTERNAL "Arrow Flight SQL include directories")

set(ARROW_FLIGHT_SQL_LIBS
    gqe arrow_flight_sql_shared arrow_flight_shared arrow_shared
    protobuf::libprotobuf gRPC::grpc++
    CACHE INTERNAL "Arrow Flight SQL link libraries")
