/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gqe/rpc/serialization/optimization_parameters.hpp>

#include <gtest/gtest.h>

namespace gqe::rpc {
namespace {

// With Boost PFR all fields are serialized uniformly, so we only need one
// sample field per supported type to verify the type dispatch is correct.

TEST(OptimizationParametersSerializationTest, RoundTripBoolField)
{
  auto p                    = gqe::make_optimization_parameters(true);
  p.join_use_hash_map_cache = true;  // default is false
  auto restored =
    deserialize_optimization_parameters(serialize_optimization_parameters(p)).ValueOrDie();
  EXPECT_EQ(restored.join_use_hash_map_cache, p.join_use_hash_map_cache);
}

TEST(OptimizationParametersSerializationTest, RoundTripIntegralField)
{
  auto p               = gqe::make_optimization_parameters(true);
  p.max_num_partitions = 16;  // default is 8
  auto restored =
    deserialize_optimization_parameters(serialize_optimization_parameters(p)).ValueOrDie();
  EXPECT_EQ(restored.max_num_partitions, p.max_num_partitions);
}

TEST(OptimizationParametersSerializationTest, RoundTripDoubleField)
{
  auto p                                        = gqe::make_optimization_parameters(true);
  p.in_memory_table_compression_ratio_threshold = 0.5;  // default is 1.0
  auto restored =
    deserialize_optimization_parameters(serialize_optimization_parameters(p)).ValueOrDie();
  EXPECT_DOUBLE_EQ(restored.in_memory_table_compression_ratio_threshold,
                   p.in_memory_table_compression_ratio_threshold);
}

TEST(OptimizationParametersSerializationTest, RoundTripStringField)
{
  auto p      = gqe::make_optimization_parameters(true);
  p.log_level = "debug";  // default is "info"
  auto restored =
    deserialize_optimization_parameters(serialize_optimization_parameters(p)).ValueOrDie();
  EXPECT_EQ(restored.log_level, p.log_level);
}

TEST(OptimizationParametersSerializationTest, RoundTripEnumField)
{
  auto p      = gqe::make_optimization_parameters(true);
  p.io_engine = gqe::io_engine_type::psync;  // default is io_uring
  auto restored =
    deserialize_optimization_parameters(serialize_optimization_parameters(p)).ValueOrDie();
  EXPECT_EQ(restored.io_engine, p.io_engine);
}

TEST(OptimizationParametersSerializationTest, RoundTripOptionalNullopt)
{
  auto p             = gqe::make_optimization_parameters(true);
  p.max_query_memory = std::nullopt;
  auto restored =
    deserialize_optimization_parameters(serialize_optimization_parameters(p)).ValueOrDie();
  EXPECT_FALSE(restored.max_query_memory.has_value());
}

TEST(OptimizationParametersSerializationTest, RoundTripOptionalSet)
{
  auto p             = gqe::make_optimization_parameters(true);
  p.max_query_memory = 4UL * 1024 * 1024 * 1024;
  auto restored =
    deserialize_optimization_parameters(serialize_optimization_parameters(p)).ValueOrDie();
  ASSERT_TRUE(restored.max_query_memory.has_value());
  EXPECT_EQ(*restored.max_query_memory, *p.max_query_memory);
}

}  // namespace
}  // namespace gqe::rpc
