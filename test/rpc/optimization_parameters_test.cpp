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

#include "../utility.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <type_traits>
#include <variant>

namespace gqe::rpc {
namespace {

static_assert(std::is_aggregate_v<gqe::detail::optimization_parameters_data>);
static_assert(!std::is_default_constructible_v<gqe::optimization_parameters>);

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
  auto p           = gqe::make_optimization_parameters(true);
  p.num_partitions = 16;  // default is 8
  auto restored =
    deserialize_optimization_parameters(serialize_optimization_parameters(p)).ValueOrDie();
  EXPECT_EQ(restored.num_partitions, p.num_partitions);
}

TEST(OptimizationParametersSerializationTest, RoundTripUnsetMaxTaskManagerMemory)
{
  auto const p = gqe::make_optimization_parameters(true);
  auto restored =
    deserialize_optimization_parameters(serialize_optimization_parameters(p)).ValueOrDie();

  EXPECT_FALSE(restored.max_task_manager_memory.has_value());
}

TEST(OptimizationParametersSerializationTest, ClearsMaxTaskManagerMemorySessionOption)
{
  auto const defaults        = gqe::make_optimization_parameters(true);
  auto const session_options = optimization_parameters_to_session_options(defaults);
  auto p                     = defaults;
  p.max_task_manager_memory  = 1UL << 30;

  auto const& unset_value = session_options.at("max_task_manager_memory");
  ASSERT_TRUE(std::holds_alternative<std::monostate>(unset_value));
  auto const status = apply_session_option(p, "max_task_manager_memory", unset_value);

  EXPECT_TRUE(status.ok());
  EXPECT_FALSE(p.max_task_manager_memory.has_value());
}

TEST(OptimizationParametersSerializationTest, RoundTripDoubleField)
{
  auto p                        = gqe::make_optimization_parameters(true);
  p.compression_ratio_threshold = 0.5;  // default is 1.0
  auto restored =
    deserialize_optimization_parameters(serialize_optimization_parameters(p)).ValueOrDie();
  EXPECT_DOUBLE_EQ(restored.compression_ratio_threshold, p.compression_ratio_threshold);
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

TEST(OptimizationParametersSerializationTest, ExecutionBackendDefaultsToLegacyTaskGraph)
{
  auto const p = gqe::make_optimization_parameters(true);
  EXPECT_EQ(p.execution_backend, gqe::execution_backend_type::legacy_task_graph);
}

TEST(OptimizationParametersSerializationTest, ExecutionBackendEnvironmentOverride)
{
  gqe_test::scoped_environment_variable execution_backend("GQE_EXECUTION_BACKEND",
                                                          "qep_task_graph");

  auto const p = gqe::make_optimization_parameters(false);
  EXPECT_EQ(p.execution_backend, gqe::execution_backend_type::qep_task_graph);
}

TEST(OptimizationParametersSerializationTest, SourceReaderEnvironmentAndRoundTrip)
{
  gqe_test::scoped_environment_variable backend("GQE_EXECUTION_BACKEND", "legacy_task_graph");
  for (auto const& [name, reader] : {std::pair{"oneshot", gqe::source_reader_type::oneshot},
                                     std::pair{"chunked", gqe::source_reader_type::chunked},
                                     std::pair{"deferred", gqe::source_reader_type::deferred}}) {
    gqe_test::scoped_environment_variable setting("GQE_SOURCE_READER", name);
    auto const p = gqe::make_optimization_parameters(false);
    EXPECT_EQ(p.source_reader, reader);
    auto restored = deserialize_optimization_parameters(serialize_optimization_parameters(p));
    ASSERT_TRUE(restored.ok()) << restored.status();
    EXPECT_EQ(restored->source_reader, reader);
  }
  gqe_test::scoped_environment_variable setting("GQE_SOURCE_READER", "automatic");
  EXPECT_THROW(std::ignore = gqe::make_optimization_parameters(false), std::invalid_argument);
}

TEST(OptimizationParametersSerializationTest, DeferredReaderIsRejectedOnQepTaskGraph)
{
  auto p              = gqe::make_optimization_parameters(true);
  p.source_reader     = gqe::source_reader_type::deferred;
  p.execution_backend = gqe::execution_backend_type::qep_task_graph;
  EXPECT_THROW(gqe::validate_optimization_parameters(p), std::invalid_argument);
}

TEST(OptimizationParametersSerializationTest, MorselMaxRowsEnvironmentOverride)
{
  gqe_test::scoped_environment_variable max_rows("GQE_MORSEL_MAX_ROWS", "123");

  auto const p = gqe::make_optimization_parameters(false);
  ASSERT_TRUE(p.morsel_max_rows.has_value());
  EXPECT_EQ(*p.morsel_max_rows, 123);
}

TEST(OptimizationParametersSerializationTest, RejectsNonPositiveMorselMaxRowsEnvironmentOverride)
{
  for (auto const* value : {"-1", "0"}) {
    gqe_test::scoped_environment_variable max_rows("GQE_MORSEL_MAX_ROWS", value);

    EXPECT_THROW(std::ignore = gqe::make_optimization_parameters(false), std::invalid_argument);
  }
}

TEST(OptimizationParametersSerializationTest, RejectsNonPositiveMorselMaxRowsSessionOption)
{
  auto p = gqe::make_optimization_parameters(true);

  auto const status = apply_session_option(p, "morsel_max_rows", int64_t{0});

  EXPECT_TRUE(status.IsInvalid());
  EXPECT_FALSE(p.morsel_max_rows.has_value());
}

TEST(OptimizationParametersSerializationTest, RejectsOutOfRangeMorselMaxRowsSessionOption)
{
  auto p                  = gqe::make_optimization_parameters(true);
  auto const out_of_range = static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()) + 1;

  auto const status = apply_session_option(p, "morsel_max_rows", out_of_range);

  EXPECT_TRUE(status.IsInvalid());
  EXPECT_FALSE(p.morsel_max_rows.has_value());
}

TEST(OptimizationParametersSerializationTest, JoinMaterializationThresholdConfiguration)
{
  auto defaults = gqe::make_optimization_parameters(true);
  EXPECT_DOUBLE_EQ(defaults.join_materialization_threshold, 0.1);
  gqe_test::scoped_environment_variable threshold("GQE_JOIN_MATERIALIZATION_THRESHOLD", "0.25");
  auto params = gqe::make_optimization_parameters(false);
  EXPECT_DOUBLE_EQ(params.join_materialization_threshold, 0.25);
  auto restored =
    deserialize_optimization_parameters(serialize_optimization_parameters(params)).ValueOrDie();
  EXPECT_DOUBLE_EQ(restored.join_materialization_threshold, 0.25);
  EXPECT_TRUE(apply_session_option(params, "join_materialization_threshold", 0.0).ok());
  EXPECT_DOUBLE_EQ(params.join_materialization_threshold, 0.0);
  for (double invalid : {-0.1,
                         1.1,
                         std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_TRUE(
      apply_session_option(params, "join_materialization_threshold", invalid).IsInvalid());
    EXPECT_DOUBLE_EQ(params.join_materialization_threshold, 0.0);
  }
}

TEST(OptimizationParametersSerializationTest, RejectsInvalidJoinMaterializationEnvironment)
{
  for (auto const* value : {"-0.1", "1.1", "nan", "inf"}) {
    gqe_test::scoped_environment_variable threshold("GQE_JOIN_MATERIALIZATION_THRESHOLD", value);
    EXPECT_THROW(std::ignore = gqe::make_optimization_parameters(false), std::invalid_argument);
  }
}

TEST(OptimizationParametersSerializationTest, RoundTripExecutionBackend)
{
  auto p              = gqe::make_optimization_parameters(true);
  p.execution_backend = gqe::execution_backend_type::qep_task_graph;
  auto restored =
    deserialize_optimization_parameters(serialize_optimization_parameters(p)).ValueOrDie();
  EXPECT_EQ(restored.execution_backend, p.execution_backend);
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
