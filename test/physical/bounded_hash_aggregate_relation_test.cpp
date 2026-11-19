/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

// Unit tests for the bounded_hash_aggregate_relation construction preconditions and its
// output-schema derivation.

#include <gqe/physical/bounded_hash_aggregate.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/types.hpp>

#include <gtest/gtest.h>

#include <cudf/aggregation.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

namespace physical = gqe::physical;

/**
 * @brief An input relation with columns [INT32, INT64, FLOAT64, DECIMAL64(scale -2)].
 */
std::shared_ptr<physical::read_relation> make_read()
{
  return std::make_shared<physical::read_relation>(
    std::vector<std::shared_ptr<physical::relation>>{},
    std::vector<std::string>{"col0", "col1", "col2", "col3"},
    "t",
    nullptr,
    std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT32},
                                 cudf::data_type{cudf::type_id::INT64},
                                 cudf::data_type{cudf::type_id::FLOAT64},
                                 cudf::data_type{cudf::type_id::DECIMAL64, -2}});
}

std::shared_ptr<physical::bounded_hash_aggregate_relation> make_relation(
  std::vector<cudf::size_type> key_indices,
  std::vector<physical::bounded_hash_aggregate_relation::value_type> values,
  int64_t group_count_bound,
  std::optional<double> desired_load_factor = std::nullopt)
{
  return std::make_shared<physical::bounded_hash_aggregate_relation>(
    make_read(), std::move(key_indices), std::move(values), group_count_bound, desired_load_factor);
}

}  // namespace

TEST(BoundedHashAggregateRelationTest, NullInputThrows)
{
  EXPECT_THROW(std::ignore = std::make_shared<physical::bounded_hash_aggregate_relation>(
                 nullptr,
                 std::vector<cudf::size_type>{0},
                 std::vector<physical::bounded_hash_aggregate_relation::value_type>{
                   {cudf::aggregation::SUM, 1, gqe::nullability::nullable}},
                 /*group_count_bound=*/42),
               std::logic_error);
}

TEST(BoundedHashAggregateRelationTest, NonPositiveBoundThrows)
{
  EXPECT_THROW(
    std::ignore = make_relation({0}, {{cudf::aggregation::SUM, 1, gqe::nullability::nullable}}, 0),
    std::logic_error);
  EXPECT_THROW(
    std::ignore = make_relation({0}, {{cudf::aggregation::SUM, 1, gqe::nullability::nullable}}, -1),
    std::logic_error);
}

TEST(BoundedHashAggregateRelationTest, OutOfRangeLoadFactorThrows)
{
  EXPECT_THROW(std::ignore = make_relation(
                 {0}, {{cudf::aggregation::SUM, 1, gqe::nullability::nullable}}, 42, 1.5),
               std::logic_error);
  EXPECT_THROW(std::ignore = make_relation(
                 {0}, {{cudf::aggregation::SUM, 1, gqe::nullability::nullable}}, 42, 0.0),
               std::logic_error);
}

TEST(BoundedHashAggregateRelationTest, KeyIndexOutsideInputSchemaThrows)
{
  EXPECT_THROW(
    std::ignore = make_relation({4}, {{cudf::aggregation::SUM, 1, gqe::nullability::nullable}}, 42),
    std::logic_error);
  EXPECT_THROW(std::ignore =
                 make_relation({-1}, {{cudf::aggregation::SUM, 1, gqe::nullability::nullable}}, 42),
               std::logic_error);
}

TEST(BoundedHashAggregateRelationTest, ValueIndexOutsideInputSchemaThrows)
{
  EXPECT_THROW(
    std::ignore = make_relation({0}, {{cudf::aggregation::SUM, 4, gqe::nullability::nullable}}, 42),
    std::logic_error);
}

/**
 * COUNT_ALL reads no column, so its value index must be nullopt. Every other kind reads one
 * column, so its value index must be engaged.
 */
TEST(BoundedHashAggregateRelationTest, ValueIndexEngagementMustMatchKind)
{
  EXPECT_THROW(std::ignore = make_relation(
                 {0}, {{cudf::aggregation::COUNT_ALL, 1, gqe::nullability::nullable}}, 42),
               std::logic_error);
  EXPECT_THROW(std::ignore =
                 make_relation({0}, {{cudf::aggregation::SUM, {}, gqe::nullability::nullable}}, 42),
               std::logic_error);
}

/**
 * Integral SUM widens to INT64, floating-point SUM keeps the source type, counts target
 * INT64 (the operator aggregates unbounded pipelined rows), and MIN keeps the source type.
 */
TEST(BoundedHashAggregateRelationTest, OutputSchemaIsKeysThenValues)
{
  auto const relation =
    make_relation({1, 0},
                  {{cudf::aggregation::SUM, 0, gqe::nullability::nullable},
                   {cudf::aggregation::SUM, 2, gqe::nullability::nullable},
                   {cudf::aggregation::COUNT_ALL, {}, gqe::nullability::nullable},
                   {cudf::aggregation::MIN, 0, gqe::nullability::nullable}},
                  42);
  auto const expected = std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT64},
                                                     cudf::data_type{cudf::type_id::INT32},
                                                     cudf::data_type{cudf::type_id::INT64},
                                                     cudf::data_type{cudf::type_id::FLOAT64},
                                                     cudf::data_type{cudf::type_id::INT64},
                                                     cudf::data_type{cudf::type_id::INT32}};
  EXPECT_EQ(relation->output_data_types(), expected);
}

/**
 * A decimal SUM accumulates into DECIMAL128 at the input's scale, following the engine's
 * Spark-style decimal widening. The operator rejects narrower decimal SUM targets.
 */
TEST(BoundedHashAggregateRelationTest, DecimalSumWidensToDecimal128)
{
  auto const relation =
    make_relation({1}, {{cudf::aggregation::SUM, 3, gqe::nullability::nullable}}, 42);
  auto const expected = std::vector<cudf::data_type>{
    cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::DECIMAL128, -2}};
  EXPECT_EQ(relation->output_data_types(), expected);
}

/**
 * A NOT NULL declaration names the value column, which COUNT_ALL does not have. Every other
 * kind accepts it, and the relation carries it unchanged.
 */
TEST(BoundedHashAggregateRelationTest, NotNullDeclarationNeedsAValueColumn)
{
  EXPECT_THROW(std::ignore = make_relation(
                 {0}, {{cudf::aggregation::COUNT_ALL, {}, gqe::nullability::not_null}}, 42),
               std::logic_error);

  auto const relation =
    make_relation({0}, {{cudf::aggregation::SUM, 1, gqe::nullability::not_null}}, 42);
  ASSERT_EQ(relation->values().size(), 1);
  EXPECT_EQ(relation->values()[0].value_nullability, gqe::nullability::not_null);
}

TEST(BoundedHashAggregateRelationTest, CarriesNoSubqueries)
{
  auto const relation =
    make_relation({0}, {{cudf::aggregation::SUM, 1, gqe::nullability::nullable}}, 42);
  EXPECT_TRUE(relation->subqueries_unsafe().empty());
}
