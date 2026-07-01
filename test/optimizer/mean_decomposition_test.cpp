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

#include <gqe/catalog.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/relation.hpp>
#include <gqe/optimizer/logical_optimization.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/types.hpp>

#include <gtest/gtest.h>

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <utility>
#include <vector>

using relation_t = gqe::logical::relation::relation_type;
using expr_t     = gqe::expression::expression_type;
using rule_t     = gqe::optimizer::logical_optimization_rule_type;
using agg_kind   = cudf::aggregation::Kind;

/**
 * @brief Test fixture for the `mean_decomposition` optimizer rule.
 *
 * Configures a single-rule optimizer pipeline over a 4-column INT32 table `t` ({a, b, c, d}).
 */
class MeanDecompositionTest : public ::testing::Test {
 protected:
  static constexpr const char* kTable = "t";

  MeanDecompositionTest()
  {
    task_manager_ctx_ = std::make_unique<gqe::task_manager_context>();
    catalog_          = std::make_unique<gqe::catalog>(task_manager_ctx_.get());

    catalog_->register_table(kTable,
                             {{"a", cudf::data_type(cudf::type_id::INT32)},
                              {"b", cudf::data_type(cudf::type_id::INT32)},
                              {"c", cudf::data_type(cudf::type_id::INT32)},
                              {"d", cudf::data_type(cudf::type_id::INT32)}},
                             gqe::storage_kind::system_memory{},
                             gqe::partitioning_schema_kind::none{});

    config_    = gqe::optimizer::optimization_configuration({rule_t::mean_decomposition}, {});
    optimizer_ = std::make_unique<gqe::optimizer::logical_optimizer>(&config_, catalog_.get());
  }

  /// @brief Build a read of the INT32 columns {a, b, c, d} over table `t`.
  std::shared_ptr<gqe::logical::read_relation> make_read()
  {
    std::vector<std::string> cols = {"a", "b", "c", "d"};
    std::vector<cudf::data_type> tys(cols.size(), cudf::data_type(cudf::type_id::INT32));
    std::vector<std::shared_ptr<gqe::logical::relation>> sq;
    return std::make_shared<gqe::logical::read_relation>(sq, cols, tys, kTable, nullptr);
  }

  /// @brief Build a `(kind, column_reference(idx))` measure.
  static std::pair<agg_kind, std::unique_ptr<gqe::expression>> measure(agg_kind kind,
                                                                       cudf::size_type idx)
  {
    return {kind, std::make_unique<gqe::column_reference_expression>(idx)};
  }

  /// @brief Build an aggregate over `make_read()` grouping by `keys` columns with `measures`.
  std::shared_ptr<gqe::logical::aggregate_relation> make_aggregate(
    std::vector<cudf::size_type> const& keys,
    std::vector<std::pair<agg_kind, std::unique_ptr<gqe::expression>>> measures)
  {
    std::vector<std::shared_ptr<gqe::logical::relation>> sq;
    std::vector<std::unique_ptr<gqe::expression>> key_exprs;
    key_exprs.reserve(keys.size());
    for (auto k : keys)
      key_exprs.push_back(std::make_unique<gqe::column_reference_expression>(k));
    return std::make_shared<gqe::logical::aggregate_relation>(
      make_read(), sq, std::move(key_exprs), std::move(measures));
  }

  /// @brief Column index of a project's `i`-th output, asserting it is a `column_reference`.
  static cudf::size_type proj_col_idx(gqe::logical::project_relation const* proj, std::size_t i)
  {
    auto const* expr =
      dynamic_cast<gqe::column_reference_expression const*>(proj->output_expressions_unsafe()[i]);
    EXPECT_NE(expr, nullptr) << "project output[" << i << "] is not a column_reference";
    return expr ? expr->column_idx() : -1;
  }

  /// @brief Assert that `expr` is exactly `column_reference(lhs) / column_reference(rhs)`.
  static void expect_divide(gqe::expression const* expr, cudf::size_type lhs, cudf::size_type rhs)
  {
    ASSERT_EQ(expr->type(), expr_t::binary_op) << "expected a binary op";
    auto const* bin = dynamic_cast<gqe::binary_op_expression const*>(expr);
    ASSERT_NE(bin, nullptr);
    EXPECT_EQ(bin->binary_operator(), cudf::binary_operator::TRUE_DIV);
    auto children = bin->children();
    ASSERT_EQ(children.size(), 2u);
    auto const* l = dynamic_cast<gqe::column_reference_expression const*>(children[0]);
    auto const* r = dynamic_cast<gqe::column_reference_expression const*>(children[1]);
    ASSERT_NE(l, nullptr);
    ASSERT_NE(r, nullptr);
    EXPECT_EQ(l->column_idx(), lhs);
    EXPECT_EQ(r->column_idx(), rhs);
  }

  /// @brief Cast a relation to `aggregate_relation`, asserting it is one.
  static gqe::logical::aggregate_relation const* as_aggregate(gqe::logical::relation const* rel)
  {
    auto const* a = dynamic_cast<gqe::logical::aggregate_relation const*>(rel);
    EXPECT_NE(a, nullptr) << "relation is not an aggregate";
    return a;
  }

  std::unique_ptr<gqe::task_manager_context> task_manager_ctx_;
  std::unique_ptr<gqe::catalog> catalog_;
  gqe::optimizer::optimization_configuration config_;
  std::unique_ptr<gqe::optimizer::logical_optimizer> optimizer_;
};

/**
 * @brief group by a, mean(b) -> project([col(0), col(1) / col(2)], aggregate([SUM(b),
 * COUNT_VALID(b)])). The output schema (FLOAT64 mean) is preserved.
 */
TEST_F(MeanDecompositionTest, SingleMeanWithKey)
{
  auto agg = make_aggregate({0}, [] {
    std::vector<std::pair<agg_kind, std::unique_ptr<gqe::expression>>> m;
    m.push_back(measure(cudf::aggregation::MEAN, 1));
    return m;
  }());

  auto expected_types = agg->data_types();
  auto optimized      = optimizer_->optimize(agg);

  ASSERT_EQ(optimized->type(), relation_t::project);
  auto* proj = static_cast<gqe::logical::project_relation*>(optimized.get());
  ASSERT_EQ(proj->output_expressions_unsafe().size(), 2u);
  EXPECT_EQ(proj_col_idx(proj, 0), 0);                        // key passthrough: a
  expect_divide(proj->output_expressions_unsafe()[1], 1, 2);  // mean = sum / count

  auto const* out = as_aggregate(proj->children_safe()[0].get());
  ASSERT_NE(out, nullptr);
  ASSERT_EQ(out->keys_unsafe().size(), 1u);
  auto measures = out->measures_unsafe();
  ASSERT_EQ(measures.size(), 2u);
  EXPECT_EQ(measures[0].first, cudf::aggregation::SUM);
  EXPECT_EQ(measures[1].first, cudf::aggregation::COUNT_VALID);
  EXPECT_NE(dynamic_cast<gqe::logical::read_relation*>(out->children_safe()[0].get()), nullptr);

  EXPECT_EQ(optimized->data_types(), expected_types);
  EXPECT_EQ(optimizer_->get_rule_count(rule_t::mean_decomposition), 1u);
}

/**
 * @brief group by a, sum(b), mean(c), sum(d): the MEAN in the middle expands in place; surrounding
 * measures keep their slots. project -> [col(0), col(1), col(2)/col(3), col(4)].
 */
TEST_F(MeanDecompositionTest, MeanMixedWithOtherMeasures)
{
  auto agg = make_aggregate({0}, [] {
    std::vector<std::pair<agg_kind, std::unique_ptr<gqe::expression>>> m;
    m.push_back(measure(cudf::aggregation::SUM, 1));
    m.push_back(measure(cudf::aggregation::MEAN, 2));
    m.push_back(measure(cudf::aggregation::SUM, 3));
    return m;
  }());

  auto expected_types = agg->data_types();
  auto optimized      = optimizer_->optimize(agg);

  ASSERT_EQ(optimized->type(), relation_t::project);
  auto* proj = static_cast<gqe::logical::project_relation*>(optimized.get());
  ASSERT_EQ(proj->output_expressions_unsafe().size(), 4u);
  EXPECT_EQ(proj_col_idx(proj, 0), 0);                        // key a
  EXPECT_EQ(proj_col_idx(proj, 1), 1);                        // sum(b)
  expect_divide(proj->output_expressions_unsafe()[2], 2, 3);  // mean(c) = sum / count
  EXPECT_EQ(proj_col_idx(proj, 3), 4);                        // sum(d)

  auto const* out = as_aggregate(proj->children_safe()[0].get());
  ASSERT_NE(out, nullptr);
  auto measures = out->measures_unsafe();
  ASSERT_EQ(measures.size(), 4u);
  EXPECT_EQ(measures[0].first, cudf::aggregation::SUM);          // b
  EXPECT_EQ(measures[1].first, cudf::aggregation::SUM);          // c (sum part of mean)
  EXPECT_EQ(measures[2].first, cudf::aggregation::COUNT_VALID);  // c (count part of mean)
  EXPECT_EQ(measures[3].first, cudf::aggregation::SUM);          // d

  EXPECT_EQ(optimized->data_types(), expected_types);
  EXPECT_EQ(optimizer_->get_rule_count(rule_t::mean_decomposition), 1u);
}

/**
 * @brief group by a, mean(b), mean(c): two MEANs each expand into adjacent SUM/COUNT pairs.
 * project -> [col(0), col(1)/col(2), col(3)/col(4)].
 */
TEST_F(MeanDecompositionTest, MultipleMeans)
{
  auto agg = make_aggregate({0}, [] {
    std::vector<std::pair<agg_kind, std::unique_ptr<gqe::expression>>> m;
    m.push_back(measure(cudf::aggregation::MEAN, 1));
    m.push_back(measure(cudf::aggregation::MEAN, 2));
    return m;
  }());

  auto expected_types = agg->data_types();
  auto optimized      = optimizer_->optimize(agg);

  ASSERT_EQ(optimized->type(), relation_t::project);
  auto* proj = static_cast<gqe::logical::project_relation*>(optimized.get());
  ASSERT_EQ(proj->output_expressions_unsafe().size(), 3u);
  EXPECT_EQ(proj_col_idx(proj, 0), 0);
  expect_divide(proj->output_expressions_unsafe()[1], 1, 2);
  expect_divide(proj->output_expressions_unsafe()[2], 3, 4);

  auto const* out = as_aggregate(proj->children_safe()[0].get());
  ASSERT_NE(out, nullptr);
  ASSERT_EQ(out->measures_unsafe().size(), 4u);

  EXPECT_EQ(optimized->data_types(), expected_types);
  EXPECT_EQ(optimizer_->get_rule_count(rule_t::mean_decomposition), 1u);
}

/**
 * @brief mean(b) with no group keys (reduction): project -> [col(0)/col(1)] over aggregate([SUM(b),
 * COUNT_VALID(b)]).
 */
TEST_F(MeanDecompositionTest, ReductionNoKeys)
{
  auto agg = make_aggregate({}, [] {
    std::vector<std::pair<agg_kind, std::unique_ptr<gqe::expression>>> m;
    m.push_back(measure(cudf::aggregation::MEAN, 1));
    return m;
  }());

  auto expected_types = agg->data_types();
  auto optimized      = optimizer_->optimize(agg);

  ASSERT_EQ(optimized->type(), relation_t::project);
  auto* proj = static_cast<gqe::logical::project_relation*>(optimized.get());
  ASSERT_EQ(proj->output_expressions_unsafe().size(), 1u);
  expect_divide(proj->output_expressions_unsafe()[0], 0, 1);

  auto const* out = as_aggregate(proj->children_safe()[0].get());
  ASSERT_NE(out, nullptr);
  EXPECT_TRUE(out->keys_unsafe().empty());
  ASSERT_EQ(out->measures_unsafe().size(), 2u);

  EXPECT_EQ(optimized->data_types(), expected_types);
  EXPECT_EQ(optimizer_->get_rule_count(rule_t::mean_decomposition), 1u);
}

/**
 * @brief group by a, sum(b): no MEAN measure -> rule does not fire, plan untouched.
 */
TEST_F(MeanDecompositionTest, NoMeanNoOp)
{
  auto agg = make_aggregate({0}, [] {
    std::vector<std::pair<agg_kind, std::unique_ptr<gqe::expression>>> m;
    m.push_back(measure(cudf::aggregation::SUM, 1));
    return m;
  }());

  auto optimized = optimizer_->optimize(agg);

  EXPECT_EQ(optimized->type(), relation_t::aggregate);
  EXPECT_EQ(optimizer_->get_rule_count(rule_t::mean_decomposition), 0u);
}

/**
 * @brief Re-optimizing an already-decomposed plan is a no-op: the rewritten aggregate has no MEAN,
 * so the second pass does not fire (the applied-rule count stays at 1).
 */
TEST_F(MeanDecompositionTest, Idempotent)
{
  auto agg = make_aggregate({0}, [] {
    std::vector<std::pair<agg_kind, std::unique_ptr<gqe::expression>>> m;
    m.push_back(measure(cudf::aggregation::MEAN, 1));
    return m;
  }());

  auto once = optimizer_->optimize(agg);
  ASSERT_EQ(once->type(), relation_t::project);
  EXPECT_EQ(optimizer_->get_rule_count(rule_t::mean_decomposition), 1u);

  auto twice = optimizer_->optimize(once);
  EXPECT_EQ(twice->type(), relation_t::project);
  // No second application: the count did not increase.
  EXPECT_EQ(optimizer_->get_rule_count(rule_t::mean_decomposition), 1u);
}

/**
 * @brief A MEAN aggregate nested below a project is rewritten in place via the visitor's
 * child-replacement path (not the root special-case). The outer project's child becomes the
 * decomposition project wrapping the SUM/COUNT aggregate.
 */
TEST_F(MeanDecompositionTest, MeanAggregateAsChildWrapped)
{
  auto agg = make_aggregate({0}, [] {
    std::vector<std::pair<agg_kind, std::unique_ptr<gqe::expression>>> m;
    m.push_back(measure(cudf::aggregation::MEAN, 1));
    return m;
  }());

  // Outer project passing the aggregate's two output columns through unchanged.
  std::vector<std::unique_ptr<gqe::expression>> outer_exprs;
  outer_exprs.push_back(std::make_unique<gqe::column_reference_expression>(0));
  outer_exprs.push_back(std::make_unique<gqe::column_reference_expression>(1));
  std::shared_ptr<gqe::logical::relation> root = std::make_shared<gqe::logical::project_relation>(
    agg, std::vector<std::shared_ptr<gqe::logical::relation>>{}, std::move(outer_exprs));

  auto optimized = optimizer_->optimize(root);

  ASSERT_EQ(optimized->type(), relation_t::project);  // outer project preserved
  auto* outer = static_cast<gqe::logical::project_relation*>(optimized.get());

  // The outer project's child is now the decomposition project (not the original aggregate).
  ASSERT_EQ(outer->children_safe()[0]->type(), relation_t::project);
  auto* inner = static_cast<gqe::logical::project_relation*>(outer->children_safe()[0].get());
  ASSERT_GE(inner->output_expressions_unsafe().size(), 2u);
  expect_divide(inner->output_expressions_unsafe()[1], 1, 2);

  auto const* out = as_aggregate(inner->children_safe()[0].get());
  ASSERT_NE(out, nullptr);
  ASSERT_EQ(out->measures_unsafe().size(), 2u);
  EXPECT_EQ(out->measures_unsafe()[0].first, cudf::aggregation::SUM);
  EXPECT_EQ(out->measures_unsafe()[1].first, cudf::aggregation::COUNT_VALID);

  EXPECT_EQ(optimizer_->get_rule_count(rule_t::mean_decomposition), 1u);
}
