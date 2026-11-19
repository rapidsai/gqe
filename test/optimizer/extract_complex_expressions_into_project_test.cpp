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
#include <gqe/logical/join.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/relation.hpp>
#include <gqe/optimizer/logical_optimization.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/types.hpp>

#include <gtest/gtest.h>

#include <cudf/aggregation.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <utility>
#include <vector>

using relation_t = gqe::logical::relation::relation_type;
using expr_t     = gqe::expression::expression_type;
using rule_t     = gqe::optimizer::logical_optimization_rule_type;

/**
 * @brief Test fixture for the `complex_expression_extraction_into_project` optimizer rule.
 *
 * Configures a single-rule optimizer pipeline over a 4-column INT32 table `t` ({a, b, c, d}).
 */
class ExtractComplexExpressionsIntoProjectTest : public ::testing::Test {
 protected:
  static constexpr const char* kTable = "t";

  ExtractComplexExpressionsIntoProjectTest()
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

    config_ = gqe::optimizer::optimization_configuration(
      {rule_t::complex_expression_extraction_into_project}, {});
    optimizer_ = std::make_unique<gqe::optimizer::logical_optimizer>(&config_, catalog_.get());
  }

  /**
   * @brief Build a read of the first `ncols` of the INT32 columns {a, b, c, d}.
   *
   * @param ncols Number of leading columns to expose (default 4).
   * @return A `read_relation` over table `t`.
   */
  std::shared_ptr<gqe::logical::read_relation> make_read(std::size_t ncols = 4)
  {
    std::vector<std::string> all_cols = {"a", "b", "c", "d"};
    std::vector<std::string> cols(all_cols.begin(), all_cols.begin() + ncols);
    std::vector<cudf::data_type> tys(ncols, cudf::data_type(cudf::type_id::INT32));
    std::vector<std::shared_ptr<gqe::logical::relation>> sq;
    return std::make_shared<gqe::logical::read_relation>(sq, cols, tys, kTable, nullptr);
  }

  /**
   * @brief Build a `column_reference` expression for column `idx`.
   *
   * @param idx Zero-based column index.
   * @return A shared `column_reference_expression`.
   */
  static std::shared_ptr<gqe::expression> col(cudf::size_type idx)
  {
    return std::make_shared<gqe::column_reference_expression>(idx);
  }

  /**
   * @brief Column index of the `i`-th output expression of a project, asserting it is a
   * `column_reference`.
   *
   * @param proj Project whose output expressions to inspect.
   * @param i Index of the output expression.
   * @return The referenced column index, or -1 if the expression is not a column reference.
   */
  static cudf::size_type proj_col_idx(gqe::logical::project_relation const* proj, std::size_t i)
  {
    auto const* expr =
      dynamic_cast<gqe::column_reference_expression const*>(proj->output_expressions_unsafe()[i]);
    EXPECT_NE(expr, nullptr) << "project output[" << i << "] is not a column_reference";
    return expr ? expr->column_idx() : -1;
  }

  /**
   * @brief Column index of a join-condition operand, asserting it is a `column_reference`.
   *
   * @param expr Operand expression to inspect.
   * @return The referenced column index, or -1 if the expression is not a column reference.
   */
  static cudf::size_type operand_col_idx(gqe::expression const* expr)
  {
    auto const* cr = dynamic_cast<gqe::column_reference_expression const*>(expr);
    EXPECT_NE(cr, nullptr) << "operand is not a column_reference";
    return cr ? cr->column_idx() : -1;
  }

  /**
   * @brief Cast a relation to `project_relation`, asserting it is one.
   *
   * @param rel Relation to cast.
   * @return The relation as a `project_relation`, or nullptr if it is not one.
   */
  static gqe::logical::project_relation const* as_project(gqe::logical::relation const* rel)
  {
    auto const* p = dynamic_cast<gqe::logical::project_relation const*>(rel);
    EXPECT_NE(p, nullptr) << "relation is not a project";
    return p;
  }

  std::unique_ptr<gqe::task_manager_context> task_manager_ctx_;
  std::unique_ptr<gqe::catalog> catalog_;
  gqe::optimizer::optimization_configuration config_;
  std::unique_ptr<gqe::optimizer::logical_optimizer> optimizer_;
};

// ---------------------------------------------------------------------------
// Aggregate
// ---------------------------------------------------------------------------

/**
 * @brief group by a, sum(a + b) -> project(read, [a, a+b]); keys=[col_ref(0)],
 * measures=[(SUM,col_ref(1))].
 */
TEST_F(ExtractComplexExpressionsIntoProjectTest, AggregateComplexMeasureExtracted)
{
  std::vector<std::shared_ptr<gqe::logical::relation>> sq;
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(std::make_unique<gqe::column_reference_expression>(0));
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> measures;
  measures.emplace_back(cudf::aggregation::SUM,
                        std::make_unique<gqe::add_expression>(col(0), col(1)));
  auto agg = std::make_shared<gqe::logical::aggregate_relation>(
    make_read(), sq, std::move(keys), std::move(measures));

  auto expected_types = agg->data_types();
  auto optimized      = optimizer_->optimize(agg);

  ASSERT_EQ(optimized->type(), relation_t::aggregate);
  auto* out = static_cast<gqe::logical::aggregate_relation*>(optimized.get());

  auto* proj = dynamic_cast<gqe::logical::project_relation*>(out->children_safe()[0].get());
  ASSERT_NE(proj, nullptr);
  ASSERT_EQ(proj->output_expressions_unsafe().size(), 2u);
  EXPECT_EQ(proj_col_idx(proj, 0), 0);                                         // key passthrough: a
  EXPECT_EQ(proj->output_expressions_unsafe()[1]->type(), expr_t::binary_op);  // a + b
  EXPECT_NE(dynamic_cast<gqe::logical::read_relation*>(proj->children_safe()[0].get()), nullptr);

  ASSERT_EQ(out->keys_unsafe().size(), 1u);
  EXPECT_EQ(operand_col_idx(out->keys_unsafe()[0]), 0);
  ASSERT_EQ(out->measures_unsafe().size(), 1u);
  EXPECT_EQ(out->measures_unsafe()[0].first, cudf::aggregation::SUM);
  EXPECT_EQ(operand_col_idx(out->measures_unsafe()[0].second), 1);

  EXPECT_EQ(out->data_types(), expected_types);
  EXPECT_EQ(optimizer_->get_rule_count(rule_t::complex_expression_extraction_into_project), 1u);
}

/**
 * @brief group by (a + b), sum(c) -> project(read, [a+b, c]); keys=[col_ref(0)],
 * measures=[(SUM,col_ref(1))].
 */
TEST_F(ExtractComplexExpressionsIntoProjectTest, AggregateComplexKeyExtracted)
{
  std::vector<std::shared_ptr<gqe::logical::relation>> sq;
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(std::make_unique<gqe::add_expression>(col(0), col(1)));
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> measures;
  measures.emplace_back(cudf::aggregation::SUM,
                        std::make_unique<gqe::column_reference_expression>(2));
  auto agg = std::make_shared<gqe::logical::aggregate_relation>(
    make_read(), sq, std::move(keys), std::move(measures));

  auto optimized = optimizer_->optimize(agg);
  auto* out      = static_cast<gqe::logical::aggregate_relation*>(optimized.get());

  auto* proj = dynamic_cast<gqe::logical::project_relation*>(out->children_safe()[0].get());
  ASSERT_NE(proj, nullptr);
  ASSERT_EQ(proj->output_expressions_unsafe().size(), 2u);
  EXPECT_EQ(proj->output_expressions_unsafe()[0]->type(), expr_t::binary_op);  // a + b
  EXPECT_EQ(proj_col_idx(proj, 1), 2);  // measure passthrough: c

  EXPECT_EQ(operand_col_idx(out->keys_unsafe()[0]), 0);
  EXPECT_EQ(operand_col_idx(out->measures_unsafe()[0].second), 1);
  EXPECT_EQ(optimizer_->get_rule_count(rule_t::complex_expression_extraction_into_project), 1u);
}

/**
 * @brief sum(a * b) with no group keys -> project(read, [a*b]); keys=[],
 * measures=[(SUM,col_ref(0))].
 */
TEST_F(ExtractComplexExpressionsIntoProjectTest, AggregateReductionNoKeys)
{
  std::vector<std::shared_ptr<gqe::logical::relation>> sq;
  std::vector<std::unique_ptr<gqe::expression>> keys;
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> measures;
  measures.emplace_back(cudf::aggregation::SUM,
                        std::make_unique<gqe::multiply_expression>(col(0), col(1)));
  auto agg = std::make_shared<gqe::logical::aggregate_relation>(
    make_read(), sq, std::move(keys), std::move(measures));

  auto optimized = optimizer_->optimize(agg);
  auto* out      = static_cast<gqe::logical::aggregate_relation*>(optimized.get());

  auto* proj = dynamic_cast<gqe::logical::project_relation*>(out->children_safe()[0].get());
  ASSERT_NE(proj, nullptr);
  ASSERT_EQ(proj->output_expressions_unsafe().size(), 1u);
  EXPECT_EQ(proj->output_expressions_unsafe()[0]->type(), expr_t::binary_op);  // a * b

  EXPECT_TRUE(out->keys_unsafe().empty());
  EXPECT_EQ(operand_col_idx(out->measures_unsafe()[0].second), 0);
  EXPECT_EQ(optimizer_->get_rule_count(rule_t::complex_expression_extraction_into_project), 1u);
}

/**
 * @brief group by c, sum(d): bare col_refs but at non-canonical positions -> canonicalized.
 * project(read, [c, d]); keys=[col_ref(0)], measures=[(SUM,col_ref(1))].
 */
TEST_F(ExtractComplexExpressionsIntoProjectTest, AggregateBareNonCanonicalCanonicalized)
{
  std::vector<std::shared_ptr<gqe::logical::relation>> sq;
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(std::make_unique<gqe::column_reference_expression>(2));  // c
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> measures;
  measures.emplace_back(cudf::aggregation::SUM,
                        std::make_unique<gqe::column_reference_expression>(3));  // d
  auto agg = std::make_shared<gqe::logical::aggregate_relation>(
    make_read(), sq, std::move(keys), std::move(measures));

  auto optimized = optimizer_->optimize(agg);
  auto* out      = static_cast<gqe::logical::aggregate_relation*>(optimized.get());

  auto* proj = dynamic_cast<gqe::logical::project_relation*>(out->children_safe()[0].get());
  ASSERT_NE(proj, nullptr);
  ASSERT_EQ(proj->output_expressions_unsafe().size(), 2u);
  EXPECT_EQ(proj_col_idx(proj, 0), 2);  // c
  EXPECT_EQ(proj_col_idx(proj, 1), 3);  // d
  EXPECT_EQ(operand_col_idx(out->keys_unsafe()[0]), 0);
  EXPECT_EQ(operand_col_idx(out->measures_unsafe()[0].second), 1);
  EXPECT_EQ(optimizer_->get_rule_count(rule_t::complex_expression_extraction_into_project), 1u);
}

/**
 * @brief group by a, sum(b): already canonical ([0] key, [1] measure) -> untouched, rule does not
 * fire.
 */
TEST_F(ExtractComplexExpressionsIntoProjectTest, AggregateAlreadyCanonicalNoOp)
{
  std::vector<std::shared_ptr<gqe::logical::relation>> sq;
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(std::make_unique<gqe::column_reference_expression>(0));
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> measures;
  measures.emplace_back(cudf::aggregation::SUM,
                        std::make_unique<gqe::column_reference_expression>(1));
  auto agg = std::make_shared<gqe::logical::aggregate_relation>(
    make_read(), sq, std::move(keys), std::move(measures));

  auto optimized = optimizer_->optimize(agg);
  auto* out      = static_cast<gqe::logical::aggregate_relation*>(optimized.get());

  EXPECT_NE(dynamic_cast<gqe::logical::read_relation*>(out->children_safe()[0].get()), nullptr);
  EXPECT_EQ(optimizer_->get_rule_count(rule_t::complex_expression_extraction_into_project), 0u);
}

// ---------------------------------------------------------------------------
// Join
// ---------------------------------------------------------------------------

/**
 * @brief (l.a + l.b) == r.b : left key complex; right key r.b is a bare col-ref already at the
 * canonical trailing position. Only the left side is projected (materializes a+b); the right side
 * stays a read. condition col_ref(2) == col_ref(4); projection {0,1,2,3} -> {0,1,3,4}.
 */
TEST_F(ExtractComplexExpressionsIntoProjectTest, JoinComplexKeyRightCanonical)
{
  std::vector<std::shared_ptr<gqe::logical::relation>> sq;
  auto condition = std::make_unique<gqe::equal_expression>(
    std::make_shared<gqe::add_expression>(col(0), col(1)), col(3));
  auto join =
    std::make_shared<gqe::logical::join_relation>(make_read(2),
                                                  make_read(2),
                                                  sq,
                                                  std::move(condition),
                                                  gqe::join_type_type::inner,
                                                  std::vector<cudf::size_type>{0, 1, 2, 3});

  auto expected_types = join->data_types();
  auto optimized      = optimizer_->optimize(join);

  ASSERT_EQ(optimized->type(), relation_t::join);
  auto* out = static_cast<gqe::logical::join_relation*>(optimized.get());

  auto* left = as_project(out->children_safe()[0].get());
  ASSERT_NE(left, nullptr);
  ASSERT_EQ(left->output_expressions_unsafe().size(), 3u);  // [a, b, a+b]
  EXPECT_EQ(proj_col_idx(left, 0), 0);
  EXPECT_EQ(proj_col_idx(left, 1), 1);
  EXPECT_EQ(left->output_expressions_unsafe()[2]->type(), expr_t::binary_op);  // a + b

  // Right key r.b is already canonical (trailing) -> no project; right stays a read.
  EXPECT_NE(dynamic_cast<gqe::logical::read_relation*>(out->children_safe()[1].get()), nullptr);

  auto* eq = dynamic_cast<gqe::binary_op_expression const*>(out->condition());
  ASSERT_NE(eq, nullptr);
  EXPECT_EQ(eq->binary_operator(), cudf::binary_operator::EQUAL);
  EXPECT_EQ(operand_col_idx(eq->children()[0]), 2);  // left key a+b materialized at left trailing
  EXPECT_EQ(operand_col_idx(eq->children()[1]), 4);  // right key r.b shifted past the appended key

  EXPECT_EQ(out->projection_indices(), (std::vector<cudf::size_type>{0, 1, 3, 4}));
  EXPECT_EQ(out->data_types(), expected_types);
  EXPECT_EQ(optimizer_->get_rule_count(rule_t::complex_expression_extraction_into_project), 1u);
}

/**
 * @brief (l.a + l.b) == r.a : left key complex; right key r.a is a bare col-ref but NOT at the
 * canonical trailing position (r.a is the first right column, not the last). Both sides are
 * projected. condition col_ref(2) == col_ref(5); projection {0,1,2,3} -> {0,1,3,4}.
 */
TEST_F(ExtractComplexExpressionsIntoProjectTest, JoinComplexBothNonCanonical)
{
  std::vector<std::shared_ptr<gqe::logical::relation>> sq;
  auto condition = std::make_unique<gqe::equal_expression>(
    std::make_shared<gqe::add_expression>(col(0), col(1)), col(2));
  auto join =
    std::make_shared<gqe::logical::join_relation>(make_read(2),
                                                  make_read(2),
                                                  sq,
                                                  std::move(condition),
                                                  gqe::join_type_type::inner,
                                                  std::vector<cudf::size_type>{0, 1, 2, 3});

  auto expected_types = join->data_types();
  auto optimized      = optimizer_->optimize(join);
  auto* out           = static_cast<gqe::logical::join_relation*>(optimized.get());

  auto* left = as_project(out->children_safe()[0].get());
  ASSERT_NE(left, nullptr);
  EXPECT_EQ(left->output_expressions_unsafe().size(), 3u);  // [a, b, a+b]
  auto* right = as_project(out->children_safe()[1].get());
  ASSERT_NE(right, nullptr);
  EXPECT_EQ(right->output_expressions_unsafe().size(), 3u);  // [a, b, a (r.a materialized)]
  EXPECT_EQ(proj_col_idx(right, 2), 0);                      // r.a materialized at right trailing

  auto* eq = dynamic_cast<gqe::binary_op_expression const*>(out->condition());
  ASSERT_NE(eq, nullptr);
  EXPECT_EQ(operand_col_idx(eq->children()[0]), 2);  // left key a+b
  EXPECT_EQ(operand_col_idx(eq->children()[1]), 5);  // right key r.a at right trailing

  EXPECT_EQ(out->projection_indices(), (std::vector<cudf::size_type>{0, 1, 3, 4}));
  EXPECT_EQ(out->data_types(), expected_types);
  EXPECT_EQ(optimizer_->get_rule_count(rule_t::complex_expression_extraction_into_project), 1u);
}

/**
 * @brief l.a == r.a AND (l.a * l.b) == r.b : the left side has a complex key, so it is projected
 * and ALL of its keys are materialized (the bare l.a too) -> [a, b, a, a*b]. The right keys r.a,
 * r.b are already canonical (trailing) -> right stays a read. projection {0,1,2,3} -> {0,1,4,5}.
 */
TEST_F(ExtractComplexExpressionsIntoProjectTest, JoinMixedKeysComplexSide)
{
  std::vector<std::shared_ptr<gqe::logical::relation>> sq;
  auto eq1 = std::make_shared<gqe::equal_expression>(col(0), col(2));
  auto eq2 = std::make_shared<gqe::equal_expression>(
    std::make_shared<gqe::multiply_expression>(col(0), col(1)), col(3));
  auto condition = std::make_unique<gqe::logical_and_expression>(eq1, eq2);
  auto join =
    std::make_shared<gqe::logical::join_relation>(make_read(2),
                                                  make_read(2),
                                                  sq,
                                                  std::move(condition),
                                                  gqe::join_type_type::inner,
                                                  std::vector<cudf::size_type>{0, 1, 2, 3});

  auto expected_types = join->data_types();
  auto optimized      = optimizer_->optimize(join);
  auto* out           = static_cast<gqe::logical::join_relation*>(optimized.get());

  auto* left = as_project(out->children_safe()[0].get());
  ASSERT_NE(left, nullptr);
  ASSERT_EQ(left->output_expressions_unsafe().size(), 4u);  // [a, b, a, a*b]
  EXPECT_EQ(proj_col_idx(left, 2), 0);                      // bare key l.a re-materialized
  EXPECT_EQ(left->output_expressions_unsafe()[3]->type(), expr_t::binary_op);  // a * b
  // Right keys r.a, r.b are already canonical -> no project; right stays a read.
  EXPECT_NE(dynamic_cast<gqe::logical::read_relation*>(out->children_safe()[1].get()), nullptr);

  EXPECT_EQ(out->projection_indices(), (std::vector<cudf::size_type>{0, 1, 4, 5}));
  EXPECT_EQ(out->data_types(), expected_types);
  EXPECT_EQ(optimizer_->get_rule_count(rule_t::complex_expression_extraction_into_project), 1u);
}

/**
 * @brief l.c == r.c with 3-column inputs: keys are bare col-refs already at the canonical trailing
 * positions (l.c at col_ref(2), r.c at col_ref(5)) -> no-op, both inputs stay reads.
 */
TEST_F(ExtractComplexExpressionsIntoProjectTest, JoinAllBareCanonicalNoOp)
{
  std::vector<std::shared_ptr<gqe::logical::relation>> sq;
  auto condition = std::make_unique<gqe::equal_expression>(col(2), col(5));
  auto join =
    std::make_shared<gqe::logical::join_relation>(make_read(3),
                                                  make_read(3),
                                                  sq,
                                                  std::move(condition),
                                                  gqe::join_type_type::inner,
                                                  std::vector<cudf::size_type>{0, 1, 2, 3, 4, 5});

  auto optimized = optimizer_->optimize(join);
  auto* out      = static_cast<gqe::logical::join_relation*>(optimized.get());

  EXPECT_NE(dynamic_cast<gqe::logical::read_relation*>(out->children_safe()[0].get()), nullptr);
  EXPECT_NE(dynamic_cast<gqe::logical::read_relation*>(out->children_safe()[1].get()), nullptr);
  EXPECT_EQ(optimizer_->get_rule_count(rule_t::complex_expression_extraction_into_project), 0u);
}

/**
 * @brief l.a == r.a (2-column inputs): keys are bare col-refs but NOT at the canonical trailing
 * positions (l.a at 0, r.a at 2 instead of the last column of each side). Both sides are projected
 * to reorder the keys to trailing. condition col_ref(2) == col_ref(5); projection {0,1,2,3} ->
 * {0,1,3,4}.
 */
TEST_F(ExtractComplexExpressionsIntoProjectTest, JoinAllBareNonCanonical)
{
  std::vector<std::shared_ptr<gqe::logical::relation>> sq;
  auto condition = std::make_unique<gqe::equal_expression>(col(0), col(2));
  auto join =
    std::make_shared<gqe::logical::join_relation>(make_read(2),
                                                  make_read(2),
                                                  sq,
                                                  std::move(condition),
                                                  gqe::join_type_type::inner,
                                                  std::vector<cudf::size_type>{0, 1, 2, 3});

  auto expected_types = join->data_types();
  auto optimized      = optimizer_->optimize(join);
  auto* out           = static_cast<gqe::logical::join_relation*>(optimized.get());

  auto* left = as_project(out->children_safe()[0].get());
  ASSERT_NE(left, nullptr);
  EXPECT_EQ(left->output_expressions_unsafe().size(), 3u);  // [a, b, a]
  auto* right = as_project(out->children_safe()[1].get());
  ASSERT_NE(right, nullptr);
  EXPECT_EQ(right->output_expressions_unsafe().size(), 3u);  // [a, b, a]

  auto* eq = dynamic_cast<gqe::binary_op_expression const*>(out->condition());
  ASSERT_NE(eq, nullptr);
  EXPECT_EQ(operand_col_idx(eq->children()[0]), 2);
  EXPECT_EQ(operand_col_idx(eq->children()[1]), 5);

  EXPECT_EQ(out->projection_indices(), (std::vector<cudf::size_type>{0, 1, 3, 4}));
  EXPECT_EQ(out->data_types(), expected_types);
  EXPECT_EQ(optimizer_->get_rule_count(rule_t::complex_expression_extraction_into_project), 1u);
}

/**
 * @brief l.a < r.a : a pure non-equality condition -> untouched.
 */
TEST_F(ExtractComplexExpressionsIntoProjectTest, JoinNonEqualityUntouched)
{
  std::vector<std::shared_ptr<gqe::logical::relation>> sq;
  auto condition = std::make_unique<gqe::less_expression>(col(0), col(2));
  auto join =
    std::make_shared<gqe::logical::join_relation>(make_read(2),
                                                  make_read(2),
                                                  sq,
                                                  std::move(condition),
                                                  gqe::join_type_type::inner,
                                                  std::vector<cudf::size_type>{0, 1, 2, 3});

  auto optimized = optimizer_->optimize(join);
  auto* out      = static_cast<gqe::logical::join_relation*>(optimized.get());

  EXPECT_NE(dynamic_cast<gqe::logical::read_relation*>(out->children_safe()[0].get()), nullptr);
  EXPECT_NE(dynamic_cast<gqe::logical::read_relation*>(out->children_safe()[1].get()), nullptr);
  EXPECT_EQ(optimizer_->get_rule_count(rule_t::complex_expression_extraction_into_project), 0u);
}

/**
 * @brief l.a == r.a AND l.b < r.b : an equi-key plus a non-equality residual -> skip the whole
 * rewrite.
 */
TEST_F(ExtractComplexExpressionsIntoProjectTest, JoinNonEquiResidualUntouched)
{
  std::vector<std::shared_ptr<gqe::logical::relation>> sq;
  auto eq        = std::make_shared<gqe::equal_expression>(col(0), col(2));
  auto lt        = std::make_shared<gqe::less_expression>(col(1), col(3));
  auto condition = std::make_unique<gqe::logical_and_expression>(eq, lt);
  auto join =
    std::make_shared<gqe::logical::join_relation>(make_read(2),
                                                  make_read(2),
                                                  sq,
                                                  std::move(condition),
                                                  gqe::join_type_type::inner,
                                                  std::vector<cudf::size_type>{0, 1, 2, 3});

  auto optimized = optimizer_->optimize(join);
  auto* out      = static_cast<gqe::logical::join_relation*>(optimized.get());

  EXPECT_NE(dynamic_cast<gqe::logical::read_relation*>(out->children_safe()[0].get()), nullptr);
  EXPECT_NE(dynamic_cast<gqe::logical::read_relation*>(out->children_safe()[1].get()), nullptr);
  EXPECT_EQ(optimizer_->get_rule_count(rule_t::complex_expression_extraction_into_project), 0u);
}

/**
 * @brief (l.a + r.a) == r.b : an operand references both inputs -> untouched.
 */
TEST_F(ExtractComplexExpressionsIntoProjectTest, JoinMixedSideOperandUntouched)
{
  std::vector<std::shared_ptr<gqe::logical::relation>> sq;
  auto condition = std::make_unique<gqe::equal_expression>(
    std::make_shared<gqe::add_expression>(col(0), col(2)), col(3));
  auto join =
    std::make_shared<gqe::logical::join_relation>(make_read(2),
                                                  make_read(2),
                                                  sq,
                                                  std::move(condition),
                                                  gqe::join_type_type::inner,
                                                  std::vector<cudf::size_type>{0, 1, 2, 3});

  auto optimized = optimizer_->optimize(join);
  auto* out      = static_cast<gqe::logical::join_relation*>(optimized.get());

  EXPECT_NE(dynamic_cast<gqe::logical::read_relation*>(out->children_safe()[0].get()), nullptr);
  EXPECT_NE(dynamic_cast<gqe::logical::read_relation*>(out->children_safe()[1].get()), nullptr);
  EXPECT_EQ(optimizer_->get_rule_count(rule_t::complex_expression_extraction_into_project), 0u);
}
