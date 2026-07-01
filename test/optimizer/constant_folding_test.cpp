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
#include <gqe/expression/literal.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/relation.hpp>
#include <gqe/optimizer/logical_optimization.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/types.hpp>

#include <gtest/gtest.h>

#include <cudf/types.hpp>

#include <memory>
#include <numeric>
#include <vector>

using relation_t = gqe::logical::relation::relation_type;

class ConstantFoldingTest : public ::testing::Test {
 protected:
  static constexpr const char* kTable = "t";

  ConstantFoldingTest()
  {
    task_manager_ctx_ = std::make_unique<gqe::task_manager_context>();
    catalog_          = std::make_unique<gqe::catalog>(task_manager_ctx_.get());

    catalog_->register_table(
      kTable,
      {{"a", cudf::data_type(cudf::type_id::INT32)}, {"b", cudf::data_type(cudf::type_id::INT32)}},
      gqe::storage_kind::system_memory{},
      gqe::partitioning_schema_kind::none{});

    config_ = gqe::optimizer::optimization_configuration(
      {gqe::optimizer::logical_optimization_rule_type::constant_folding}, {});
    optimizer_ = std::make_unique<gqe::optimizer::logical_optimizer>(&config_, catalog_.get());
  }

  // Read with 2 INT32 columns {a, b}.
  std::shared_ptr<gqe::logical::read_relation> make_read(
    std::unique_ptr<gqe::expression> partial_filter = nullptr)
  {
    std::vector<std::string> cols    = {"a", "b"};
    std::vector<cudf::data_type> tys = {cudf::data_type(cudf::type_id::INT32),
                                        cudf::data_type(cudf::type_id::INT32)};
    std::vector<std::shared_ptr<gqe::logical::relation>> sq;
    return std::make_shared<gqe::logical::read_relation>(
      sq, cols, tys, kTable, std::move(partial_filter));
  }

  // filter(child, literal(true), identity projection over all child columns).
  std::shared_ptr<gqe::logical::filter_relation> make_trivial_filter(
    std::shared_ptr<gqe::logical::relation> child)
  {
    auto ncols = child->data_types().size();
    std::vector<cudf::size_type> indices(ncols);
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<std::shared_ptr<gqe::logical::relation>> sq;
    return std::make_shared<gqe::logical::filter_relation>(
      child, sq, std::make_unique<gqe::literal_expression<bool>>(true), std::move(indices));
  }

  // A real (non-trivial) condition: literal_int(1) < literal_int(2).
  static std::unique_ptr<gqe::expression> make_real_condition()
  {
    return std::make_unique<gqe::less_expression>(
      std::make_shared<gqe::literal_expression<int32_t>>(1),
      std::make_shared<gqe::literal_expression<int32_t>>(2));
  }

  // Extract the column_idx from the i-th output expression of a project_relation.
  static cudf::size_type col_ref_idx(gqe::logical::project_relation const* proj, std::size_t i)
  {
    auto const* expr =
      dynamic_cast<gqe::column_reference_expression const*>(proj->output_expressions_unsafe()[i]);
    EXPECT_NE(expr, nullptr) << "output_expressions[" << i << "] is not a column_reference";
    return expr ? expr->column_idx() : -1;
  }

  std::unique_ptr<gqe::task_manager_context> task_manager_ctx_;
  std::unique_ptr<gqe::catalog> catalog_;
  gqe::optimizer::optimization_configuration config_;
  std::unique_ptr<gqe::optimizer::logical_optimizer> optimizer_;
};

// filter(read, literal(true), {0,1}) → project(read, [col_ref(0), col_ref(1)]).
// projection_indices are part of the output contract and must be preserved as a project_relation.
TEST_F(ConstantFoldingTest, FilterRootConvertedToProject)
{
  auto plan      = make_trivial_filter(make_read());
  auto optimized = optimizer_->optimize(plan);

  auto* proj = dynamic_cast<gqe::logical::project_relation*>(optimized.get());
  ASSERT_NE(proj, nullptr);
  ASSERT_EQ(proj->output_expressions_unsafe().size(), 2u);
  EXPECT_EQ(col_ref_idx(proj, 0), 0);
  EXPECT_EQ(col_ref_idx(proj, 1), 1);
  EXPECT_NE(dynamic_cast<gqe::logical::read_relation*>(proj->children_safe()[0].get()), nullptr);
}

// filter(read, literal(true), {}) → project(read, {}) — zero-column output preserves schema.
TEST_F(ConstantFoldingTest, FilterRootWithEmptyProjectionElided)
{
  std::vector<std::shared_ptr<gqe::logical::relation>> sq;
  auto plan = std::make_shared<gqe::logical::filter_relation>(
    make_read(),
    sq,
    std::make_unique<gqe::literal_expression<bool>>(true),
    std::vector<cudf::size_type>{});

  auto optimized = optimizer_->optimize(plan);

  auto* proj = dynamic_cast<gqe::logical::project_relation*>(optimized.get());
  ASSERT_NE(proj, nullptr);
  EXPECT_EQ(proj->output_expressions_unsafe().size(), 0u);
  EXPECT_NE(dynamic_cast<gqe::logical::read_relation*>(proj->children_safe()[0].get()), nullptr);
  EXPECT_EQ(dynamic_cast<gqe::logical::filter_relation*>(optimized.get()), nullptr);
}

// filter(filter(read, literal(true), {0,1}), real_condition, {0,1})
// → filter(project(read, [col_ref(0), col_ref(1)]), real_condition, {0,1}).
// Inner trivial filter is converted to project; outer real filter is preserved.
TEST_F(ConstantFoldingTest, FilterChildConvertedToProject)
{
  auto inner = make_trivial_filter(make_read());

  auto ncols = inner->data_types().size();
  std::vector<cudf::size_type> indices(ncols);
  std::iota(indices.begin(), indices.end(), 0);
  std::vector<std::shared_ptr<gqe::logical::relation>> sq;
  auto outer = std::make_shared<gqe::logical::filter_relation>(
    inner, sq, make_real_condition(), std::move(indices));

  auto optimized = optimizer_->optimize(outer);

  auto* f = dynamic_cast<gqe::logical::filter_relation*>(optimized.get());
  ASSERT_NE(f, nullptr);
  EXPECT_NE(dynamic_cast<gqe::logical::project_relation*>(f->children_safe()[0].get()), nullptr);
  EXPECT_EQ(dynamic_cast<gqe::logical::filter_relation*>(f->children_safe()[0].get()), nullptr);
}

// literal(false) condition must not be rewritten.
TEST_F(ConstantFoldingTest, FilterFalsePreserved)
{
  std::vector<cudf::size_type> indices = {0, 1};
  std::vector<std::shared_ptr<gqe::logical::relation>> sq;
  auto plan = std::make_shared<gqe::logical::filter_relation>(
    make_read(), sq, std::make_unique<gqe::literal_expression<bool>>(false), indices);

  auto optimized = optimizer_->optimize(plan);
  EXPECT_NE(dynamic_cast<gqe::logical::filter_relation*>(optimized.get()), nullptr);
}

// literal_expression<int32_t>(42) condition must not be rewritten.
TEST_F(ConstantFoldingTest, FilterNonBoolPreserved)
{
  std::vector<cudf::size_type> indices = {0, 1};
  std::vector<std::shared_ptr<gqe::logical::relation>> sq;
  auto plan = std::make_shared<gqe::logical::filter_relation>(
    make_read(), sq, std::make_unique<gqe::literal_expression<int32_t>>(42), indices);

  auto optimized = optimizer_->optimize(plan);
  EXPECT_NE(dynamic_cast<gqe::logical::filter_relation*>(optimized.get()), nullptr);
}

// A real Boolean expression must not be rewritten.
TEST_F(ConstantFoldingTest, FilterRealConditionPreserved)
{
  std::vector<cudf::size_type> indices = {0, 1};
  std::vector<std::shared_ptr<gqe::logical::relation>> sq;
  auto plan = std::make_shared<gqe::logical::filter_relation>(
    make_read(), sq, make_real_condition(), indices);

  auto optimized = optimizer_->optimize(plan);
  EXPECT_NE(dynamic_cast<gqe::logical::filter_relation*>(optimized.get()), nullptr);
}

// filter(read, literal(true), {1,0}) → project(read, [col_ref(1), col_ref(0)]).
// Non-identity projection is preserved in the correct (swapped) order.
TEST_F(ConstantFoldingTest, FilterNonIdentityProjectionConvertedToProject)
{
  std::vector<cudf::size_type> swapped = {1, 0};
  std::vector<std::shared_ptr<gqe::logical::relation>> sq;
  auto plan = std::make_shared<gqe::logical::filter_relation>(
    make_read(), sq, std::make_unique<gqe::literal_expression<bool>>(true), swapped);

  auto optimized = optimizer_->optimize(plan);

  auto* proj = dynamic_cast<gqe::logical::project_relation*>(optimized.get());
  ASSERT_NE(proj, nullptr);
  ASSERT_EQ(proj->output_expressions_unsafe().size(), 2u);
  EXPECT_EQ(col_ref_idx(proj, 0), 1);
  EXPECT_EQ(col_ref_idx(proj, 1), 0);
}

// read_relation with literal(true) partial_filter → partial_filter cleared.
TEST_F(ConstantFoldingTest, ReadPartialFilterCleared)
{
  auto plan      = make_read(std::make_unique<gqe::literal_expression<bool>>(true));
  auto optimized = optimizer_->optimize(plan);

  auto* read = dynamic_cast<gqe::logical::read_relation*>(optimized.get());
  ASSERT_NE(read, nullptr);
  EXPECT_EQ(read->partial_filter_unsafe(), nullptr);
}

// read_relation with literal(false) partial_filter → partial_filter preserved.
TEST_F(ConstantFoldingTest, ReadPartialFilterFalsePreserved)
{
  auto plan      = make_read(std::make_unique<gqe::literal_expression<bool>>(false));
  auto optimized = optimizer_->optimize(plan);

  auto* read = dynamic_cast<gqe::logical::read_relation*>(optimized.get());
  ASSERT_NE(read, nullptr);
  EXPECT_NE(read->partial_filter_unsafe(), nullptr);
}

// filter(join(read, read, cond, inner, {0,1,2,3}), literal(true), {0,1,2,3})
// → project(join, [col_ref(0), col_ref(1), col_ref(2), col_ref(3)]).
// This is the post_join_filter case produced by parse_join_relation.
TEST_F(ConstantFoldingTest, JoinPostFilterConvertedToProject)
{
  auto left  = make_read();
  auto right = make_read();

  // Join: 2+2 = 4 output columns.
  std::vector<cudf::size_type> join_proj = {0, 1, 2, 3};
  std::vector<std::shared_ptr<gqe::logical::relation>> sq;
  auto join = std::make_shared<gqe::logical::join_relation>(
    left, right, sq, make_real_condition(), gqe::join_type_type::inner, join_proj);

  // Wrap with trivial-true filter (as the parser would emit).
  std::vector<cudf::size_type> filter_proj = {0, 1, 2, 3};
  auto plan                                = std::make_shared<gqe::logical::filter_relation>(
    join, sq, std::make_unique<gqe::literal_expression<bool>>(true), filter_proj);

  auto optimized = optimizer_->optimize(plan);

  auto* proj = dynamic_cast<gqe::logical::project_relation*>(optimized.get());
  ASSERT_NE(proj, nullptr);
  ASSERT_EQ(proj->output_expressions_unsafe().size(), 4u);
  for (std::size_t i = 0; i < 4; ++i)
    EXPECT_EQ(col_ref_idx(proj, i), static_cast<cudf::size_type>(i));
  EXPECT_NE(dynamic_cast<gqe::logical::join_relation*>(proj->children_safe()[0].get()), nullptr);
  EXPECT_EQ(dynamic_cast<gqe::logical::filter_relation*>(proj->children_safe()[0].get()), nullptr);
}
