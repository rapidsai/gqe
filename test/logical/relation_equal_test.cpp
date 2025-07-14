/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/fetch.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/relation.hpp>
#include <gqe/logical/set.hpp>
#include <gqe/logical/sort.hpp>
#include <gqe/logical/window.hpp>
#include <gqe/logical/write.hpp>
#include <gqe/types.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/types.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

using measures_type =
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>>;

class RelationEqualTest : public ::testing::Test {
 protected:
  RelationEqualTest()
  {
    int32_literal_one = std::make_shared<gqe::literal_expression<int32_t>>(1);

    col_0 = std::make_shared<gqe::column_reference_expression>(0);
    col_1 = std::make_shared<gqe::column_reference_expression>(1);
    col_2 = std::make_shared<gqe::column_reference_expression>(2);

    std::vector<std::string> col_names_0 = {"a", "b"};
    auto col_types_0                     = {cudf::data_type(cudf::type_id::INT32),
                                            cudf::data_type(cudf::type_id::FLOAT32)};
    read_rel_0                           = std::make_shared<gqe::logical::read_relation>(
      empty_relations(), std::move(col_names_0), std::move(col_types_0), "t0", nullptr);
    std::vector<std::string> col_names_0_dup = {"a", "b"};
    auto col_types_0_dup                     = {cudf::data_type(cudf::type_id::INT32),
                                                cudf::data_type(cudf::type_id::FLOAT32)};
    read_rel_0_dup                           = std::make_shared<gqe::logical::read_relation>(
      empty_relations(), std::move(col_names_0_dup), std::move(col_types_0_dup), "t0", nullptr);

    std::vector<std::string> col_names_1 = {"a", "c"};
    auto col_types_1                     = {cudf::data_type(cudf::type_id::INT32),
                                            cudf::data_type(cudf::type_id::BOOL8)};
    read_rel_1                           = std::make_shared<gqe::logical::read_relation>(
      empty_relations(), std::move(col_names_1), std::move(col_types_1), "t1", nullptr);

    std::vector<std::string> col_names_2 = {"d", "e"};
    auto col_types_2                     = {cudf::data_type(cudf::type_id::INT32),
                                            cudf::data_type(cudf::type_id::FLOAT32)};
    read_rel_2                           = std::make_shared<gqe::logical::read_relation>(
      empty_relations(), std::move(col_names_2), std::move(col_types_2), "t2", nullptr);
  }

  std::vector<std::shared_ptr<gqe::logical::relation>> empty_relations() { return {}; }

  std::vector<std::shared_ptr<gqe::logical::relation>> non_empty_relations()
  {
    std::vector<std::shared_ptr<gqe::logical::relation>> subq_relations;
    subq_relations.push_back(read_rel_0);
    return subq_relations;
  }

  std::vector<std::unique_ptr<gqe::expression>> duplicate_expressions(
    const std::vector<std::unique_ptr<gqe::expression>>& exprs)
  {
    std::vector<std::unique_ptr<gqe::expression>> new_exprs;
    new_exprs.reserve(exprs.size());
    for (auto& expr : exprs) {
      new_exprs.push_back(expr->clone());
    }
    return new_exprs;
  }

  measures_type duplicate_measures(const measures_type& measures)
  {
    measures_type new_measures;
    new_measures.reserve(measures.size());
    for (auto& measure : measures) {
      new_measures.push_back(std::make_pair(measure.first, measure.second->clone()));
    }
    return new_measures;
  }

  std::shared_ptr<gqe::literal_expression<int32_t>> int32_literal_one;

  std::shared_ptr<gqe::column_reference_expression> col_0;
  std::shared_ptr<gqe::column_reference_expression> col_1;
  std::shared_ptr<gqe::column_reference_expression> col_2;

  std::shared_ptr<gqe::logical::read_relation> read_rel_0;
  std::shared_ptr<gqe::logical::read_relation> read_rel_1;
  std::shared_ptr<gqe::logical::read_relation> read_rel_2;
  std::shared_ptr<gqe::logical::read_relation> read_rel_0_dup;
};

TEST_F(RelationEqualTest, Aggregate)
{
  std::vector<std::unique_ptr<gqe::expression>> keys_0;
  std::vector<std::unique_ptr<gqe::expression>> keys_1;
  std::vector<std::unique_ptr<gqe::expression>> keys_0_dup;
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> measures_0;
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> measures_1;
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> measures_0_dup;

  keys_0.push_back(std::make_unique<gqe::column_reference_expression>(0));
  measures_0.emplace_back(cudf::aggregation::SUM,
                          std::make_unique<gqe::column_reference_expression>(1));
  std::unique_ptr<gqe::logical::aggregate_relation> agg_rel_0 =
    std::make_unique<gqe::logical::aggregate_relation>(
      std::move(read_rel_0), empty_relations(), std::move(keys_0), std::move(measures_0));

  keys_0_dup.push_back(std::make_unique<gqe::column_reference_expression>(0));
  measures_0_dup.emplace_back(cudf::aggregation::SUM,
                              std::make_unique<gqe::column_reference_expression>(1));
  std::unique_ptr<gqe::logical::aggregate_relation> agg_rel_0_dup =
    std::make_unique<gqe::logical::aggregate_relation>(std::move(read_rel_0_dup),
                                                       empty_relations(),
                                                       std::move(keys_0_dup),
                                                       std::move(measures_0_dup));

  keys_1.push_back(std::make_unique<gqe::column_reference_expression>(1));
  measures_1.emplace_back(cudf::aggregation::SUM,
                          std::make_unique<gqe::column_reference_expression>(0));
  std::unique_ptr<gqe::logical::aggregate_relation> agg_rel_1 =
    std::make_unique<gqe::logical::aggregate_relation>(
      std::move(read_rel_1), empty_relations(), std::move(keys_1), std::move(measures_1));
  std::unique_ptr<gqe::logical::aggregate_relation> agg_rel_2 =
    std::make_unique<gqe::logical::aggregate_relation>(std::move(read_rel_1),
                                                       non_empty_relations(),
                                                       duplicate_expressions(keys_1),
                                                       duplicate_measures(measures_1));

  // TODO: Add more tests
  EXPECT_FALSE(*agg_rel_0 == *agg_rel_1);
  EXPECT_FALSE(*agg_rel_1 == *agg_rel_2);
  EXPECT_EQ(*agg_rel_0, *agg_rel_0_dup);
}

TEST_F(RelationEqualTest, Fetch)
{
  std::unique_ptr<gqe::logical::fetch_relation> fetch_0 =
    std::make_unique<gqe::logical::fetch_relation>(read_rel_0, 0, 100);
  std::unique_ptr<gqe::logical::fetch_relation> fetch_0_dup =
    std::make_unique<gqe::logical::fetch_relation>(read_rel_0_dup, 0, 100);
  std::unique_ptr<gqe::logical::fetch_relation> fetch_1 =
    std::make_unique<gqe::logical::fetch_relation>(read_rel_1, 0, 100);
  std::unique_ptr<gqe::logical::fetch_relation> fetch_2 =
    std::make_unique<gqe::logical::fetch_relation>(read_rel_0, 1, 100);
  std::unique_ptr<gqe::logical::fetch_relation> fetch_3 =
    std::make_unique<gqe::logical::fetch_relation>(read_rel_0, 0, 50);
  EXPECT_FALSE(*fetch_0 == *fetch_1);  // different input
  EXPECT_FALSE(*fetch_0 == *fetch_2);  // different offset
  EXPECT_FALSE(*fetch_0 == *fetch_3);  // different count
  EXPECT_EQ(*fetch_0, *fetch_0_dup);
}

TEST_F(RelationEqualTest, Filter)
{
  auto cond_0 = std::make_unique<gqe::equal_expression>(col_0, int32_literal_one);
  auto cond_1 = std::make_unique<gqe::equal_expression>(col_1, int32_literal_one);

  std::vector<cudf::size_type> projection_indices_0 = {0};
  std::vector<cudf::size_type> projection_indices_1 = {1};
  std::vector<cudf::size_type> projection_indices_2 = {0, 1};

  std::unique_ptr<gqe::logical::filter_relation> filter_0 =
    std::make_unique<gqe::logical::filter_relation>(
      read_rel_0, empty_relations(), cond_0->clone(), projection_indices_0);
  std::unique_ptr<gqe::logical::filter_relation> filter_0_dup =
    std::make_unique<gqe::logical::filter_relation>(
      read_rel_0_dup, empty_relations(), cond_0->clone(), projection_indices_0);
  std::unique_ptr<gqe::logical::filter_relation> filter_1 =
    std::make_unique<gqe::logical::filter_relation>(
      read_rel_1, empty_relations(), std::move(cond_0), projection_indices_0);
  std::unique_ptr<gqe::logical::filter_relation> filter_2 =
    std::make_unique<gqe::logical::filter_relation>(
      read_rel_1, empty_relations(), cond_1->clone(), projection_indices_0);
  std::unique_ptr<gqe::logical::filter_relation> filter_3 =
    std::make_unique<gqe::logical::filter_relation>(
      read_rel_1, non_empty_relations(), cond_1->clone(), projection_indices_0);
  std::unique_ptr<gqe::logical::filter_relation> filter_4 =
    std::make_unique<gqe::logical::filter_relation>(
      read_rel_1, non_empty_relations(), cond_1->clone(), projection_indices_1);
  std::unique_ptr<gqe::logical::filter_relation> filter_5 =
    std::make_unique<gqe::logical::filter_relation>(
      read_rel_1, non_empty_relations(), cond_1->clone(), projection_indices_2);
  EXPECT_FALSE(*filter_0 == *filter_1);  // different input
  EXPECT_FALSE(*filter_1 == *filter_2);  // different condition
  EXPECT_FALSE(*filter_2 == *filter_3);  // different suquery_relations
  EXPECT_FALSE(*filter_3 == *filter_4);  // different projection indices
  EXPECT_FALSE(*filter_4 == *filter_5);  // different projection indices length
  EXPECT_EQ(*filter_0, *filter_0_dup);
}

TEST_F(RelationEqualTest, Join)
{
  auto cond_0     = std::make_unique<gqe::equal_expression>(col_0, col_2);
  auto cond_1     = std::make_unique<gqe::equal_expression>(col_1, int32_literal_one);
  auto cond_0_dup = std::make_unique<gqe::equal_expression>(col_0, col_2);
  std::vector<cudf::size_type> projection_indices_0 = {1, 3};
  std::vector<cudf::size_type> projection_indices_1 = {1, 4};
  std::vector<cudf::size_type> projection_indices_2 = {1, 4, 5};

  std::unique_ptr<gqe::logical::join_relation> join_0 =
    std::make_unique<gqe::logical::join_relation>(read_rel_0,
                                                  read_rel_1,
                                                  empty_relations(),
                                                  cond_0->clone(),
                                                  gqe::join_type_type::inner,
                                                  projection_indices_0);
  std::unique_ptr<gqe::logical::join_relation> join_0_dup =
    std::make_unique<gqe::logical::join_relation>(read_rel_0,
                                                  read_rel_1,
                                                  empty_relations(),
                                                  cond_0->clone(),
                                                  gqe::join_type_type::inner,
                                                  projection_indices_0);
  std::unique_ptr<gqe::logical::join_relation> join_1 =
    std::make_unique<gqe::logical::join_relation>(read_rel_1,
                                                  read_rel_0,
                                                  empty_relations(),
                                                  std::move(cond_0),
                                                  gqe::join_type_type::inner,
                                                  projection_indices_0);
  std::unique_ptr<gqe::logical::join_relation> join_2 =
    std::make_unique<gqe::logical::join_relation>(read_rel_1,
                                                  read_rel_0,
                                                  empty_relations(),
                                                  cond_1->clone(),
                                                  gqe::join_type_type::inner,
                                                  projection_indices_0);
  std::unique_ptr<gqe::logical::join_relation> join_3 =
    std::make_unique<gqe::logical::join_relation>(read_rel_1,
                                                  read_rel_0,
                                                  empty_relations(),
                                                  cond_1->clone(),
                                                  gqe::join_type_type::full,
                                                  projection_indices_0);
  std::unique_ptr<gqe::logical::join_relation> join_4 =
    std::make_unique<gqe::logical::join_relation>(read_rel_1,
                                                  read_rel_0,
                                                  empty_relations(),
                                                  cond_1->clone(),
                                                  gqe::join_type_type::full,
                                                  projection_indices_1);
  std::unique_ptr<gqe::logical::join_relation> join_5 =
    std::make_unique<gqe::logical::join_relation>(read_rel_1,
                                                  read_rel_0,
                                                  empty_relations(),
                                                  cond_1->clone(),
                                                  gqe::join_type_type::full,
                                                  projection_indices_2);
  std::unique_ptr<gqe::logical::join_relation> join_6 =
    std::make_unique<gqe::logical::join_relation>(read_rel_1,
                                                  read_rel_0,
                                                  non_empty_relations(),
                                                  std::move(cond_1),
                                                  gqe::join_type_type::full,
                                                  projection_indices_2);
  EXPECT_FALSE(*join_0 == *join_1);  // different input order
  EXPECT_FALSE(*join_1 == *join_2);  // different condition
  EXPECT_FALSE(*join_2 == *join_3);  // different join type
  EXPECT_FALSE(*join_3 == *join_4);  // different projection indices
  EXPECT_FALSE(*join_4 == *join_5);  // different projection indices length
  EXPECT_FALSE(*join_5 == *join_6);  // different subquery relations
  EXPECT_EQ(*join_0, *join_0_dup);
}

TEST_F(RelationEqualTest, Project)
{
  std::vector<std::unique_ptr<gqe::expression>> expressions_0;
  expressions_0.push_back(col_0->clone());
  expressions_0.push_back(col_1->clone());

  std::vector<std::unique_ptr<gqe::expression>> expressions_1;
  expressions_1.push_back(col_1->clone());
  expressions_1.push_back(col_0->clone());

  auto project_0 = std::make_unique<gqe::logical::project_relation>(
    read_rel_0, empty_relations(), duplicate_expressions(expressions_0));
  auto project_1 = std::make_unique<gqe::logical::project_relation>(
    read_rel_1, empty_relations(), duplicate_expressions(expressions_0));
  auto project_2 = std::make_unique<gqe::logical::project_relation>(
    read_rel_1, empty_relations(), duplicate_expressions(expressions_1));
  auto project_3 = std::make_unique<gqe::logical::project_relation>(
    read_rel_1, non_empty_relations(), duplicate_expressions(expressions_1));
  auto project_0_dup = std::make_unique<gqe::logical::project_relation>(
    read_rel_0, empty_relations(), duplicate_expressions(expressions_0));

  EXPECT_FALSE(*project_0 == *project_1);  // different input relation
  EXPECT_FALSE(*project_1 == *project_2);  // different output expressions
  EXPECT_FALSE(*project_2 == *project_3);  // different subquery relations
  EXPECT_EQ(*project_0, *project_0_dup);
}

TEST_F(RelationEqualTest, Read)
{
  std::vector<std::string> col_names_2 = {"a", "c"};
  auto col_types_2 = {cudf::data_type(cudf::type_id::INT32), cudf::data_type(cudf::type_id::BOOL8)};
  auto read_rel_2  = std::make_shared<gqe::logical::read_relation>(
    non_empty_relations(), std::move(col_names_2), std::move(col_types_2), "t1", nullptr);
  EXPECT_FALSE(*read_rel_0 == *read_rel_1);
  EXPECT_FALSE(*read_rel_1 == *read_rel_2);
  EXPECT_EQ(*read_rel_0, *read_rel_0_dup);
}

TEST_F(RelationEqualTest, Set)
{
  auto set_0 = std::make_unique<gqe::logical::set_relation>(
    read_rel_0, read_rel_2, gqe::logical::set_relation::set_operator_type::set_minus);
  auto set_1 = std::make_unique<gqe::logical::set_relation>(
    read_rel_2, read_rel_0, gqe::logical::set_relation::set_operator_type::set_minus);
  auto set_2 = std::make_unique<gqe::logical::set_relation>(
    read_rel_2, read_rel_0, gqe::logical::set_relation::set_operator_type::set_union);
  auto set_0_dup = std::make_unique<gqe::logical::set_relation>(
    read_rel_0, read_rel_2, gqe::logical::set_relation::set_operator_type::set_minus);

  EXPECT_FALSE(*set_0 == *set_1);  // different input order
  EXPECT_FALSE(*set_1 == *set_2);  // different set operation
  EXPECT_EQ(*set_0, *set_0_dup);
}

TEST_F(RelationEqualTest, Sort)
{
  std::vector<std::unique_ptr<gqe::expression>> sort_exprs_0;
  std::vector<std::unique_ptr<gqe::expression>> sort_exprs_1;
  sort_exprs_0.push_back(col_0->clone());
  sort_exprs_0.push_back(col_1->clone());
  sort_exprs_1.push_back(col_1->clone());
  sort_exprs_1.push_back(col_0->clone());

  auto sort_0 = std::make_unique<gqe::logical::sort_relation>(
    read_rel_0,
    empty_relations(),
    std::vector<cudf::order>({cudf::order::ASCENDING, cudf::order::DESCENDING}),
    std::vector<cudf::null_order>({cudf::null_order::BEFORE, cudf::null_order::BEFORE}),
    duplicate_expressions(sort_exprs_0));
  auto sort_1 = std::make_unique<gqe::logical::sort_relation>(
    read_rel_0,
    empty_relations(),
    std::vector<cudf::order>({cudf::order::ASCENDING, cudf::order::ASCENDING}),
    std::vector<cudf::null_order>({cudf::null_order::BEFORE, cudf::null_order::BEFORE}),
    duplicate_expressions(sort_exprs_0));
  auto sort_2 = std::make_unique<gqe::logical::sort_relation>(
    read_rel_0,
    empty_relations(),
    std::vector<cudf::order>({cudf::order::ASCENDING, cudf::order::ASCENDING}),
    std::vector<cudf::null_order>({cudf::null_order::BEFORE, cudf::null_order::AFTER}),
    duplicate_expressions(sort_exprs_0));
  auto sort_3 = std::make_unique<gqe::logical::sort_relation>(
    read_rel_0,
    empty_relations(),
    std::vector<cudf::order>({cudf::order::ASCENDING, cudf::order::ASCENDING}),
    std::vector<cudf::null_order>({cudf::null_order::BEFORE, cudf::null_order::AFTER}),
    duplicate_expressions(sort_exprs_1));
  auto sort_4 = std::make_unique<gqe::logical::sort_relation>(
    read_rel_1,
    empty_relations(),
    std::vector<cudf::order>({cudf::order::ASCENDING, cudf::order::ASCENDING}),
    std::vector<cudf::null_order>({cudf::null_order::BEFORE, cudf::null_order::AFTER}),
    duplicate_expressions(sort_exprs_1));
  auto sort_5 = std::make_unique<gqe::logical::sort_relation>(
    read_rel_1,
    non_empty_relations(),
    std::vector<cudf::order>({cudf::order::ASCENDING, cudf::order::ASCENDING}),
    std::vector<cudf::null_order>({cudf::null_order::BEFORE, cudf::null_order::AFTER}),
    duplicate_expressions(sort_exprs_1));
  auto sort_0_dup = std::make_unique<gqe::logical::sort_relation>(
    read_rel_0,
    empty_relations(),
    std::vector<cudf::order>({cudf::order::ASCENDING, cudf::order::DESCENDING}),
    std::vector<cudf::null_order>({cudf::null_order::BEFORE, cudf::null_order::BEFORE}),
    duplicate_expressions(sort_exprs_0));

  EXPECT_FALSE(*sort_0 == *sort_1);  // different sort orders
  EXPECT_FALSE(*sort_1 == *sort_2);  // different null orders
  EXPECT_FALSE(*sort_2 == *sort_3);  // different sort expressions
  EXPECT_FALSE(*sort_3 == *sort_4);  // different input
  EXPECT_FALSE(*sort_4 == *sort_5);  // different subquery relations
  EXPECT_EQ(*sort_0, *sort_0_dup);
}

TEST_F(RelationEqualTest, Window)
{
  std::vector<std::unique_ptr<gqe::expression>> arguments_0;
  std::vector<std::unique_ptr<gqe::expression>> order_by_0;
  std::vector<std::unique_ptr<gqe::expression>> partition_by_0;

  std::vector<std::unique_ptr<gqe::expression>> arguments_1;
  std::vector<std::unique_ptr<gqe::expression>> order_by_1;
  std::vector<std::unique_ptr<gqe::expression>> partition_by_1;

  arguments_0.push_back(col_0->clone());
  order_by_0.push_back(col_0->clone());
  partition_by_0.push_back(col_1->clone());

  arguments_1.push_back(col_1->clone());
  order_by_1.push_back(col_1->clone());
  partition_by_1.push_back(col_0->clone());

  auto window_0 = std::make_unique<gqe::logical::window_relation>(
    read_rel_0,
    empty_relations(),
    cudf::aggregation::Kind::RANK,
    duplicate_expressions(arguments_0),
    duplicate_expressions(order_by_0),
    duplicate_expressions(partition_by_0),
    std::vector<cudf::order>({cudf::order::ASCENDING}),
    gqe::window_frame_bound::unbounded(),
    gqe::window_frame_bound::unbounded());
  auto window_1 = std::make_unique<gqe::logical::window_relation>(
    read_rel_1,
    empty_relations(),
    cudf::aggregation::Kind::RANK,
    duplicate_expressions(arguments_0),
    duplicate_expressions(order_by_0),
    duplicate_expressions(partition_by_0),
    std::vector<cudf::order>({cudf::order::ASCENDING}),
    gqe::window_frame_bound::unbounded(),
    gqe::window_frame_bound::unbounded());
  auto window_2 = std::make_unique<gqe::logical::window_relation>(
    read_rel_0,
    empty_relations(),
    cudf::aggregation::Kind::ROW_NUMBER,
    duplicate_expressions(arguments_0),
    duplicate_expressions(order_by_0),
    duplicate_expressions(partition_by_0),
    std::vector<cudf::order>({cudf::order::ASCENDING}),
    gqe::window_frame_bound::unbounded(),
    gqe::window_frame_bound::unbounded());
  auto window_3 = std::make_unique<gqe::logical::window_relation>(
    read_rel_0,
    empty_relations(),
    cudf::aggregation::Kind::RANK,
    duplicate_expressions(arguments_1),
    duplicate_expressions(order_by_1),
    duplicate_expressions(partition_by_1),
    std::vector<cudf::order>({cudf::order::ASCENDING}),
    gqe::window_frame_bound::unbounded(),
    gqe::window_frame_bound::unbounded());
  auto window_4 = std::make_unique<gqe::logical::window_relation>(
    read_rel_0,
    empty_relations(),
    cudf::aggregation::Kind::RANK,
    duplicate_expressions(arguments_0),
    duplicate_expressions(order_by_0),
    duplicate_expressions(partition_by_0),
    std::vector<cudf::order>({cudf::order::DESCENDING}),
    gqe::window_frame_bound::unbounded(),
    gqe::window_frame_bound::unbounded());
  auto window_5 = std::make_unique<gqe::logical::window_relation>(
    read_rel_0,
    empty_relations(),
    cudf::aggregation::Kind::RANK,
    duplicate_expressions(arguments_0),
    duplicate_expressions(order_by_0),
    duplicate_expressions(partition_by_0),
    std::vector<cudf::order>({cudf::order::DESCENDING}),
    gqe::window_frame_bound::bounded(1),
    gqe::window_frame_bound::unbounded());
  auto window_6 = std::make_unique<gqe::logical::window_relation>(
    read_rel_0,
    empty_relations(),
    cudf::aggregation::Kind::RANK,
    duplicate_expressions(arguments_0),
    duplicate_expressions(order_by_0),
    duplicate_expressions(partition_by_0),
    std::vector<cudf::order>({cudf::order::DESCENDING}),
    gqe::window_frame_bound::bounded(2),
    gqe::window_frame_bound::unbounded());
  auto window_7 = std::make_unique<gqe::logical::window_relation>(
    read_rel_0,
    non_empty_relations(),
    cudf::aggregation::Kind::RANK,
    duplicate_expressions(arguments_0),
    duplicate_expressions(order_by_0),
    duplicate_expressions(partition_by_0),
    std::vector<cudf::order>({cudf::order::DESCENDING}),
    gqe::window_frame_bound::bounded(2),
    gqe::window_frame_bound::unbounded());
  auto window_0_dup = std::make_unique<gqe::logical::window_relation>(
    read_rel_0,
    empty_relations(),
    cudf::aggregation::Kind::RANK,
    duplicate_expressions(arguments_0),
    duplicate_expressions(order_by_0),
    duplicate_expressions(partition_by_0),
    std::vector<cudf::order>({cudf::order::ASCENDING}),
    gqe::window_frame_bound::unbounded(),
    gqe::window_frame_bound::unbounded());
  EXPECT_FALSE(*window_0 == *window_1);  // different input
  EXPECT_FALSE(*window_1 == *window_2);  // different aggregate kind
  EXPECT_FALSE(*window_2 == *window_3);  // different arguments, order_by, partition_by
  EXPECT_FALSE(*window_3 == *window_4);  // different orders
  EXPECT_FALSE(*window_4 == *window_5);  // different bound types
  EXPECT_FALSE(*window_5 == *window_6);  // different bound values
  EXPECT_FALSE(*window_6 == *window_7);  // different subquery relations
  EXPECT_EQ(*window_0, *window_0_dup);
}

TEST_F(RelationEqualTest, Write)
{
  std::vector<std::string> col_names_0 = {"a", "b"};
  auto col_types_0                     = {cudf::data_type(cudf::type_id::INT32),
                                          cudf::data_type(cudf::type_id::FLOAT32)};
  auto write_0                         = std::make_shared<gqe::logical::write_relation>(
    read_rel_0, std::move(col_names_0), std::move(col_types_0), "t0");
  std::vector<std::string> col_names_1 = {"a", "c"};
  auto col_types_1 = {cudf::data_type(cudf::type_id::INT32), cudf::data_type(cudf::type_id::BOOL8)};
  auto write_1     = std::make_shared<gqe::logical::write_relation>(
    read_rel_1, std::move(col_names_1), std::move(col_types_1), "t0");
  std::vector<std::string> col_names_0_dup = {"a", "b"};
  auto col_types_0_dup                     = {cudf::data_type(cudf::type_id::INT32),
                                              cudf::data_type(cudf::type_id::FLOAT32)};
  auto write_0_dup                         = std::make_shared<gqe::logical::write_relation>(
    read_rel_0, std::move(col_names_0_dup), std::move(col_types_0_dup), "t0");
  EXPECT_FALSE(*write_0 == *write_1);
  EXPECT_EQ(*write_0, *write_0_dup);
}
