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

#include "utilities.hpp"

#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/window.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/query_context.hpp>

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cstdint>
#include <memory>
#include <vector>

using int32_column_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;
using bool_column_wrapper  = cudf::test::fixed_width_column_wrapper<bool>;

class WindowOrderByPartitionBy : public ::testing::Test {
 protected:
  void construct_window_task()
  {
    constexpr int32_t stage_id       = 0;
    constexpr int32_t input_task_id  = 0;
    constexpr int32_t filter_task_id = 1;

    gqe::query_context qctx(gqe::optimization_parameters(true));

    int64_column_wrapper input_col_0({1, 1, 2, 1, 2, 2, 1});
    int64_column_wrapper input_col_1({1, 2, 4, 3, 5, 6, 2});
    int32_column_wrapper input_col_2({6, 5, 4, 3, 2, 1, 7});
    int64_column_wrapper input_col_3({0, 1, 2, 3, 4, 5, 6});

    std::vector<std::unique_ptr<cudf::column>> input_columns;
    input_columns.push_back(input_col_0.release());
    input_columns.push_back(input_col_1.release());
    input_columns.push_back(input_col_2.release());
    input_columns.push_back(input_col_3.release());

    auto input_task = std::make_shared<gqe::test::executed_task>(
      &qctx, input_task_id, stage_id, std::make_unique<cudf::table>(std::move(input_columns)));

    std::unique_ptr<gqe::expression> arguments{
      std::make_unique<gqe::column_reference_expression>(1)};
    std::vector<std::unique_ptr<gqe::expression>> arguments_vec;
    arguments_vec.push_back(std::move(arguments));

    std::unique_ptr<gqe::expression> partition_by{
      std::make_unique<gqe::column_reference_expression>(0)};
    std::vector<std::unique_ptr<gqe::expression>> partition_by_vec;
    partition_by_vec.push_back(std::move(partition_by));

    std::unique_ptr<gqe::expression> order_by{
      std::make_unique<gqe::column_reference_expression>(2)};
    std::vector<std::unique_ptr<gqe::expression>> order_by_vec;
    order_by_vec.push_back(std::move(order_by));

    std::unique_ptr<gqe::expression> ident_col{
      std::make_unique<gqe::column_reference_expression>(3)};
    std::vector<std::unique_ptr<gqe::expression>> ident_col_vec;
    ident_col_vec.push_back(std::move(ident_col));

    std::vector<cudf::order> order_dirs;
    order_dirs.push_back(cudf::order::ASCENDING);

    gqe::window_frame_bound::unbounded window_lower_bound;
    gqe::window_frame_bound::bounded window_upper_bound(0);

    window_task = std::make_unique<gqe::window_task>(&qctx,
                                                     filter_task_id,
                                                     stage_id,
                                                     std::move(input_task),
                                                     cudf::aggregation::Kind::SUM,
                                                     std::move(ident_col_vec),
                                                     std::move(arguments_vec),
                                                     std::move(partition_by_vec),
                                                     std::move(order_by_vec),
                                                     std::move(order_dirs),
                                                     window_lower_bound,
                                                     window_upper_bound);
  }

  std::unique_ptr<gqe::window_task> window_task;
};

class WindowOrderBy : public ::testing::Test {
 protected:
  void construct_window_task()
  {
    constexpr int32_t stage_id       = 0;
    constexpr int32_t input_task_id  = 0;
    constexpr int32_t filter_task_id = 1;

    gqe::query_context qctx(gqe::optimization_parameters(true));

    int64_column_wrapper input_col_0({1, 2, 3, 4, 5, 6, 2});
    int32_column_wrapper input_col_1({6, 5, 4, 3, 2, 1, 7});
    int64_column_wrapper input_col_2({0, 1, 2, 3, 4, 5, 6});

    std::vector<std::unique_ptr<cudf::column>> input_columns;
    input_columns.push_back(input_col_0.release());
    input_columns.push_back(input_col_1.release());
    input_columns.push_back(input_col_2.release());

    auto input_task = std::make_shared<gqe::test::executed_task>(
      &qctx, input_task_id, stage_id, std::make_unique<cudf::table>(std::move(input_columns)));

    std::unique_ptr<gqe::expression> arguments{
      std::make_unique<gqe::column_reference_expression>(0)};
    std::vector<std::unique_ptr<gqe::expression>> arguments_vec;
    arguments_vec.push_back(std::move(arguments));

    std::vector<std::unique_ptr<gqe::expression>> partition_by_vec;

    std::unique_ptr<gqe::expression> order_by{
      std::make_unique<gqe::column_reference_expression>(1)};
    std::vector<std::unique_ptr<gqe::expression>> order_by_vec;
    order_by_vec.push_back(std::move(order_by));

    std::unique_ptr<gqe::expression> ident_col{
      std::make_unique<gqe::column_reference_expression>(2)};
    std::vector<std::unique_ptr<gqe::expression>> ident_col_vec;
    ident_col_vec.push_back(std::move(ident_col));

    std::vector<cudf::order> order_dirs;
    order_dirs.push_back(cudf::order::ASCENDING);

    gqe::window_frame_bound::unbounded window_lower_bound;
    gqe::window_frame_bound::bounded window_upper_bound(0);

    window_task = std::make_unique<gqe::window_task>(&qctx,
                                                     filter_task_id,
                                                     stage_id,
                                                     std::move(input_task),
                                                     cudf::aggregation::Kind::SUM,
                                                     std::move(ident_col_vec),
                                                     std::move(arguments_vec),
                                                     std::move(partition_by_vec),
                                                     std::move(order_by_vec),
                                                     std::move(order_dirs),
                                                     window_lower_bound,
                                                     window_upper_bound);
  }

  std::unique_ptr<gqe::window_task> window_task;
};

class WindowOrderByPartitionByRank : public ::testing::Test {
 protected:
  void construct_window_task()
  {
    constexpr int32_t stage_id       = 0;
    constexpr int32_t input_task_id  = 0;
    constexpr int32_t filter_task_id = 1;

    gqe::query_context qctx(gqe::optimization_parameters(true));

    int64_column_wrapper input_col_0({1, 1, 2, 3, 2, 2, 1, 1});
    int32_column_wrapper input_col_1({5, 5, 4, 3, 2, 1, 0, 6});
    int64_column_wrapper input_col_2({0, 1, 2, 3, 4, 5, 6, 7});

    std::vector<std::unique_ptr<cudf::column>> input_columns;
    input_columns.push_back(input_col_0.release());
    input_columns.push_back(input_col_1.release());
    input_columns.push_back(input_col_2.release());

    auto input_task = std::make_shared<gqe::test::executed_task>(
      &qctx, input_task_id, stage_id, std::make_unique<cudf::table>(std::move(input_columns)));

    std::vector<std::unique_ptr<gqe::expression>> arguments_vec;

    std::unique_ptr<gqe::expression> partition_by{
      std::make_unique<gqe::column_reference_expression>(0)};
    std::vector<std::unique_ptr<gqe::expression>> partition_by_vec;
    partition_by_vec.push_back(std::move(partition_by));

    std::unique_ptr<gqe::expression> order_by{
      std::make_unique<gqe::column_reference_expression>(1)};
    std::vector<std::unique_ptr<gqe::expression>> order_by_vec;
    order_by_vec.push_back(std::move(order_by));

    std::unique_ptr<gqe::expression> ident_col{
      std::make_unique<gqe::column_reference_expression>(2)};
    std::vector<std::unique_ptr<gqe::expression>> ident_col_vec;
    ident_col_vec.push_back(std::move(ident_col));

    std::vector<cudf::order> order_dirs;
    order_dirs.push_back(cudf::order::ASCENDING);

    gqe::window_frame_bound::unbounded window_lower_bound;
    gqe::window_frame_bound::bounded window_upper_bound(0);

    window_task = std::make_unique<gqe::window_task>(&qctx,
                                                     filter_task_id,
                                                     stage_id,
                                                     std::move(input_task),
                                                     cudf::aggregation::Kind::RANK,
                                                     std::move(ident_col_vec),
                                                     std::move(arguments_vec),
                                                     std::move(partition_by_vec),
                                                     std::move(order_by_vec),
                                                     std::move(order_dirs),
                                                     window_lower_bound,
                                                     window_upper_bound);
  }

  std::unique_ptr<gqe::window_task> window_task;
};

class WindowOrderByRank : public ::testing::Test {
 protected:
  void construct_window_task()
  {
    constexpr int32_t stage_id       = 0;
    constexpr int32_t input_task_id  = 0;
    constexpr int32_t filter_task_id = 1;

    gqe::query_context qctx(gqe::optimization_parameters(true));

    int32_column_wrapper input_col_0({5, 5, 4, 3, 2, 1, 0, 6});
    int64_column_wrapper input_col_1({0, 1, 2, 3, 4, 5, 6, 7});

    std::vector<std::unique_ptr<cudf::column>> input_columns;
    input_columns.push_back(input_col_0.release());
    input_columns.push_back(input_col_1.release());

    auto input_task = std::make_shared<gqe::test::executed_task>(
      &qctx, input_task_id, stage_id, std::make_unique<cudf::table>(std::move(input_columns)));

    std::vector<std::unique_ptr<gqe::expression>> arguments_vec;

    std::vector<std::unique_ptr<gqe::expression>> partition_by_vec;

    std::unique_ptr<gqe::expression> order_by{
      std::make_unique<gqe::column_reference_expression>(0)};
    std::vector<std::unique_ptr<gqe::expression>> order_by_vec;
    order_by_vec.push_back(std::move(order_by));

    std::unique_ptr<gqe::expression> ident_col{
      std::make_unique<gqe::column_reference_expression>(1)};
    std::vector<std::unique_ptr<gqe::expression>> ident_col_vec;
    ident_col_vec.push_back(std::move(ident_col));

    std::vector<cudf::order> order_dirs;
    order_dirs.push_back(cudf::order::ASCENDING);

    gqe::window_frame_bound::unbounded window_lower_bound;
    gqe::window_frame_bound::bounded window_upper_bound(0);

    window_task = std::make_unique<gqe::window_task>(&qctx,
                                                     filter_task_id,
                                                     stage_id,
                                                     std::move(input_task),
                                                     cudf::aggregation::Kind::RANK,
                                                     std::move(ident_col_vec),
                                                     std::move(arguments_vec),
                                                     std::move(partition_by_vec),
                                                     std::move(order_by_vec),
                                                     std::move(order_dirs),
                                                     window_lower_bound,
                                                     window_upper_bound);
  }

  std::unique_ptr<gqe::window_task> window_task;
};

TEST_F(WindowOrderByPartitionBy, WindowOrderByPartitionBy)
{
  construct_window_task();

  window_task->execute();
  auto window_result = window_task->result();
  ASSERT_EQ(window_result.has_value(), true);

  int64_column_wrapper ref_col_0({3, 1, 0, 6, 5, 4, 2});
  int64_column_wrapper ref_col_1({3, 5, 6, 8, 6, 11, 15});

  std::vector<std::unique_ptr<cudf::column>> ref_columns;
  ref_columns.push_back(ref_col_0.release());
  ref_columns.push_back(ref_col_1.release());

  cudf::table ref_table(std::move(ref_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(window_result.value(), ref_table.view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(window_result.value(), ref_table.view());
}

TEST_F(WindowOrderBy, WindowOrderBy)
{
  construct_window_task();

  window_task->execute();
  auto window_result = window_task->result();
  ASSERT_EQ(window_result.has_value(), true);

  int64_column_wrapper ref_col_0({5, 4, 3, 2, 1, 0, 6});
  int64_column_wrapper ref_col_1({6, 11, 15, 18, 20, 21, 23});

  std::vector<std::unique_ptr<cudf::column>> ref_columns;
  ref_columns.push_back(ref_col_0.release());
  ref_columns.push_back(ref_col_1.release());

  cudf::table ref_table(std::move(ref_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(window_result.value(), ref_table.view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(window_result.value(), ref_table.view());
}

TEST_F(WindowOrderByPartitionByRank, WindowOrderByPartitionByRank)
{
  construct_window_task();

  window_task->execute();
  auto window_result = window_task->result();
  ASSERT_EQ(window_result.has_value(), true);

  int64_column_wrapper ref_col_0({6, 0, 1, 7, 5, 4, 2, 3});
  int32_column_wrapper ref_col_1({1, 2, 2, 4, 1, 2, 3, 1});

  std::vector<std::unique_ptr<cudf::column>> ref_columns;
  ref_columns.push_back(ref_col_0.release());
  ref_columns.push_back(ref_col_1.release());

  cudf::table ref_table(std::move(ref_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(window_result.value(), ref_table.view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(window_result.value(), ref_table.view());
}

TEST_F(WindowOrderByRank, WindowOrderByRank)
{
  construct_window_task();

  window_task->execute();
  auto window_result = window_task->result();
  ASSERT_EQ(window_result.has_value(), true);

  // 0 comes before 1 in ref_col_0 due to stable sorting order
  int64_column_wrapper ref_col_0({6, 5, 4, 3, 2, 0, 1, 7});
  int32_column_wrapper ref_col_1({1, 2, 3, 4, 5, 6, 6, 8});

  std::vector<std::unique_ptr<cudf::column>> ref_columns;
  ref_columns.push_back(ref_col_0.release());
  ref_columns.push_back(ref_col_1.release());

  cudf::table ref_table(std::move(ref_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(window_result.value(), ref_table.view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(window_result.value(), ref_table.view());
}
