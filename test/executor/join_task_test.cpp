/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/executor/join.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>

#include <cudf/column/column.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cstdint>
#include <memory>
#include <vector>

using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

/*
 * This test suite constructs the input tables with handpicked values. Both the left and the right
 * table have 2 columns, 1 key column and 1 payload column.
 */
class SingleKeyColumnJoinTest : public ::testing::Test {
 protected:
  void construct_join_task(gqe::join_type_type join_type,
                           std::vector<cudf::size_type> projection_indices)
  {
    int64_column_wrapper left_key({2, 1, 1, 3, 4, 1});
    int64_column_wrapper left_payload({0, 1, 2, 3, 4, 5});
    int64_column_wrapper right_key({3, 1, 5, 1, 2});
    int64_column_wrapper right_payload({0, 1, 2, 3, 4});

    std::vector<std::unique_ptr<cudf::column>> left_table_columns;
    left_table_columns.push_back(left_key.release());
    left_table_columns.push_back(left_payload.release());
    auto left_table = std::make_unique<cudf::table>(std::move(left_table_columns));

    std::vector<std::unique_ptr<cudf::column>> right_table_columns;
    right_table_columns.push_back(right_key.release());
    right_table_columns.push_back(right_payload.release());
    auto right_table = std::make_unique<cudf::table>(std::move(right_table_columns));

    constexpr int32_t left_task_id  = 0;
    constexpr int32_t right_task_id = 1;
    constexpr int32_t join_task_id  = 2;
    constexpr int32_t stage_id      = 0;

    auto left_task =
      std::make_shared<gqe::test::executed_task>(left_task_id, stage_id, std::move(left_table));
    auto right_task =
      std::make_shared<gqe::test::executed_task>(right_task_id, stage_id, std::move(right_table));
    auto join_condition = std::make_unique<gqe::equal_expression>(
      std::make_shared<gqe::column_reference_expression>(0),
      std::make_shared<gqe::column_reference_expression>(2));

    join_task = std::make_unique<gqe::join_task>(join_task_id,
                                                 stage_id,
                                                 left_task,
                                                 right_task,
                                                 join_type,
                                                 std::move(join_condition),
                                                 std::move(projection_indices));
  }

  std::unique_ptr<gqe::join_task> join_task;
};

TEST_F(SingleKeyColumnJoinTest, InnerJoin)
{
  construct_join_task(gqe::join_type_type::inner, {0, 1, 3});

  join_task->execute();

  auto join_result = join_task->result();
  ASSERT_EQ(join_result.has_value(), true);
  auto join_result_sorted = cudf::sort(*join_result);

  int64_column_wrapper ref_result_col0({1, 1, 1, 1, 1, 1, 2, 3});
  int64_column_wrapper ref_result_col1({1, 1, 2, 2, 5, 5, 0, 3});
  int64_column_wrapper ref_result_col2({1, 3, 1, 3, 1, 3, 4, 0});

  std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
  ref_result_columns.push_back(ref_result_col0.release());
  ref_result_columns.push_back(ref_result_col1.release());
  ref_result_columns.push_back(ref_result_col2.release());

  auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table->view());
}

TEST_F(SingleKeyColumnJoinTest, LeftJoin)
{
  construct_join_task(gqe::join_type_type::left, {0, 1, 3});

  join_task->execute();

  auto join_result = join_task->result();
  ASSERT_EQ(join_result.has_value(), true);
  auto join_result_sorted = cudf::sort(*join_result);

  int64_column_wrapper ref_result_col0({1, 1, 1, 1, 1, 1, 2, 3, 4});
  int64_column_wrapper ref_result_col1({1, 1, 2, 2, 5, 5, 0, 3, 4});
  int64_column_wrapper ref_result_col2({1, 3, 1, 3, 1, 3, 4, 0, 0},
                                       {true, true, true, true, true, true, true, true, false});

  std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
  ref_result_columns.push_back(ref_result_col0.release());
  ref_result_columns.push_back(ref_result_col1.release());
  ref_result_columns.push_back(ref_result_col2.release());

  auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table->view());
}

TEST_F(SingleKeyColumnJoinTest, LeftSemiJoin)
{
  construct_join_task(gqe::join_type_type::left_semi, {0, 1});

  join_task->execute();

  auto join_result = join_task->result();
  ASSERT_EQ(join_result.has_value(), true);
  auto join_result_sorted = cudf::sort(*join_result);

  int64_column_wrapper ref_result_col0({1, 1, 1, 2, 3});
  int64_column_wrapper ref_result_col1({1, 2, 5, 0, 3});

  std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
  ref_result_columns.push_back(ref_result_col0.release());
  ref_result_columns.push_back(ref_result_col1.release());

  auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table->view());
}

TEST_F(SingleKeyColumnJoinTest, LeftAntiJoin)
{
  construct_join_task(gqe::join_type_type::left_anti, {0, 1});

  join_task->execute();

  auto join_result = join_task->result();
  ASSERT_EQ(join_result.has_value(), true);
  auto join_result_sorted = cudf::sort(*join_result);

  int64_column_wrapper ref_result_col0({4});
  int64_column_wrapper ref_result_col1({4});

  std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
  ref_result_columns.push_back(ref_result_col0.release());
  ref_result_columns.push_back(ref_result_col1.release());

  auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table->view());
}

TEST_F(SingleKeyColumnJoinTest, FullJoin)
{
  construct_join_task(gqe::join_type_type::full, {0, 1, 2, 3});

  join_task->execute();

  auto join_result = join_task->result();
  ASSERT_EQ(join_result.has_value(), true);
  auto join_result_sorted = cudf::sort(*join_result);

  int64_column_wrapper ref_result_col0(
    {0, 1, 1, 1, 1, 1, 1, 2, 3, 4}, {false, true, true, true, true, true, true, true, true, true});
  int64_column_wrapper ref_result_col1(
    {0, 1, 1, 2, 2, 5, 5, 0, 3, 4}, {false, true, true, true, true, true, true, true, true, true});
  int64_column_wrapper ref_result_col2(
    {5, 1, 1, 1, 1, 1, 1, 2, 3, 0}, {true, true, true, true, true, true, true, true, true, false});
  int64_column_wrapper ref_result_col3(
    {2, 1, 3, 1, 3, 1, 3, 4, 0, 0}, {true, true, true, true, true, true, true, true, true, false});

  std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
  ref_result_columns.push_back(ref_result_col0.release());
  ref_result_columns.push_back(ref_result_col1.release());
  ref_result_columns.push_back(ref_result_col2.release());
  ref_result_columns.push_back(ref_result_col3.release());

  auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table->view());
}

class SingleKeyColumnNullsEqualJoinTest : public ::testing::Test {
 protected:
  void construct_join_task(gqe::join_type_type join_type,
                           std::vector<cudf::size_type> projection_indices,
                           bool nulls_equal)
  {
    int64_column_wrapper left_key({0, 1, 1, 3, 4, 1}, {false, true, true, true, true, true});
    int64_column_wrapper left_payload({0, 1, 2, 3, 4, 5});
    int64_column_wrapper right_key({3, 1, 5, 1, 0}, {true, true, true, true, false});
    int64_column_wrapper right_payload({0, 1, 2, 3, 4});

    std::vector<std::unique_ptr<cudf::column>> left_table_columns;
    left_table_columns.push_back(left_key.release());
    left_table_columns.push_back(left_payload.release());
    auto left_table = std::make_unique<cudf::table>(std::move(left_table_columns));

    std::vector<std::unique_ptr<cudf::column>> right_table_columns;
    right_table_columns.push_back(right_key.release());
    right_table_columns.push_back(right_payload.release());
    auto right_table = std::make_unique<cudf::table>(std::move(right_table_columns));

    constexpr int32_t left_task_id  = 0;
    constexpr int32_t right_task_id = 1;
    constexpr int32_t join_task_id  = 2;
    constexpr int32_t stage_id      = 0;

    auto left_task =
      std::make_shared<gqe::test::executed_task>(left_task_id, stage_id, std::move(left_table));
    auto right_task =
      std::make_shared<gqe::test::executed_task>(right_task_id, stage_id, std::move(right_table));
    std::unique_ptr<gqe::expression> join_condition;
    if (nulls_equal) {
      join_condition = std::make_unique<gqe::nulls_equal_expression>(
        std::make_shared<gqe::column_reference_expression>(0),
        std::make_shared<gqe::column_reference_expression>(2));
    } else {
      join_condition = std::make_unique<gqe::equal_expression>(
        std::make_shared<gqe::column_reference_expression>(0),
        std::make_shared<gqe::column_reference_expression>(2));
    }

    join_task = std::make_unique<gqe::join_task>(join_task_id,
                                                 stage_id,
                                                 left_task,
                                                 right_task,
                                                 join_type,
                                                 std::move(join_condition),
                                                 std::move(projection_indices));
  }

  std::unique_ptr<gqe::join_task> join_task;
};

TEST_F(SingleKeyColumnNullsEqualJoinTest, NullsEqual)
{
  construct_join_task(gqe::join_type_type::inner, {0, 1, 3}, true);

  join_task->execute();

  auto join_result = join_task->result();
  ASSERT_EQ(join_result.has_value(), true);
  auto join_result_sorted = cudf::sort(*join_result);

  int64_column_wrapper ref_result_col0({0, 1, 1, 1, 1, 1, 1, 3},
                                       {false, true, true, true, true, true, true, true});
  int64_column_wrapper ref_result_col1({0, 1, 1, 2, 2, 5, 5, 3});
  int64_column_wrapper ref_result_col2({4, 1, 3, 1, 3, 1, 3, 0});

  std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
  ref_result_columns.push_back(ref_result_col0.release());
  ref_result_columns.push_back(ref_result_col1.release());
  ref_result_columns.push_back(ref_result_col2.release());

  auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table->view());
}

TEST_F(SingleKeyColumnNullsEqualJoinTest, NullsNotEqual)
{
  construct_join_task(gqe::join_type_type::inner, {0, 1, 3}, false);

  join_task->execute();

  auto join_result = join_task->result();
  ASSERT_EQ(join_result.has_value(), true);
  auto join_result_sorted = cudf::sort(*join_result);

  int64_column_wrapper ref_result_col0({1, 1, 1, 1, 1, 1, 3});
  int64_column_wrapper ref_result_col1({1, 1, 2, 2, 5, 5, 3});
  int64_column_wrapper ref_result_col2({1, 3, 1, 3, 1, 3, 0});

  std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
  ref_result_columns.push_back(ref_result_col0.release());
  ref_result_columns.push_back(ref_result_col1.release());
  ref_result_columns.push_back(ref_result_col2.release());

  auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table->view());
}

// TODO: Add a test on multi column join keys
