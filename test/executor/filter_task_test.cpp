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

#include <gqe/executor/filter.hpp>
#include <gqe/expression/column_reference.hpp>

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cstdint>
#include <memory>
#include <vector>

using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;
using bool_column_wrapper  = cudf::test::fixed_width_column_wrapper<bool>;

class FilterTest : public ::testing::Test {
 protected:
  void construct_filter_task()
  {
    constexpr int32_t stage_id       = 0;
    constexpr int32_t input_task_id  = 0;
    constexpr int32_t filter_task_id = 1;

    int64_column_wrapper input_col_0({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    int64_column_wrapper input_col_1({11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    bool_column_wrapper input_col_2(
      {true, false, true, false, true, false, true, false, true, false});

    std::vector<std::unique_ptr<cudf::column>> input_columns;
    input_columns.push_back(input_col_0.release());
    input_columns.push_back(input_col_1.release());
    input_columns.push_back(input_col_2.release());

    auto input_task = std::make_shared<gqe::test::executed_task>(
      input_task_id, stage_id, std::make_unique<cudf::table>(std::move(input_columns)));

    std::unique_ptr<gqe::expression> condition{
      std::make_unique<gqe::column_reference_expression>(2)};

    filter_task = std::make_unique<gqe::filter_task>(
      filter_task_id, stage_id, std::move(input_task), std::move(condition));
  }

  std::unique_ptr<gqe::filter_task> filter_task;
};

TEST_F(FilterTest, FilterOddsOnly)
{
  construct_filter_task();

  filter_task->execute();
  auto filter_result = filter_task->result();
  ASSERT_EQ(filter_result.has_value(), true);

  int64_column_wrapper ref_col_0({1, 3, 5, 7, 9});
  int64_column_wrapper ref_col_1({11, 13, 15, 17, 19});
  bool_column_wrapper ref_col_2({true, true, true, true, true});

  std::vector<std::unique_ptr<cudf::column>> ref_columns;
  ref_columns.push_back(ref_col_0.release());
  ref_columns.push_back(ref_col_1.release());
  ref_columns.push_back(ref_col_2.release());

  cudf::table ref_table(std::move(ref_columns));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(filter_result.value(), ref_table.view());
}