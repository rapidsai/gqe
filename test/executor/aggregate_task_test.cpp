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

#include <gqe/executor/aggregate.hpp>
#include <gqe/expression/column_reference.hpp>

#include <cudf/column/column.hpp>
#include <cudf/sorting.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cstdint>
#include <memory>
#include <vector>

class HandCodedValuesAggregationTest : public ::testing::Test {
 protected:
  void construct_aggregate_task(
    std::vector<std::unique_ptr<gqe::expression>> keys,
    std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values)
  {
    constexpr int32_t stage_id          = 0;
    constexpr int32_t input_task_id     = 0;
    constexpr int32_t aggregate_task_id = 1;

    cudf::test::strings_column_wrapper input_col_0(
      {"apple", "orange", "apple", "apple", "apple", "orange", "orange", "apple"},
      {true, false, true, true, false, true, true, true});
    cudf::test::fixed_width_column_wrapper<int32_t> input_col_1({0, 1, 0, 1, 1, 0, 1, 1});
    cudf::test::fixed_width_column_wrapper<int64_t> input_col_2(
      {0, 1, 2, 4, 8, 16, 32, 64}, {true, true, false, true, true, true, true, true});

    std::vector<std::unique_ptr<cudf::column>> input_columns;
    input_columns.push_back(input_col_0.release());
    input_columns.push_back(input_col_1.release());
    input_columns.push_back(input_col_2.release());

    auto input_task = std::make_shared<gqe::test::executed_task>(
      input_task_id, stage_id, std::make_unique<cudf::table>(std::move(input_columns)));

    aggregate_task = std::make_unique<gqe::aggregate_task>(
      aggregate_task_id, stage_id, std::move(input_task), std::move(keys), std::move(values));
  }

  std::unique_ptr<gqe::aggregate_task> aggregate_task;
};

TEST_F(HandCodedValuesAggregationTest, Reduction)
{
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::SUM,
                      std::make_unique<gqe::column_reference_expression>(1));
  values.emplace_back(cudf::aggregation::SUM,
                      std::make_unique<gqe::column_reference_expression>(2));
  values.emplace_back(cudf::aggregation::MAX,
                      std::make_unique<gqe::column_reference_expression>(2));

  construct_aggregate_task({}, std::move(values));
  aggregate_task->execute();

  cudf::test::fixed_width_column_wrapper<int64_t> ref_col_0({5});
  cudf::test::fixed_width_column_wrapper<int64_t> ref_col_1({125});
  cudf::test::fixed_width_column_wrapper<int64_t> ref_col_2({64});

  std::vector<std::unique_ptr<cudf::column>> ref_columns;
  ref_columns.push_back(ref_col_0.release());
  ref_columns.push_back(ref_col_1.release());
  ref_columns.push_back(ref_col_2.release());

  auto ref_table = std::make_unique<cudf::table>(std::move(ref_columns));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(aggregate_task->result().value(), ref_table->view());
}

TEST_F(HandCodedValuesAggregationTest, Groupby)
{
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(std::make_unique<gqe::column_reference_expression>(0));
  keys.push_back(std::make_unique<gqe::column_reference_expression>(1));

  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.push_back(
    std::make_pair(cudf::aggregation::SUM, std::make_unique<gqe::column_reference_expression>(2)));

  construct_aggregate_task(std::move(keys), std::move(values));
  aggregate_task->execute();
  auto aggregate_result_sorted = cudf::sort(aggregate_task->result().value());

  cudf::test::strings_column_wrapper ref_col_0({"", "apple", "apple", "orange", "orange"},
                                               {false, true, true, true, true});
  cudf::test::fixed_width_column_wrapper<int32_t> ref_col_1({1, 0, 1, 0, 1});
  cudf::test::fixed_width_column_wrapper<int64_t> ref_col_2({9, 0, 68, 16, 32});

  std::vector<std::unique_ptr<cudf::column>> ref_columns;
  ref_columns.push_back(ref_col_0.release());
  ref_columns.push_back(ref_col_1.release());
  ref_columns.push_back(ref_col_2.release());

  auto ref_table = std::make_unique<cudf::table>(std::move(ref_columns));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(aggregate_result_sorted->view(), ref_table->view());
}
