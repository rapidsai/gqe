/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/executor/groupby.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/column/column.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>

#include <random>

template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
void gen_keys_data(std::vector<T>& column)
{
  auto rand_gen = std::default_random_engine();
  std::uniform_int_distribution<T> dist(1, 100);

  std::generate(column.begin(), column.end(), [&rand_gen, &dist]() { return dist(rand_gen); });
}

template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
void gen_values_data(std::vector<T>& column)
{
  auto rand_gen = std::default_random_engine();
  std::uniform_int_distribution<T> dist(-10000, 10000);

  std::generate(column.begin(), column.end(), [&rand_gen, &dist]() { return dist(rand_gen); });
}

template <typename Keys, typename Results>
auto get_output_table(Keys agg_keys, Results agg_results)
{
  auto result_columns = agg_keys->release();

  for (auto& agg_result : agg_results) {
    assert(agg_result.results.size() == 1);
    result_columns.push_back(std::move(agg_result.results[0]));
  }

  return std::make_unique<cudf::table>(std::move(result_columns));
}

TEST(StressTest, Groupby)
{
  int num_of_rows = 100000;

  std::vector<int64_t> keys_data(num_of_rows);
  std::vector<int64_t> values_data(num_of_rows);

  gen_keys_data(keys_data);
  gen_values_data(values_data);

  cudf::test::fixed_width_column_wrapper<int64_t> keys_column(keys_data.begin(), keys_data.end());
  cudf::test::fixed_width_column_wrapper<int64_t> values_column(values_data.begin(),
                                                                values_data.end());

  auto keys_column_view   = cudf::column_view(keys_column);
  auto values_column_view = cudf::column_view(values_column);
  cudf::groupby::aggregation_request req;
  req.values = values_column_view;
  req.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  std::vector<cudf::groupby::aggregation_request> reqs;
  reqs.push_back(std::move(req));

  cudf::groupby::groupby cudf_groupby_obj(cudf::table_view(std::vector({keys_column_view})));
  auto [expected_agg_keys, expected_agg_results] = cudf_groupby_obj.aggregate(reqs);

  gqe::groupby::groupby gqe_groupby_obj(cudf::table_view(std::vector({keys_column_view})));
  auto [actual_agg_keys, actual_agg_results] = gqe_groupby_obj.aggregate(reqs, {});

  auto expected_result =
    get_output_table(std::move(expected_agg_keys), std::move(expected_agg_results));
  auto actual_result = get_output_table(std::move(actual_agg_keys), std::move(actual_agg_results));

  auto sorted_result_table   = cudf::sort(actual_result->view());
  auto sorted_expected_table = cudf::sort(expected_result->view());

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(sorted_expected_table->view(),
                                          sorted_result_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sorted_expected_table->view(), sorted_result_table->view());
}
