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

#include <gqe/executor/groupby.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/aggregation.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace {

[[nodiscard]] std::vector<cudf::groupby::aggregation_request> make_min_requests(
  cudf::column_view values)
{
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
  aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.push_back(cudf::groupby::aggregation_request{values, std::move(aggregations)});
  return requests;
}

void expect_groupby_result(std::unique_ptr<cudf::table> const& result_keys,
                           std::vector<cudf::groupby::aggregation_result> const& result_values,
                           cudf::table_view expected)
{
  auto const actual =
    cudf::table_view{{result_keys->view().column(0), result_values[0].results[0]->view()}};
  auto sorted_actual   = cudf::sort(actual);
  auto sorted_expected = cudf::sort(expected);
  CUDF_TEST_EXPECT_TABLES_EQUAL(sorted_actual->view(), sorted_expected->view());
}

}  // namespace

TEST(GroupbyFallbackTest, ActiveMaskFiltersKeysAndValuesOnExplicitStream)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys{1, 1, 2, 2};
  cudf::test::strings_column_wrapper values{"z", "a", "x", "b"};
  cudf::test::fixed_width_column_wrapper<bool> active_mask{true, false, false, true};

  auto requests = make_min_requests(values);

  rmm::cuda_stream stream;
  rmm::mr::tracking_resource_adaptor result_mr{cudf::get_current_device_resource_ref()};
  gqe::groupby::groupby grouped{cudf::table_view{{keys}}};
  auto [result_keys, result_values] = grouped.aggregate(requests, active_mask, stream, result_mr);
  stream.synchronize();

  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{1, 2};
  cudf::test::strings_column_wrapper expected_values{"z", "b"};
  expect_groupby_result(
    result_keys, result_values, cudf::table_view{{expected_keys, expected_values}});
  EXPECT_TRUE(result_mr.get_outstanding_allocations().contains(
    const_cast<void*>(result_keys->view().column(0).head())));
}

TEST(GroupbyFallbackTest, EmptyActiveMaskIncludesAllRows)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys{1, 1, 2, 2};
  cudf::test::strings_column_wrapper values{"z", "a", "x", "b"};

  auto requests = make_min_requests(values);

  gqe::groupby::groupby grouped{cudf::table_view{{keys}}};
  auto [result_keys, result_values] =
    grouped.aggregate(requests, cudf::column_view{}, cudf::get_current_device_resource_ref());

  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{1, 2};
  cudf::test::strings_column_wrapper expected_values{"a", "b"};
  expect_groupby_result(
    result_keys, result_values, cudf::table_view{{expected_keys, expected_values}});
}

TEST(GroupbyFallbackTest, AllFalseMaskProducesEmptyResult)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys{1, 1, 2, 2};
  cudf::test::strings_column_wrapper values{"z", "a", "x", "b"};
  cudf::test::fixed_width_column_wrapper<bool> active_mask{false, false, false, false};

  auto requests = make_min_requests(values);

  gqe::groupby::groupby grouped{cudf::table_view{{keys}}};
  auto [result_keys, result_values] =
    grouped.aggregate(requests, active_mask, cudf::get_current_device_resource_ref());

  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{};
  cudf::test::strings_column_wrapper expected_values{};
  expect_groupby_result(
    result_keys, result_values, cudf::table_view{{expected_keys, expected_values}});
}

TEST(GroupbyFallbackTest, ZeroRowInputProducesEmptyResult)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys{};
  cudf::test::strings_column_wrapper values{};

  auto requests = make_min_requests(values);

  gqe::groupby::groupby grouped{cudf::table_view{{keys}}};
  auto [result_keys, result_values] =
    grouped.aggregate(requests, cudf::column_view{}, cudf::get_current_device_resource_ref());

  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{};
  cudf::test::strings_column_wrapper expected_values{};
  expect_groupby_result(
    result_keys, result_values, cudf::table_view{{expected_keys, expected_values}});
}
