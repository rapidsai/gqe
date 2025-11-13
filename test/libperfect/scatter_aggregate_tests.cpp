/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/types.hpp>

#include <gtest/gtest.h>

#include <scatter_aggregate.hpp>

#include <vector>

struct scatter_aggregate_test : public cudf::test::BaseFixture {};

template <typename T>
static cudf::column_view to_column_view(const rmm::device_uvector<T>& rmm_device_uvector)
{
  return cudf::column_view(cudf::data_type(cudf::type_to_id<T>()),
                           rmm_device_uvector.size(),
                           rmm_device_uvector.data(),
                           nullptr,
                           0);
}

// Template for parameterized test with different data types
template <typename T>
class scatter_aggregate_single_column_test : public scatter_aggregate_test {};

TYPED_TEST_SUITE_P(scatter_aggregate_single_column_test);

TYPED_TEST_P(scatter_aggregate_single_column_test, basic_single_column)
{
  using T = TypeParam;

  // Test cases for different data types
  std::vector<std::pair<std::vector<T>, std::vector<cudf::size_type>>> test_cases = {
    {{30, 20, 30, 10, 20, 20, 10, 30, 30, 20}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {{}, {}},
    {{30, 20, 30, 10, 20, 20, 10, 30, 30, 20}, {0, 1, 2, 3, 2, 1, 0, 0, 2, 2}}

  };

  std::vector<std::tuple<cudf::aggregation::Kind, T, T (*)(T, T)>> aggregation_kinds = {
    // triple of cudf aggregation, initial value, and binary operation on accumulator and value.
    {cudf::aggregation::SUM, 0, [](T a, T b) { return a + b; }},
    {cudf::aggregation::PRODUCT, 1, [](T a, T b) { return a * b; }},
    {cudf::aggregation::MIN,
     std::numeric_limits<T>::max(),
     [](T a, T b) { return std::min(a, b); }},
    {cudf::aggregation::COUNT_VALID, 0, [](T a, T b) { return a + 1; }},
    {cudf::aggregation::COUNT_ALL, 0, [](T a, T b) { return a + 1; }}};

  for (const auto& [aggregation_kind, initial_value, binary_operation] : aggregation_kinds) {
    for (const auto& [keys_data, groups_data] : test_cases) {
      cudf::test::fixed_width_column_wrapper<T> keys(keys_data.begin(), keys_data.end());
      rmm::device_uvector<cudf::size_type> group_ids(groups_data.size(), rmm::cuda_stream_default);
      cudaMemcpy(group_ids.data(),
                 groups_data.data(),
                 groups_data.size() * sizeof(cudf::size_type),
                 cudaMemcpyHostToDevice);
      cudf::column_view empty_mask;
      cudf::size_type groups_count =
        groups_data.size() > 0 ? *std::max_element(groups_data.begin(), groups_data.end()) + 1 : 0;
      cudf::type_id output_type_id = cudf::type_to_id<T>();
      auto result                  = libperfect::scatter_aggregate(
        keys, group_ids, empty_mask, std::nullopt, aggregation_kind, groups_count, output_type_id);
      auto [host_result, _] = cudf::test::to_host<T>(result);
      for (int i = 0; i < groups_count; i++) {
        T expected_result = initial_value;
        for (uint j = 0; j < groups_data.size(); j++) {
          if (groups_data[j] == i) {
            expected_result = binary_operation(expected_result, keys_data[j]);
          }
        }
        EXPECT_EQ(host_result[i], expected_result);
      }
    }
  }
}

// Register test cases for different data types
REGISTER_TYPED_TEST_SUITE_P(scatter_aggregate_single_column_test, basic_single_column);

// Define the data types to test
using TestTypes = ::testing::Types<int32_t, int64_t, float, double>;

// Instantiate tests for each data type
INSTANTIATE_TYPED_TEST_SUITE_P(DataTypeTests, scatter_aggregate_single_column_test, TestTypes);

TEST_F(scatter_aggregate_test, string_input_throws_exception)
{
  // Test with string input - should throw an exception
  std::vector<std::string> string_data{"apple", "banana", "apple", "banana", "cherry"};
  std::vector<cudf::size_type> groups_data{0, 1, 0, 1, 2};

  cudf::test::strings_column_wrapper string_keys(string_data.begin(), string_data.end());
  rmm::device_uvector<cudf::size_type> group_ids(groups_data.size(), rmm::cuda_stream_default);
  cudaMemcpy(group_ids.data(),
             groups_data.data(),
             groups_data.size() * sizeof(cudf::size_type),
             cudaMemcpyHostToDevice);

  cudf::column_view empty_mask;
  cudf::size_type groups_count =
    groups_data.size() > 0 ? *std::max_element(groups_data.begin(), groups_data.end()) + 1 : 0;
  cudf::type_id output_type_id = cudf::type_to_id<std::string>();

  // Expect an exception to be thrown because libperfect::scatter_aggregate does not accept string
  // columns
  EXPECT_THROW(libperfect::scatter_aggregate(string_keys,
                                             group_ids,
                                             empty_mask,
                                             std::nullopt,
                                             cudf::aggregation::SUM,
                                             groups_count,
                                             output_type_id),
               std::invalid_argument)
    << "Expected an exception to be thrown because libperfect::scatter_aggregate does not accept "
       "string columns";
}
