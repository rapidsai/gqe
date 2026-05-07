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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>
#include <gtest/gtest.h>

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/types.hpp>

#include <unique_indices.hpp>

#include <vector>

struct unique_indices_test : public cudf::test::BaseFixture {};

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
class unique_indices_single_column_test : public unique_indices_test {};

TYPED_TEST_SUITE_P(unique_indices_single_column_test);

TYPED_TEST_P(unique_indices_single_column_test, basic_single_column)
{
  using T = TypeParam;

  // Test cases for different data types
  std::vector<std::vector<T>> test_cases = {
    // Original test case
    std::vector<T>{30, 20, 30, 10, 20, 20, 10, 30, 30, 20},
    // Additional test cases
    std::vector<T>{1, 2, 3, 4, 5},             // All unique
    std::vector<T>{1, 1, 1, 1, 1},             // All same
    std::vector<T>{5, 4, 3, 2, 1},             // Reverse order
    std::vector<T>{1, 2, 1, 2, 1, 2},          // Alternating pattern
    std::vector<T>{},                          // Empty case
    std::vector<T>{42},                        // Single element
    std::vector<T>{1, 2, 3, 2, 1, 3, 1, 2, 3}  // Complex pattern
  };

  for (const auto& keys_data : test_cases) {
    cudf::test::fixed_width_column_wrapper<T> keys(keys_data.begin(), keys_data.end());

    std::vector<cudf::column_view> key_columns{keys};
    cudf::column_view empty_mask;

    auto [unique_indices, group_indices] = libperfect::unique_indices(key_columns, empty_mask);

    // The length of the unique_indices should be the same as the number of unique elements in
    // keys_data.
    EXPECT_EQ(unique_indices.size(), std::set(keys_data.begin(), keys_data.end()).size());

    // Use the unique indices and group indices to reconstruct the original data.
    // It should be true that:
    // key_columns[i] == key_columns[unique_indices[group_indices[i]]] for all i 0 to keys.size()
    // - 1.
    auto unique_indices_table = cudf::table_view({to_column_view(unique_indices)});
    auto reconstructed_key_indices_table =
      cudf::gather(unique_indices_table, to_column_view(group_indices));
    auto key_indices_column       = reconstructed_key_indices_table->get_column(0);
    auto keys_table               = cudf::table_view({keys});
    auto reconstructed_keys_table = cudf::gather(keys_table, key_indices_column);

    CUDF_TEST_EXPECT_TABLES_EQUAL(keys_table, *reconstructed_keys_table);
  }
}

// Register test cases for different data types
REGISTER_TYPED_TEST_SUITE_P(unique_indices_single_column_test, basic_single_column);

// Define the data types to test
using TestTypes = ::testing::Types<int32_t, int64_t, float, double>;

// Instantiate tests for each data type
INSTANTIATE_TYPED_TEST_SUITE_P(DataTypeTests, unique_indices_single_column_test, TestTypes);

TEST_F(unique_indices_test, mixed_data_types_int_double)
{
  std::vector<int32_t> int_data{1, 2, 1, 2, 3, 1, 2, 4};
  std::vector<double> double_data{10.5, 20.5, 10.5, 20.5, 30.5, 10.5, 20.5, 30.5};

  // Test with mixed data types: int32_t and double
  cudf::test::fixed_width_column_wrapper<int32_t> int_keys(int_data.begin(), int_data.end());
  cudf::test::fixed_width_column_wrapper<double> double_keys(double_data.begin(),
                                                             double_data.end());

  std::vector<cudf::column_view> key_columns{int_keys, double_keys};
  cudf::column_view empty_mask;

  auto [unique_indices, group_indices] = libperfect::unique_indices(key_columns, empty_mask);

  // Calculate expected number of unique combinations
  std::set<std::pair<int32_t, double>> unique_combinations;

  for (size_t i = 0; i < int_data.size(); ++i) {
    unique_combinations.insert({int_data[i], double_data[i]});
  }

  EXPECT_EQ(unique_indices.size(), unique_combinations.size());

  // Use the unique indices and group indices to reconstruct the original data
  auto unique_indices_table = cudf::table_view({to_column_view(unique_indices)});
  auto reconstructed_key_indices_table =
    cudf::gather(unique_indices_table, to_column_view(group_indices));
  auto key_indices_column       = reconstructed_key_indices_table->get_column(0);
  auto keys_table               = cudf::table_view(key_columns);
  auto reconstructed_keys_table = cudf::gather(keys_table, key_indices_column);

  CUDF_TEST_EXPECT_TABLES_EQUAL(keys_table, *reconstructed_keys_table);
}

TEST_F(unique_indices_test, mixed_data_types_int_string)
{
  std::vector<int32_t> int_data{1, 2, 1, 2, 3, 1, 2, 4};
  std::vector<std::string> string_data{
    "apple", "banana", "apple", "banana", "cherry", "apple", "banana", "cherry"};

  // Test with mixed data types: int32_t and string
  cudf::test::strings_column_wrapper string_keys(string_data.begin(), string_data.end());
  cudf::test::fixed_width_column_wrapper<int32_t> int_keys(int_data.begin(), int_data.end());

  std::vector<cudf::column_view> key_columns{int_keys, string_keys};
  cudf::column_view empty_mask;

  // Expect an exception to be thrown because libperfect::unique_indices does not accept string
  // columns
  EXPECT_THROW(libperfect::unique_indices(key_columns, empty_mask), std::invalid_argument);
}
