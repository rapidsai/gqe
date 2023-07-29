/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <cudf/column/column.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <gqe/executor/unique_key_inner_join.hpp>

#include <random>

template <typename T>
void gen_keys_data(std::vector<T>& left_keys_data, std::vector<T>& right_keys_data)
{
  auto rand_gen = std::default_random_engine();
  auto dist     = std::uniform_int_distribution<T>();

  // Left keys should be unique
  std::set<T> unique_numbers;
  while (unique_numbers.size() < 5) {
    unique_numbers.insert(dist(rand_gen));
  }
  std::copy(unique_numbers.begin(), unique_numbers.end(), std::back_inserter(left_keys_data));
  std::uniform_int_distribution<> dist_range(0, 10);

  // Right keys have some values from left_keys_data
  // others are randomly generated
  for (int i = 0; i < 5; i++) {
    int index = dist_range(rand_gen);
    if (index < 5)
      right_keys_data.push_back(left_keys_data[index]);
    else
      right_keys_data.push_back(dist(rand_gen));
  }
}

void make_tables_check_equality(cudf::column_view result_left_column,
                                cudf::column_view result_right_column,
                                std::vector<cudf::size_type>& expected_left_indices,
                                std::vector<cudf::size_type>& expected_right_indices)
{
  cudf::table_view result_table({result_left_column, result_right_column});
  auto sorted_result_table = cudf::sort(result_table);

  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_left_column(
    expected_left_indices.begin(), expected_left_indices.end());
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_right_column(
    expected_right_indices.begin(), expected_right_indices.end());

  cudf::table_view expected_table({expected_left_column, expected_right_column});

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_table, sorted_result_table->view());
}

TEST(UniqueKeyInnerJoinTest, Int64Test)
{
  std::vector<int64_t> left_keys_data;
  std::vector<int64_t> right_keys_data;

  gen_keys_data(left_keys_data, right_keys_data);

  cudf::test::fixed_width_column_wrapper<int64_t> left_keys_column(left_keys_data.begin(),
                                                                   left_keys_data.end());
  cudf::test::fixed_width_column_wrapper<int64_t> right_keys_column(right_keys_data.begin(),
                                                                    right_keys_data.end());

  cudf::table_view left_keys_table_view({left_keys_column});
  cudf::table_view right_keys_table_view({right_keys_column});

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_result_indices;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> right_result_indices;

  std::tie(left_result_indices, right_result_indices) = gqe::unique_key_inner_join(
    left_keys_table_view, right_keys_table_view, cudf::null_equality::UNEQUAL);

  auto result_left_column =
    std::make_unique<cudf::column>(std::move(*left_result_indices), rmm::device_buffer{}, 0);
  auto result_right_column =
    std::make_unique<cudf::column>(std::move(*right_result_indices), rmm::device_buffer{}, 0);

  std::vector<cudf::size_type> expected_left_indices  = {0, 0, 4, 4};
  std::vector<cudf::size_type> expected_right_indices = {0, 2, 1, 3};

  make_tables_check_equality(result_left_column->view(),
                             result_right_column->view(),
                             expected_left_indices,
                             expected_right_indices);
}

TEST(UniqueKeyInnerJoinTest, Int32Test)
{
  std::vector<int32_t> left_keys_data;
  std::vector<int32_t> right_keys_data;

  gen_keys_data(left_keys_data, right_keys_data);

  cudf::test::fixed_width_column_wrapper<int32_t> left_keys_column(left_keys_data.begin(),
                                                                   left_keys_data.end());
  cudf::test::fixed_width_column_wrapper<int32_t> right_keys_column(right_keys_data.begin(),
                                                                    right_keys_data.end());

  cudf::table_view left_keys_table_view({left_keys_column});
  cudf::table_view right_keys_table_view({right_keys_column});

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_result_indices;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> right_result_indices;

  std::tie(left_result_indices, right_result_indices) = gqe::unique_key_inner_join(
    left_keys_table_view, right_keys_table_view, cudf::null_equality::UNEQUAL);

  auto result_left_column =
    std::make_unique<cudf::column>(std::move(*left_result_indices), rmm::device_buffer{}, 0);
  auto result_right_column =
    std::make_unique<cudf::column>(std::move(*right_result_indices), rmm::device_buffer{}, 0);

  std::vector<cudf::size_type> expected_left_indices  = {4, 4};
  std::vector<cudf::size_type> expected_right_indices = {0, 2};

  make_tables_check_equality(result_left_column->view(),
                             result_right_column->view(),
                             expected_left_indices,
                             expected_right_indices);
}

TEST(UniqueKeyInnerJoinTest, SentinelTest)
{
  int64_t sentinel                     = std::numeric_limits<int64_t>::min();
  std::vector<int64_t> left_keys_data  = {sentinel, 1, 2, 3, 4, 5};
  std::vector<int64_t> right_keys_data = {sentinel, 3, 4, 6, 7, 8};

  cudf::test::fixed_width_column_wrapper<int64_t> left_keys_column(left_keys_data.begin(),
                                                                   left_keys_data.end());
  cudf::test::fixed_width_column_wrapper<int64_t> right_keys_column(right_keys_data.begin(),
                                                                    right_keys_data.end());

  cudf::table_view left_keys_table_view({left_keys_column});
  cudf::table_view right_keys_table_view({right_keys_column});

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_result_indices;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> right_result_indices;

  std::tie(left_result_indices, right_result_indices) = gqe::unique_key_inner_join(
    left_keys_table_view, right_keys_table_view, cudf::null_equality::UNEQUAL);
  auto result_left_column =
    std::make_unique<cudf::column>(std::move(*left_result_indices), rmm::device_buffer{}, 0);
  auto result_right_column =
    std::make_unique<cudf::column>(std::move(*right_result_indices), rmm::device_buffer{}, 0);

  std::vector<cudf::size_type> expected_left_indices  = {0, 3, 4};
  std::vector<cudf::size_type> expected_right_indices = {0, 1, 2};

  make_tables_check_equality(result_left_column->view(),
                             result_right_column->view(),
                             expected_left_indices,
                             expected_right_indices);
}

TEST(UniqueKeyInnerJoinTest, EqualNullTest)
{
  std::vector<int64_t> left_keys_data   = {0, 1, 2, 3, 4, 5};
  std::vector<int64_t> left_keys_valid  = {0, 1, 1, 1, 1, 1};
  std::vector<int64_t> right_keys_data  = {0, 3, 4, 6, 7, 8};
  std::vector<int64_t> right_keys_valid = {0, 1, 1, 1, 1, 1};

  cudf::test::fixed_width_column_wrapper<int64_t> left_keys_column(
    left_keys_data.begin(), left_keys_data.end(), left_keys_valid.begin());
  cudf::test::fixed_width_column_wrapper<int64_t> right_keys_column(
    right_keys_data.begin(), right_keys_data.end(), right_keys_valid.begin());

  cudf::table_view left_keys_table_view({left_keys_column});
  cudf::table_view right_keys_table_view({right_keys_column});

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_result_indices;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> right_result_indices;

  std::tie(left_result_indices, right_result_indices) = gqe::unique_key_inner_join(
    left_keys_table_view, right_keys_table_view, cudf::null_equality::EQUAL);
  auto result_left_column =
    std::make_unique<cudf::column>(std::move(*left_result_indices), rmm::device_buffer{}, 0);
  auto result_right_column =
    std::make_unique<cudf::column>(std::move(*right_result_indices), rmm::device_buffer{}, 0);

  std::vector<cudf::size_type> expected_left_indices  = {0, 3, 4};
  std::vector<cudf::size_type> expected_right_indices = {0, 1, 2};

  make_tables_check_equality(result_left_column->view(),
                             result_right_column->view(),
                             expected_left_indices,
                             expected_right_indices);
}

TEST(UniqueKeyInnerJoinTest, UnequalNullTest)
{
  std::vector<int64_t> left_keys_data   = {0, 1, 2, 3, 4, 5};
  std::vector<int64_t> left_keys_valid  = {0, 1, 1, 1, 1, 1};
  std::vector<int64_t> right_keys_data  = {0, 3, 4, 6, 7, 8};
  std::vector<int64_t> right_keys_valid = {0, 1, 1, 1, 1, 1};

  cudf::test::fixed_width_column_wrapper<int64_t> left_keys_column(
    left_keys_data.begin(), left_keys_data.end(), left_keys_valid.begin());
  cudf::test::fixed_width_column_wrapper<int64_t> right_keys_column(
    right_keys_data.begin(), right_keys_data.end(), right_keys_valid.begin());

  cudf::table_view left_keys_table_view({left_keys_column});
  cudf::table_view right_keys_table_view({right_keys_column});

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_result_indices;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> right_result_indices;

  std::tie(left_result_indices, right_result_indices) = gqe::unique_key_inner_join(
    left_keys_table_view, right_keys_table_view, cudf::null_equality::UNEQUAL);
  auto result_left_column =
    std::make_unique<cudf::column>(std::move(*left_result_indices), rmm::device_buffer{}, 0);
  auto result_right_column =
    std::make_unique<cudf::column>(std::move(*right_result_indices), rmm::device_buffer{}, 0);

  std::vector<cudf::size_type> expected_left_indices  = {3, 4};
  std::vector<cudf::size_type> expected_right_indices = {1, 2};

  make_tables_check_equality(result_left_column->view(),
                             result_right_column->view(),
                             expected_left_indices,
                             expected_right_indices);
}