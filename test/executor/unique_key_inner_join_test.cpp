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

#include <cudf/join.hpp>

#include <cudf/column/column.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <gqe/executor/unique_key_inner_join.hpp>

#include <random>

using TestType =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

template <typename T, typename RandDist>
void gen_keys_from_dist(std::vector<T>& build_keys_data,
                        std::vector<T>& probe_keys_data,
                        int num_keys,
                        RandDist dist)
{
  auto rand_gen = std::default_random_engine();

  // build keys should be unique
  std::set<T> unique_numbers;
  while (static_cast<int>(unique_numbers.size()) < num_keys) {
    unique_numbers.insert(dist(rand_gen));
  }
  std::copy(unique_numbers.begin(), unique_numbers.end(), std::back_inserter(build_keys_data));
  std::uniform_int_distribution<> dist_range(0, num_keys * 2);

  // probe keys have some values from build_keys_data
  // others are randomly generated
  for (int i = 0; i < num_keys; i++) {
    int index = dist_range(rand_gen);
    if (index < static_cast<int>(num_keys))
      probe_keys_data.push_back(build_keys_data[index]);
    else
      probe_keys_data.push_back(dist(rand_gen));
  }
}

template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
void gen_keys_data(std::vector<T>& build_keys_data, std::vector<T>& probe_keys_data, int num_keys)
{
  auto dist = std::uniform_int_distribution<T>();

  gen_keys_from_dist(build_keys_data, probe_keys_data, num_keys, dist);
}

template <typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
void gen_keys_data(std::vector<T>& build_keys_data, std::vector<T>& probe_keys_data, int num_keys)
{
  auto dist = std::uniform_real_distribution<T>();

  gen_keys_from_dist(build_keys_data, probe_keys_data, num_keys, dist);
}

void make_tables_check_equality(cudf::column_view result_build_column,
                                cudf::column_view result_probe_column,
                                cudf::column_view expected_build_column,
                                cudf::column_view expected_probe_column)
{
  cudf::table_view result_table({result_build_column, result_probe_column});
  auto sorted_result_table = cudf::sort(result_table);

  cudf::table_view expected_table({expected_build_column, expected_probe_column});
  auto sorted_expected_table = cudf::sort(expected_table);

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(sorted_expected_table->view(),
                                          sorted_result_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sorted_expected_table->view(), sorted_result_table->view());
}

void check_equality(rmm::device_uvector<cudf::size_type>* build_result_indices,
                    rmm::device_uvector<cudf::size_type>* probe_result_indices,
                    std::vector<cudf::size_type> build_expected_indices,
                    std::vector<cudf::size_type> probe_expected_indices)
{
  cudf::column_view result_build_column_view(cudf::data_type{cudf::type_id::INT32},
                                             build_result_indices->size(),
                                             build_result_indices->data(),
                                             nullptr,
                                             0);
  cudf::column_view result_probe_column_view(cudf::data_type{cudf::type_id::INT32},
                                             probe_result_indices->size(),
                                             probe_result_indices->data(),
                                             nullptr,
                                             0);

  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_build_column(
    build_expected_indices.begin(), build_expected_indices.end());
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_probe_column(
    probe_expected_indices.begin(), probe_expected_indices.end());

  cudf::column_view expected_build_column_view = expected_build_column;
  cudf::column_view expected_probe_column_view = expected_probe_column;

  make_tables_check_equality(result_build_column_view,
                             result_probe_column_view,
                             expected_build_column_view,
                             expected_probe_column_view);
}

void check_equality(rmm::device_uvector<cudf::size_type>* build_result_indices,
                    rmm::device_uvector<cudf::size_type>* probe_result_indices,
                    rmm::device_uvector<cudf::size_type>* build_expected_indices,
                    rmm::device_uvector<cudf::size_type>* probe_expected_indices)
{
  cudf::column_view result_build_column_view(cudf::data_type{cudf::type_id::INT32},
                                             build_result_indices->size(),
                                             build_result_indices->data(),
                                             nullptr,
                                             0);
  cudf::column_view result_probe_column_view(cudf::data_type{cudf::type_id::INT32},
                                             probe_result_indices->size(),
                                             probe_result_indices->data(),
                                             nullptr,
                                             0);
  cudf::column_view expected_build_column_view(cudf::data_type{cudf::type_id::INT32},
                                               build_expected_indices->size(),
                                               build_expected_indices->data(),
                                               nullptr,
                                               0);
  cudf::column_view expected_probe_column_view(cudf::data_type{cudf::type_id::INT32},
                                               probe_expected_indices->size(),
                                               probe_expected_indices->data(),
                                               nullptr,
                                               0);
  make_tables_check_equality(result_build_column_view,
                             result_probe_column_view,
                             expected_build_column_view,
                             expected_probe_column_view);
}

template <typename T>
struct UniqueKeyInnerJoinTest : public cudf::test::BaseFixture {
};

TYPED_TEST_SUITE(UniqueKeyInnerJoinTest, TestType);

TYPED_TEST(UniqueKeyInnerJoinTest, Basic)
{
  using T      = TypeParam;
  int num_keys = 100;
  std::vector<T> build_keys_data;
  std::vector<T> probe_keys_data;

  gen_keys_data(build_keys_data, probe_keys_data, num_keys);

  cudf::test::fixed_width_column_wrapper<T> build_keys_column(build_keys_data.begin(),
                                                              build_keys_data.end());
  cudf::test::fixed_width_column_wrapper<T> probe_keys_column(probe_keys_data.begin(),
                                                              probe_keys_data.end());

  cudf::table_view build_keys_table_view({build_keys_column});
  cudf::table_view probe_keys_table_view({probe_keys_column});

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> build_result_indices;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> probe_result_indices;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> build_expected_indices;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> probe_expected_indices;

  std::tie(build_result_indices, probe_result_indices) = gqe::unique_key_inner_join(
    build_keys_table_view, probe_keys_table_view, cudf::null_equality::UNEQUAL);

  std::tie(build_expected_indices, probe_expected_indices) =
    cudf::inner_join(build_keys_table_view, probe_keys_table_view, cudf::null_equality::UNEQUAL);

  check_equality(build_result_indices.get(),
                 probe_result_indices.get(),
                 build_expected_indices.get(),
                 probe_expected_indices.get());
}

TYPED_TEST(UniqueKeyInnerJoinTest, Multicol)
{
  using T      = TypeParam;
  int num_keys = 100;
  std::vector<T> build_keys_col_0_data;
  std::vector<T> probe_keys_col_0_data;

  std::vector<T> build_keys_col_1_data;
  std::vector<T> probe_keys_col_1_data;

  gen_keys_data(build_keys_col_0_data, probe_keys_col_0_data, num_keys);
  gen_keys_data(build_keys_col_1_data, probe_keys_col_1_data, num_keys);

  cudf::test::fixed_width_column_wrapper<T> build_keys_col_0(build_keys_col_0_data.begin(),
                                                             build_keys_col_0_data.end());
  cudf::test::fixed_width_column_wrapper<T> build_keys_col_1(build_keys_col_1_data.begin(),
                                                             build_keys_col_1_data.end());
  cudf::test::fixed_width_column_wrapper<T> probe_keys_col_0(probe_keys_col_0_data.begin(),
                                                             probe_keys_col_0_data.end());
  cudf::test::fixed_width_column_wrapper<T> probe_keys_col_1(probe_keys_col_1_data.begin(),
                                                             probe_keys_col_1_data.end());

  cudf::table_view build_keys_table_view({build_keys_col_0, build_keys_col_1});
  cudf::table_view probe_keys_table_view({probe_keys_col_0, probe_keys_col_1});

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> build_result_indices;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> probe_result_indices;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> build_expected_indices;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> probe_expected_indices;

  std::tie(build_result_indices, probe_result_indices) = gqe::unique_key_inner_join(
    build_keys_table_view, probe_keys_table_view, cudf::null_equality::UNEQUAL);

  std::tie(build_expected_indices, probe_expected_indices) =
    cudf::inner_join(build_keys_table_view, probe_keys_table_view, cudf::null_equality::UNEQUAL);

  check_equality(build_result_indices.get(),
                 probe_result_indices.get(),
                 build_expected_indices.get(),
                 probe_expected_indices.get());
}

TEST(UniqueKeyInnerJoinTest, EqualNullTest)
{
  std::vector<int64_t> build_keys_data  = {0, 1, 2, 3, 4, 5};
  std::vector<int64_t> build_keys_valid = {0, 1, 1, 1, 1, 1};
  std::vector<int64_t> probe_keys_data  = {0, 3, 4, 6, 7, 8};
  std::vector<int64_t> probe_keys_valid = {0, 1, 1, 1, 1, 1};

  cudf::test::fixed_width_column_wrapper<int64_t> build_keys_column(
    build_keys_data.begin(), build_keys_data.end(), build_keys_valid.begin());
  cudf::test::fixed_width_column_wrapper<int64_t> probe_keys_column(
    probe_keys_data.begin(), probe_keys_data.end(), probe_keys_valid.begin());

  cudf::table_view build_keys_table_view({build_keys_column});
  cudf::table_view probe_keys_table_view({probe_keys_column});

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> build_result_indices;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> probe_result_indices;

  std::tie(build_result_indices, probe_result_indices) = gqe::unique_key_inner_join(
    build_keys_table_view, probe_keys_table_view, cudf::null_equality::EQUAL);

  std::vector<cudf::size_type> build_expected_indices = {0, 3, 4};
  std::vector<cudf::size_type> probe_expected_indices = {0, 1, 2};

  check_equality(build_result_indices.get(),
                 probe_result_indices.get(),
                 build_expected_indices,
                 probe_expected_indices);
}

TEST(UniqueKeyInnerJoinTest, UnequalNullTest)
{
  std::vector<int64_t> build_keys_data  = {0, 1, 2, 3, 4, 5};
  std::vector<int64_t> build_keys_valid = {0, 1, 1, 1, 1, 1};
  std::vector<int64_t> probe_keys_data  = {0, 3, 4, 6, 7, 8};
  std::vector<int64_t> probe_keys_valid = {0, 1, 1, 1, 1, 1};

  cudf::test::fixed_width_column_wrapper<int64_t> build_keys_column(
    build_keys_data.begin(), build_keys_data.end(), build_keys_valid.begin());
  cudf::test::fixed_width_column_wrapper<int64_t> probe_keys_column(
    probe_keys_data.begin(), probe_keys_data.end(), probe_keys_valid.begin());

  cudf::table_view build_keys_table_view({build_keys_column});
  cudf::table_view probe_keys_table_view({probe_keys_column});

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> build_result_indices;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> probe_result_indices;

  std::tie(build_result_indices, probe_result_indices) = gqe::unique_key_inner_join(
    build_keys_table_view, probe_keys_table_view, cudf::null_equality::UNEQUAL);

  std::vector<cudf::size_type> build_expected_indices = {3, 4};
  std::vector<cudf::size_type> probe_expected_indices = {1, 2};

  check_equality(build_result_indices.get(),
                 probe_result_indices.get(),
                 build_expected_indices,
                 probe_expected_indices);
}
