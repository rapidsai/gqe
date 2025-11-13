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

#include <gqe/executor/mark_join.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf_test/column_utilities.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <numeric>  // iota

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

class mark_join_test : public ::testing::Test {
 protected:
  void SetUp() override
  {
    stream = cudf::get_default_stream();
    mr     = rmm::mr::get_current_device_resource();
  }

  void TearDown() override
  {
    stream = {};
    mr     = nullptr;
  }

  void check(rmm::device_uvector<cudf::size_type> const& result,
             column_wrapper<cudf::size_type> const& expected)
  {
    auto result_cv = cudf::column_view(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                                       result.size(),
                                       result.data(),
                                       nullptr,
                                       0);
    auto result_tv = cudf::table_view({result_cv});
    auto sorted    = cudf::sort(result_tv);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, sorted->view().column(0));
  }

  rmm::cuda_stream_view stream;
  rmm::mr::device_memory_resource* mr;
};

class semi_equality : public mark_join_test {};

class anti_equality : public mark_join_test {};

class semi_mixed : public mark_join_test {};

class anti_mixed : public mark_join_test {};

TEST_F(semi_equality, one_column)
{
  // Match in rows 0, 1, and 3.
  column_wrapper<int32_t> left_equi_col0{0, 1, 2, 0};
  column_wrapper<int32_t> right_equi_col0{0, 1, 3, 4};

  auto left_equi  = cudf::table_view{{left_equi_col0}};
  auto right_equi = cudf::table_view{{right_equi_col0}};

  auto result =
    gqe::left_semi_mark_join(left_equi, right_equi, cudf::null_equality::EQUAL, 0.5, stream, mr);
  stream.synchronize();

  column_wrapper<cudf::size_type> expected{0, 1, 3};
  check(*result, expected);
}

TEST_F(semi_equality, two_column)
{
  // Match in rows 0, 1, and 3.
  column_wrapper<int32_t> left_equi_col0{0, 1, 2, 0};
  column_wrapper<int32_t> right_equi_col0{0, 1, 3, 4};

  // match in rows 1
  column_wrapper<int32_t> left_equi_col1{0, 1, 2, 0};
  column_wrapper<int32_t> right_equi_col1{5, 1, 5, 5};

  auto left_equi  = cudf::table_view{{left_equi_col0, left_equi_col1}};
  auto right_equi = cudf::table_view{{right_equi_col0, right_equi_col1}};

  auto result =
    gqe::left_semi_mark_join(left_equi, right_equi, cudf::null_equality::EQUAL, 0.5, stream, mr);
  stream.synchronize();

  column_wrapper<cudf::size_type> expected{1};
  check(*result, expected);
}

TEST_F(anti_equality, one_column)
{
  // Match in rows 0, 1, and 3.
  column_wrapper<int32_t> left_equi_col0{0, 1, 2, 0, 6};
  column_wrapper<int32_t> right_equi_col0{0, 1, 3, 4, 5};

  auto left_equi  = cudf::table_view{{left_equi_col0}};
  auto right_equi = cudf::table_view{{right_equi_col0}};

  auto result =
    gqe::left_anti_mark_join(left_equi, right_equi, cudf::null_equality::EQUAL, 0.5, stream, mr);
  stream.synchronize();

  column_wrapper<cudf::size_type> expected{2, 4};
  check(*result, expected);
}

TEST_F(semi_mixed, one_equality_one_condition)
{
  // Match in rows 1 and 3.
  column_wrapper<int32_t> left_equi_col0{0, 1, 2, 0};
  column_wrapper<int32_t> left_cond_col0{0, 1, 2, 5};
  column_wrapper<int32_t> right_equi_col0{0, 1, 3, 1};
  column_wrapper<int32_t> right_cond_col0{0, 2, 3, 3};

  auto left_equi  = cudf::table_view{{left_equi_col0}};
  auto right_equi = cudf::table_view{{right_equi_col0}};
  auto left_cond  = cudf::table_view{{left_cond_col0}};
  auto right_cond = cudf::table_view{{right_cond_col0}};

  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto binary_predicate =
    cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, left_ref, right_ref);

  auto result = gqe::mixed_left_semi_mark_join(left_equi,
                                               right_equi,
                                               left_cond,
                                               right_cond,
                                               binary_predicate,
                                               cudf::null_equality::EQUAL,
                                               0.5,
                                               stream,
                                               mr);
  stream.synchronize();

  column_wrapper<cudf::size_type> expected{1, 3};
  check(*result, expected);
}

TEST_F(semi_mixed, asymmetric_condition)
{
  // Match in 3.
  column_wrapper<int32_t> left_equi_col0{0, 1, 2, 0};
  column_wrapper<int32_t> left_cond_col0{0, 1, 2, 5};
  column_wrapper<int32_t> right_equi_col0{0, 1, 3, 1};
  column_wrapper<int32_t> right_cond_col0{0, 2, 3, 3};

  auto left_equi  = cudf::table_view{{left_equi_col0}};
  auto right_equi = cudf::table_view{{right_equi_col0}};
  auto left_cond  = cudf::table_view{{left_cond_col0}};
  auto right_cond = cudf::table_view{{right_cond_col0}};

  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto binary_predicate =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref, right_ref);

  auto result = gqe::mixed_left_semi_mark_join(left_equi,
                                               right_equi,
                                               left_cond,
                                               right_cond,
                                               binary_predicate,
                                               cudf::null_equality::EQUAL,
                                               0.5,
                                               stream,
                                               mr);
  stream.synchronize();

  column_wrapper<cudf::size_type> expected{3};
  check(*result, expected);
}

TEST_F(semi_mixed, mixed_data_types)
{
  // Match in 3.
  column_wrapper<int32_t> left_equi_col0{0, 1, 2, 0};
  column_wrapper<double> left_cond_col0{0., 1., 2., 5.};
  column_wrapper<int32_t> left_cond_col1{0, 1, 2, 5};

  column_wrapper<int32_t> right_equi_col0{0, 1, 3, 1};
  column_wrapper<double> right_cond_col0{0., 2., 3., 3.};
  column_wrapper<int32_t> right_cond_col1{0, 2, 3, 3};

  auto left_equi  = cudf::table_view{{left_equi_col0}};
  auto right_equi = cudf::table_view{{right_equi_col0}};
  auto left_cond  = cudf::table_view{{left_cond_col0, left_cond_col1}};
  auto right_cond = cudf::table_view{{right_cond_col0, right_cond_col1}};

  auto left_ref0  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref0 = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto left_ref1  = cudf::ast::column_reference(1, cudf::ast::table_reference::LEFT);
  auto right_ref1 = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);

  auto cond0     = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref0, right_ref0);
  auto cond1     = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_ref1, right_ref1);
  auto predicate = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, cond0, cond1);

  auto result = gqe::mixed_left_semi_mark_join(left_equi,
                                               right_equi,
                                               left_cond,
                                               right_cond,
                                               predicate,
                                               cudf::null_equality::EQUAL,
                                               0.5,
                                               stream,
                                               mr);
  stream.synchronize();

  column_wrapper<cudf::size_type> expected{3};
  check(*result, expected);
}

TEST_F(semi_mixed, nullable_columns)
{
  // Match in rows 1, 3, and 4, and 6. Expect 4 because null equality for equi, and then unequal
  // cond. 6 is just a valid match. 5, and 7 are other combinations of 4/6 that should not work.
  column_wrapper<int32_t> left_equi_col0({0, 1, 2, 0, 9, 9, 9, 9},
                                         {true, true, true, true, false, false, true, true});
  column_wrapper<int32_t> left_cond_col0({0, 1, 2, 5, 9, 9, 9, 9},
                                         {true, true, true, true, true, false, true, false});
  column_wrapper<int32_t> right_equi_col0({0, 1, 3, 1, 9, 9, 9, 9},
                                          {true, true, true, true, false, false, true, true});
  column_wrapper<int32_t> right_cond_col0({0, 2, 3, 3, 8, 8, 8, 8},
                                          {true, true, true, true, true, false, true, false});

  auto left_equi  = cudf::table_view{{left_equi_col0}};
  auto right_equi = cudf::table_view{{right_equi_col0}};
  auto left_cond  = cudf::table_view{{left_cond_col0}};
  auto right_cond = cudf::table_view{{right_cond_col0}};

  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto binary_predicate =
    cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, left_ref, right_ref);

  auto result = gqe::mixed_left_semi_mark_join(left_equi,
                                               right_equi,
                                               left_cond,
                                               right_cond,
                                               binary_predicate,
                                               cudf::null_equality::EQUAL,
                                               0.5,
                                               stream,
                                               mr);
  stream.synchronize();

  column_wrapper<cudf::size_type> expected{1, 3, 4, 6};
  check(*result, expected);
}

TEST_F(semi_mixed, test_write_buffer_flush)
{
  // Large prime number to ensure in-loop flush and finalizer flush code paths are taken.
  const size_t size = 4999999;

  // Equality conditions match exactly once, inequality condition never matches. I.e., result is the
  // array [0..size].
  std::vector<cudf::size_type> equi_vec(size);
  std::iota(equi_vec.begin(), equi_vec.end(), 0);
  std::vector<cudf::size_type> right_cond_vec(size, 5000000);

  column_wrapper<int32_t> left_equi_col0(equi_vec.begin(), equi_vec.end());
  column_wrapper<int32_t> left_cond_col0(equi_vec.begin(), equi_vec.end());
  column_wrapper<int32_t> right_equi_col0(equi_vec.begin(), equi_vec.end());
  column_wrapper<int32_t> right_cond_col0(right_cond_vec.begin(), right_cond_vec.end());

  auto left_equi  = cudf::table_view{{left_equi_col0}};
  auto right_equi = cudf::table_view{{right_equi_col0}};
  auto left_cond  = cudf::table_view{{left_cond_col0}};
  auto right_cond = cudf::table_view{{right_cond_col0}};

  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto binary_predicate =
    cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, left_ref, right_ref);

  auto result = gqe::mixed_left_semi_mark_join(left_equi,
                                               right_equi,
                                               left_cond,
                                               right_cond,
                                               binary_predicate,
                                               cudf::null_equality::EQUAL,
                                               0.5,
                                               stream,
                                               mr);
  stream.synchronize();

  std::vector<cudf::size_type> expected_vec(size);
  std::iota(expected_vec.begin(), expected_vec.end(), 0);
  column_wrapper<cudf::size_type> expected(expected_vec.begin(), expected_vec.end());
  check(*result, expected);
}

TEST_F(anti_mixed, one_equality_one_cond)
{
  // Match in rows 1 and 3.
  column_wrapper<int32_t> left_equi_col0{0, 1, 2, 0, 6};
  column_wrapper<int32_t> left_cond_col0{0, 1, 2, 5, 6};
  column_wrapper<int32_t> right_equi_col0{0, 1, 3, 1, 6};
  column_wrapper<int32_t> right_cond_col0{0, 2, 3, 3, 6};

  auto left_equi  = cudf::table_view{{left_equi_col0}};
  auto right_equi = cudf::table_view{{right_equi_col0}};
  auto left_cond  = cudf::table_view{{left_cond_col0}};
  auto right_cond = cudf::table_view{{right_cond_col0}};

  auto left_ref  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto right_ref = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto binary_predicate =
    cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, left_ref, right_ref);

  auto result = gqe::mixed_left_anti_mark_join(left_equi,
                                               right_equi,
                                               left_cond,
                                               right_cond,
                                               binary_predicate,
                                               cudf::null_equality::EQUAL,
                                               0.5,
                                               stream,
                                               mr);
  stream.synchronize();

  column_wrapper<cudf::size_type> expected{0, 2, 4};
  check(*result, expected);
}
