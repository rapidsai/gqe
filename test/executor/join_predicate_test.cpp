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

#include "../../src/executor/join_predicate.hpp"

#include <gqe/executor/eval.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/literal.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/join/mixed_join.hpp>
#include <cudf/sorting.hpp>
#include <rmm/device_buffer.hpp>

#include <array>
#include <memory>
#include <stdexcept>
#include <utility>

using int32_column_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;

/** @brief Exercises join AST bindings through cuDF execution and checked reference failures. */
class JoinPredicateTest : public cudf::test::BaseFixture {};

TEST_F(JoinPredicateTest, SparseReorderedInputsHaveIndependentRowGeometry)
{
  int32_column_wrapper left_values{5, 15, 25};
  int32_column_wrapper left_keys{1, 1, 1};
  int32_column_wrapper right_keys{1, 1};
  int32_column_wrapper right_values{10, 20};
  cudf::table_view const left({left_values, left_keys});
  cudf::table_view const right({right_keys, right_values});
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  // Logical left width is 5, including an unmaterialized trailing column.
  // Left column 3 and right column 0 occupy different physical positions.
  gqe::logical_and_expression residual(
    std::make_shared<gqe::less_expression>(std::make_shared<gqe::column_reference_expression>(3),
                                           std::make_shared<gqe::column_reference_expression>(5)),
    std::make_shared<gqe::greater_expression>(
      std::make_shared<gqe::column_reference_expression>(3),
      std::make_shared<gqe::literal_expression<int32_t>>(0)));
  auto predicate = [&] {
    std::array<cudf::size_type, 2> const left_indices{3, 1};
    std::array<cudf::size_type, 2> const right_indices{2, 0};
    gqe::expression_binding const left_binding(left_indices);
    gqe::expression_binding const right_binding(right_indices);
    gqe::expression_input const left_input(left, left_binding, left.num_rows());
    gqe::expression_input const right_input(right, right_binding, right.num_rows());
    return gqe::detail::make_join_predicate(residual, 5, left_input, right_input, stream, mr);
  }();

  // The bindings have expired, but the AST and its literal still own everything they need.
  auto matches = cudf::mixed_left_semi_join(left.select({1}),
                                            right.select({0}),
                                            left,
                                            right,
                                            predicate.root(),
                                            cudf::null_equality::UNEQUAL,
                                            stream,
                                            mr);
  cudf::column positions(std::move(*matches), rmm::device_buffer(), 0);
  auto sorted = cudf::sort(cudf::table_view({positions.view()}));
  int32_column_wrapper expected{0, 1};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->view().column(0), expected);
}

TEST_F(JoinPredicateTest, IdentityBindingsPreserveExistingConversion)
{
  int32_column_wrapper left_values{5, 25};
  int32_column_wrapper right_values{10};
  cudf::table_view const left({left_values});
  cudf::table_view const right({right_values});
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  gqe::less_expression residual(std::make_shared<gqe::column_reference_expression>(0),
                                std::make_shared<gqe::column_reference_expression>(1));
  auto const original = gqe::detail::make_join_predicate(residual, 1, stream, mr);
  auto const bound    = gqe::detail::make_join_predicate(
    residual, 1, gqe::expression_input(left), gqe::expression_input(right), stream, mr);
  // Equal constant keys make this an existence test for the residual on each left row.
  int32_column_wrapper left_keys{1, 1};
  int32_column_wrapper right_keys{1};
  for (auto const* predicate : {&original, &bound}) {
    auto matches = cudf::mixed_left_semi_join(cudf::table_view({left_keys}),
                                              cudf::table_view({right_keys}),
                                              left,
                                              right,
                                              predicate->root(),
                                              cudf::null_equality::UNEQUAL,
                                              stream,
                                              mr);
    cudf::column positions(std::move(*matches), rmm::device_buffer(), 0);
    int32_column_wrapper expected{0};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(positions.view(), expected);
  }
}

TEST_F(JoinPredicateTest, RejectsMissingAndOutOfRangeReferencesOnEitherSide)
{
  int32_column_wrapper values{1};
  cudf::table_view const table({values});
  std::array<cudf::size_type, 1> const indices{1};
  gqe::expression_binding const binding(indices);
  gqe::expression_input const left(table, binding, table.num_rows());
  gqe::expression_input const right(table, binding, table.num_rows());
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  // Left 0 and right 0 are missing. Combined index 5 is past the logical schema.
  for (auto index : {0, 3, 5}) {
    gqe::column_reference_expression reference(index);
    EXPECT_THROW((void)gqe::detail::make_join_predicate(reference, 3, left, right, stream, mr),
                 std::out_of_range);
  }
  gqe::column_reference_expression valid(1);
  EXPECT_THROW((void)gqe::detail::make_join_predicate(valid, -1, left, right, stream, mr),
               std::invalid_argument);
  gqe::column_reference_expression negative(-1);
  EXPECT_THROW((void)gqe::detail::make_join_predicate(negative, 3, left, right, stream, mr),
               std::invalid_argument);
}
