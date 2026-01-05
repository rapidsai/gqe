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

#include "groupby_test_util.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>
#include <gtest/gtest.h>

#include <cudf/detail/aggregation/aggregation.hpp>

template <typename V>
struct groupby_argmax_test : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(groupby_argmax_test, cudf::test::FixedWidthTypes);

TYPED_TEST(groupby_argmax_test, basic)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::ARGMAX>;

  if (std::is_same_v<V, bool>) return;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals{9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{0, 1, 2};

  auto agg = cudf::make_argmax_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_argmax_test, zero_valid_values)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::ARGMAX>;

  if (std::is_same_v<V, bool>) return;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> vals({3, 4, 5}, cudf::test::iterators::all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0}, cudf::test::iterators::all_nulls());

  auto agg = cudf::make_argmax_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_argmax_test, null_keys_and_values)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::ARGMAX>;

  if (std::is_same_v<V, bool>) return;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                                 {1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<V> vals({9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 4},
                                                 {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0});

  //  {1, 1,     2, 2, 2,   3, 3,    4}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 3, 4}, {1, 1, 1, 0, 1});
  //  {6, 3,     5, 4, 0,   2, 1,    -}
  cudf::test::fixed_width_column_wrapper<R> expect_vals({3, 4, 7, 2, 0}, {1, 1, 1, 1, 0});

  auto agg = cudf::make_argmax_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

struct groupby_argmax_string_test : public cudf::test::BaseFixture {};

TEST_F(groupby_argmax_string_test, basic)
{
  using K = int32_t;
  using R = cudf::detail::target_type_t<cudf::string_view, cudf::aggregation::ARGMAX>;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::strings_column_wrapper vals{
    "año", "bit", "₹1", "aaa", "zit", "bat", "aab", "$1", "€1", "wut"};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0, 4, 2});

  auto agg = cudf::make_argmax_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TEST_F(groupby_argmax_string_test, zero_valid_values)
{
  using K = int32_t;
  using R = cudf::detail::target_type_t<cudf::string_view, cudf::aggregation::ARGMAX>;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::strings_column_wrapper vals({"año", "bit", "₹1"}, cudf::test::iterators::all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0}, cudf::test::iterators::all_nulls());

  auto agg = cudf::make_argmax_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

CUDF_TEST_PROGRAM_MAIN()
