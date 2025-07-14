/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
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
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>

#include <gqe/executor/groupby.hpp>

#include <gtest/gtest.h>

using namespace cudf::test::iterators;

template <typename V>
struct groupby_keys_test : public cudf::test::BaseFixture {};

using supported_types = cudf::test::
  Types<int8_t, int16_t, int32_t, int64_t, float, double, numeric::decimal32, numeric::decimal64>;

TYPED_TEST_SUITE(groupby_keys_test, supported_types);

TYPED_TEST(groupby_keys_test, basic)
{
  using K = TypeParam;
  using V = int32_t;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::COUNT_VALID>;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys        { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  cudf::test::fixed_width_column_wrapper<K> expect_keys { 1, 2, 3 };
  cudf::test::fixed_width_column_wrapper<R> expect_vals { 3, 4, 3 };
  // clang-format on

  auto agg = cudf::make_count_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_keys_test, include_null_keys)
{
  using K = TypeParam;
  using V = int32_t;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::SUM>;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys(       { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                                        { 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4};

                                                    //  { 1, 1, 1,  2, 2, 2, 2,  3, 3,  4,  -}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({ 1,        2,           3,     4,  3},
                                                        { 1,        1,           1,     1,  0});
                                                    //  { 0, 3, 6,  1, 4, 5, 9,  2, 8,  -,  -}
  cudf::test::fixed_width_column_wrapper<R> expect_vals { 9,        19,          10,    4,  7};
  // clang-format on

  auto agg = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), cudf::null_policy::INCLUDE);
}

struct groupby_string_keys_test : public cudf::test::BaseFixture {};

TEST_F(groupby_string_keys_test, basic)
{
  using V = int32_t;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::SUM>;

  // clang-format off
  cudf::test::strings_column_wrapper        keys        { "aaa", "año", "₹1", "aaa", "año", "año", "aaa", "₹1", "₹1", "año"};
  cudf::test::fixed_width_column_wrapper<V> vals        {     0,     1,    2,     3,     4,     5,     6,    7,    8,     9};

  cudf::test::strings_column_wrapper        expect_keys({ "aaa", "año", "₹1" });
  cudf::test::fixed_width_column_wrapper<R> expect_vals {     9,    19,   17 };
  // clang-format on

  auto agg = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}
// clang-format on

struct groupby_dictionary_keys_test : public cudf::test::BaseFixture {};

TEST_F(groupby_dictionary_keys_test, basic)
{
  using K = std::string;
  using V = int32_t;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::SUM>;

  // clang-format off
  cudf::test::dictionary_column_wrapper<K> keys { "aaa", "año", "₹1", "aaa", "año", "año", "aaa", "₹1", "₹1", "año"};
  cudf::test::fixed_width_column_wrapper<V> vals{     0,     1,    2,     3,     4,     5,     6,    7,    8,     9};
  cudf::test::dictionary_column_wrapper<K>expect_keys  ({ "aaa", "año", "₹1" });
  cudf::test::fixed_width_column_wrapper<R> expect_vals({     9,    19,   17 });
  // clang-format on

  test_single_agg(
    keys, vals, expect_keys, expect_vals, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  test_single_agg(
    keys, vals, expect_keys, expect_vals, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
}
