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

#include <cudf/detail/aggregation/aggregation.hpp>

#include <gtest/gtest.h>

using namespace cudf::test::iterators;

template <typename V>
struct groupby_argmin_test : public cudf::test::BaseFixture {
};

TYPED_TEST_SUITE(groupby_argmin_test, cudf::test::FixedWidthTypes);

TYPED_TEST(groupby_argmin_test, basic)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::ARGMIN>;

  if (std::is_same_v<V, bool>) return;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals{9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{6, 9, 8};

  auto agg = cudf::make_argmin_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_argmin_test, zero_valid_values)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::ARGMIN>;

  if (std::is_same_v<V, bool>) return;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> vals({3, 4, 5}, all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0}, all_nulls());

  auto agg = cudf::make_argmin_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_argmin_test, null_keys_and_values)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::ARGMIN>;

  if (std::is_same_v<V, bool>) return;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                                 {1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<V> vals({9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 4},
                                                 {1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0});

  //  { 1, 1,     2, 2, 2,   3, 3,    4}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 3, 4}, {1, 1, 1, 0, 1});
  //  { 9, 6,     8, 5, 0,   7, 1,    -}
  cudf::test::fixed_width_column_wrapper<R> expect_vals({3, 9, 8, 7, 0}, {1, 1, 1, 1, 0});

  auto agg = cudf::make_argmin_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

struct groupby_argmin_string_test : public cudf::test::BaseFixture {
};

TEST_F(groupby_argmin_string_test, basic)
{
  using K = int32_t;
  using R = cudf::detail::target_type_t<cudf::string_view, cudf::aggregation::ARGMIN>;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::strings_column_wrapper vals{
    "año", "bit", "₹1", "aaa", "zit", "bat", "aab", "$1", "€1", "wut"};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({3, 5, 7});

  auto agg = cudf::make_argmin_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TEST_F(groupby_argmin_string_test, zero_valid_values)
{
  using K = int32_t;
  using R = cudf::detail::target_type_t<cudf::string_view, cudf::aggregation::ARGMIN>;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::strings_column_wrapper vals({"año", "bit", "₹1"}, all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0}, all_nulls());

  auto agg = cudf::make_argmin_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}
