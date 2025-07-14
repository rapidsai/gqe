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
#include <cudf/dictionary/update_keys.hpp>

#include <gqe/executor/groupby.hpp>

#include <gtest/gtest.h>

#include <limits>

using namespace cudf::test::iterators;

template <typename V>
struct groupby_min_test : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(groupby_min_test, cudf::test::FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(groupby_min_test, basic)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::MIN>;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0, 1, 2});

  auto agg = cudf::make_min_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_min_test, empty_cols)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::MIN>;

  cudf::test::fixed_width_column_wrapper<K> keys{};
  cudf::test::fixed_width_column_wrapper<V> vals{};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_min_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_min_test, zero_valid_values)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::MIN>;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> vals({3, 4, 5}, all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0}, all_nulls());

  auto agg = cudf::make_min_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_min_test, null_keys_and_values)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, cudf::aggregation::MIN>;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                                 {1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<V> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4},
                                                 {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

  //  { 1, 1,     2, 2, 2,   3, 3,    4}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 3, 4}, {1, 1, 1, 0, 1});
  //  { 3, 6,     1, 4, 9,   2, 8,    -}
  cudf::test::fixed_width_column_wrapper<R> expect_vals({3, 1, 2, 7, 0}, {1, 1, 1, 1, 0});

  auto agg = cudf::make_min_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

template <typename T>
struct GroupByMinFixedPointTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(GroupByMinFixedPointTest, PartialFixedPointTypes);

TYPED_TEST(GroupByMinFixedPointTest, GroupByHashMinDecimalAsValue)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using K          = int32_t;

  for (auto const i : {2, 1, 0, -1, -2}) {
    auto const scale = scale_type{i};
    // clang-format off
    auto const keys  = cudf::test::fixed_width_column_wrapper<K>{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    auto const vals  = fp_wrapper{                              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, scale};
    // clang-format on

    auto const expect_keys     = cudf::test::fixed_width_column_wrapper<K>{1, 2, 3};
    auto const expect_vals_min = fp_wrapper{{0, 1, 2}, scale};

    auto agg6 = cudf::make_min_aggregation<cudf::groupby_aggregation>();
    test_single_agg(keys, vals, expect_keys, expect_vals_min, std::move(agg6));
  }
}

template <typename V>
struct groupby_min_floating_point_test : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(groupby_min_floating_point_test, cudf::test::FloatingPointTypes);

TYPED_TEST(groupby_min_floating_point_test, values_with_nan)
{
  using T          = TypeParam;
  using int32s_col = cudf::test::fixed_width_column_wrapper<int32_t>;
  using floats_col = cudf::test::fixed_width_column_wrapper<T, int32_t>;

  auto constexpr nan = std::numeric_limits<T>::quiet_NaN();

  auto const keys = int32s_col{1, 1};
  auto const vals = floats_col{nan, nan};

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  requests[0].values = vals;
  requests[0].aggregations.emplace_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());

  // Without properly handling NaN, this will hang forever in hash-based aggregate (which is the
  // default back-end for min/max in groupby context).
  // This test is just to verify that the aggregate operation does not hang.
  auto gb_obj = gqe::groupby::groupby(cudf::table_view({keys}));
  cudf::column_view active_mask;
  auto const result = gb_obj.aggregate(requests, active_mask);

  EXPECT_EQ(result.first->num_rows(), 1);
}
