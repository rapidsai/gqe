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

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cudf_test/type_lists.hpp>

enum class force_use_sort_impl : bool { NO, YES };

// For decimal128, decimal64 cudf's sort based groupby must be used
using PartialFixedPointTypes = cudf::test::Types<numeric::decimal32>;

void test_single_agg(cudf::column_view const& keys,
                     cudf::column_view const& values,
                     cudf::column_view const& expect_keys,
                     cudf::column_view const& expect_vals,
                     std::unique_ptr<cudf::groupby_aggregation>&& agg,
                     cudf::null_policy include_null_keys          = cudf::null_policy::EXCLUDE,
                     cudf::sorted keys_are_sorted                 = cudf::sorted::NO,
                     std::vector<cudf::order> const& column_order = {},
                     std::vector<cudf::null_order> const& null_precedence = {},
                     cudf::sorted reference_keys_are_sorted               = cudf::sorted::NO);

void test_sum_agg(cudf::column_view const& keys,
                  cudf::column_view const& values,
                  cudf::column_view const& expected_keys,
                  cudf::column_view const& expected_values);
