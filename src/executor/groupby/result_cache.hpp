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

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/utilities/error.hpp>

#include <functional>
#include <unordered_map>
#include <utility>

namespace gqe::groupby::detail {

struct pair_column_aggregation_equal_to {
  bool operator()(
    std::pair<cudf::column_view, std::reference_wrapper<cudf::aggregation const>> const& lhs,
    std::pair<cudf::column_view, std::reference_wrapper<cudf::aggregation const>> const& rhs) const
  {
    return cudf::detail::is_shallow_equivalent(lhs.first, rhs.first) and
           lhs.second.get().is_equal(rhs.second.get());
  }
};

struct pair_column_aggregation_hash {
  size_t operator()(
    std::pair<cudf::column_view, std::reference_wrapper<cudf::aggregation const>> const& key) const
  {
    return cudf::hashing::detail::hash_combine(cudf::detail::shallow_hash(key.first),
                                               key.second.get().do_hash());
  }
};

class result_cache {
 public:
  result_cache()                        = delete;
  ~result_cache()                       = default;
  result_cache(result_cache const&)     = delete;
  result_cache(result_cache&&)          = delete;
  result_cache& operator=(result_cache) = delete;

  explicit result_cache(size_t num_columns) : _cache(num_columns) {}

  [[nodiscard]] bool has_result(cudf::column_view const& input, cudf::aggregation const& agg) const
  {
    return _cache.contains({input, std::cref(agg)});
  }

  void add_result(cudf::column_view const& input,
                  cudf::aggregation const& agg,
                  std::unique_ptr<cudf::column>&& col)
  {
    // We can't guarantee that agg will outlive the cache, so we need to take ownership of a copy.
    // To allow lookup by reference, make the key a reference and keep the owner in the value pair.
    auto owned_agg  = agg.clone();
    auto const& key = *owned_agg;
    _cache.try_emplace({input, std::cref(key)}, std::move(owned_agg), std::move(col));
  }

  [[nodiscard]] cudf::column_view get_result(cudf::column_view const& input,
                                             cudf::aggregation const& agg) const
  {
    auto const it = _cache.find({input, std::cref(agg)});
    CUDF_EXPECTS(it != _cache.end(), "Result not found in cache.");
    return it->second.second->view();
  }

  std::unique_ptr<cudf::column> release_result(cudf::column_view const& input,
                                               cudf::aggregation const& agg)
  {
    auto node = _cache.extract({input, std::cref(agg)});
    CUDF_EXPECTS(not node.empty(), "Result not found in cache.");
    return std::move(node.mapped().second);
  }

 private:
  std::unordered_map<std::pair<cudf::column_view, std::reference_wrapper<cudf::aggregation const>>,
                     std::pair<std::unique_ptr<cudf::aggregation>, std::unique_ptr<cudf::column>>,
                     pair_column_aggregation_hash,
                     pair_column_aggregation_equal_to>
    _cache;
};

}  // namespace gqe::groupby::detail
