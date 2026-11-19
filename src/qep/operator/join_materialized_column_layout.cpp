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

#include <gqe/qep/detail/join_materialized_column_layout.hpp>

#include <gqe/utility/error.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <span>
#include <stdexcept>
#include <vector>

namespace gqe::qep::detail {

namespace {

/** @brief Append valid unseen logical indices to a materialized column layout. */
void append_unique_indices(std::span<cudf::size_type const> indices,
                           cudf::size_type logical_column_count,
                           std::vector<cudf::size_type>& logical_indices,
                           std::vector<cudf::size_type>& logical_to_materialized)
{
  for (auto const index : indices) {
    GQE_EXPECTS(index >= 0 && index < logical_column_count,
                "join_materialized_column_layout: logical column index is out of range",
                std::invalid_argument);
    auto& materialized = logical_to_materialized[static_cast<std::size_t>(index)];
    if (materialized >= 0) { continue; }
    materialized = static_cast<cudf::size_type>(logical_indices.size());
    logical_indices.push_back(index);
  }
}

}  // namespace

join_materialized_column_layout::join_materialized_column_layout(
  cudf::size_type logical_column_count,
  std::span<cudf::size_type const> key_indices,
  std::span<cudf::size_type const> projected_payload_indices)
  : _logical_column_count{logical_column_count}
{
  GQE_EXPECTS(logical_column_count >= 0,
              "join_materialized_column_layout: logical column count must be non-negative",
              std::invalid_argument);
  _logical_to_materialized.assign(static_cast<std::size_t>(logical_column_count),
                                  cudf::size_type{-1});
  _logical_indices.reserve(key_indices.size() + projected_payload_indices.size());
  append_unique_indices(
    key_indices, logical_column_count, _logical_indices, _logical_to_materialized);
  append_unique_indices(
    projected_payload_indices, logical_column_count, _logical_indices, _logical_to_materialized);
}

cudf::size_type join_materialized_column_layout::materialized_index(
  cudf::size_type logical_index) const
{
  GQE_EXPECTS(logical_index >= 0 && logical_index < _logical_column_count,
              "join_materialized_column_layout: logical column index is out of range");
  auto const materialized = _logical_to_materialized[static_cast<std::size_t>(logical_index)];
  GQE_EXPECTS(materialized >= 0,
              "join_materialized_column_layout: logical column is absent from the materialized "
              "relation");
  return materialized;
}

std::vector<cudf::size_type> join_materialized_column_layout::remap(
  std::span<cudf::size_type const> logical_indices) const
{
  std::vector<cudf::size_type> materialized;
  materialized.reserve(logical_indices.size());
  std::ranges::transform(logical_indices, std::back_inserter(materialized), [&](auto index) {
    return materialized_index(index);
  });
  return materialized;
}

}  // namespace gqe::qep::detail
