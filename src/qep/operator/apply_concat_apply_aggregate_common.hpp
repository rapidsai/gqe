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

#include <gqe/qep/column_materialization.hpp>
#include <gqe/qep/shapes/logical_table.hpp>
#include <gqe/qep/shapes/masked_table.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/query_context.hpp>
#include <gqe/utility/error.hpp>

#include <cuda/stream>
#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <rmm/resource_ref.hpp>

#include <algorithm>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gqe {
namespace qep {
namespace detail {

/**
 * @brief Resolved regular-table data and optional QEP valid mask for one apply-concat-apply input.
 *
 * The views borrow their storage from the source state container, which must outlive every use of
 * this object.
 */
struct aggregate_input_view {
  cudf::table_view table;  ///< Data columns consumed by the aggregate.

  cudf::column_view mask;  ///< QEP valid mask, or an empty view for unmasked input.

  bool has_mask;  ///< Whether the input used the masked-table state shape.
};

/** @brief Logical-column selection and materialized indices for a partial grouped aggregate. */
struct aggregate_input_layout {
  std::vector<cudf::size_type>
    materialized_indices;  ///< Unique logical input indices in materialization order.

  std::vector<cudf::size_type>
    materialized_key_indices;  ///< Grouping-key indices into the materialized table.

  std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>>
    materialized_value_indices;  ///< Aggregate specifications over the materialized table.
};

/**
 * @brief Find a logical index in the selection, appending it when absent.
 *
 * @param[in,out] selected_indices Logical indices in materialization order.
 * @param[in] logical_index Logical column to select.
 * @return Position of the logical column in the selection.
 */
[[nodiscard]] inline cudf::size_type find_or_append_logical_index(
  std::vector<cudf::size_type>& selected_indices, cudf::size_type logical_index)
{
  auto const found = std::ranges::find(selected_indices, logical_index);
  if (found != selected_indices.end()) {
    return static_cast<cudf::size_type>(std::distance(selected_indices.begin(), found));
  }
  auto const index = static_cast<cudf::size_type>(selected_indices.size());
  selected_indices.push_back(logical_index);
  return index;
}

/**
 * @brief Build one selective-materialization layout shared by the hash and perfect-hash aggregate
 * tasks.
 *
 * Repeated logical indices map to the same materialized column while repeated key and value
 * specifications remain repeated in their original order.
 */
[[nodiscard]] inline aggregate_input_layout make_aggregate_input_layout(
  std::span<cudf::size_type const> key_indices,
  std::span<std::pair<cudf::aggregation::Kind, cudf::size_type> const> value_indices)
{
  aggregate_input_layout result;
  result.materialized_indices.reserve(key_indices.size() + value_indices.size());

  result.materialized_key_indices.reserve(key_indices.size());
  for (auto const key_index : key_indices) {
    result.materialized_key_indices.push_back(
      find_or_append_logical_index(result.materialized_indices, key_index));
  }
  result.materialized_value_indices.reserve(value_indices.size());
  for (auto const& [kind, value_index] : value_indices) {
    result.materialized_value_indices.emplace_back(
      kind, find_or_append_logical_index(result.materialized_indices, value_index));
  }
  return result;
}

/** @brief Owner of selectively materialized aggregate columns and relation mask metadata. */
struct materialized_aggregate_input {
  column_materialization columns;  ///< Selected columns and their asynchronous dependencies.

  cudf::column_view mask;  ///< QEP valid mask, or an empty view for unmasked input.

  cudf::size_type logical_column_count;  ///< Column count before selective materialization.

  bool has_mask;  ///< Whether the logical input carries a QEP valid mask.

  /** @brief Return the selected columns and mask as partial-aggregate input. */
  [[nodiscard]] aggregate_input_view view() const
  {
    return aggregate_input_view{columns.table_view(), mask, has_mask};
  }
};

/**
 * @brief Selectively materialize a column-bearing logical relation for partial aggregation.
 *
 * @param[in] inputs Aggregate input state container.
 * @param[in] materialized_indices Unique logical columns required by the aggregate.
 * @param[in] stream CUDA stream used for materialization.
 * @param[in] mr Memory resource used for every materialization allocation.
 * @return Materialized columns with the original relation mask metadata.
 * @throws std::invalid_argument If the input is not a column-bearing logical relation.
 */
[[nodiscard]] inline materialized_aggregate_input materialize_aggregate_input(
  state_container_view inputs,
  std::span<cudf::size_type const> materialized_indices,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref mr)
{
  auto const logical = logical_table_view::try_from(inputs);
  GQE_EXPECTS(logical.has_value(),
              "partial aggregate input is not a logical relation",
              std::invalid_argument);
  auto columns    = materialize_columns(*logical, materialized_indices, stream_handle, mr);
  auto const mask = logical->valid_mask();
  return materialized_aggregate_input{std::move(columns),
                                      mask.value_or(cudf::column_view{}),
                                      logical->num_columns(),
                                      mask.has_value()};
}

/** @brief Optional owner that keeps a normalized active mask alive alongside its view. */
struct normalized_active_mask {
  std::unique_ptr<cudf::column> owner;  ///< Normalized mask owner, or null when no copy was needed.

  cudf::column_view view;  ///< Mask supplied to the aggregate operation.
};

/**
 * @brief Resolve a regular-table or masked-table state container for aggregation.
 *
 * @param[in] inputs Aggregate input state container.
 * @param[in] is_mask_allowed Whether the caller accepts a masked-table input.
 * @return Resolved data columns, mask, and shape indicator.
 * @throws std::invalid_argument If the input shape is unsupported.
 */
[[nodiscard]] inline aggregate_input_view resolve_aggregate_input(state_container_view inputs,
                                                                  bool is_mask_allowed)
{
  if (auto const masked = masked_table_view::try_from(inputs); masked.has_value()) {
    GQE_EXPECTS(
      is_mask_allowed, "aggregate input must be an unmasked regular table", std::invalid_argument);
    return aggregate_input_view{masked->columns, masked->mask, true};
  }

  auto const table = to_table_view(inputs);
  GQE_EXPECTS(table.has_value() && table->num_columns() > 0,
              "aggregate input is not a non-empty regular table",
              std::invalid_argument);
  return aggregate_input_view{*table, cudf::column_view{}, false};
}

/**
 * @brief Replace null active-mask entries with false values when required.
 *
 * @param[in] mask Active mask to normalize.
 * @param[in] stream CUDA stream used for device work.
 * @param[in] mr Memory resource used for the normalized mask allocation.
 * @return Active-mask view and any owner required to keep it alive.
 */
[[nodiscard]] inline normalized_active_mask normalize_active_mask(cudf::column_view mask,
                                                                  query_stream const& stream_handle,
                                                                  rmm::device_async_resource_ref mr)
{
  if (mask.is_empty() || mask.null_count() == 0) { return normalized_active_mask{nullptr, mask}; }

  auto owner = make_non_nullable_valid_mask(mask, stream_handle, mr);
  auto view  = owner->view();
  return normalized_active_mask{std::move(owner), view};
}

}  // namespace detail
}  // namespace qep
}  // namespace gqe
