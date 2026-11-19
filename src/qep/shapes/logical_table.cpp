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

#include <gqe/qep/shapes/logical_table.hpp>

#include <gqe/utility/error.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <variant>

namespace gqe {
namespace qep {

logical_table_view::logical_table_view(state_container_view columns,
                                       cudf::size_type row_count,
                                       std::optional<cudf::column_view> mask,
                                       shared_state const* mask_state) noexcept
  : _columns{columns}, _row_count{row_count}, _mask{mask}, _mask_state{mask_state}
{
}

std::optional<logical_table_view> logical_table_view::try_from(state_container_view container)
{
  if (container.empty()) { return std::nullopt; }
  GQE_EXPECTS(std::ranges::none_of(container, [](auto const& slot) { return slot == nullptr; }),
              "logical_table_view::try_from: container has a null state slot");

  if (std::holds_alternative<state_kind::row_count>(*container.front())) { return std::nullopt; }

  auto const mask                = to_valid_mask_view(container.front());
  std::size_t const first_column = mask.has_value() ? 1 : 0;
  if (first_column == container.size()) {
    GQE_EXPECTS(false,
                "logical_table_view::try_from: valid mask must be followed by at least one "
                "logical column");
  }

  auto const columns = container.subspan(first_column);
  bool const claims_relation =
    mask.has_value() || std::ranges::any_of(columns, [](auto const& slot) {
      return std::holds_alternative<state_kind::cudf_column>(*slot) ||
             std::holds_alternative<state_kind::cudf_column_view>(*slot) ||
             std::holds_alternative<state_kind::indirect_column>(*slot);
    });
  if (!claims_relation) { return std::nullopt; }

  GQE_EXPECTS(
    columns.size() <= static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
    "logical_table_view::try_from: logical column count exceeds cudf::size_type");

  std::optional<cudf::size_type> row_count;
  if (mask.has_value()) { row_count = mask->size(); }
  for (auto const& state : columns) {
    cudf::size_type column_rows;
    if (auto const direct = to_column_view(state); direct.has_value()) {
      column_rows = direct->size();
    } else if (auto const* indirect = try_get_indirect_column(state)) {
      column_rows = indirect->gather_source()->position_list()->column().size();
    } else {
      GQE_EXPECTS(false,
                  "logical_table_view::try_from: logical column has an unsupported state kind");
    }

    if (!row_count.has_value()) {
      row_count = column_rows;
    } else {
      GQE_EXPECTS(*row_count == column_rows,
                  "logical_table_view::try_from: logical column row counts do not match");
    }
  }

  return logical_table_view(
    columns, *row_count, mask, mask.has_value() ? &container.front() : nullptr);
}

shared_state const& logical_table_view::column_state(cudf::size_type logical_column_index) const
{
  GQE_EXPECTS(logical_column_index >= 0 && logical_column_index < num_columns(),
              "logical_table_view::column_state: logical column index is out of range",
              std::out_of_range);
  return _columns[static_cast<std::size_t>(logical_column_index)];
}

cudf::data_type logical_table_view::column_type(cudf::size_type logical_column_index) const
{
  auto const& state = column_state(logical_column_index);
  if (auto const direct = to_column_view(state); direct.has_value()) { return direct->type(); }

  auto const* indirect = try_get_indirect_column(state);
  GQE_EXPECTS(indirect != nullptr,
              "logical_table_view::column_type: validated logical column has unsupported state");
  return indirect->gather_source()
    ->relation_source()
    ->column(indirect->source_column_index())
    .type();
}

}  // namespace qep
}  // namespace gqe
