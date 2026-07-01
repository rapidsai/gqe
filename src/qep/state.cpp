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

#include <gqe/qep/state.hpp>

#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>

#include <memory>
#include <stdexcept>
#include <utility>

namespace gqe {
namespace qep {

// -----------------------------------------------------------------------------
// task_private_state
// -----------------------------------------------------------------------------

task_private_state::~task_private_state() = default;

// -----------------------------------------------------------------------------
// to_table_view
// -----------------------------------------------------------------------------

std::optional<cudf::table_view> to_table_view(state_container_view container)
{
  if (container.empty()) { return std::nullopt; }
  for (auto const& s : container) {
    GQE_EXPECTS(s != nullptr, "to_table_view: null state slot");
  }

  // Skip a leading `row_count` slot.
  std::size_t const first_column =
    std::holds_alternative<state_kind::row_count>(*container[0]) ? 1 : 0;

  // Pass 1: validate the shape and count convertible slots. Short-circuit on any
  // non-column slot — no allocation happens on the rejection path.
  std::size_t column_count = 0;
  for (std::size_t i = first_column; i < container.size(); ++i) {
    auto const& slot = *container[i];
    if (!std::holds_alternative<state_kind::cudf_column>(slot) &&
        !std::holds_alternative<state_kind::cudf_column_view>(slot)) {
      return std::nullopt;
    }
    ++column_count;
  }

  // Pass 2: build the table_view. The shape is already validated.
  std::vector<cudf::column_view> columns;
  columns.reserve(column_count);
  for (std::size_t i = first_column; i < container.size(); ++i) {
    auto const& slot = *container[i];
    if (auto const* owned = std::get_if<state_kind::cudf_column>(&slot)) {
      columns.push_back(owned->column->view());
    } else {
      columns.push_back(std::get<state_kind::cudf_column_view>(slot).column);
    }
  }

  return cudf::table_view(std::move(columns));
}

// -----------------------------------------------------------------------------
// get_row_count
// -----------------------------------------------------------------------------

cudf::size_type get_row_count(state_container_view container)
{
  GQE_EXPECTS(!container.empty(), "get_row_count: container is empty");
  GQE_EXPECTS(container[0] != nullptr, "get_row_count: null state at index 0");
  return std::visit(
    gqe::utility::overloaded{
      [](state_kind::cudf_column const& s) { return s.column->size(); },
      [](state_kind::cudf_column_view const& s) { return s.column.size(); },
      [](state_kind::valid_mask const& s) { return s.column->size(); },
      [](state_kind::row_count const& s) { return s.value; },
      [](auto const&) -> cudf::size_type {
        throw std::logic_error("get_row_count: state at index 0 carries no row count");
      },
    },
    *container[0]);
}

// -----------------------------------------------------------------------------
// make_empty_state_container
// -----------------------------------------------------------------------------

state_container make_empty_state_container(std::span<cudf::data_type const> data_types)
{
  GQE_EXPECTS(!data_types.empty(),
              "make_empty_state_container: data_types must be non-empty (use "
              "make_row_count_container(0) for column-less zero-row results)");

  std::vector<std::unique_ptr<cudf::column>> empty_columns;
  empty_columns.reserve(data_types.size());
  for (auto const& dtype : data_types) {
    empty_columns.push_back(cudf::make_empty_column(dtype));
  }
  cudf::table empty_table{std::move(empty_columns)};

  return state_container_builder().add_state(std::move(empty_table)).build();
}

// -----------------------------------------------------------------------------
// make_mutable_state_copy
// -----------------------------------------------------------------------------

state_container make_mutable_state_copy(state_container_view src)
{
  return state_container(src.begin(), src.end());
}

// -----------------------------------------------------------------------------
// state_container_builder
// -----------------------------------------------------------------------------

state_container_builder::state_container_builder() : _states{} {}

state_container_builder& state_container_builder::add_state(state_kind::type&& state)
{
  _states.push_back(make_shared_state(std::move(state)));
  return *this;
}

state_container_builder& state_container_builder::add_state(shared_state state)
{
  GQE_EXPECTS(
    state != nullptr, "state_container_builder::add_state: null state", std::invalid_argument);
  _states.push_back(std::move(state));
  return *this;
}

state_container_builder& state_container_builder::add_state(cudf::table&& table)
{
  auto columns = table.release();
  _states.reserve(_states.size() + columns.size());

  for (auto& column : columns) {
    _states.push_back(make_shared_state(state_kind::cudf_column{std::move(column)}));
  }
  return *this;
}

state_container_builder& state_container_builder::add_state(cudf::table_view table_view)
{
  _states.reserve(_states.size() + table_view.num_columns());

  for (auto const& column : table_view) {
    _states.push_back(make_shared_state(state_kind::cudf_column_view{column}));
  }
  return *this;
}

state_container state_container_builder::build() { return std::move(_states); }

}  // namespace qep
}  // namespace gqe
