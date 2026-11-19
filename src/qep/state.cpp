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

#include <gqe/qep/shapes/deferred_column.hpp>
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/source_reference.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace gqe {
namespace qep {

namespace {

[[nodiscard]] std::vector<query_stream> deduplicate_deallocation_streams(
  std::span<query_stream const> deallocation_streams)
{
  std::vector<query_stream> result;
  result.reserve(deallocation_streams.size());
  detail::append_unique_deallocation_streams(result, deallocation_streams);
  return result;
}

[[nodiscard]] std::vector<query_stream> union_deallocation_streams(
  std::span<query_stream const> lhs, std::span<query_stream const> rhs)
{
  auto result = deduplicate_deallocation_streams(lhs);
  result.reserve(result.size() + rhs.size());
  detail::append_unique_deallocation_streams(result, rhs);
  return result;
}

[[nodiscard]] bool directly_owns_column_storage(state_kind::type const& state)
{
  return std::holds_alternative<state_kind::cudf_column>(state) ||
         std::holds_alternative<state_kind::valid_mask>(state);
}

/** @brief RAII owner for one CUDA event used to install a stream-ordering fan-out. */
class cuda_event {
 public:
  cuda_event() { GQE_CUDA_TRY(cudaEventCreateWithFlags(&_event, cudaEventDisableTiming)); }
  ~cuda_event()
  {
    if (_event != nullptr) { GQE_CUDA_TRY_NO_THROW(cudaEventDestroy(_event)); }
  }

  cuda_event(cuda_event const&)            = delete;
  cuda_event& operator=(cuda_event const&) = delete;

  [[nodiscard]] cudaEvent_t get() const noexcept { return _event; }

  void destroy()
  {
    GQE_CUDA_TRY(cudaEventDestroy(_event));
    _event = nullptr;
  }

 private:
  cudaEvent_t _event{};
};

/** @brief Control block that retains an anchor and a complete visible shared state. */
struct anchored_state_holder {
  shared_state anchor;
  shared_state visible;
};

}  // namespace

void detail::append_unique_deallocation_streams(std::vector<query_stream>& destination,
                                                std::span<query_stream const> source)
{
  for (auto const& deallocation_stream : source) {
    if (std::ranges::none_of(destination, [&deallocation_stream](auto const& candidate) {
          return candidate.ref().get() == deallocation_stream.ref().get();
        })) {
      destination.push_back(deallocation_stream);
    }
  }
}

class detail::shared_state_factory {
 public:
  [[nodiscard]] static shared_state make(std::shared_ptr<state_kind::type const> state,
                                         std::vector<query_stream> deallocation_streams)
  {
    return shared_state(std::move(state), std::move(deallocation_streams));
  }
};

// -----------------------------------------------------------------------------
// task_private_state
// -----------------------------------------------------------------------------

task_private_state::~task_private_state() = default;

// -----------------------------------------------------------------------------
// typed and generic state wrappers
// -----------------------------------------------------------------------------

position_list_state::position_list_state(std::unique_ptr<cudf::column> column,
                                         std::span<query_stream const> deallocation_streams)
  : _deallocation_streams{deduplicate_deallocation_streams(deallocation_streams)},
    _column{std::move(column)}
{
  GQE_EXPECTS(
    _column != nullptr, "position_list_state: column must not be null", std::invalid_argument);
  GQE_EXPECTS(!deallocation_streams.empty(),
              "position_list_state: owned column storage requires a deallocation stream",
              std::invalid_argument);
  GQE_EXPECTS(_column->type().id() == cudf::type_to_id<cudf::size_type>(),
              "position_list_state: column must have cudf::size_type elements",
              std::invalid_argument);
  GQE_EXPECTS(!_column->has_nulls(),
              "position_list_state: column must not contain nulls",
              std::invalid_argument);
}

position_list_handle make_position_list(std::unique_ptr<cudf::column> positions,
                                        query_stream const& deallocation_stream)
{
  return make_position_list(std::move(positions),
                            std::span<query_stream const>{&deallocation_stream, 1});
}

position_list_handle make_position_list(std::unique_ptr<cudf::column> positions,
                                        std::span<query_stream const> deallocation_streams)
{
  return std::make_shared<position_list_state const>(std::move(positions), deallocation_streams);
}

namespace state_kind {

row_count::row_count(cudf::size_type value) : value{value}
{
  GQE_EXPECTS(value >= 0, "row_count: value must not be negative", std::invalid_argument);
}

valid_mask::valid_mask(std::unique_ptr<cudf::column> column) : _column{std::move(column)}
{
  GQE_EXPECTS(_column != nullptr, "valid_mask: column must not be null", std::invalid_argument);
  GQE_EXPECTS(_column->type().id() == cudf::type_id::BOOL8,
              "valid_mask: column must have BOOL8 elements",
              std::invalid_argument);
}

valid_mask_view::valid_mask_view(cudf::column_view column) : _column{column}
{
  GQE_EXPECTS(_column.type().id() == cudf::type_id::BOOL8,
              "valid_mask_view: column must have BOOL8 elements",
              std::invalid_argument);
}

position_list::position_list(position_list_handle state) : _state{std::move(state)}
{
  GQE_EXPECTS(_state != nullptr, "position_list: state must not be null", std::invalid_argument);
}

indirect_column::indirect_column(std::shared_ptr<indirect_column_state const> state)
  : _state{std::move(state)}
{
  GQE_EXPECTS(_state != nullptr, "indirect_column: state must not be null", std::invalid_argument);
}

keyed_accumulator::keyed_accumulator(keyed_accumulator_handle state) : _state{std::move(state)}
{
  GQE_EXPECTS(_state != nullptr, "keyed_accumulator: state must not be null");
}

reference::reference(std::shared_ptr<source_reference const> state) : _state{std::move(state)}
{
  GQE_EXPECTS(_state != nullptr, "reference: state must not be null", std::invalid_argument);
}

deferred_column::deferred_column(std::shared_ptr<deferred_column_state const> state)
  : _state{std::move(state)}
{
  GQE_EXPECTS(_state != nullptr, "deferred_column: state must not be null", std::invalid_argument);
}

}  // namespace state_kind

// -----------------------------------------------------------------------------
// shared state ownership and ordering
// -----------------------------------------------------------------------------

shared_state make_shared_state(state_kind::type state)
{
  GQE_EXPECTS(!directly_owns_column_storage(state),
              "make_shared_state: directly owned column storage requires an explicit "
              "deallocation stream",
              std::invalid_argument);

  std::vector<query_stream> deallocation_streams;
  if (auto const* positions = std::get_if<state_kind::position_list>(&state)) {
    deallocation_streams =
      deduplicate_deallocation_streams(positions->state()->deallocation_streams());
  } else if (auto const* indirect = std::get_if<state_kind::indirect_column>(&state)) {
    deallocation_streams =
      deduplicate_deallocation_streams(indirect->state()->gather_source()->deallocation_streams());
  }
  return detail::shared_state_factory::make(
    std::make_shared<state_kind::type const>(std::move(state)), std::move(deallocation_streams));
}

shared_state make_shared_state(state_kind::type state, query_stream const& deallocation_stream)
{
  return make_shared_state(std::move(state),
                           std::span<query_stream const>{&deallocation_stream, 1});
}

shared_state make_shared_state(state_kind::type state,
                               std::span<query_stream const> deallocation_streams)
{
  GQE_EXPECTS(directly_owns_column_storage(state),
              "make_shared_state: explicit deallocation streams require directly owned column "
              "storage",
              std::invalid_argument);
  GQE_EXPECTS(!deallocation_streams.empty(),
              "make_shared_state: owned column storage requires a deallocation stream",
              std::invalid_argument);
  return detail::shared_state_factory::make(
    std::make_shared<state_kind::type const>(std::move(state)),
    deduplicate_deallocation_streams(deallocation_streams));
}

shared_state make_anchored_state(shared_state anchor, shared_state visible)
{
  GQE_EXPECTS(anchor != nullptr, "make_anchored_state: anchor must not be null");
  GQE_EXPECTS(visible != nullptr, "make_anchored_state: visible state must not be null");
  auto deallocation_streams =
    union_deallocation_streams(anchor.deallocation_streams(), visible.deallocation_streams());
  auto holder = std::make_shared<anchored_state_holder>(
    anchored_state_holder{std::move(anchor), std::move(visible)});
  auto aliased = std::shared_ptr<state_kind::type const>(holder, holder->visible.get());
  return detail::shared_state_factory::make(std::move(aliased), std::move(deallocation_streams));
}

shared_state make_anchored_state(shared_state anchor, state_kind::cudf_column_view visible)
{
  return make_anchored_state(std::move(anchor),
                             make_shared_state(state_kind::type{std::move(visible)}));
}

shared_state make_anchored_state(shared_state anchor, state_kind::valid_mask_view visible)
{
  return make_anchored_state(std::move(anchor),
                             make_shared_state(state_kind::type{std::move(visible)}));
}

void order_deallocation_after_stream_use(state_container_view dependencies,
                                         query_stream const& consuming_stream)
{
  auto const consuming_stream_ref = consuming_stream.ref();
  try {
    // Explicit stream handles have stable identities across host threads. PTDS tokens do not, but
    // the public contract requires PTDS-backed buffers to be deallocated on this calling thread;
    // within that restriction, raw-value deduplication refers to this thread's one PTDS.
    std::vector<query_stream> foreign_deallocation_streams;
    for (auto const& dependency : dependencies) {
      GQE_EXPECTS(dependency != nullptr,
                  "order_deallocation_after_stream_use: dependency must not be null");
      detail::append_unique_deallocation_streams(foreign_deallocation_streams,
                                                 dependency.deallocation_streams());
    }
    std::erase_if(foreign_deallocation_streams, [consuming_stream_ref](auto const& stream) {
      return stream.ref().get() == consuming_stream_ref.get();
    });
    if (foreign_deallocation_streams.empty()) { return; }

    cuda_event event;
    GQE_CUDA_TRY(cudaEventRecord(event.get(), consuming_stream_ref.get()));
    for (auto const& deallocation_stream : foreign_deallocation_streams) {
      GQE_CUDA_TRY(cudaStreamWaitEvent(deallocation_stream.ref().get(), event.get()));
    }
    event.destroy();
  } catch (...) {
    auto const synchronization_error = cudaStreamSynchronize(consuming_stream_ref.get());
    if (synchronization_error != cudaSuccess) { GQE_CUDA_TRY_NO_THROW(synchronization_error); }
    throw;
  }
}

// -----------------------------------------------------------------------------
// to_column_view / to_valid_mask_view / to_table_view
// -----------------------------------------------------------------------------

std::optional<cudf::column_view> to_column_view(shared_state const& state)
{
  GQE_EXPECTS(state != nullptr, "to_column_view: null state slot");
  if (auto const* owned = std::get_if<state_kind::cudf_column>(state.get())) {
    GQE_EXPECTS(owned->column != nullptr, "to_column_view: cudf_column carries null storage");
    return owned->column->view();
  }
  if (auto const* borrowed = std::get_if<state_kind::cudf_column_view>(state.get())) {
    return borrowed->column;
  }
  return std::nullopt;
}

indirect_column_state const* try_get_indirect_column(shared_state const& state)
{
  GQE_EXPECTS(state != nullptr, "try_get_indirect_column: null state slot");
  auto const* indirect = std::get_if<state_kind::indirect_column>(state.get());
  return indirect != nullptr ? indirect->state().get() : nullptr;
}

std::optional<cudf::column_view> to_valid_mask_view(shared_state const& state)
{
  GQE_EXPECTS(state != nullptr, "to_valid_mask_view: null state slot");
  if (auto const* owned = std::get_if<state_kind::valid_mask>(state.get())) {
    return owned->column().view();
  }
  if (auto const* borrowed = std::get_if<state_kind::valid_mask_view>(state.get())) {
    return borrowed->column();
  }
  return std::nullopt;
}

std::optional<cudf::table_view> to_table_view(state_container_view container)
{
  if (container.empty()) { return std::nullopt; }

  // Skip a leading `row_count` slot.
  std::size_t const first_column =
    container[0] != nullptr && std::holds_alternative<state_kind::row_count>(*container[0]) ? 1 : 0;

  // Pass 1: validate the shape and count convertible slots. Short-circuit on any
  // non-column slot — no allocation happens on the rejection path.
  std::size_t column_count = 0;
  for (std::size_t i = first_column; i < container.size(); ++i) {
    if (!to_column_view(container[i]).has_value()) { return std::nullopt; }
    ++column_count;
  }

  // Pass 2: build the table_view. The shape is already validated.
  std::vector<cudf::column_view> columns;
  columns.reserve(column_count);
  for (std::size_t i = first_column; i < container.size(); ++i) {
    columns.push_back(*to_column_view(container[i]));
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
      [](state_kind::valid_mask const& s) { return s.column().size(); },
      [](state_kind::valid_mask_view const& s) { return s.column().size(); },
      [](state_kind::position_list const& s) { return s.state()->column().size(); },
      [](state_kind::indirect_column const& s) {
        return s.state()->gather_source()->position_list()->column().size();
      },
      [](state_kind::reference const& s) { return s.state()->num_rows(); },
      [](state_kind::deferred_column const& s) { return s.state()->source()->num_rows(); },
      [](state_kind::row_count const& s) { return s.value; },
      [](state_kind::keyed_accumulator const&) -> cudf::size_type {
        throw std::logic_error(
          "get_row_count: keyed_accumulator carries no host-readable row count; its group count "
          "is device-side only");
      },
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

  state_container result;
  result.reserve(data_types.size());
  for (auto const& dtype : data_types) {
    // The factory-created empty column owns no device buffers, so an empty deallocation-stream set
    // is complete. Keep this narrow exception separate from general owning-state construction.
    auto state = std::make_shared<state_kind::type const>(
      state_kind::cudf_column{cudf::make_empty_column(dtype)});
    result.push_back(detail::shared_state_factory::make(std::move(state), {}));
  }
  return result;
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

state_container_builder::state_container_builder() : _valid_mask{}, _states{} {}

state_container_builder& state_container_builder::add_state(state_kind::type&& state)
{
  return add_state(make_shared_state(std::move(state)));
}

state_container_builder& state_container_builder::add_state(state_kind::type&& state,
                                                            query_stream const& deallocation_stream)
{
  return add_state(make_shared_state(std::move(state), deallocation_stream));
}

state_container_builder& state_container_builder::add_state(shared_state state)
{
  GQE_EXPECTS(
    state != nullptr, "state_container_builder::add_state: null state", std::invalid_argument);

  bool const is_valid_mask = to_valid_mask_view(state).has_value();
  if (is_valid_mask) {
    GQE_EXPECTS(!_valid_mask.has_value(),
                "state_container_builder::add_state: duplicate valid mask",
                std::invalid_argument);
    _valid_mask = std::move(state);
  } else {
    _states.push_back(std::move(state));
  }
  return *this;
}

state_container_builder& state_container_builder::add_state(cudf::table&& table,
                                                            query_stream const& deallocation_stream)
{
  auto columns = table.release();
  _states.reserve(_states.size() + columns.size());

  for (auto& column : columns) {
    _states.push_back(
      make_shared_state(state_kind::cudf_column{std::move(column)}, deallocation_stream));
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

state_container state_container_builder::build()
{
  if (!_valid_mask.has_value()) { return std::exchange(_states, {}); }

  state_container result;
  result.reserve(_states.size() + 1);
  result.push_back(std::move(*_valid_mask));
  result.insert(
    result.end(), std::make_move_iterator(_states.begin()), std::make_move_iterator(_states.end()));
  _valid_mask.reset();
  _states.clear();
  return result;
}

}  // namespace qep
}  // namespace gqe
