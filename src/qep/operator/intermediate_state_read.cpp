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

#include <gqe/qep/operator/intermediate_state_read.hpp>

#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/shapes/masked_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>

#include <cudf/copying.hpp>

#include <algorithm>
#include <atomic>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace gqe {
namespace qep {
namespace {

/** @brief Plain-table input recognized during initialization. */
struct plain_table {
  cudf::table_view columns;
};

/** @brief Masked-table input recognized during initialization. */
struct masked_table {
  cudf::column_view mask;
  cudf::table_view columns;
};

/** @brief Count-only input recognized during initialization. */
struct count_only {
  cudf::size_type row_count;
};

/** @brief Indirect-table input recognized during initialization. */
struct indirect_table {
  indirect_table_view table;
};

/** @brief Supported materialized input shapes. */
using input_shape = std::variant<plain_table, masked_table, indirect_table, count_only>;

/** @brief Precomputed plain-table output chunks. */
struct plain_chunks {
  std::vector<cudf::table_view> chunks;
};

/** @brief Precomputed masked-table output chunks. */
struct masked_chunks {
  std::vector<cudf::column_view> masks;
  std::vector<cudf::table_view> columns;
};

/** @brief Precomputed count-only output chunks. */
struct count_only_chunks {
  std::vector<cudf::size_type> row_counts;
};

/** @brief Precomputed indirect-table output chunks. */
struct indirect_chunks {
  std::vector<state_container> chunks;
};

/** @brief Precomputed chunks for one supported input shape. */
using precomputed_chunks =
  std::variant<plain_chunks, masked_chunks, indirect_chunks, count_only_chunks>;

[[nodiscard]] std::size_t get_chunk_count(precomputed_chunks const& chunks)
{
  return std::visit(utility::overloaded{
                      [](plain_chunks const& plain) { return plain.chunks.size(); },
                      [](masked_chunks const& masked) { return masked.columns.size(); },
                      [](indirect_chunks const& indirect) { return indirect.chunks.size(); },
                      [](count_only_chunks const& count) { return count.row_counts.size(); },
                    },
                    chunks);
}

/**
 * @brief State shared by concurrent calls to `intermediate_state_read_task::next`.
 *
 * `input` retains the source slots for borrowed output views. For bounded indirect input, `chunks`
 * owns the chunk-specific position lists and retains their original source states. Both containers
 * are immutable after construction, while `next_chunk` atomically assigns each chunk to one
 * caller. The executor owns this iterator and stream-orders its destruction.
 */
struct intermediate_state_read_state final : task_private_state {
  intermediate_state_read_state(state_container input, precomputed_chunks chunks)
    : input(std::move(input)), chunks(std::move(chunks))
  {
  }

  state_container input;
  precomputed_chunks chunks;
  std::atomic<std::size_t> next_chunk{0};
};

/** @brief Half-open row range `[begin, end)` for one output chunk. */
struct row_range {
  cudf::size_type begin;
  cudf::size_type end;
};

[[nodiscard]] std::size_t compute_num_chunks(input_shape const& shape,
                                             cudf::size_type total_rows,
                                             cudf::size_type rows_per_chunk)
{
  if (std::holds_alternative<count_only>(shape)) { return 1; }
  if (total_rows == 0) { return 1; }
  return static_cast<std::size_t>(1 + (total_rows - 1) / rows_per_chunk);
}

[[nodiscard]] row_range get_row_range(std::size_t chunk_index,
                                      cudf::size_type total_rows,
                                      cudf::size_type rows_per_chunk)
{
  auto const begin =
    static_cast<cudf::size_type>(chunk_index * static_cast<std::size_t>(rows_per_chunk));
  auto const end = begin + std::min(rows_per_chunk, total_rows - begin);
  return row_range{begin, end};
}

[[nodiscard]] std::vector<cudf::size_type> make_slice_indices(std::size_t num_chunks,
                                                              cudf::size_type total_rows,
                                                              cudf::size_type rows_per_chunk)
{
  std::vector<cudf::size_type> indices;
  indices.reserve(num_chunks * 2);
  for (std::size_t chunk_index = 0; chunk_index < num_chunks; ++chunk_index) {
    auto const range = get_row_range(chunk_index, total_rows, rows_per_chunk);
    indices.push_back(range.begin);
    indices.push_back(range.end);
  }
  return indices;
}

[[nodiscard]] precomputed_chunks make_precomputed_chunks(input_shape const& shape,
                                                         cudf::size_type total_rows,
                                                         cudf::size_type rows_per_chunk,
                                                         std::size_t num_chunks,
                                                         query_stream const& stream_handle,
                                                         rmm::device_async_resource_ref mr)
{
  auto const stream = stream_handle.ref();
  if (num_chunks == 1) {
    return std::visit(
      utility::overloaded{
        [](plain_table const& table) { return precomputed_chunks{plain_chunks{{table.columns}}}; },
        [](masked_table const& table) {
          return precomputed_chunks{masked_chunks{{table.mask}, {table.columns}}};
        },
        [&](indirect_table const& table) {
          std::vector<state_container> chunks;
          chunks.push_back(slice_indirect_table(table.table, 0, total_rows, stream_handle, mr));
          return precomputed_chunks{indirect_chunks{std::move(chunks)}};
        },
        [](count_only const& count) {
          return precomputed_chunks{count_only_chunks{{count.row_count}}};
        },
      },
      shape);
  }

  return std::visit(
    utility::overloaded{
      [&](plain_table const& table) -> precomputed_chunks {
        auto const indices = make_slice_indices(num_chunks, total_rows, rows_per_chunk);
        return plain_chunks{cudf::slice(table.columns, indices, stream)};
      },
      [&](masked_table const& table) -> precomputed_chunks {
        auto const indices = make_slice_indices(num_chunks, total_rows, rows_per_chunk);
        auto masks         = cudf::slice(table.mask, indices, stream);
        auto columns       = cudf::slice(table.columns, indices, stream);
        return masked_chunks{std::move(masks), std::move(columns)};
      },
      [&](indirect_table const& table) -> precomputed_chunks {
        std::vector<state_container> chunks;
        chunks.reserve(num_chunks);
        for (std::size_t chunk_index = 0; chunk_index < num_chunks; ++chunk_index) {
          auto const range = get_row_range(chunk_index, total_rows, rows_per_chunk);
          chunks.push_back(
            slice_indirect_table(table.table, range.begin, range.end, stream_handle, mr));
        }
        return indirect_chunks{std::move(chunks)};
      },
      [](count_only const&) -> precomputed_chunks {
        throw std::logic_error(
          "intermediate_state_read_task: count-only input reached cuDF slicing");
      },
    },
    shape);
}

[[nodiscard]] intermediate_state_read_state& get_iterator_state(state_container_view iterator)
{
  GQE_EXPECTS(iterator.size() == 1,
              "intermediate_state_read_task::next: iterator must contain one state slot");
  GQE_EXPECTS(iterator.front() != nullptr,
              "intermediate_state_read_task::next: iterator contains a null state slot");
  auto* private_state = std::get_if<state_kind::task_private>(iterator.front().get());
  GQE_EXPECTS(private_state != nullptr,
              "intermediate_state_read_task::next: iterator is not task-private state");
  GQE_EXPECTS(private_state->data != nullptr,
              "intermediate_state_read_task::next: iterator task-private state is null");
  auto* state = dynamic_cast<intermediate_state_read_state*>(private_state->data.get());
  GQE_EXPECTS(state != nullptr,
              "intermediate_state_read_task::next: iterator belongs to another task");
  return *state;
}

[[nodiscard]] state_container make_plain_output(intermediate_state_read_state const& state,
                                                cudf::table_view columns)
{
  state_container output;
  output.reserve(static_cast<std::size_t>(columns.num_columns()));
  for (cudf::size_type i = 0; i < columns.num_columns(); ++i) {
    output.push_back(make_anchored_state(state.input[static_cast<std::size_t>(i)],
                                         state_kind::cudf_column_view{columns.column(i)}));
  }
  return output;
}

[[nodiscard]] state_container make_masked_output(intermediate_state_read_state const& state,
                                                 cudf::column_view mask,
                                                 cudf::table_view columns)
{
  state_container_builder builder;
  builder.add_state(
    make_anchored_state(state.input.front(), state_kind::valid_mask_view{std::move(mask)}));
  for (cudf::size_type i = 0; i < columns.num_columns(); ++i) {
    auto const input_index = static_cast<std::size_t>(i) + 1;
    builder.add_state(make_anchored_state(state.input[input_index],
                                          state_kind::cudf_column_view{columns.column(i)}));
  }
  return builder.build();
}

}  // namespace

intermediate_state_read_task::intermediate_state_read_task(single_output_t)
  : iterate_task(task_ordering::preserve), _max_rows_per_output(std::nullopt)
{
}

intermediate_state_read_task::intermediate_state_read_task(cudf::size_type max_rows_per_output)
  : _max_rows_per_output(max_rows_per_output)
{
  GQE_EXPECTS(max_rows_per_output > 0,
              "intermediate_state_read_task: max_rows_per_output must be positive",
              std::invalid_argument);
}

state_container intermediate_state_read_task::initialize(state_container_view inputs,
                                                         context_reference,
                                                         query_stream const& stream_handle,
                                                         rmm::device_async_resource_ref mr) const
{
  input_shape shape;
  cudf::size_type row_count;

  if (auto const count = try_row_count(inputs)) {
    row_count = *count;
    shape     = count_only{*count};
  } else if (auto indirect = indirect_table_view::try_from(inputs); indirect.has_value()) {
    row_count = indirect->row_count;
    shape     = indirect_table{std::move(*indirect)};
  } else if (auto const masked = masked_table_view::try_from(inputs); masked.has_value()) {
    row_count = masked->row_count;
    shape     = masked_table{masked->mask, masked->columns};
  } else {
    auto const table = to_table_view(inputs);
    GQE_EXPECTS(
      table.has_value() && static_cast<std::size_t>(table->num_columns()) == inputs.size(),
      "intermediate_state_read_task::initialize: input is not a plain table");
    row_count = table->num_rows();
    shape     = plain_table{*table};
  }

  auto input = make_mutable_state_copy(inputs);
  auto const rows_per_chunk =
    _max_rows_per_output.value_or(std::max(row_count, cudf::size_type{1}));
  auto const num_chunks = compute_num_chunks(shape, row_count, rows_per_chunk);
  auto chunks =
    make_precomputed_chunks(shape, row_count, rows_per_chunk, num_chunks, stream_handle, mr);
  auto iterator =
    std::make_unique<intermediate_state_read_state>(std::move(input), std::move(chunks));
  return state_container_builder().add_state(state_kind::task_private{std::move(iterator)}).build();
}

std::optional<state_container> intermediate_state_read_task::next(
  state_container_view iterator,
  context_reference,
  query_stream const&,
  rmm::device_async_resource_ref) const
{
  auto& state = get_iterator_state(iterator);

  auto const num_chunks  = get_chunk_count(state.chunks);
  auto const chunk_index = state.next_chunk.fetch_add(1, std::memory_order_relaxed);
  if (chunk_index >= num_chunks) { return std::nullopt; }

  return std::visit(utility::overloaded{
                      [&](plain_chunks const& chunks) {
                        return make_plain_output(state, chunks.chunks[chunk_index]);
                      },
                      [&](masked_chunks const& chunks) {
                        return make_masked_output(
                          state, chunks.masks[chunk_index], chunks.columns[chunk_index]);
                      },
                      [&](indirect_chunks const& chunks) {
                        return make_mutable_state_copy(chunks.chunks[chunk_index]);
                      },
                      [&](count_only_chunks const& chunks) {
                        return make_row_count_container(chunks.row_counts[chunk_index]);
                      },
                    },
                    state.chunks);
}

std::unique_ptr<task> intermediate_state_read_task::clone() const
{
  if (_max_rows_per_output) {
    return std::make_unique<intermediate_state_read_task>(*_max_rows_per_output);
  }
  return std::make_unique<intermediate_state_read_task>(single_output);
}

}  // namespace qep
}  // namespace gqe
