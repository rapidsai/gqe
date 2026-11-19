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

#include <gqe/storage/chunked_in_memory_writer.hpp>

#include "in_memory_common.hpp"

#include <gqe/optimizer/statistics.hpp>
#include <gqe/qep/shapes/write_statistics.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/query_context.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <algorithm>
#include <cstdint>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gqe::storage {

namespace {

/**
 * @brief Accumulates statistics for one in-memory write fold.
 *
 * Concurrent `next()` calls append through the internally synchronized
 * `table_statistics_manager`. The executor calls `finalize()` only after those calls complete, so
 * its final statistics snapshot does not race with an append.
 */
struct chunked_in_memory_writer_private : qep::task_private_state {
  explicit chunked_in_memory_writer_private(std::span<cudf::data_type const> data_types)
    : statistics(table_statistics{0, data_types})
  {
  }

  table_statistics_manager statistics;
};

}  // namespace

chunked_in_memory_write_task::chunked_in_memory_write_task(
  row_group_appender_t row_group_appender,
  memory_kind::type memory_kind,
  std::vector<cudf::size_type> destination_column_indexes,
  std::vector<cudf::data_type> data_types)
  : _row_group_appender(std::move(row_group_appender)),
    _memory_kind(memory_kind),
    _destination_column_indexes(std::move(destination_column_indexes)),
    _data_types(std::move(data_types))
{
  GQE_EXPECTS(!_data_types.empty(),
              "chunked_in_memory_write_task: data_types must be non-empty",
              std::invalid_argument);
  GQE_EXPECTS(
    _destination_column_indexes.size() == _data_types.size(),
    "chunked_in_memory_write_task: destination_column_indexes and data_types must have the same "
    "length",
    std::invalid_argument);

  std::vector<bool> seen(_destination_column_indexes.size(), false);
  for (auto const destination_idx : _destination_column_indexes) {
    GQE_EXPECTS(destination_idx >= 0 && static_cast<std::size_t>(destination_idx) < seen.size(),
                "chunked_in_memory_write_task: destination column index is out of range",
                std::invalid_argument);
    GQE_EXPECTS(!seen[destination_idx],
                "chunked_in_memory_write_task: destination column indexes must be a permutation",
                std::invalid_argument);
    seen[destination_idx] = true;
  }

  GQE_EXPECTS(
    std::ranges::none_of(_data_types,
                         [](cudf::data_type type) {
                           return type.id() != cudf::type_id::EMPTY && cudf::is_nested(type);
                         }),
    "chunked_in_memory_write_task: nested LIST and STRUCT columns are not supported because "
    "data_types does not describe their child schemas",
    std::invalid_argument);
}

std::unique_ptr<qep::task> chunked_in_memory_write_task::clone() const
{
  return std::make_unique<chunked_in_memory_write_task>(
    _row_group_appender, _memory_kind, _destination_column_indexes, _data_types);
}

qep::state_container chunked_in_memory_write_task::initialize(
  context_reference ctx_ref,
  gqe::query_stream const& /* stream_handle */,
  rmm::device_async_resource_ref /* mr */) const
{
  utility::nvtx_scoped_range range{"chunked_in_memory_write_task::initialize"};

  GQE_EXPECTS(ctx_ref._query_context != nullptr,
              "chunked_in_memory_write_task::initialize: query context must not be null",
              std::invalid_argument);
  GQE_EXPECTS(ctx_ref._task_manager_context != nullptr,
              "chunked_in_memory_write_task::initialize: task-manager context must not be null",
              std::invalid_argument);

  auto const compression_config = detail::make_compression_configuration(*ctx_ref._query_context);
  GQE_EXPECTS(compression_config.primary_compression_format == compression_format::none ||
                ctx_ref._query_context->parameters.zone_map_partition_size > 0,
              "Compressed in-memory writes require zone_map_partition_size > 0 because "
              "compressed_sliced_column is the only supported compressed storage layout.",
              std::invalid_argument);

  if (ctx_ref._query_context->parameters.zone_map_partition_size > 0) {
    for (std::size_t source_idx = 0; source_idx < _destination_column_indexes.size();
         ++source_idx) {
      GQE_EXPECTS(
        _destination_column_indexes[source_idx] == static_cast<cudf::size_type>(source_idx),
        "chunked_in_memory_write_task::initialize: zone maps currently do not support reordered "
        "destination columns",
        std::invalid_argument);
    }
  }

  return qep::state_container_builder()
    .add_state(qep::state_kind::task_private{
      std::make_unique<chunked_in_memory_writer_private>(_data_types)})
    .build();
}

void chunked_in_memory_write_task::next(qep::state_container_view inputs,
                                        qep::state_container_view accumulator,
                                        context_reference ctx_ref,
                                        gqe::query_stream const& stream_handle,
                                        rmm::device_async_resource_ref /* mr */) const
{
  auto const stream = stream_handle.ref();
  utility::nvtx_scoped_range range{"chunked_in_memory_write_task::next"};

  auto* private_slot = std::get_if<qep::state_kind::task_private>(accumulator[0].get());
  GQE_EXPECTS(private_slot != nullptr,
              "chunked_in_memory_write_task::next: private slot must not be null",
              std::logic_error);

  auto const table = qep::to_table_view(inputs);
  GQE_EXPECTS(table.has_value(),
              "chunked_in_memory_write_task::next: input is not a plain table",
              std::invalid_argument);
  GQE_EXPECTS(static_cast<std::size_t>(table->num_columns()) == _data_types.size(),
              "chunked_in_memory_write_task::next: input chunk has wrong number of columns",
              std::invalid_argument);
  for (std::size_t i = 0; i < _data_types.size(); ++i) {
    if (table->column(i).type() != _data_types[i]) {
      throw std::invalid_argument(std::format(
        "chunked_in_memory_write_task::next: input chunk column type does not match the "
        "expected output schema at column {}: expected ({}, scale={}), got ({}, scale={})",
        i,
        cudf::type_to_name(_data_types[i]),
        _data_types[i].scale(),
        cudf::type_to_name(table->column(i).type()),
        table->column(i).type().scale()));
    }
  }

  GQE_EXPECTS(ctx_ref._query_context != nullptr,
              "chunked_in_memory_write_task::next: query context must not be null",
              std::invalid_argument);
  GQE_EXPECTS(ctx_ref._task_manager_context != nullptr,
              "chunked_in_memory_write_task::next: task-manager context must not be null",
              std::invalid_argument);

  auto& state = static_cast<chunked_in_memory_writer_private&>(*private_slot->data);
  auto const compression_config = detail::make_compression_configuration(*ctx_ref._query_context);
  auto const partition_size     = ctx_ref._query_context->parameters.zone_map_partition_size;
  auto const table_mr = ctx_ref._task_manager_context->get_table_memory_resource(_memory_kind);

  // The appender is a small non-owning handle. Give each concurrent call its own handle object
  // while all copies continue to publish into the same destination table.
  auto row_group_appender         = _row_group_appender;
  auto const row_group_statistics = std::visit(
    [&](auto& row_group_appender) {
      return detail::write_row_group(row_group_appender,
                                     *table,
                                     table_mr,
                                     _memory_kind,
                                     _destination_column_indexes,
                                     _data_types,
                                     compression_config,
                                     partition_size,
                                     stream);
    },
    row_group_appender);

  if (row_group_statistics) { state.statistics.append_table_statistics(*row_group_statistics); }
}

qep::state_container chunked_in_memory_write_task::finalize(
  qep::state_container&& accumulator,
  context_reference /* ctx_ref */,
  gqe::query_stream const& /* stream_handle */,
  rmm::device_async_resource_ref /* mr */) const
{
  utility::nvtx_scoped_range range{"chunked_in_memory_write_task::finalize"};

  auto* private_slot = std::get_if<qep::state_kind::task_private>(accumulator[0].get());
  GQE_EXPECTS(private_slot != nullptr,
              "chunked_in_memory_write_task::finalize: private slot must not be null",
              std::logic_error);
  auto& state = static_cast<chunked_in_memory_writer_private&>(*private_slot->data);

  return qep::make_write_statistics_container(state.statistics.statistics());
}

}  // namespace gqe::storage
