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

#include <gqe/storage/in_memory_resolve.hpp>

#include "in_memory_common.hpp"

#include <gqe/qep/shapes/deferred_column.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/query_context.hpp>
#include <gqe/storage/in_memory.hpp>
#include <gqe/storage/source_references.hpp>
#include <gqe/storage/table_provider.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace gqe::storage {

namespace {

/**
 * @brief Deferred source materializing columns of one in-memory row group.
 *
 * Retains the resolved table's `shared_ptr` so the row-group pointer stays valid until the source
 * is destroyed, and the resolve-time execution context the output-table builder needs. Both are
 * query-lifetime objects, so the source must be materialized or dropped before its query
 * completes. Each decode builds the requested column subset in one batched H2D copy and
 * decompression on the consumer's stream. The pruning result is retained by value, so every
 * request constructs its builder from intact inputs.
 */
class in_memory_deferred_source final : public qep::deferred_table_source_state {
 public:
  in_memory_deferred_source(cudf::size_type num_rows,
                            context_reference ctx_ref,
                            std::shared_ptr<table> table_anchor,
                            row_group_interface const* row_group,
                            pruning_result_t pruning,
                            std::vector<cudf::size_type> column_indexes,
                            std::vector<cudf::data_type> output_data_types)
    : qep::deferred_table_source_state{num_rows, std::move(output_data_types)},
      _ctx_ref{ctx_ref},
      _table_anchor{std::move(table_anchor)},
      _row_group{row_group},
      _pruning{std::move(pruning)},
      _column_indexes{std::move(column_indexes)}
  {
    // The builder dereferences both contexts at materialization time.
    GQE_EXPECTS(_ctx_ref._query_context != nullptr && _ctx_ref._task_manager_context != nullptr,
                "in_memory_deferred_source: query and task-manager contexts must not be null",
                std::invalid_argument);
    // With use_overlap_mtx the builder allocates the copied-to output buffers on the shared
    // copy-engine stream rather than the materialization stream, which would falsify the
    // deallocation stream recorded for the returned columns.
    GQE_EXPECTS(!_ctx_ref._query_context->parameters.use_overlap_mtx,
                "in_memory_deferred_source: use_overlap_mtx is not supported with deferred sources",
                std::invalid_argument);
  }

 protected:
  [[nodiscard]] std::vector<std::unique_ptr<cudf::column>> materialize_columns_impl(
    std::span<cudf::size_type const> source_column_indices,
    query_stream const& stream_handle,
    rmm::device_async_resource_ref mr) const override
  {
    auto const stream = stream_handle.ref();
    utility::nvtx_scoped_range range{"in_memory_deferred_source::materialize"};

    std::vector<cudf::size_type> table_columns;
    std::vector<cudf::data_type> types;
    table_columns.reserve(source_column_indices.size());
    types.reserve(source_column_indices.size());
    for (auto const index : source_column_indices) {
      table_columns.push_back(_column_indexes[static_cast<std::size_t>(index)]);
      types.push_back(column_type(index));
    }

    auto pruning_results = std::make_unique<pruning_results_t>();
    pruning_results->emplace_back(_row_group, _pruning);
    output_table_builder builder{
      _ctx_ref, std::move(table_columns), std::move(types), std::move(pruning_results)};
    return builder.build(stream, mr)->release();
  }

 private:
  context_reference _ctx_ref;
  std::shared_ptr<table> _table_anchor;
  row_group_interface const* _row_group;
  pruning_result_t _pruning;
  std::vector<cudf::size_type> _column_indexes;
};

}  // namespace

in_memory_resolve_task::in_memory_resolve_task(table_provider const* table_provider,
                                               std::vector<cudf::size_type> column_indexes,
                                               std::vector<cudf::data_type> output_data_types)
  : _table_provider{table_provider},
    _column_indexes{std::move(column_indexes)},
    _output_data_types{std::move(output_data_types)}
{
  GQE_EXPECTS(_table_provider != nullptr,
              "in_memory_resolve_task: table_provider must not be null",
              std::invalid_argument);
  GQE_EXPECTS(_column_indexes.size() == _output_data_types.size(),
              "in_memory_resolve_task: column_indexes and output_data_types must have the same "
              "length",
              std::invalid_argument);
}

std::optional<qep::state_container> in_memory_resolve_task::next(
  qep::state_container_view inputs,
  context_reference ctx_ref,
  query_stream const& /* stream_handle */,
  rmm::device_async_resource_ref /* mr */) const
{
  utility::nvtx_scoped_range range{"in_memory_resolve_task::next"};

  GQE_EXPECTS(inputs.size() == 1,
              "in_memory_resolve_task: expected exactly one input state",
              std::invalid_argument);
  GQE_EXPECTS(inputs[0] != nullptr, "in_memory_resolve_task: null input state slot");

  if (qep::try_row_count(inputs)) {
    // Count-only chunks pass through unchanged, so count-only plans tolerate an inserted resolve
    // task.
    return qep::make_mutable_state_copy(inputs);
  } else {
    GQE_EXPECTS(!_column_indexes.empty(),
                "in_memory_resolve_task: an empty projection accepts only count-only chunks",
                std::invalid_argument);
    auto const* reference_slot = std::get_if<qep::state_kind::reference>(inputs[0].get());
    GQE_EXPECTS(reference_slot != nullptr,
                "in_memory_resolve_task: input is not a reference chunk",
                std::invalid_argument);
    auto const* ref = dynamic_cast<in_memory_reference const*>(reference_slot->state().get());
    GQE_EXPECTS(ref != nullptr,
                "in_memory_resolve_task: reference does not address an in-memory source",
                std::invalid_argument);

    if (ref->is_empty()) {
      return qep::make_empty_state_container(_output_data_types);
    } else {
      return resolve(*ref, ctx_ref);
    }
  }
}

qep::state_container in_memory_resolve_task::resolve(in_memory_reference const& reference,
                                                     context_reference ctx_ref) const
{
  auto const* ref = &reference;

  auto resolved_table = _table_provider->get_table(ref->table_name());
  GQE_EXPECTS(resolved_table != nullptr,
              "in_memory_resolve_task: table provider returned a null table");
  auto in_memory_table =
    std::dynamic_pointer_cast<in_memory_table_interface>(std::move(resolved_table));
  GQE_EXPECTS(in_memory_table != nullptr,
              "in_memory_resolve_task: referenced table is not an in-memory table");

  row_group_interface const* row_group = nullptr;
  {
    std::shared_lock guard{in_memory_table->row_group_latch()};
    GQE_EXPECTS(ref->row_group_index() < in_memory_table->num_row_groups(),
                "in_memory_resolve_task: reference row-group index is out of range",
                std::out_of_range);
    row_group = in_memory_table->row_group_at(ref->row_group_index());
  }

  // References carry no schema, so fail fast if the resolved row group does not match the plan's
  // projection instead of reading mistyped bytes at materialize time.
  for (std::size_t i = 0; i < _column_indexes.size(); ++i) {
    auto const column_index = _column_indexes[i];
    GQE_EXPECTS(column_index >= 0 && static_cast<int64_t>(column_index) < row_group->num_columns(),
                "in_memory_resolve_task: column index is outside the resolved row-group schema",
                std::out_of_range);

    auto const& column = row_group->get_column(column_index);
    GQE_EXPECTS(detail::cudf_type_of(column) == _output_data_types[i],
                "in_memory_resolve_task: requested output type does not match the resolved column "
                "type",
                std::invalid_argument);

    auto const nulls_supported = column.layout() == column_layout::contiguous &&
                                 _output_data_types[i].id() == cudf::type_id::STRING;
    GQE_EXPECTS(column.null_count() == 0 || nulls_supported,
                "in_memory_resolve_task: projected column materialization does not support null "
                "values",
                std::invalid_argument);
  }

  // The pruning result's row count is the source of the reference's row count, so comparing the
  // two is vacuous. Validate the pruning extent against the resolved row group instead, so a
  // mismatched table fails loudly rather than slicing past the row group's end.
  auto const& partitions = ref->pruning().consolidated_partitions();
  GQE_EXPECTS(!partitions.empty(),
              "in_memory_resolve_task: reference pruning result selects no partitions");
  auto const max_extent =
    std::ranges::max(partitions, {}, [](auto const& partition) { return partition.end; }).end;
  GQE_EXPECTS(static_cast<std::int64_t>(max_extent) <= row_group->size(),
              "in_memory_resolve_task: reference pruning extent exceeds the resolved row group");

  auto source = std::make_shared<in_memory_deferred_source>(ref->num_rows(),
                                                            ctx_ref,
                                                            std::move(in_memory_table),
                                                            row_group,
                                                            ref->pruning(),
                                                            _column_indexes,
                                                            _output_data_types);
  return qep::make_deferred_column_container(std::move(source));
}

std::unique_ptr<qep::task> in_memory_resolve_task::clone() const
{
  return std::make_unique<in_memory_resolve_task>(
    _table_provider, _column_indexes, _output_data_types);
}

}  // namespace gqe::storage
