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

#include <gqe/storage/oneshot_parquet_reader.hpp>

#include "parquet_common.hpp"
#include "source_work_split.hpp"

#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>

#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/types.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gqe {
namespace storage {

namespace {

/**
 * @brief File selection and one-emission state owned by an initialized task.
 */
struct oneshot_iterator_private : public qep::task_private_state {
  std::mutex mutex;  ///< Serializes concurrent `next()` calls.
  bool emitted = false;
  std::optional<detail::hive_file_filter_info>
    file_filter;                        ///< Set for column-bearing reads during `initialize()`.
  cudf::size_type count_only_rows = 0;  ///< Used only in count-only mode.
};

}  // namespace

oneshot_parquet_read_task::oneshot_parquet_read_task(std::vector<std::string> file_paths,
                                                     std::vector<std::string> column_names,
                                                     std::vector<cudf::data_type> output_data_types,
                                                     std::optional<hive_pruner_options> pruner)
  : _file_paths(std::move(file_paths)),
    _column_names(std::move(column_names)),
    _output_data_types(std::move(output_data_types)),
    _pruner(std::move(pruner))
{
  GQE_EXPECTS(_column_names.size() == _output_data_types.size(),
              "oneshot_parquet_read_task: column_names and output_data_types must have the same "
              "length",
              std::invalid_argument);
  if (_pruner) {
    GQE_EXPECTS(_pruner->partition_column_idx >= 0 &&
                  static_cast<std::size_t>(_pruner->partition_column_idx) < _column_names.size(),
                "oneshot_parquet_read_task: hive_pruner_options::partition_column_idx is out of "
                "range",
                std::invalid_argument);
  }
}

qep::state_container oneshot_parquet_read_task::initialize(qep::state_container_view inputs,
                                                           context_reference /* ctx_ref */,
                                                           gqe::query_stream const& stream_handle,
                                                           rmm::device_async_resource_ref mr) const
{
  gqe::utility::nvtx_scoped_range range{"oneshot_parquet_read_task::initialize"};

  auto private_state = std::make_unique<oneshot_iterator_private>();

  if (_column_names.empty()) {
    // Count-only mode, read parquet metadata to determine the total row count.
    if (_file_paths.empty()) {
      // An empty split is valid and must still emit one chunk. Avoid constructing a Parquet
      // metadata reader with no sources; `next()` will emit `[row_count{0}]`.
      private_state->count_only_rows = 0;
    } else {
      auto const metadata = cudf::io::read_parquet_metadata(cudf::io::source_info{_file_paths});
      auto const num_rows = metadata.num_rows();
      GQE_EXPECTS(num_rows <= std::numeric_limits<cudf::size_type>::max(),
                  "oneshot_parquet_read_task: total row count is too large",
                  std::overflow_error);
      private_state->count_only_rows = static_cast<cudf::size_type>(num_rows);
    }
  } else if (_pruner) {
    auto const haystack = qep::to_table_view(inputs);
    GQE_EXPECTS(haystack.has_value(),
                "oneshot_parquet_read_task: haystack input is not a plain table",
                std::invalid_argument);
    private_state->file_filter = detail::filter_hive_files_by_haystack(
      _file_paths,
      _pruner->partition_column_idx,
      _column_names[static_cast<std::size_t>(_pruner->partition_column_idx)],
      *haystack,
      stream_handle.ref(),
      mr);
  } else {
    private_state->file_filter = detail::all_files_non_partitioned(_file_paths.size());
  }

  return qep::state_container_builder()
    .add_state(qep::state_kind::task_private{std::move(private_state)})
    .build();
}

std::optional<qep::state_container> oneshot_parquet_read_task::next(
  qep::state_container_view iterator,
  context_reference ctx_ref,
  gqe::query_stream const& stream_handle,
  rmm::device_async_resource_ref mr) const
{
  auto const stream = stream_handle.ref();
  gqe::utility::nvtx_scoped_range range{"oneshot_parquet_read_task::next"};

  auto* private_state = std::get_if<qep::state_kind::task_private>(iterator[0].get());
  assert(private_state != nullptr);
  auto* iter_data = static_cast<oneshot_iterator_private*>(private_state->data.get());

  {
    std::lock_guard<std::mutex> const guard{iter_data->mutex};
    if (iter_data->emitted) { return std::nullopt; }
    iter_data->emitted = true;
  }

  if (_column_names.empty()) { return qep::make_row_count_container(iter_data->count_only_rows); }

  assert(iter_data->file_filter.has_value());
  auto table = detail::read_pruned_parquet(
    ctx_ref, *iter_data->file_filter, _file_paths, _column_names, _output_data_types, stream, mr);

  return qep::state_container_builder().add_state(std::move(*table), stream_handle).build();
}

std::size_t oneshot_parquet_read_task::work_unit_count() const noexcept
{
  return _file_paths.size();
}

std::unique_ptr<qep::iterate_task> oneshot_parquet_read_task::split(
  qep::source_work_range range) const
{
  return std::make_unique<oneshot_parquet_read_task>(
    detail::select_source_work(_file_paths, range), _column_names, _output_data_types, _pruner);
}

std::unique_ptr<qep::task> oneshot_parquet_read_task::clone() const
{
  return split({0, work_unit_count()});
}

}  // namespace storage
}  // namespace gqe
