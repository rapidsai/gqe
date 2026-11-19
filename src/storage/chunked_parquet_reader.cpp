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

#include <gqe/storage/chunked_parquet_reader.hpp>

#include "parquet_common.hpp"
#include "source_work_split.hpp"

#include <gqe/context_reference.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/storage/parquet.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>

#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>

#include <cassert>
#include <cstdint>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gqe {
namespace storage {

namespace {

struct parquet_iterator_private : public qep::task_private_state {
  std::mutex mutex;  ///< Must be acquired by every `next` call before reading or mutating the
                     ///< rest of this state. Not required in `initialize` or `finalize` — the
                     ///< executor guarantees they run single-threaded.

  std::unique_ptr<cudf::io::chunked_parquet_reader>
    reader;  ///< Backing chunked reader. Non-null in data-read mode; null in count-only mode
             ///< or when Hive pruning eliminated every file.

  std::vector<cudf::data_type>
    output_data_types;  ///< Expected output type per column. Empty in count-only mode.

  cudf::size_type count_only_rows =
    0;  ///< Total dataset row count recovered from Parquet metadata, used in count-only mode
        ///< to populate the single emitted `[row_count{N}]` chunk.

  bool single_output_emitted =
    false;  ///< Tracks the single output in count-only mode or when no files remain to read.

  std::optional<detail::hive_partitioning_state> hive;  ///< Set when Hive pruning is active.
                                                        ///< Drives per-chunk partition column
                                                        ///< synthesis in `next`.
};

}  // namespace

chunked_parquet_read_task::chunked_parquet_read_task(std::vector<std::string> file_paths,
                                                     std::vector<std::string> column_names,
                                                     std::vector<cudf::data_type> output_data_types,
                                                     std::optional<hive_pruner_options> pruner)
  : _file_paths(std::move(file_paths)),
    _column_names(std::move(column_names)),
    _output_data_types(std::move(output_data_types)),
    _pruner(std::move(pruner))
{
  GQE_EXPECTS(_column_names.size() == _output_data_types.size(),
              "chunked_parquet_read_task: column_names and output_data_types must have the same "
              "length",
              std::invalid_argument);
  if (_pruner) {
    GQE_EXPECTS(_pruner->partition_column_idx >= 0 &&
                  static_cast<std::size_t>(_pruner->partition_column_idx) < _column_names.size(),
                "chunked_parquet_read_task: hive_pruner_options::partition_column_idx is out of "
                "range",
                std::invalid_argument);
  }
}

qep::state_container chunked_parquet_read_task::initialize(qep::state_container_view inputs,
                                                           context_reference ctx_ref,
                                                           gqe::query_stream const& stream_handle,
                                                           rmm::device_async_resource_ref mr) const
{
  auto const stream = stream_handle.ref();
  gqe::utility::nvtx_scoped_range initialize_range{"chunked_parquet_read_task::initialize"};

  auto private_state               = std::make_unique<parquet_iterator_private>();
  private_state->output_data_types = _output_data_types;
  if (_file_paths.empty()) {
    return qep::state_container_builder()
      .add_state(qep::state_kind::task_private{std::move(private_state)})
      .build();
  }

  if (_column_names.empty()) {
    // Count-only mode: skip the chunked reader (it has no pages to traverse) and recover the
    // dataset's total row count from the Parquet metadata.
    auto const meta = cudf::io::read_parquet_metadata(cudf::io::source_info{_file_paths});

    GQE_EXPECTS(meta.num_rows() < std::numeric_limits<cudf::size_type>::max(),
                "chunked_parquet_read_task: total row count is too large",
                std::invalid_argument);

    private_state->count_only_rows = static_cast<cudf::size_type>(meta.num_rows());

    return qep::state_container_builder()
      .add_state(qep::state_kind::task_private{std::move(private_state)})
      .build();
  }

  // Files and column list passed to the chunked reader. The Hive branch overrides these with a
  // pruned file list and a column list that excludes the partition column.
  std::vector<std::string> reader_file_paths   = _file_paths;
  std::vector<std::string> reader_column_names = _column_names;

  if (_pruner) {
    auto const partition_column_name = _column_names[_pruner->partition_column_idx];

    // Resolve the haystack from the predecessor's output state container. The QEP-builder
    // wires a single haystack-producing predecessor whose output is shaped
    // `[cudf_column]` (single integral column).
    auto const haystack_table = qep::to_table_view(inputs);
    GQE_EXPECTS(haystack_table.has_value(),
                "chunked_parquet_read_task: haystack input is not a plain table",
                std::invalid_argument);

    // Parse Hive keys per file and intersect with the haystack. `filter_hive_files_by_haystack`
    // validates the haystack shape/type and drops nulls internally.
    auto const info = detail::filter_hive_files_by_haystack(_file_paths,
                                                            _pruner->partition_column_idx,
                                                            partition_column_name,
                                                            *haystack_table,
                                                            stream_handle.ref(),
                                                            mr);

    // Strict mode: every file must encode the partition key in its path. Files that don't match
    // the `<column>=<value>` pattern land in `non_partitioned_files`; reject the whole dataset
    // rather than silently reading a mixed layout. (Hive's `__HIVE_DEFAULT_PARTITION__` files
    // carry a null key and are dropped by `filter_hive_files_by_haystack`, not surfaced here.)
    if (!info.non_partitioned_files.empty()) {
      throw std::runtime_error(
        std::format("chunked_parquet_read_task: file `{}` does not match Hive partitioning pattern "
                    "`{}=<value>` (strict mode; mixed datasets are not supported)",
                    _file_paths[info.non_partitioned_files.front()],
                    partition_column_name));
    }

    if (info.partitioned_files.empty()) {
      // No surviving files. Leave `reader` null so `next` emits one schema-carrying empty chunk.

      return qep::state_container_builder()
        .add_state(qep::state_kind::task_private{std::move(private_state)})
        .build();
    }

    detail::hive_partitioning_state hive;
    hive.partition_column_idx = _pruner->partition_column_idx;
    hive.partition_keys.reserve(info.partitioned_files.size());
    hive.rows_per_file.reserve(info.partitioned_files.size());
    reader_file_paths.clear();
    reader_file_paths.reserve(info.partitioned_files.size());

    for (auto const& [file_idx, key_ptr] : info.partitioned_files) {
      reader_file_paths.push_back(_file_paths[file_idx]);
      hive.partition_keys.push_back(*std::static_pointer_cast<int64_t>(key_ptr));
      auto const file_meta =
        cudf::io::read_parquet_metadata(cudf::io::source_info{_file_paths[file_idx]});
      hive.rows_per_file.push_back(static_cast<cudf::size_type>(file_meta.num_rows()));
    }

    // Drop the partition column from the on-disk read; synthesis happens per-chunk in `next`.
    reader_column_names.erase(reader_column_names.begin() + _pruner->partition_column_idx);

    private_state->hive = std::move(hive);
  }
  auto const& query_ctx = *ctx_ref._query_context;

  auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info{reader_file_paths})
                   .column_names(reader_column_names)
                   .build();

  private_state->reader =
    std::make_unique<cudf::io::chunked_parquet_reader>(query_ctx.parameters.parquet_chunk_read_size,
                                                       query_ctx.parameters.parquet_pass_read_size,
                                                       options,
                                                       stream,
                                                       mr);
  return qep::state_container_builder()
    .add_state(qep::state_kind::task_private{std::move(private_state)})
    .build();
}

std::optional<qep::state_container> chunked_parquet_read_task::next(
  qep::state_container_view iterator,
  context_reference /* ctx_ref */,
  gqe::query_stream const& stream_handle,
  rmm::device_async_resource_ref mr) const
{
  auto const stream = stream_handle.ref();
  gqe::utility::nvtx_scoped_range range{"chunked_parquet_read_task::next"};

  auto* private_state = std::get_if<qep::state_kind::task_private>(iterator[0].get());
  assert(private_state != nullptr);
  // The `task_private` slot was populated in `initialize` with this concrete type, so a
  // `static_cast` is sufficient.
  auto* iter_data = static_cast<parquet_iterator_private*>(private_state->data.get());

  std::optional<cudf::io::table_with_metadata> chunk;
  std::optional<detail::hive_chunk_segments> hive_segments;
  std::optional<std::size_t> partition_column_idx;

  {
    // The chunked reader is internally stateful and not safe under concurrent calls to
    // `read_chunk`. Concurrent `next` calls are permitted by the contract, so serialize access
    // to the reader and Hive cursor state with the mutex.
    std::lock_guard<std::mutex> const guard{iter_data->mutex};

    if (iter_data->reader == nullptr && iter_data->single_output_emitted) {
      return std::nullopt;
    } else if (iter_data->reader == nullptr) {
      iter_data->single_output_emitted = true;
      if (!iter_data->output_data_types.empty()) {
        return qep::make_empty_state_container(iter_data->output_data_types);
      }
      return qep::make_row_count_container(iter_data->count_only_rows);
    } else if (!iter_data->reader->has_next()) {
      return std::nullopt;
    }

    chunk = iter_data->reader->read_chunk();

    if (iter_data->hive) {
      hive_segments = detail::compute_hive_chunk_segments(*iter_data->hive, chunk->tbl->num_rows());
      partition_column_idx = static_cast<std::size_t>(iter_data->hive->partition_column_idx);
    }
  }

  assert(chunk.has_value());

  if (hive_segments) {
    // Synthesize the partition column for this chunk and splice it back at the original
    // index, so the result table layout matches `_output_data_types` (and `_column_names`).
    assert(partition_column_idx.has_value());
    auto loaded_columns = chunk->tbl->release();
    loaded_columns.insert(loaded_columns.begin() + *partition_column_idx,
                          construct_partition_key_column(_output_data_types[*partition_column_idx],
                                                         std::move(hive_segments->keys),
                                                         std::move(hive_segments->rows),
                                                         stream,
                                                         mr));
    chunk->tbl = std::make_unique<cudf::table>(std::move(loaded_columns));
  }

  auto enforced =
    detail::enforce_data_types(std::move(chunk->tbl), iter_data->output_data_types, stream, mr);

  return qep::state_container_builder().add_state(std::move(*enforced), stream_handle).build();
}

void chunked_parquet_read_task::finalize(qep::state_container&& iterator,
                                         context_reference /* ctx_ref */,
                                         gqe::query_stream const& /* stream_handle */,
                                         rmm::device_async_resource_ref /* mr */) const
{
  auto* private_state = std::get_if<qep::state_kind::task_private>(iterator[0].get());
  assert(private_state != nullptr);
  auto* iter_data = static_cast<parquet_iterator_private*>(private_state->data.get());

  // This gets called after all next calls have returned, so we don't need to synchronize here.
  iter_data->reader.reset();
}

std::size_t chunked_parquet_read_task::work_unit_count() const noexcept
{
  return _file_paths.size();
}

std::unique_ptr<qep::iterate_task> chunked_parquet_read_task::split(
  qep::source_work_range range) const
{
  return std::make_unique<chunked_parquet_read_task>(
    detail::select_source_work(_file_paths, range), _column_names, _output_data_types, _pruner);
}

std::unique_ptr<qep::task> chunked_parquet_read_task::clone() const
{
  return split({0, work_unit_count()});
}

}  // namespace storage
}  // namespace gqe
