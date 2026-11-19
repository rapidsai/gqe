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

#include <gqe/storage/parquet_reference_iterate.hpp>

#include "source_work_split.hpp"

#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/query_context.hpp>
#include <gqe/storage/parquet_footer_cache.hpp>
#include <gqe/storage/source_references.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/utilities/span.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gqe::storage {

namespace {

using hybrid_scan_reader = cudf::io::parquet::experimental::hybrid_scan_reader;

struct parquet_reference_iterator_private : qep::task_private_state {
  std::atomic<std::size_t> next_index{0};
  std::atomic<bool> single_output_emitted{false};
  std::optional<cudf::size_type> count_only_rows;
  std::vector<std::shared_ptr<parquet_reference const>> references;
};

/**
 * @brief Sum the row count of one contiguous row-group span.
 */
[[nodiscard]] cudf::size_type sum_span_rows(cudf::io::parquet::FileMetaData const& metadata,
                                            std::size_t begin,
                                            std::size_t count)
{
  // Bounding every partial sum by cudf's row limit keeps the accumulation far from
  // INT64_MAX, so corrupt footers cannot drive the addition into signed overflow.
  auto constexpr max_rows = std::int64_t{std::numeric_limits<cudf::size_type>::max()};
  auto total_rows         = std::int64_t{0};
  for (auto idx = begin; idx < begin + count; ++idx) {
    auto const& row_group = metadata.row_groups[idx];
    GQE_EXPECTS(row_group.num_rows >= 0,
                "parquet_reference_iterate_task: row group has a negative row count",
                std::logic_error);
    GQE_EXPECTS(row_group.num_rows <= max_rows - total_rows,
                "parquet_reference_iterate_task: referenced span exceeds cuDF's row limit",
                std::overflow_error);
    total_rows += row_group.num_rows;
  }
  return static_cast<cudf::size_type>(total_rows);
}

}  // namespace

parquet_reference_iterate_task::parquet_reference_iterate_task(
  std::vector<std::string> file_paths,
  std::vector<std::string> column_names,
  std::vector<cudf::data_type> output_data_types,
  std::optional<hive_pruner_options> pruner,
  std::shared_ptr<parquet_footer_cache> footer_cache)
  : _file_paths(std::move(file_paths)),
    _column_names(std::move(column_names)),
    _output_data_types(std::move(output_data_types)),
    _pruner(std::move(pruner)),
    _footer_cache(std::move(footer_cache))
{
  GQE_EXPECTS(_column_names.size() == _output_data_types.size(),
              "parquet_reference_iterate_task: column_names and output_data_types must have the "
              "same length",
              std::invalid_argument);
  GQE_EXPECTS(_footer_cache != nullptr,
              "parquet_reference_iterate_task: footer_cache must not be null",
              std::invalid_argument);
  GQE_EXPECTS(!_column_names.empty() || !_pruner,
              "parquet_reference_iterate_task: Hive pruning is not supported in count-only mode",
              std::invalid_argument);
  if (_pruner) {
    GQE_EXPECTS(_pruner->partition_column_idx >= 0 &&
                  static_cast<std::size_t>(_pruner->partition_column_idx) < _column_names.size(),
                "parquet_reference_iterate_task: hive_pruner_options::partition_column_idx is out "
                "of range",
                std::invalid_argument);
  }
}

parquet_reference_iterate_task::~parquet_reference_iterate_task() = default;

qep::state_container parquet_reference_iterate_task::initialize(
  qep::state_container_view inputs,
  context_reference ctx_ref,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref mr) const
{
  utility::nvtx_scoped_range range{"parquet_reference_iterate_task::initialize"};

  GQE_EXPECTS(ctx_ref._query_context != nullptr,
              "parquet_reference_iterate_task: query context must not be null",
              std::invalid_argument);

  auto private_state = std::make_unique<parquet_reference_iterator_private>();

  if (_column_names.empty()) {
    // Bound every partial sum by cudf's row limit so corrupt footers cannot drive the addition
    // into signed overflow.
    auto constexpr max_rows = std::int64_t{std::numeric_limits<cudf::size_type>::max()};
    auto total_rows         = std::int64_t{0};
    for (auto const& path : _file_paths) {
      auto const file_rows = _footer_cache->get_or_load(path)->num_rows;
      GQE_EXPECTS(file_rows >= 0,
                  "parquet_reference_iterate_task: file footer has a negative row count",
                  std::logic_error);
      GQE_EXPECTS(file_rows <= max_rows - total_rows,
                  "parquet_reference_iterate_task: total row count exceeds cuDF's row limit",
                  std::overflow_error);
      total_rows += file_rows;
    }
    private_state->count_only_rows = static_cast<cudf::size_type>(total_rows);
    return qep::state_container_builder()
      .add_state(qep::state_kind::task_private{std::move(private_state)})
      .build();
  }

  // Selected files paired with their Hive partition key, when pruning is configured.
  std::vector<std::pair<std::size_t, std::optional<std::int64_t>>> selected_files;
  selected_files.reserve(_file_paths.size());
  for (std::size_t i = 0; i < _file_paths.size(); ++i) {
    selected_files.emplace_back(i, std::nullopt);
  }
  std::vector<std::string> reader_column_names = _column_names;

  if (_pruner) {
    auto const partition_column_name = _column_names[_pruner->partition_column_idx];
    reader_column_names.erase(reader_column_names.begin() + _pruner->partition_column_idx);

    auto const haystack_table = qep::to_table_view(inputs);
    GQE_EXPECTS(haystack_table.has_value(),
                "parquet_reference_iterate_task: haystack input is not a plain table",
                std::invalid_argument);

    auto const info = detail::filter_hive_files_by_haystack(_file_paths,
                                                            _pruner->partition_column_idx,
                                                            partition_column_name,
                                                            *haystack_table,
                                                            stream_handle.ref(),
                                                            mr);
    if (!info.non_partitioned_files.empty()) {
      throw std::runtime_error(std::format(
        "parquet_reference_iterate_task: file `{}` does not match Hive partitioning pattern "
        "`{}=<value>` (strict mode; mixed datasets are not supported)",
        _file_paths[info.non_partitioned_files.front()],
        partition_column_name));
    }

    selected_files.clear();
    selected_files.reserve(info.partitioned_files.size());
    for (auto const& [file_idx, partition_key] : info.partitioned_files) {
      // The filter stores each surviving file's parsed key as a type-erased shared value.
      selected_files.emplace_back(file_idx, *static_cast<std::int64_t const*>(partition_key.get()));
    }
  } else {
    GQE_EXPECTS(inputs.empty(),
                "parquet_reference_iterate_task: source task expects no initialize inputs",
                std::invalid_argument);
  }

  auto const options = cudf::io::parquet_reader_options::builder()
                         .column_names(std::move(reader_column_names))
                         .build();
  auto const pass_read_limit = ctx_ref._query_context->parameters.parquet_pass_read_size;

  for (auto const& [file_idx, hive_partition_key] : selected_files) {
    auto const& path = _file_paths[file_idx];

    auto const file   = _footer_cache->get_or_load(path);
    auto setup_reader = hybrid_scan_reader{*file, options};

    auto const row_groups = setup_reader.all_row_groups(options);
    if (row_groups.empty()) { continue; }

    auto const passes = setup_reader.construct_row_group_passes(
      cudf::host_span<cudf::size_type const>{row_groups.data(), row_groups.size()},
      pass_read_limit);
    for (auto const& pass : passes) {
      if (pass.empty()) { continue; }

      for (std::size_t pass_idx = 0; pass_idx < pass.size(); ++pass_idx) {
        auto const row_group = pass[pass_idx];
        GQE_EXPECTS(row_group >= 0 && static_cast<std::size_t>(row_group) < file->row_groups.size(),
                    "parquet_reference_iterate_task: generated an unavailable row group",
                    std::logic_error);
        if (pass_idx > 0) {
          GQE_EXPECTS(pass[pass_idx - 1] < std::numeric_limits<cudf::size_type>::max() &&
                        row_group == pass[pass_idx - 1] + 1,
                      "parquet_reference_iterate_task: generated row-group pass is not contiguous",
                      std::logic_error);
        }
      }

      auto const span_rows =
        sum_span_rows(*file, static_cast<std::size_t>(pass.front()), pass.size());
      if (span_rows == 0) { continue; }

      private_state->references.push_back(
        parquet_reference::make(path,
                                pass.front(),
                                static_cast<cudf::size_type>(pass.size()),
                                span_rows,
                                hive_partition_key));
    }
  }

  return qep::state_container_builder()
    .add_state(qep::state_kind::task_private{std::move(private_state)})
    .build();
}

std::optional<qep::state_container> parquet_reference_iterate_task::next(
  qep::state_container_view iterator,
  context_reference /* ctx_ref */,
  query_stream const& /* stream_handle */,
  rmm::device_async_resource_ref /* mr */) const
{
  utility::nvtx_scoped_range range{"parquet_reference_iterate_task::next"};

  auto& iter_data = qep::get_task_private_state<parquet_reference_iterator_private>(iterator, 0);

  // Three modes: emit the count once, emit the empty-input reference once, or claim one reference.
  if (iter_data.count_only_rows) {
    if (iter_data.single_output_emitted.exchange(true, std::memory_order_relaxed)) {
      return std::nullopt;
    }
    return qep::make_row_count_container(*iter_data.count_only_rows);
  } else if (iter_data.references.empty()) {
    if (iter_data.single_output_emitted.exchange(true, std::memory_order_relaxed)) {
      return std::nullopt;
    }
    return qep::state_container_builder()
      .add_state(qep::state_kind::reference{parquet_reference::make_empty()})
      .build();
  } else {
    auto const work_index = iter_data.next_index.fetch_add(1, std::memory_order_relaxed);
    if (work_index >= iter_data.references.size()) { return std::nullopt; }
    return qep::state_container_builder()
      .add_state(qep::state_kind::reference{iter_data.references[work_index]})
      .build();
  }
}

std::size_t parquet_reference_iterate_task::work_unit_count() const noexcept
{
  return _file_paths.size();
}

std::unique_ptr<qep::iterate_task> parquet_reference_iterate_task::split(
  qep::source_work_range range) const
{
  return std::make_unique<parquet_reference_iterate_task>(
    detail::select_source_work(_file_paths, range),
    _column_names,
    _output_data_types,
    _pruner,
    _footer_cache);
}

std::unique_ptr<qep::task> parquet_reference_iterate_task::clone() const
{
  return split({0, work_unit_count()});
}

}  // namespace gqe::storage
