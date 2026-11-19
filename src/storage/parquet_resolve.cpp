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

#include <gqe/storage/parquet_resolve.hpp>

#include "parquet_common.hpp"

#include <gqe/qep/shapes/deferred_column.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/query_context.hpp>
#include <gqe/storage/parquet_footer_cache.hpp>
#include <gqe/storage/source_references.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/span.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace gqe::storage {

namespace {

using hybrid_scan_reader = cudf::io::parquet::experimental::hybrid_scan_reader;

/**
 * @brief Return the top-level column names of a Parquet file schema.
 *
 * The schema is stored depth-first with the root at index zero, so each top-level column is a
 * child of the root and its own subtree is skipped by counting descendants.
 */
[[nodiscard]] std::vector<std::string_view> top_level_column_names(
  cudf::io::parquet::FileMetaData const& metadata)
{
  std::vector<std::string_view> names;
  if (metadata.schema.empty()) { return names; }
  std::size_t index = 1;
  for (int32_t child = 0; child < metadata.schema.front().num_children; ++child) {
    GQE_EXPECTS(index < metadata.schema.size(),
                "parquet_resolve_task: file schema is truncated",
                std::logic_error);
    names.push_back(metadata.schema[index].name);
    auto pending = static_cast<std::int64_t>(metadata.schema[index].num_children);
    ++index;
    while (pending > 0) {
      GQE_EXPECTS(index < metadata.schema.size(),
                  "parquet_resolve_task: file schema is truncated",
                  std::logic_error);
      pending += metadata.schema[index].num_children - 1;
      ++index;
    }
  }
  return names;
}

/**
 * @brief Deferred source materializing columns of one Parquet row-group span.
 *
 * Retains the cached parsed footer, so every decode constructs its hybrid-scan reader without
 * re-parsing. Each decode reads and decodes only the requested columns' chunks. The byte-range
 * read's host-side completion wait blocks that call. A requested Hive partition column is
 * synthesized from the reference's mint-time key without reading file bytes.
 */
class parquet_deferred_source final : public qep::deferred_table_source_state {
 public:
  parquet_deferred_source(cudf::size_type num_rows,
                          std::shared_ptr<cudf::io::parquet::FileMetaData const> file,
                          std::shared_ptr<parquet_reference const> reference,
                          std::vector<std::string> column_names,
                          std::vector<cudf::data_type> output_data_types,
                          std::optional<std::size_t> partition_column_idx,
                          std::size_t chunk_read_limit,
                          std::size_t pass_read_limit)
    : qep::deferred_table_source_state{num_rows, std::move(output_data_types)},
      _file{std::move(file)},
      _reference{std::move(reference)},
      _column_names{std::move(column_names)},
      _partition_column_idx{partition_column_idx},
      _chunk_read_limit{chunk_read_limit},
      _pass_read_limit{pass_read_limit}
  {
  }

 protected:
  [[nodiscard]] std::vector<std::unique_ptr<cudf::column>> materialize_columns_impl(
    std::span<cudf::size_type const> source_column_indices,
    query_stream const& stream_handle,
    rmm::device_async_resource_ref mr) const override
  {
    auto const stream = stream_handle.ref();
    utility::nvtx_scoped_range range{"parquet_deferred_source::materialize"};

    // A requested Hive partition column is synthesized; every other requested column is decoded
    // from the file in one pass over its column chunks.
    auto const is_partition_column = [&](cudf::size_type index) {
      return _partition_column_idx.has_value() &&
             static_cast<std::size_t>(index) == *_partition_column_idx;
    };

    std::vector<std::string> reader_names;
    for (auto const index : source_column_indices) {
      if (!is_partition_column(index)) {
        reader_names.push_back(_column_names[static_cast<std::size_t>(index)]);
      }
    }

    std::vector<std::unique_ptr<cudf::column>> decoded_columns;
    if (!reader_names.empty()) { decoded_columns = decode_file_columns(reader_names, stream, mr); }

    std::vector<std::unique_ptr<cudf::column>> result;
    std::vector<cudf::data_type> types;
    result.reserve(source_column_indices.size());
    types.reserve(source_column_indices.size());
    std::size_t next_decoded = 0;
    for (auto const index : source_column_indices) {
      if (is_partition_column(index)) {
        GQE_EXPECTS(_reference->hive_partition_key().has_value(),
                    "parquet_deferred_source: partition column requested without a mint-time key");
        result.push_back(construct_partition_key_column(
          column_type(index), {*_reference->hive_partition_key()}, {num_rows()}, stream, mr));
      } else {
        result.push_back(std::move(decoded_columns[next_decoded++]));
      }
      types.push_back(column_type(index));
    }

    return detail::enforce_data_types(
             std::make_unique<cudf::table>(std::move(result)), types, stream, mr)
      ->release();
  }

 private:
  [[nodiscard]] std::vector<std::unique_ptr<cudf::column>> decode_file_columns(
    std::vector<std::string> const& reader_names,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const
  {
    auto const options =
      cudf::io::parquet_reader_options::builder().column_names(reader_names).build();

    std::vector<cudf::size_type> row_groups(
      static_cast<std::size_t>(_reference->row_group_count()));
    std::iota(row_groups.begin(), row_groups.end(), _reference->row_group_begin());
    auto const row_group_span =
      cudf::host_span<cudf::size_type const>{row_groups.data(), row_groups.size()};

    auto datasource  = cudf::io::datasource::create(_reference->file_path());
    auto reader      = hybrid_scan_reader{*_file, options};
    auto byte_ranges = reader.all_column_chunks_byte_ranges(row_group_span, options);
    auto [compressed_buffers, column_chunk_data, read_future] =
      cudf::io::parquet::fetch_byte_ranges_to_device_async(
        *datasource,
        cudf::host_span<cudf::io::parquet::byte_range_info const>{byte_ranges.data(),
                                                                  byte_ranges.size()},
        cudf::io::parquet::io_submission_policy::SERIALIZE,
        stream,
        mr);
    read_future.get();

    reader.setup_chunking_for_all_columns(_chunk_read_limit,
                                          _pass_read_limit,
                                          row_group_span,
                                          cudf::host_span<cudf::device_span<uint8_t const> const>{
                                            column_chunk_data.data(), column_chunk_data.size()},
                                          options,
                                          stream,
                                          mr);
    std::vector<std::unique_ptr<cudf::table>> chunks;
    while (reader.has_next_table_chunk()) {
      chunks.push_back(reader.materialize_all_columns_chunk().tbl);
    }
    GQE_EXPECTS(!chunks.empty(),
                "parquet_deferred_source: referenced span produced no table",
                std::logic_error);

    std::unique_ptr<cudf::table> output;
    if (chunks.size() == 1) {
      output = std::move(chunks.front());
    } else {
      std::vector<cudf::table_view> views;
      views.reserve(chunks.size());
      std::transform(
        chunks.begin(), chunks.end(), std::back_inserter(views), [](auto const& chunk) {
          return chunk->view();
        });
      output = cudf::concatenate(views, stream, mr);
    }

    GQE_EXPECTS(output->num_rows() == num_rows(),
                "parquet_deferred_source: materialized row count differs from footer metadata",
                std::logic_error);
    return output->release();
  }

  std::shared_ptr<cudf::io::parquet::FileMetaData const> _file;
  std::shared_ptr<parquet_reference const> _reference;
  std::vector<std::string> _column_names;
  std::optional<std::size_t> _partition_column_idx;
  std::size_t _chunk_read_limit;
  std::size_t _pass_read_limit;
};

}  // namespace

parquet_resolve_task::parquet_resolve_task(std::shared_ptr<parquet_footer_cache> footer_cache,
                                           std::vector<std::string> column_names,
                                           std::vector<cudf::data_type> output_data_types,
                                           std::optional<hive_pruner_options> pruner)
  : _footer_cache{std::move(footer_cache)},
    _column_names{std::move(column_names)},
    _output_data_types{std::move(output_data_types)},
    _pruner{std::move(pruner)}
{
  GQE_EXPECTS(_footer_cache != nullptr,
              "parquet_resolve_task: footer_cache must not be null",
              std::invalid_argument);
  GQE_EXPECTS(_column_names.size() == _output_data_types.size(),
              "parquet_resolve_task: column_names and output_data_types must have the same length",
              std::invalid_argument);
  if (_pruner) {
    GQE_EXPECTS(_pruner->partition_column_idx >= 0 &&
                  static_cast<std::size_t>(_pruner->partition_column_idx) < _column_names.size(),
                "parquet_resolve_task: hive_pruner_options::partition_column_idx is out of range",
                std::invalid_argument);
  }
}

std::optional<qep::state_container> parquet_resolve_task::next(
  qep::state_container_view inputs,
  context_reference ctx_ref,
  query_stream const& /* stream_handle */,
  rmm::device_async_resource_ref /* mr */) const
{
  utility::nvtx_scoped_range range{"parquet_resolve_task::next"};

  GQE_EXPECTS(ctx_ref._query_context != nullptr,
              "parquet_resolve_task: query context must not be null",
              std::invalid_argument);
  GQE_EXPECTS(inputs.size() == 1,
              "parquet_resolve_task: expected exactly one input state",
              std::invalid_argument);
  GQE_EXPECTS(inputs[0] != nullptr, "parquet_resolve_task: null input state slot");

  if (qep::try_row_count(inputs)) {
    // Count-only chunks pass through unchanged, so count-only plans tolerate an inserted resolve
    // task.
    return qep::make_mutable_state_copy(inputs);
  } else {
    GQE_EXPECTS(!_column_names.empty(),
                "parquet_resolve_task: an empty projection accepts only count-only chunks",
                std::invalid_argument);
    auto const* reference_slot = std::get_if<qep::state_kind::reference>(inputs[0].get());
    GQE_EXPECTS(reference_slot != nullptr,
                "parquet_resolve_task: input is not a reference chunk",
                std::invalid_argument);
    auto reference = std::dynamic_pointer_cast<parquet_reference const>(reference_slot->state());
    GQE_EXPECTS(reference != nullptr,
                "parquet_resolve_task: reference does not address a Parquet source",
                std::invalid_argument);

    if (reference->is_empty()) {
      return qep::make_empty_state_container(_output_data_types);
    } else {
      return resolve(std::move(reference), ctx_ref);
    }
  }
}

qep::state_container parquet_resolve_task::resolve(
  std::shared_ptr<parquet_reference const> reference, context_reference ctx_ref) const
{
  GQE_EXPECTS(!_pruner || reference->hive_partition_key().has_value(),
              "parquet_resolve_task: Hive synthesis requires a reference with a partition key",
              std::invalid_argument);

  auto file = _footer_cache->get_or_load(reference->file_path());

  auto const begin = static_cast<std::size_t>(reference->row_group_begin());
  auto const count = static_cast<std::size_t>(reference->row_group_count());
  GQE_EXPECTS(begin < file->row_groups.size() && count <= file->row_groups.size() - begin,
              "parquet_resolve_task: reference contains an unavailable row group",
              std::invalid_argument);

  auto span_rows = std::int64_t{0};
  for (auto idx = begin; idx < begin + count; ++idx) {
    span_rows += file->row_groups[idx].num_rows;
  }
  GQE_EXPECTS(span_rows == reference->num_rows(),
              "parquet_resolve_task: reference row count differs from footer metadata");

  auto partition_column_idx = std::optional<std::size_t>{};
  if (_pruner) { partition_column_idx = static_cast<std::size_t>(_pruner->partition_column_idx); }

  // References carry no schema, so fail fast if the plan projects a column the resolved file does
  // not have, instead of surfacing a reader error inside a consumer's materialization. Stored types
  // are enforced by cast at materialization, as in every Parquet read path.
  auto const file_columns = top_level_column_names(*file);
  for (std::size_t i = 0; i < _column_names.size(); ++i) {
    if (partition_column_idx.has_value() && i == *partition_column_idx) { continue; }
    GQE_EXPECTS(std::ranges::find(file_columns, _column_names[i]) != file_columns.end(),
                std::format("parquet_resolve_task: projected column `{}` is not in the file schema",
                            _column_names[i]),
                std::invalid_argument);
  }

  auto const& parameters = ctx_ref._query_context->parameters;
  auto source            = std::make_shared<parquet_deferred_source>(reference->num_rows(),
                                                          std::move(file),
                                                          std::move(reference),
                                                          _column_names,
                                                          _output_data_types,
                                                          partition_column_idx,
                                                          parameters.parquet_chunk_read_size,
                                                          parameters.parquet_pass_read_size);
  return qep::make_deferred_column_container(std::move(source));
}

std::unique_ptr<qep::task> parquet_resolve_task::clone() const
{
  return std::make_unique<parquet_resolve_task>(
    _footer_cache, _column_names, _output_data_types, _pruner);
}

}  // namespace gqe::storage
