/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <gqe/node_manager/stats_tables.hpp>

#include <gqe/types.hpp>

#include <cudf/types.hpp>

#include <arrow/api.h>

#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace gqe::node_manager {

namespace {

cudf::data_type int64_type() { return cudf::data_type{cudf::type_id::INT64}; }
cudf::data_type string_type() { return cudf::data_type{cudf::type_id::STRING}; }

/// The seven fields of a `fixed_width_compression_statistics`, in declaration order. Appended to a
/// schema with a prefix so the aggregate (no prefix), `offsets_`, and `chars_` blocks stay aligned.
void append_compression_columns(std::vector<column_traits>& columns, std::string const& prefix)
{
  columns.emplace_back(prefix + "num_compressed_row_groups", int64_type());
  columns.emplace_back(prefix + "compressed_size", int64_type());
  columns.emplace_back(prefix + "uncompressed_size", int64_type());
  columns.emplace_back(prefix + "primary_compressed_size", int64_type());
  columns.emplace_back(prefix + "secondary_compressed_size", int64_type());
  columns.emplace_back(prefix + "num_primary_compressed_row_groups", int64_type());
  columns.emplace_back(prefix + "num_secondary_compressed_row_groups", int64_type());
}

}  // namespace

bool is_stats_table(std::string_view name)
{
  return name == table_stats_table_name || name == column_stats_table_name;
}

std::vector<column_traits> table_stats_schema()
{
  std::vector<column_traits> columns;
  columns.emplace_back("table_name", string_type());
  columns.emplace_back("num_rows", int64_type());
  columns.emplace_back("num_row_groups", int64_type());
  columns.emplace_back("num_columns", int64_type());
  columns.emplace_back("compressed_size", int64_type());
  columns.emplace_back("uncompressed_size", int64_type());
  columns.emplace_back("primary_compressed_size", int64_type());
  columns.emplace_back("secondary_compressed_size", int64_type());
  return columns;
}

std::vector<column_traits> column_stats_schema()
{
  std::vector<column_traits> columns;
  // Identity.
  columns.emplace_back("table_name", string_type());
  columns.emplace_back("column_id", int64_type());
  columns.emplace_back("column_name", string_type());
  columns.emplace_back("column_type", string_type());
  // Type-agnostic aggregate (sizes summed across buffers; counts null for string columns).
  append_compression_columns(columns, "");
  // Per-buffer breakdown for string columns (null for fixed-width columns).
  append_compression_columns(columns, "offsets_");
  append_compression_columns(columns, "chars_");
  return columns;
}

void register_stats_tables(catalog& catalog)
{
  catalog.register_system_table(std::string{table_stats_table_name},
                                table_stats_schema(),
                                storage_kind::shared_segment_memory{},
                                partitioning_schema_kind::none{});
  catalog.register_system_table(std::string{column_stats_table_name},
                                column_stats_schema(),
                                storage_kind::shared_segment_memory{},
                                partitioning_schema_kind::none{});
}

// ============================================================================
// Projection of catalog statistics into record batches
// ============================================================================

namespace {

void check(arrow::Status const& status, std::string_view what)
{
  if (!status.ok()) {
    throw std::runtime_error(std::format("stats projection: {}: {}", what, status.ToString()));
  }
}

/// Map a cuDF data type to the Arrow type used in the projected batch. Only the types used by the
/// statistics schemas are supported.
std::shared_ptr<arrow::DataType> arrow_type_for(cudf::data_type type)
{
  switch (type.id()) {
    case cudf::type_id::INT64: return arrow::int64();
    case cudf::type_id::STRING: return arrow::utf8();
    default:
      throw std::logic_error(std::format("stats projection: unsupported column type id {}",
                                         static_cast<int>(type.id())));
  }
}

/// Build the Arrow schema for a statistics table from its column traits (single source of truth).
std::shared_ptr<arrow::Schema> arrow_schema_for(std::vector<column_traits> const& columns)
{
  arrow::FieldVector fields;
  fields.reserve(columns.size());
  for (auto const& column : columns) {
    fields.push_back(arrow::field(column.name, arrow_type_for(column.data_type)));
  }
  return arrow::schema(std::move(fields));
}

/// Human-readable name for a cuDF data type, used for the `column_type` column.
std::string cudf_type_to_string(cudf::data_type type)
{
  switch (type.id()) {
    case cudf::type_id::BOOL8: return "BOOL8";
    case cudf::type_id::INT8: return "INT8";
    case cudf::type_id::INT16: return "INT16";
    case cudf::type_id::INT32: return "INT32";
    case cudf::type_id::INT64: return "INT64";
    case cudf::type_id::UINT8: return "UINT8";
    case cudf::type_id::UINT16: return "UINT16";
    case cudf::type_id::UINT32: return "UINT32";
    case cudf::type_id::UINT64: return "UINT64";
    case cudf::type_id::FLOAT32: return "FLOAT32";
    case cudf::type_id::FLOAT64: return "FLOAT64";
    case cudf::type_id::STRING: return "STRING";
    case cudf::type_id::TIMESTAMP_DAYS: return "TIMESTAMP_DAYS";
    case cudf::type_id::TIMESTAMP_SECONDS: return "TIMESTAMP_SECONDS";
    case cudf::type_id::TIMESTAMP_MILLISECONDS: return "TIMESTAMP_MILLISECONDS";
    case cudf::type_id::TIMESTAMP_MICROSECONDS: return "TIMESTAMP_MICROSECONDS";
    case cudf::type_id::TIMESTAMP_NANOSECONDS: return "TIMESTAMP_NANOSECONDS";
    default: return std::format("type_id({})", static_cast<int>(type.id()));
  }
}

/// Fold a column's per-buffer compression statistics into a single aggregate. For a fixed-width
/// column this is its single buffer; for a string column it is `offsets + chars` (sizes sum). The
/// caller decides which aggregate fields are meaningful (see the schema notes on null counts).
fixed_width_compression_statistics fold_buffers(column_compression_statistics const& stats,
                                                bool& is_string)
{
  if (auto const* fixed = std::get_if<fixed_width_compression_statistics>(&stats)) {
    is_string = false;
    return *fixed;
  }
  auto const& str                              = std::get<string_compression_statistics>(stats);
  is_string                                    = true;
  fixed_width_compression_statistics aggregate = str.offsets_stats;
  aggregate += str.chars_stats;
  return aggregate;
}

/// A nullable INT64 output column accumulated row-by-row, then finished into an Arrow array.
struct int64_column {
  std::vector<std::int64_t> values;
  std::vector<std::uint8_t> valid;

  void push(std::int64_t value)
  {
    values.push_back(value);
    valid.push_back(1);
  }
  void push_null()
  {
    values.push_back(0);
    valid.push_back(0);
  }
  std::shared_ptr<arrow::Array> finish()
  {
    arrow::Int64Builder builder;
    // Pointer overload: `valid` holds one byte per element (non-zero = valid).
    check(
      builder.AppendValues(values.data(), static_cast<std::int64_t>(values.size()), valid.data()),
      "append int64 values");
    std::shared_ptr<arrow::Array> array;
    check(builder.Finish(&array), "finish int64 column");
    return array;
  }
};

/// A non-null UTF-8 output column.
struct string_column {
  std::vector<std::string> values;

  void push(std::string value) { values.push_back(std::move(value)); }
  std::shared_ptr<arrow::Array> finish()
  {
    arrow::StringBuilder builder;
    check(builder.AppendValues(values), "append string values");
    std::shared_ptr<arrow::Array> array;
    check(builder.Finish(&array), "finish string column");
    return array;
  }
};

/// The seven fields of a `fixed_width_compression_statistics`, as parallel nullable INT64 columns,
/// in the order produced by `append_compression_columns`.
struct compression_block {
  int64_column num_compressed_row_groups;
  int64_column compressed_size;
  int64_column uncompressed_size;
  int64_column primary_compressed_size;
  int64_column secondary_compressed_size;
  int64_column num_primary_compressed_row_groups;
  int64_column num_secondary_compressed_row_groups;

  /// Append all seven fields from a single buffer's statistics.
  void push(fixed_width_compression_statistics const& fixed)
  {
    num_compressed_row_groups.push(static_cast<std::int64_t>(fixed.num_compressed_row_groups));
    compressed_size.push(fixed.compressed_size);
    uncompressed_size.push(fixed.uncompressed_size);
    primary_compressed_size.push(fixed.primary_compressed_size);
    secondary_compressed_size.push(fixed.secondary_compressed_size);
    num_primary_compressed_row_groups.push(
      static_cast<std::int64_t>(fixed.num_primary_compressed_row_groups));
    num_secondary_compressed_row_groups.push(
      static_cast<std::int64_t>(fixed.num_secondary_compressed_row_groups));
  }

  /// Append the aggregate of a string column: sizes are summed across buffers; the row-group-count
  /// fields are null because counts are not meaningfully combined across buffers (the authoritative
  /// per-buffer counts live in the offsets/chars blocks).
  void push_aggregate_sizes_only(fixed_width_compression_statistics const& summed)
  {
    num_compressed_row_groups.push_null();
    compressed_size.push(summed.compressed_size);
    uncompressed_size.push(summed.uncompressed_size);
    primary_compressed_size.push(summed.primary_compressed_size);
    secondary_compressed_size.push(summed.secondary_compressed_size);
    num_primary_compressed_row_groups.push_null();
    num_secondary_compressed_row_groups.push_null();
  }

  /// Append a fully-null row (e.g. an offsets/chars block for a non-string column).
  void push_null()
  {
    num_compressed_row_groups.push_null();
    compressed_size.push_null();
    uncompressed_size.push_null();
    primary_compressed_size.push_null();
    secondary_compressed_size.push_null();
    num_primary_compressed_row_groups.push_null();
    num_secondary_compressed_row_groups.push_null();
  }

  /// Finish into seven arrays, appended in schema order to @p out.
  void finish_into(std::vector<std::shared_ptr<arrow::Array>>& out)
  {
    out.push_back(num_compressed_row_groups.finish());
    out.push_back(compressed_size.finish());
    out.push_back(uncompressed_size.finish());
    out.push_back(primary_compressed_size.finish());
    out.push_back(secondary_compressed_size.finish());
    out.push_back(num_primary_compressed_row_groups.finish());
    out.push_back(num_secondary_compressed_row_groups.finish());
  }
};

/// Registered tables to project: every in-memory table except the statistics tables themselves.
///
/// Only in-memory tables carry row-group and compression statistics. File-backed (parquet) tables
/// are excluded — this covers both user external tables (CREATE EXTERNAL TABLE) and COPY's
/// transient
/// "__pq_*" parquet staging tables, neither of which has meaningful in-memory statistics.
std::vector<std::string> projected_table_names(catalog const& catalog)
{
  std::vector<std::string> names;
  for (auto const& name : catalog.table_names()) {
    if (is_stats_table(name)) { continue; }
    if (storage_kind::is_file_backed(catalog.storage_kind(name))) { continue; }
    names.push_back(name);
  }
  return names;
}

}  // namespace

std::shared_ptr<arrow::RecordBatch> project_table_stats(catalog const& catalog)
{
  string_column table_name;
  int64_column num_rows;
  int64_column num_row_groups;
  int64_column num_columns;
  int64_column compressed_size;
  int64_column uncompressed_size;
  int64_column primary_compressed_size;
  int64_column secondary_compressed_size;

  auto const names = projected_table_names(catalog);
  for (auto const& name : names) {
    auto const stats = catalog.statistics(name)->statistics();

    std::int64_t table_compressed = 0, table_uncompressed = 0, table_primary = 0,
                 table_secondary = 0;
    for (auto const& column : stats.column_stats) {
      bool is_string = false;
      fixed_width_compression_statistics const aggregate =
        fold_buffers(column.compression_stats, is_string);
      table_compressed += aggregate.compressed_size;
      table_uncompressed += aggregate.uncompressed_size;
      table_primary += aggregate.primary_compressed_size;
      table_secondary += aggregate.secondary_compressed_size;
    }

    table_name.push(name);
    num_rows.push(stats.num_rows);
    num_row_groups.push(static_cast<std::int64_t>(stats.num_row_groups));
    num_columns.push(static_cast<std::int64_t>(stats.num_columns));
    compressed_size.push(table_compressed);
    uncompressed_size.push(table_uncompressed);
    primary_compressed_size.push(table_primary);
    secondary_compressed_size.push(table_secondary);
  }

  std::vector<std::shared_ptr<arrow::Array>> arrays{table_name.finish(),
                                                    num_rows.finish(),
                                                    num_row_groups.finish(),
                                                    num_columns.finish(),
                                                    compressed_size.finish(),
                                                    uncompressed_size.finish(),
                                                    primary_compressed_size.finish(),
                                                    secondary_compressed_size.finish()};
  return arrow::RecordBatch::Make(
    arrow_schema_for(table_stats_schema()), static_cast<std::int64_t>(names.size()), arrays);
}

std::shared_ptr<arrow::RecordBatch> project_column_stats(catalog const& catalog)
{
  string_column table_name;
  int64_column column_id;
  string_column column_name;
  string_column column_type;
  compression_block aggregate;
  compression_block offsets;
  compression_block chars;

  std::int64_t row_count = 0;
  for (auto const& name : projected_table_names(catalog)) {
    auto const stats         = catalog.statistics(name)->statistics();
    auto const& column_names = catalog.column_names(name);
    auto const column_types  = catalog.column_types(name);

    for (auto const& column : stats.column_stats) {
      auto const id = column.column_id;

      table_name.push(name);
      column_id.push(static_cast<std::int64_t>(id));
      column_name.push(id < column_names.size() ? column_names[id] : std::string{});
      column_type.push(id < column_types.size() ? cudf_type_to_string(column_types[id])
                                                : std::string{});

      if (auto const* str = std::get_if<string_compression_statistics>(&column.compression_stats)) {
        fixed_width_compression_statistics summed = str->offsets_stats;
        summed += str->chars_stats;
        aggregate.push_aggregate_sizes_only(summed);
        offsets.push(str->offsets_stats);
        chars.push(str->chars_stats);
      } else {
        auto const& fixed = std::get<fixed_width_compression_statistics>(column.compression_stats);
        aggregate.push(fixed);
        offsets.push_null();
        chars.push_null();
      }
      ++row_count;
    }
  }

  std::vector<std::shared_ptr<arrow::Array>> arrays{
    table_name.finish(), column_id.finish(), column_name.finish(), column_type.finish()};
  aggregate.finish_into(arrays);
  offsets.finish_into(arrays);
  chars.finish_into(arrays);

  return arrow::RecordBatch::Make(arrow_schema_for(column_stats_schema()), row_count, arrays);
}

}  // namespace gqe::node_manager
