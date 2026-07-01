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

#include <gqe/utility/cudf_to_arrow.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/interop.hpp>

#include <arrow/array/util.h>
#include <arrow/c/bridge.h>
#include <arrow/table.h>

#include <format>
#include <span>
#include <stdexcept>

namespace {

std::shared_ptr<arrow::Schema> cudf_schema_to_arrow(cudf::table_view table,
                                                    std::span<std::string const> column_names)
{
  std::vector<cudf::column_metadata> metadata;
  metadata.reserve(column_names.size());
  for (auto const& name : column_names) {
    metadata.emplace_back(name);
  }

  auto arrow_schema = cudf::to_arrow_schema(
    table, cudf::host_span<cudf::column_metadata const>{metadata.data(), metadata.size()});
  return arrow::ImportSchema(arrow_schema.get()).ValueOrDie();
}

}  // namespace

namespace gqe::utility {

std::shared_ptr<arrow::Schema> build_arrow_schema(std::vector<std::string> const& column_names,
                                                  std::vector<cudf::data_type> const& column_types)
{
  // GQE internally represents Decimal columns as FLOAT64. Map them back to
  // Decimal128 so that Flight SQL clients see the original type and perform
  // exact decimal arithmetic when planning queries. Without this, DataFusion
  // on the client does float64 arithmetic (e.g., 0.06 + 0.01 = 0.06999...)
  // instead of exact decimal arithmetic, which causes TPC-H Q6 and Q15 to
  // return wrong results.
  // TODO: Remove this workaround once GQE has native decimal support.
  arrow::FieldVector fields;
  fields.reserve(column_names.size());
  for (std::size_t i = 0; i < column_names.size(); ++i) {
    auto const& name = column_names[i];
    auto const& type = column_types[i];
    if (type.id() == cudf::type_id::FLOAT64) {
      fields.push_back(arrow::field(name, arrow::decimal128(15, 2)));
    } else {
      cudf::column_view col(type, 0, nullptr, nullptr, 0);
      auto one_col_schema = cudf_schema_to_arrow(cudf::table_view{{col}}, {&name, 1});
      fields.push_back(one_col_schema->field(0));
    }
  }
  return arrow::schema(std::move(fields));
}

std::shared_ptr<arrow::RecordBatch> cudf_table_to_arrow(std::span<cudf::table_view const> views,
                                                        std::span<std::string const> column_names)
{
  if (views.empty()) { throw std::runtime_error("No table views provided for Arrow conversion"); }

  auto num_columns = views[0].num_columns();
  std::vector<cudf::column_metadata> metadata;
  metadata.reserve(num_columns);
  if (column_names.empty()) {
    for (cudf::size_type i = 0; i < num_columns; ++i) {
      metadata.emplace_back(std::to_string(i));
    }
  } else {
    for (auto const& name : column_names) {
      metadata.emplace_back(name);
    }
  }

  auto metadata_span =
    cudf::host_span<cudf::column_metadata const>(metadata.data(), metadata.size());

  // Convert each partition to an Arrow RecordBatch via the C Data Interface
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  batches.reserve(views.size());
  for (auto const& view : views) {
    auto c_schema      = cudf::to_arrow_schema(view, metadata_span);
    auto schema_result = arrow::ImportSchema(c_schema.get());
    if (!schema_result.ok()) {
      throw std::runtime_error(
        std::format("Failed to import Arrow schema: {}", schema_result.status().ToString()));
    }
    auto schema = schema_result.ValueOrDie();

    auto c_device_array = cudf::to_arrow_host(view);

    auto batch_result = arrow::ImportRecordBatch(&c_device_array->array, schema);
    if (!batch_result.ok()) {
      throw std::runtime_error(
        std::format("Failed to import Arrow RecordBatch: {}", batch_result.status().ToString()));
    }
    batches.push_back(batch_result.ValueOrDie());
  }

  // Concatenate all partitions into a single RecordBatch
  auto table_result = arrow::Table::FromRecordBatches(batches);
  if (!table_result.ok()) {
    throw std::runtime_error(
      std::format("Failed to create Arrow Table: {}", table_result.status().ToString()));
  }

  auto table = table_result.ValueOrDie();

  arrow::TableBatchReader reader(*table);
  std::shared_ptr<arrow::RecordBatch> batch;
  auto read_status = reader.ReadNext(&batch);
  if (!read_status.ok()) {
    throw std::runtime_error(
      std::format("Failed to read Arrow RecordBatch: {}", read_status.ToString()));
  }

  // ReadNext returns null at end-of-stream (e.g. 0-row table).
  if (!batch) { batch = arrow::RecordBatch::MakeEmpty(table->schema()).ValueOrDie(); }

  return batch;
}

}  // namespace gqe::utility
