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

#pragma once

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <arrow/record_batch.h>
#include <arrow/type.h>

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gqe::utility {

/**
 * @brief Convert one or more cudf::table_view partitions to a single Arrow RecordBatch.
 * Column names default to string indices ("0", "1", "2", ...).
 *
 * Copies column data from device (GPU) to host (CPU) memory using cudf::to_arrow,
 * concatenates all partitions, and returns a single RecordBatch suitable for Flight SQL
 * streaming.
 *
 * @param views        The cudf table views (may reside on device memory).
 * @param column_names Column names for the output schema.
 * @return An Arrow RecordBatch with the concatenated data on the host.
 */
[[nodiscard]] std::shared_ptr<arrow::RecordBatch> cudf_table_to_arrow(
  std::span<cudf::table_view const> views, std::span<std::string const> column_names = {});

/**
 * @brief Build an Arrow Schema from column names and cudf data types.
 *
 * Convenience wrapper around `cudf_schema_to_arrow` that constructs an empty
 * table_view from the given types.
 *
 * @param column_names Column names for the output schema.
 * @param column_types Corresponding cudf data types.
 * @return An Arrow Schema.
 */
[[nodiscard]] std::shared_ptr<arrow::Schema> build_arrow_schema(
  std::vector<std::string> const& column_names, std::vector<cudf::data_type> const& column_types);

}  // namespace gqe::utility
