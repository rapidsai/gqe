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

#include <gqe/storage/source_references.hpp>

#include <gqe/utility/error.hpp>

#include <memory>
#include <stdexcept>
#include <utility>

namespace gqe {
namespace storage {

// -----------------------------------------------------------------------------
// in_memory_reference
// -----------------------------------------------------------------------------

in_memory_reference::in_memory_reference(std::string table_name,
                                         std::optional<std::size_t> row_group_index,
                                         std::optional<pruning_result_t> pruning)
  : _table_name{std::move(table_name)},
    _row_group_index{row_group_index},
    _pruning{std::move(pruning)}
{
  GQE_EXPECTS(!_table_name.empty(),
              "in_memory_reference: table_name must not be empty",
              std::invalid_argument);
}

std::shared_ptr<in_memory_reference const> in_memory_reference::make(std::string table_name,
                                                                     std::size_t row_group_index,
                                                                     pruning_result_t pruning)
{
  return std::shared_ptr<in_memory_reference const>{
    new in_memory_reference{std::move(table_name), row_group_index, std::move(pruning)}};
}

std::shared_ptr<in_memory_reference const> in_memory_reference::make_empty(std::string table_name)
{
  return std::shared_ptr<in_memory_reference const>{
    new in_memory_reference{std::move(table_name), std::nullopt, std::nullopt}};
}

std::size_t in_memory_reference::row_group_index() const
{
  GQE_EXPECTS(_row_group_index.has_value(),
              "in_memory_reference: the empty-input reference has no row-group index");
  return *_row_group_index;
}

pruning_result_t const& in_memory_reference::pruning() const
{
  GQE_EXPECTS(_pruning.has_value(),
              "in_memory_reference: the empty-input reference has no pruning result");
  return *_pruning;
}

// -----------------------------------------------------------------------------
// parquet_reference
// -----------------------------------------------------------------------------

parquet_reference::parquet_reference(std::string file_path,
                                     cudf::size_type row_group_begin,
                                     cudf::size_type row_group_count,
                                     cudf::size_type num_rows,
                                     std::optional<std::int64_t> hive_partition_key)
  : _file_path{std::move(file_path)},
    _row_group_begin{row_group_begin},
    _row_group_count{row_group_count},
    _num_rows{num_rows},
    _hive_partition_key{hive_partition_key}
{
  GQE_EXPECTS(_row_group_begin >= 0,
              "parquet_reference: row_group_begin must not be negative",
              std::invalid_argument);
  GQE_EXPECTS(_row_group_count >= 0,
              "parquet_reference: row_group_count must not be negative",
              std::invalid_argument);
  GQE_EXPECTS(
    _num_rows >= 0, "parquet_reference: num_rows must not be negative", std::invalid_argument);
  GQE_EXPECTS(!_file_path.empty() || _row_group_count == 0,
              "parquet_reference: file_path must not be empty",
              std::invalid_argument);
}

std::shared_ptr<parquet_reference const> parquet_reference::make(
  std::string file_path,
  cudf::size_type row_group_begin,
  cudf::size_type row_group_count,
  cudf::size_type num_rows,
  std::optional<std::int64_t> hive_partition_key)
{
  GQE_EXPECTS(row_group_count > 0,
              "parquet_reference: row_group_count must be positive; use make_empty for the "
              "empty-input reference",
              std::invalid_argument);
  GQE_EXPECTS(num_rows > 0,
              "parquet_reference: num_rows must be positive; zero-row spans are represented by "
              "the empty-input reference",
              std::invalid_argument);
  return std::shared_ptr<parquet_reference const>{new parquet_reference{
    std::move(file_path), row_group_begin, row_group_count, num_rows, hive_partition_key}};
}

std::shared_ptr<parquet_reference const> parquet_reference::make_empty()
{
  return std::shared_ptr<parquet_reference const>{
    new parquet_reference{std::string{}, 0, 0, 0, std::nullopt}};
}

}  // namespace storage
}  // namespace gqe
