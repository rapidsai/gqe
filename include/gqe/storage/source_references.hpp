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

#include <gqe/qep/source_reference.hpp>
#include <gqe/storage/pruning.hpp>

#include <cudf/types.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace gqe {
namespace storage {

/**
 * @brief Portable reference to one in-memory row group.
 *
 * Addresses a row group by table name and positional index instead of a rank-local pointer, so
 * the reference can be resolved on any process that can reach the table through a
 * `storage::table_provider`. Row-group indices are stable for the lifetime of a query:
 * `in_memory_table_interface` guarantees appends never reorder or displace existing row groups.
 *
 * # Lifetime
 *
 * A reference is valid only within the query execution that minted it. Like every GQE read path,
 * it relies on the engine-wide contract that table contents are not replaced or cleared while a
 * query reads them; resolving a reference after the table changes is undefined. References must
 * not be retained across queries.
 *
 * # Pruning Payload
 *
 * Every row-group reference carries the row group's `pruning_result_t`, evaluated once by the
 * minting task. Materialization reads only the pruned partitions, and `num_rows()` is derived
 * from the same result. Only the empty-input reference has no payload, and it has zero rows.
 */
class in_memory_reference final : public qep::source_reference {
 public:
  /**
   * @brief Build a reference to one row group.
   *
   * @param[in] table_name Catalog name the resolving rank looks up.
   * @param[in] row_group_index Positional row-group index in the table.
   * @param[in] pruning Zone-map pruning result for the row group.
   *
   * @throws std::invalid_argument If `table_name` is empty.
   *
   * @return A shared immutable reference.
   */
  [[nodiscard]] static std::shared_ptr<in_memory_reference const> make(std::string table_name,
                                                                       std::size_t row_group_index,
                                                                       pruning_result_t pruning);

  /**
   * @brief Build the portable empty-input reference.
   *
   * Emitted when pruning removes every row group, so a split still yields exactly one
   * schema-carrying chunk after the resolve task maps it to an empty table.
   *
   * @param[in] table_name Catalog name the resolving rank looks up.
   *
   * @throws std::invalid_argument If `table_name` is empty.
   *
   * @return A shared immutable empty reference.
   */
  [[nodiscard]] static std::shared_ptr<in_memory_reference const> make_empty(
    std::string table_name);

  /**
   * @brief Return the retained row count, derived from the pruning result. Zero for the
   * empty-input reference.
   */
  [[nodiscard]] cudf::size_type num_rows() const noexcept override
  {
    return _pruning.has_value() ? _pruning->num_rows() : 0;
  }

  [[nodiscard]] std::string const& table_name() const noexcept { return _table_name; }

  /**
   * @brief Return whether this is the empty-input reference.
   */
  [[nodiscard]] bool is_empty() const noexcept { return !_row_group_index.has_value(); }

  /**
   * @brief Return the row-group index.
   *
   * @throws std::logic_error If this is the empty-input reference.
   */
  [[nodiscard]] std::size_t row_group_index() const;

  /**
   * @brief Return the row group's pruning result.
   *
   * @throws std::logic_error If this is the empty-input reference.
   */
  [[nodiscard]] pruning_result_t const& pruning() const;

 private:
  in_memory_reference(std::string table_name,
                      std::optional<std::size_t> row_group_index,
                      std::optional<pruning_result_t> pruning);

  std::string _table_name;
  std::optional<std::size_t> _row_group_index;
  std::optional<pruning_result_t> _pruning;
};

/**
 * @brief Portable reference to a span of consecutive row groups in one Parquet file.
 *
 * The span is contiguous in row-group index space; the projected column chunks read within it
 * need not be adjacent in the file. Addresses the span by file path and row-group indices only,
 * so any rank with filesystem access can resolve it. The reference carries no reader, datasource,
 * or parsed footer; a resolving rank obtains footer metadata from its own cache or by one footer
 * read.
 *
 * A zero-length span is the portable empty-input reference; the resolve task maps it to a
 * schema-carrying empty table.
 */
class parquet_reference final : public qep::source_reference {
 public:
  /**
   * @brief Build a reference to a row-group span.
   *
   * @param[in] file_path Path of the Parquet file.
   * @param[in] row_group_begin First row-group index of the span. Must not be negative.
   * @param[in] row_group_count Number of consecutive row groups. Must be positive; the
   * empty-input reference is built by `make_empty`.
   * @param[in] num_rows Exact row count of the span, from footer metadata. Must be positive.
   * @param[in] hive_partition_key Optional Hive partition-key value parsed from the file path at
   * mint time.
   *
   * @throws std::invalid_argument If `file_path` is empty, or `row_group_count` or `num_rows`
   * is not positive.
   *
   * @return A shared immutable reference.
   */
  [[nodiscard]] static std::shared_ptr<parquet_reference const> make(
    std::string file_path,
    cudf::size_type row_group_begin,
    cudf::size_type row_group_count,
    cudf::size_type num_rows,
    std::optional<std::int64_t> hive_partition_key = std::nullopt);

  /**
   * @brief Build the portable empty-input reference.
   *
   * Emitted when the file list is empty or pruning removes every file, so a split still yields
   * exactly one schema-carrying chunk after the resolve task maps it to an empty table.
   *
   * @return A shared immutable empty reference.
   */
  [[nodiscard]] static std::shared_ptr<parquet_reference const> make_empty();

  [[nodiscard]] cudf::size_type num_rows() const noexcept override { return _num_rows; }

  [[nodiscard]] std::string const& file_path() const noexcept { return _file_path; }
  [[nodiscard]] cudf::size_type row_group_begin() const noexcept { return _row_group_begin; }
  [[nodiscard]] cudf::size_type row_group_count() const noexcept { return _row_group_count; }
  [[nodiscard]] std::optional<std::int64_t> const& hive_partition_key() const noexcept
  {
    return _hive_partition_key;
  }

  /**
   * @brief Return whether this is the empty-input reference.
   */
  [[nodiscard]] bool is_empty() const noexcept { return _row_group_count == 0; }

 private:
  parquet_reference(std::string file_path,
                    cudf::size_type row_group_begin,
                    cudf::size_type row_group_count,
                    cudf::size_type num_rows,
                    std::optional<std::int64_t> hive_partition_key);

  std::string _file_path;
  cudf::size_type _row_group_begin;
  cudf::size_type _row_group_count;
  cudf::size_type _num_rows;
  std::optional<std::int64_t> _hive_partition_key;
};

}  // namespace storage
}  // namespace gqe
