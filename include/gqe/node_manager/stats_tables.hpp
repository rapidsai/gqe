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

#pragma once

#include <gqe/catalog.hpp>

#include <arrow/type_fwd.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace gqe::node_manager {

/**
 * @file
 * @brief System tables that expose table and column statistics as queryable SQL tables.
 *
 * GQE maintains per-table compression statistics in the node manager's catalog. To make them
 * accessible to clients (e.g. `gqe-cli` / `gqe-python`) that can only issue SQL, the statistics are
 * projected into two regular tables, `_gqe_table_stats` and `_gqe_column_stats`, which clients
 * query with ordinary `SELECT` statements. The tables use `shared_segment_memory` storage
 * (host-shared, uncompressed) so the task managers that execute the read can map and read them.
 *
 * This header is the single source of truth for the two tables' names and column schemas; both
 * registration (here) and projection/materialization of their contents refer to these schemas so
 * the column order stays consistent.
 */

/// Name of the table-level statistics table.
inline constexpr std::string_view table_stats_table_name = "_gqe_table_stats";

/// Name of the column-level statistics table.
inline constexpr std::string_view column_stats_table_name = "_gqe_column_stats";

/**
 * @brief Whether @p name refers to one of the statistics system tables.
 */
[[nodiscard]] bool is_stats_table(std::string_view name);

/**
 * @brief Column schema of @ref table_stats_table_name, in storage order.
 */
[[nodiscard]] std::vector<column_traits> table_stats_schema();

/**
 * @brief Column schema of @ref column_stats_table_name, in storage order.
 *
 * The layout is: identity columns, then a type-agnostic aggregate block (sizes summed across a
 * column's buffers; for string columns the row-group-count fields are null and authoritative counts
 * live in the breakdown), then the `offsets_*` and `chars_*` per-buffer breakdown blocks (null for
 * non-string columns).
 */
[[nodiscard]] std::vector<column_traits> column_stats_schema();

/**
 * @brief Register @ref table_stats_table_name and @ref column_stats_table_name in @p catalog.
 *
 * Registers both tables as empty `shared_segment_memory` tables so they are always queryable.
 * Intended to be called once during node manager startup. Their contents are populated lazily on
 * read.
 *
 * @throw std::logic_error if either table name is already registered.
 */
void register_stats_tables(catalog& catalog);

/**
 * @brief Project the catalog's current statistics into a record batch for @ref
 * table_stats_table_name.
 *
 * Produces one row per registered table (excluding the statistics tables themselves), with columns
 * matching @ref table_stats_schema. Sizes are summed across each table's columns and buffers.
 *
 * @throw std::runtime_error if the Arrow batch cannot be built.
 */
[[nodiscard]] std::shared_ptr<arrow::RecordBatch> project_table_stats(catalog const& catalog);

/**
 * @brief Project the catalog's current statistics into a record batch for @ref
 * column_stats_table_name.
 *
 * Produces one row per (table, column), with columns matching @ref column_stats_schema. The
 * aggregate block sums sizes across a column's buffers; row-group-count fields are null for string
 * columns (their authoritative per-buffer counts live in the `offsets_*` / `chars_*` blocks, which
 * are null for non-string columns).
 *
 * @throw std::runtime_error if the Arrow batch cannot be built.
 */
[[nodiscard]] std::shared_ptr<arrow::RecordBatch> project_column_stats(catalog const& catalog);

}  // namespace gqe::node_manager
