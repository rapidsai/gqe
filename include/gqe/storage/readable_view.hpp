/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/context_reference.hpp>
#include <gqe/executor/read.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/qep/task.hpp>
#include <gqe/types.hpp>

#include <memory>
#include <optional>
#include <stdexcept>

namespace gqe {

namespace storage {

class table_provider;

/**
 * @brief Storage tasks implementing one physical QEP read.
 *
 * The caller connects `source` to `resolve` when present and uses the resolver as the read's
 * terminal. Deferred readers need both tasks so portable references are resolved before reaching
 * relational consumers. Other reader families contain only a source.
 */
struct qep_read_pipeline {
  /** @brief Source iterator selected by the requested reader family. */
  std::unique_ptr<qep::iterate_task> source;

  /** @brief Stateless transform that resolves source references into deferred columns. */
  std::unique_ptr<qep::stateless_transform_task> resolve;
};

/**
 * @brief Data access method to read the data in a table.
 *
 * The readable view provides a uniform interface to read the data in a table.
 * Subtypes implement the read method for the concrete table kind.
 */
class readable_view {
 public:
  /**
   * @brief Parameters forwarded to exactly one read task constructor.
   */
  struct task_parameters {
    int32_t task_id; /**< Globally unique identifier of the task. */
    std::unique_ptr<gqe::expression> partial_filter =
      nullptr; /**< Used to support predicate pushdown. Note that a row that satisfies
                * the predicate is guaranteed to be included in the loaded table, but a row that
                * does not satisfy the predicate may or may not be excluded. If such exclusion needs
                * to be guaranteed, an extra filter task is needed. If this argument is nullptr, no
                * rows will be filtered out.
                */
    std::vector<std::shared_ptr<gqe::task>> subquery_tasks =
      {}; /**< Subquery tasks that may be referenced by a subquery expression. A
           * relation index `i` in a subquery expression refers to `subquery_expressions[i]`.
           */
  };

  virtual ~readable_view() = default;

  /**
   * @brief Return multiple read tasks for the table kind.
   *
   * @param[in] task_parameters The parameters per read task.
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] stage_id Stage of the current task.
   * @param[in] column_names Columns to be loaded.
   * @param[in] data_types Expected data types of each column. If the actual data type of a loaded
   * column is different from expected, the column will be casted to the data type specified. Must
   * have the same length as `column_names`.
   *
   * == Thread Safety ==
   *
   * Implementations guarantee thread safety while writes to the table occur. In particular,
   * repeated reads of the table size may be inconsistent while table appends are underway.
   */
  virtual std::vector<std::unique_ptr<read_task_base>> get_read_tasks(
    std::vector<task_parameters>&& task_parameters,
    context_reference ctx_ref,
    int32_t stage_id,
    std::vector<std::string> column_names,
    std::vector<cudf::data_type> data_types) = 0;

  /**
   * @brief Construct the storage tasks for a QEP read using an explicit reader family.
   *
   * Splitting a returned source keeps its reader family and configuration. Deferred Parquet tasks
   * share a footer cache. Deferred in-memory tasks retain the catalog table identity and provider.
   * Their resolver rejects contexts with `use_overlap_mtx=true`. Chunked and deferred in-memory
   * readers require fixed host or device source residency and reject managed memory, whose
   * residency may change.
   *
   * Hint: Sources that support executor-selected streaming source ranges should implement
   * `qep::splittable_source_trait`. Operator scans with static executor-specific work units instead
   * implement `qep::static_iterate_split_trait`.
   *
   * @param[in] reader Reader family selected by QEP lowering.
   * @param[in] table_name Catalog identity used by deferred in-memory references.
   * @param[in] provider Provider used to resolve deferred in-memory references. For that family,
   *   it must outlive the tasks and every emitted deferred state. May be null unless an in-memory
   *   table is read with the deferred family.
   * @param[in] column_names Columns to load, in output order. An empty list selects count-only
   *   mode where supported.
   * @param[in] data_types Output type per column. Must have the same length as `column_names`.
   * @param[in] partition_column_idx Optional partition column for predicate pushdown. When set,
   *   the source expects a single materialized haystack predecessor as input #0. Storage kinds
   *   that do not support this pruning throw.
   * @param[in] partial_filter Conservative predicate-pushdown hint. Exact filtering remains the
   *   downstream filter's responsibility. Implementations may ignore it.
   *
   * @return The source and optional resolver. Neither task retains the readable view.
   * @throws std::logic_error If the storage kind does not support the requested QEP read.
   * @throws std::invalid_argument If the reader family or its configuration is invalid.
   */
  [[nodiscard]] virtual qep_read_pipeline get_read_pipeline(
    source_reader_type /*reader*/,
    std::string /*table_name*/,
    table_provider const* /*provider*/,
    std::vector<std::string> /*column_names*/,
    std::vector<cudf::data_type> /*data_types*/,
    std::optional<cudf::size_type> /*partition_column_idx*/ = std::nullopt,
    std::unique_ptr<gqe::expression> /*partial_filter*/     = nullptr)
  {
    throw std::logic_error("QEP read pipeline is not supported for this storage kind");
  }
};

};  // namespace storage

};  // namespace gqe
