/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/storage/zone_map.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace gqe {
namespace storage {

class row_group_interface;

/**
 * @brief Indicate which partitions of an in-memory table remain after partition pruning and which
 * can be skipped.
 *
 * Depending on the column type, the projected columns of an in_memory_read_task compute the same
 * information over the list of partitions returned by the zone map evaluation. For example,
 * contiguous columns consolidate individual partitions into maximally large partitions, whereas
 * compressed sliced columns require individual partition indices. All column classes require the
 * number of rows remaining after partition pruning.
 *
 * This class precomputes these values, so that they are only computed once when accessed by
 * multiple columns.
 */
class pruning_result_t {
 public:
  explicit pruning_result_t(const std::vector<zone_map::partition>& partitions,
                            cudf::size_type partition_size);

  [[nodiscard]] cudf::size_type partition_size() const;

  /// @brief Return all partitions which are not pruned.
  [[nodiscard]] const std::vector<zone_map::partition>& candidate_partitions() const;

  /// @brief Return maximally consolidated partitions.
  [[nodiscard]] const std::vector<zone_map::partition>& consolidated_partitions() const;

  /// @brief Return partition ids
  [[nodiscard]] const std::vector<size_t>& partition_indexes() const;

  /// @brief Return the number of rows after pruning.
  [[nodiscard]] cudf::size_type num_rows() const;

 private:
  std::vector<zone_map::partition> _partitions;
  cudf::size_type _partition_size;
  std::vector<zone_map::partition> _candidate_partitions;
  std::vector<zone_map::partition> _consolidated_partitions;
  std::vector<size_t> _partition_indexes;
  cudf::size_type _num_rows;
};

/**
 * @brief Pair a row group with its partition pruning result.
 */
using row_group_with_pruning_result_t = std::pair<const row_group_interface*, pruning_result_t>;

/**
 * @brief Pruning results of all row groups of an in-memory table
 */
using pruning_results_t = std::vector<row_group_with_pruning_result_t>;

namespace detail {

/**
 * @brief Evaluate conservative zone-map pruning for an in-memory row-group range.
 *
 * If @ref partial_filter exists, evaluate it on the zone maps of all row groups, and pair each row
 * group with the evaluation result. Otherwise, pair each row group with a partition which covers
 * the entire row group.
 */
[[nodiscard]] std::unique_ptr<pruning_results_t> evaluate_in_memory_pruning(
  std::span<row_group_interface const* const> row_groups,
  std::span<cudf::size_type const> column_indexes,
  std::optional<arrow::compute::Expression> const& partial_filter,
  bool use_partition_pruning,
  cudf::size_type partition_size);

}  // namespace detail

}  // namespace storage
}  // namespace gqe
