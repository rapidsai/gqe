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

#include <gqe/storage/pruning.hpp>

#include <gqe/storage/in_memory.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/logger.hpp>

#include <algorithm>
#include <span>
#include <stdexcept>

namespace gqe {

namespace storage {

pruning_result_t::pruning_result_t(const std::vector<zone_map::partition>& partitions,
                                   cudf::size_type partition_size)
  : _partitions(partitions), _partition_size(partition_size)
{
  // Compute candidate partitions
  _candidate_partitions.reserve(partitions.size());
  std::copy_if(partitions.begin(),
               partitions.end(),
               std::back_inserter(_candidate_partitions),
               [](const auto& partition) { return !partition.pruned; });

  // Compute consolidated partitions
  _consolidated_partitions = zone_map::consolidate_partitions(partitions);
  auto it                  = std::remove_if(_consolidated_partitions.begin(),
                           _consolidated_partitions.end(),
                           [](const zone_map::partition& partition) { return partition.pruned; });
  _consolidated_partitions.erase(it, _consolidated_partitions.end());

  // Compute vector of partition indexes when partitioning is enabled. Unpartitioned results keep
  // only row ranges; compressed-sliced materialization requires a positive partition size.
  if (_partition_size > 0) {
    size_t num_partition_indexes = 0;
    for (const auto& partition : _consolidated_partitions) {
      const size_t start_idx = partition.start / _partition_size;
      const size_t end_idx   = gqe::utility::divide_round_up(partition.end, _partition_size);
      num_partition_indexes += end_idx - start_idx;
    }
    _partition_indexes.reserve(num_partition_indexes);

    for (const auto& partition : _consolidated_partitions) {
      size_t start_idx = partition.start / _partition_size;
      size_t end_idx   = gqe::utility::divide_round_up(partition.end, _partition_size);
      for (size_t partition_idx = start_idx; partition_idx < end_idx; ++partition_idx) {
        _partition_indexes.emplace_back(partition_idx);
      }
    }
  }

  // Compute total number of rows
  _num_rows = 0;
  for (const auto& partition : _consolidated_partitions) {
    _num_rows += partition.end - partition.start;
  }
}

cudf::size_type pruning_result_t::partition_size() const { return _partition_size; }

const std::vector<zone_map::partition>& pruning_result_t::candidate_partitions() const
{
  return _candidate_partitions;
}

const std::vector<zone_map::partition>& pruning_result_t::consolidated_partitions() const
{
  return _consolidated_partitions;
}

const std::vector<size_t>& pruning_result_t::partition_indexes() const
{
  return _partition_indexes;
}

cudf::size_type pruning_result_t::num_rows() const { return _num_rows; }

namespace detail {

std::unique_ptr<pruning_results_t> evaluate_in_memory_pruning(
  std::span<row_group_interface const* const> row_groups,
  std::span<cudf::size_type const> column_indexes,
  std::optional<arrow::compute::Expression> const& partial_filter,
  bool use_partition_pruning,
  cudf::size_type partition_size)
{
  GQE_EXPECTS(partition_size >= 0, "partition_size must be non-negative");

  auto pruning_results = std::make_unique<pruning_results_t>();
  // TODO Check for partition_size and use_partition_pruning should be dropped
  // If partial_filter has a value, these are implicit
  // Need to update the oneshot_iterate_task
  if (partial_filter && partition_size != 0 && use_partition_pruning) {
    pruning_results->reserve(row_groups.size());
    std::for_each(row_groups.begin(),
                  row_groups.end(),
                  [&pruning_results, &partial_filter, partition_size](auto const* rg) {
                    if (!rg->zone_map()) {
                      throw std::logic_error("Row group should have a zone map but none exists");
                    }
                    auto const partitions = rg->zone_map()->evaluate(*partial_filter);
                    const auto has_unpruned_partitions =
                      std::find_if(partitions.begin(), partitions.end(), [](const auto& partition) {
                        return !partition.pruned;
                      }) != partitions.end();
                    if (has_unpruned_partitions) {
                      // TODO Change output of evaluate to pruning_result?
                      pruning_results->emplace_back(rg,
                                                    pruning_result_t{partitions, partition_size});
                    }
                  });
    GQE_LOG_DEBUG("Row groups after pruning: {}", pruning_results->size());
    return pruning_results;
  } else {
    GQE_LOG_DEBUG("No partial filter; processing all row groups");
    pruning_results->reserve(row_groups.size());
    std::transform(
      row_groups.begin(),
      row_groups.end(),
      std::back_inserter(*pruning_results),
      [column_indexes, partition_size](auto const* rg) {
        std::vector<cudf::size_type> null_counts(rg->num_columns(), 0);
        std::for_each(column_indexes.begin(), column_indexes.end(), [&null_counts, &rg](auto i) {
          null_counts[i] = rg->get_column(i).null_count();
        });
        const zone_map::partition partition{.pruned      = false,
                                            .start       = 0,
                                            .end         = static_cast<cudf::size_type>(rg->size()),
                                            .null_counts = null_counts};
        // When zone map partition size is disabled
        // (<= 0), compressed_sliced columns use one slice of row-group size. Avoid _partition_size
        // == 0 in pruning_result_t
        const auto const_row_group_rows = static_cast<cudf::size_type>(rg->size());
        const cudf::size_type effective_partition_size =
          partition_size > 0 ? partition_size
                             : std::max(static_cast<cudf::size_type>(1), const_row_group_rows);
        return row_group_with_pruning_result_t{
          rg, pruning_result_t{{partition}, effective_partition_size}};
      });
    return pruning_results;
  }
}

}  // namespace detail

}  // namespace storage
}  // namespace gqe
