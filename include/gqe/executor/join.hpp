/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <gqe/executor/query_context.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/types.hpp>

#include <cudf/join.hpp>
#include <cudf/types.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace gqe {

/**
 * @brief Owner of the hash map used in hash joins.
 *
 * When the build table is the same in multiple join tasks, an object of this class is helpful for
 * owning the hash map, so that it does not need to be rebuilt for each task.
 */
class join_hash_map_cache {
 public:
  /**
   * @brief Whether the left or the right table should be used as the build table.
   */
  enum class build_location { left, right };

  /**
   * @brief Construct an object to hold the hash map used in joins.
   *
   * @param[in] build_side Indicate which side the hash map is built from.
   */
  join_hash_map_cache(build_location build_side) : _build_side(build_side) {}

  /**
   * @brief Return which side the hash map is built from.
   */
  build_location build_side() const noexcept { return _build_side; }

  /**
   * @brief Build the hash map from key columns of the build table.
   *
   * This function could be called multiple times with the same build table keys. The hash map
   * will be cached after the first call, and the subsequent calls would be served with the cached
   * hash map.
   *
   * @param[in] build_keys Key columns of the build table. Note that if this function is called
   * multiple times, this argument needs to refer to the same build table. Otherwise, the behavior
   * is undefined.
   * @param[in] compare_nulls Whether NULL join keys should be compared as equal. Note that if
   * this function is called multiple times, this argument should be the same. Otherwise, the
   * behavior is undefined.
   *
   * @return A hash join object that can be subsequently probed.
   */
  cudf::hash_join const* hash_map(cudf::table_view build_keys,
                                  cudf::null_equality compare_nulls) const;

 private:
  build_location _build_side;
  mutable std::unique_ptr<cudf::hash_join> _hash_map;
};

class join_task : public task {
 public:
  /**
   * @brief Construct a new join task.
   *
   * In `condition`, the column indices of the right table come after the left table. For example,
   * suppose the left table has 3 columns. The equality join of column 1 from the left table with
   * the column 0 from the right table and column 0 from the left table with column 2 from the right
   * table can be represented by
   * `AND(Equal(ColumnReference 1, ColumnReference 3), Equal(ColumnReference 0, ColumnReference 5))`
   *
   * For each Equal expression ("=") in `condition`, the left child expression is evaluated on the
   * left table, and the right child expression is evaluated on the right table. Trying to reference
   * a right (left) table column in the left (right) child expression is an undefined behavior. In
   * the example above, `ColumnReference 1` and `ColumnReference 0` would be evaluated on the left
   * table, whereas `ColumnReference 3` and `ColumnReference 5` are evaluated on the right table.
   *
   * @param[in] query_context The query context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] left Left table to be joined.
   * @param[in] right Right table to be joined.
   * @param[in] join_type Type of the join.
   * @param[in] condition A boolean expression to define when a left tuple matches with a right
   * tuple.
   * @param[in] projection_indices Column indices to materialize after the join. The rest of columns
   * are discarded.
   * @param[in] hash_map_cache If supplied, the hash map used for the join can be loaded from the
   * cache instead of reconstructed.
   */
  join_task(query_context* query_context,
            int32_t task_id,
            int32_t stage_id,
            std::shared_ptr<task> left,
            std::shared_ptr<task> right,
            join_type_type join_type,
            std::unique_ptr<expression> condition,
            std::vector<cudf::size_type> projection_indices,
            std::shared_ptr<join_hash_map_cache> hash_map_cache = nullptr);

  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

 private:
  join_type_type _join_type;
  std::unique_ptr<expression> _condition;
  std::vector<cudf::size_type> _projection_indices;
  std::shared_ptr<join_hash_map_cache> _hash_map_cache;
};

}  // namespace gqe
