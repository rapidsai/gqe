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

#include <gqe/context_reference.hpp>
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
  mutable std::shared_mutex
    _hash_map_latch;  //> The latch guards the hash map object. It needs to be acquired to allocate
                      // and free the hash map. It does not need to be acquired for (thread-safe
                      // parallel) read-write or read-only access to hash map entries. Note: The
                      // current implementation allocates and builds the map in a single step,
                      // inside the critical section.
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
   * @param[in] ctx_ref The context in which the current task is running in.
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
   * @param[in] materialize_output If `true`, emit the materialized table with the same number of
   * columns as the size of `projection_indices`. If `false`, emit the position lists.
   * @param[in] build_unique_keys_pol If `build_unique_keys_policy::right`, build on right side and
   * assume unique keys. Similarly if `build_unique_keys_policy::left`.
   * @param[in] perfect_hashing If `true`, use perfect hashing for the join.
   */
  join_task(context_reference ctx_ref,
            int32_t task_id,
            int32_t stage_id,
            std::shared_ptr<task> left,
            std::shared_ptr<task> right,
            join_type_type join_type,
            std::unique_ptr<expression> condition,
            std::vector<cudf::size_type> projection_indices,
            std::shared_ptr<join_hash_map_cache> hash_map_cache = nullptr,
            bool materialize_output                             = true,
            gqe::unique_keys_policy unique_keys_pol             = gqe::unique_keys_policy::none,
            bool perfect_hashing                                = false);

  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

  /**
   * @brief Return a unique_keys_policy indicating whether the unique keys optimization can
   * be enabled with building on the right or left.
   */
  [[nodiscard]] gqe::unique_keys_policy unique_keys_policy() const noexcept
  {
    return _unique_keys_policy;
  }

 private:
  join_type_type _join_type;
  std::unique_ptr<expression> _condition;
  std::vector<cudf::size_type> _projection_indices;
  std::shared_ptr<join_hash_map_cache> _hash_map_cache;
  bool _materialize_output;
  gqe::unique_keys_policy _unique_keys_policy;
  bool _perfect_hashing;
};

namespace detail {

/**
 * @brief Set the boolean mask at specific indices to true.
 *
 * Requires the indices to be unique. Otherwise, the behavior is undefined.
 * Does nothing, if indices is empty.
 *
 * @param[in] boolean_mask Boolean mask column to be set. The column must have type `BOOL8`.
 * @param[in] indices Row indices at which the boolean mask is set to true.
 */
void set_boolean_mask(cudf::mutable_column_view boolean_mask, cudf::column_view indices);

/**
 * @brief Increment the counts column at specific indices by 1.
 *
 * Requires the indices to be unique. Otherwise, the behavior is undefined.
 * Does nothing, if indices is empty.
 *
 * @param[in] counts Counts column to be incremented. The column must have type `INT32`.
 * @param[in] indices Row indices at which the counts are incremented by 1.
 */
void increment_counts(cudf::mutable_column_view counts, cudf::column_view indices);

}  // namespace detail

/**
 * @brief Merge multiple position lists and materialize the semi/anti join result.
 *
 * For left-semi join, a row from the left table is included in the output if and only if the row
 * index is in at least one of the position list.
 *
 * For left-anti join, a row from the left table is included in the output if and only if the row
 * index is in all position lists.
 *
 * This task is used for materializing the join output when broadcasting the left side in a left
 * semi/anti join. Note that this task does not check the positions are valid.
 */
class materialize_join_from_position_lists_task : public task {
 public:
  /**
   * @brief Construct a materialize-join-from-position-lists task.
   *
   * @param[in] ctx_ref The context in which the current task is running.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] left_table Left table of the join
   * @param[in] position_lists Position lists to be merged.
   * @param[in] join_type Type of the join. Currently, only left semi and left anti join are
   * supported.
   * @param[in] projection_indices Column indices to materialize.
   */
  materialize_join_from_position_lists_task(context_reference ctx_ref,
                                            int32_t task_id,
                                            int32_t stage_id,
                                            std::shared_ptr<task> left_table,
                                            std::vector<std::shared_ptr<task>> position_lists,
                                            join_type_type join_type,
                                            std::vector<cudf::size_type> projection_indices);

  void execute() override;

 private:
  join_type_type _join_type;
  std::vector<cudf::size_type> _projection_indices;
};

}  // namespace gqe
