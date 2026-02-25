/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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
#include <gqe/executor/mark_join.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/executor/unique_key_inner_join.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/types.hpp>

#include <cudf/join/hash_join.hpp>
#include <cudf/types.hpp>

#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <vector>

namespace gqe {

/**
 * @brief Base class for join hash table implementations that can be cached.
 *
 * All cached join implementations support probing with equality-only join conditions.
 * Some implementations extend this to also support mixed conditions (see mixed_join_interface).
 */
class join_interface {
 public:
  virtual ~join_interface() = default;

  /**
   * @brief Probe the cached hash table with equality-only join conditions.
   *
   * @param[in] probe_keys Key columns of the probe table.
   * @param[in] probe_mask Optional boolean mask to filter probe rows.
   * @param[in] join_type Type of join to perform.
   * @return A pair of device vectors containing (probe_indices, build_indices).
   */
  virtual std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
                    std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
  probe(cudf::table_view const& probe_keys,
        cudf::column_view const& probe_mask,
        join_type_type join_type) const = 0;
};

/**
 * @brief Extended join implementation that supports mixed (equality + non-equality) conditions.
 *
 * This class extends join_interface to add support for non-equality predicates in addition to
 * equality-based key matching. Implementations that support mixed conditions should inherit
 * from this class.
 */
class mixed_join_interface : public join_interface {
 public:
  /**
   * @brief Probe the hash table with mixed join conditions (equality + non-equality predicates).
   *
   * @param[in] probe_keys Key columns of the probe table for equality matching.
   * @param[in] probe_mask Optional boolean mask to filter probe rows.
   * @param[in] left_conditional Full left table for evaluating non-equality conditions.
   * @param[in] right_conditional Full right table for evaluating non-equality conditions.
   * @param[in] binary_predicate AST expression for non-equality join conditions.
   * @param[in] join_type Type of join to perform.
   * @return A pair of device vectors containing (probe_indices, build_indices).
   */
  virtual std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
                    std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
  probe_mixed(cudf::table_view const& probe_keys,
              cudf::column_view const& probe_mask,
              cudf::table_view const& left_conditional,
              cudf::table_view const& right_conditional,
              cudf::ast::expression const* binary_predicate,
              join_type_type join_type) const = 0;
};

/**
 * @brief Hash join implementation using cudf::hash_join.
 *
 * Supports inner, left, and full equality joins.
 */
class hash_join_impl : public join_interface {
 public:
  ~hash_join_impl() override = default;
  hash_join_impl(cudf::table_view const& build,
                 cudf::null_equality compare_nulls = cudf::null_equality::EQUAL,
                 rmm::cuda_stream_view stream      = cudf::get_default_stream());

  std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
            std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
  probe(cudf::table_view const& probe_keys,
        cudf::column_view const& probe_mask,
        join_type_type join_type) const override;

 private:
  mutable std::unique_ptr<cudf::hash_join> _hash_join;
};

/**
 * @brief Unique key join implementation for tables with unique build keys.
 *
 * Optimized for inner joins when the build table has unique keys.
 */
class unique_key_join_impl : public join_interface {
 public:
  ~unique_key_join_impl() override = default;
  unique_key_join_impl(cudf::table_view const& build,
                       cudf::column_view const& build_mask = cudf::column_view(),
                       cudf::null_equality compare_nulls   = cudf::null_equality::EQUAL,
                       rmm::cuda_stream_view stream        = cudf::get_default_stream());

  std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
            std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
  probe(cudf::table_view const& probe_keys,
        cudf::column_view const& probe_mask,
        join_type_type join_type) const override;

 private:
  mutable std::unique_ptr<gqe::unique_key_join> _unique_key_join;
};

/**
 * @brief Mark join implementation for semi/anti joins.
 *
 * Supports both equality-only and mixed join conditions for left_semi and left_anti joins.
 * Inherits from mixed_join_interface to provide the probe_mixed capability.
 */
class mark_join_impl : public mixed_join_interface {
 public:
  ~mark_join_impl() override = default;
  mark_join_impl(cudf::table_view const& build,
                 cudf::column_view const& build_mask,
                 bool is_cached,
                 cudf::null_equality compare_nulls,
                 rmm::cuda_stream_view stream = cudf::get_default_stream());

  std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
            std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
  probe(cudf::table_view const& probe_keys,
        cudf::column_view const& probe_mask,
        join_type_type join_type) const override;

  std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
            std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
  probe_mixed(cudf::table_view const& probe_keys,
              cudf::column_view const& probe_mask,
              cudf::table_view const& left_conditional,
              cudf::table_view const& right_conditional,
              cudf::ast::expression const* binary_predicate,
              join_type_type join_type) const override;

  /**
   * @brief Compute position list from cached hash map for semi/anti joins.
   *
   * @param[in] join_type Type of join (must be left_semi or left_anti).
   * @return Device vector containing matching row indices.
   */
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> compute_positions_list_from_cached_map(
    join_type_type join_type) const;

 private:
  mutable std::unique_ptr<gqe::mark_join> _mark_join;
};

/**
 * @brief Thread-safe cache for join hash tables.
 *
 * Caches the hash table built from the build side of a join so multiple probe tasks
 * can reuse it without rebuilding. The cache is thread-safe and supports lazy
 * initialization via constructor arguments or a builder function.
 *
 * Usage:
 * @code
 * auto cache = std::make_shared<join_hash_map_cache>(join_hash_map_cache::build_location::left);
 *
 * Direct construction with constructor arguments:
 * auto& impl = cache->get_or_create<hash_join_impl>(build_keys, compare_nulls);
 * impl.probe(probe_keys, mask, join_type);
 *
 * Deferred construction with builder (e.g., filter mask evaluated inside lock):
 * auto& impl = cache->get_or_create_fn<mark_join_impl>([&]() {
 *     auto [mask, col] = get_mask_from_filter(params, table, std::move(expr), offset);
 *     return std::make_unique<mark_join_impl>(keys, mask, true, compare_nulls);
 * });
 * @endcode
 */
class join_hash_map_cache {
 public:
  /**
   * @brief Specifies which side of the join should be used as the build table.
   */
  enum class build_location { left, right };

  /**
   * @brief Construct a cache for join hash tables.
   *
   * @param[in] build_side Which side the hash table is built from.
   */
  explicit join_hash_map_cache(build_location build_side) : _build_side(build_side) {}

  /**
   * @brief Return which side the hash table is built from.
   */
  build_location build_side() const noexcept { return _build_side; }

  /**
   * @brief Get the cached implementation, or create it with the given constructor arguments.
   *
   * This method is thread-safe. The first caller creates the implementation by forwarding
   * the arguments to T's constructor; subsequent callers receive the cached instance.
   *
   * @tparam T The expected implementation type (e.g., hash_join_impl, mark_join_impl).
   *           Must be derived from join_interface.
   * @tparam Args Constructor argument types for T.
   * @param[in] args Arguments to forward to T's constructor (only used if cache is empty).
   * @return Reference to the cached implementation of type T.
   * @throws std::logic_error if the cached implementation doesn't match the requested type.
   */
  template <typename T, typename... Args>
  T const& get_or_create(Args&&... args) const
  {
    static_assert(std::is_base_of_v<join_interface, T>, "T must be derived from join_interface");
    {
      std::shared_lock lock(_mutex);
      if (_impl) {
        auto* result = dynamic_cast<T const*>(_impl.get());
        if (!result) { throw std::logic_error("Cached join implementation has unexpected type"); }
        return *result;
      }
    }
    std::unique_lock lock(_mutex);
    if (!_impl) {
      _impl = std::make_unique<T>(std::forward<Args>(args)...);
      cudf::get_default_stream().synchronize();
    }
    auto* result = dynamic_cast<T const*>(_impl.get());
    if (!result) { throw std::logic_error("Cached join implementation has unexpected type"); }
    return *result;
  }

  /**
   * @brief Get the cached implementation, or create it using a builder function.
   *
   * This method is thread-safe. The builder is only invoked if the cache is empty,
   * and is called while holding the lock. This is useful when creating the implementation
   * requires expensive setup (e.g., evaluating filter masks) that should only happen once.
   *
   * @tparam T The expected implementation type (e.g., hash_join_impl, mark_join_impl).
   *           Must be derived from join_interface.
   * @tparam Builder A callable that returns std::unique_ptr<T>.
   * @param[in] builder Function to create the implementation (only called if cache is empty).
   * @return Reference to the cached implementation of type T.
   * @throws std::logic_error if the cached implementation doesn't match the requested type.
   */
  template <typename T, typename Builder>
  T const& get_or_create_fn(Builder&& builder) const
  {
    static_assert(std::is_base_of_v<join_interface, T>, "T must be derived from join_interface");
    {
      std::shared_lock lock(_mutex);
      if (_impl) {
        auto* result = dynamic_cast<T const*>(_impl.get());
        if (!result) { throw std::logic_error("Cached join implementation has unexpected type"); }
        return *result;
      }
    }
    std::unique_lock lock(_mutex);
    if (!_impl) {
      _impl = builder();
      cudf::get_default_stream().synchronize();
    }
    auto* result = dynamic_cast<T const*>(_impl.get());
    if (!result) { throw std::logic_error("Cached join implementation has unexpected type"); }
    return *result;
  }

  /**
   * @brief Get the cached implementation without creating it.
   *
   * @tparam T The expected implementation type.
   * @return Pointer to the cached implementation of type T, or nullptr if not cached
   *         or if the cached type doesn't match.
   */
  template <typename T>
  T const* get() const
  {
    static_assert(std::is_base_of_v<join_interface, T>, "T must be derived from join_interface");
    std::shared_lock lock(_mutex);
    return dynamic_cast<T const*>(_impl.get());
  }

 private:
  build_location _build_side;
  mutable std::unique_ptr<join_interface> _impl;
  mutable std::shared_mutex _mutex;
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
   * @param[in] left_filter_condition A boolean expression to filter the left table before the join.
   * @param[in] right_filter_condition A boolean expression to filter the right table before the
   * @param[in] mark_join If `true`, use the mark join implementation for left semi and anti joins.
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
            bool perfect_hashing                                = false,
            std::unique_ptr<expression> left_filter_condition   = nullptr,
            std::unique_ptr<expression> right_filter_condition  = nullptr,
            bool mark_join                                      = true);

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
  std::unique_ptr<expression> _left_filter_condition;
  std::unique_ptr<expression> _right_filter_condition;
  bool _mark_join;
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

/**
 * @brief Fill the range with a sequence of numbers.
 *
 * @param[in] begin Beginning of the range.
 * @param[in] end End of the range.
 * @param[in] start Starting value, default is 0.
 * @param[in] step Step size, default is 1.
 */
template <typename T>
void sequence(T* begin, T* end, T start = 0, T step = 1);

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
 *
 * In the case that the hash map cache is initialized, this function will also compute the positions
 * lists directly instead of pulling the indices list from prior tasks.
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
  std::shared_ptr<join_hash_map_cache> _hash_map_cache;
  bool _mark_join;
};

class extract_mark_join_positions_task : public task {
 public:
  /**
   * @brief Extract the mark join positions list from the local cached hash map.
   *
   * @param ctx_ref The context in which the current task is running.
   * @param task_id Globally unique identifier of the task.
   * @param stage_id Stage of the current task.
   * @param join_tasks The mark join tasks that contain the cached hash map. This would not be
   * dependency, but it would be stored as a member to keep the join tasks alive.
   * @param join_type Type of the join. Only left semi and left anti join are supported.
   * @param hash_map_cache The cache that contains the hash map for mark join. This would not be
   * dependency, but it would be stored as a member to keep the cache alive.
   */
  extract_mark_join_positions_task(context_reference ctx_ref,
                                   int32_t task_id,
                                   int32_t stage_id,
                                   std::vector<std::shared_ptr<task>> join_tasks,
                                   join_type_type join_type,
                                   std::shared_ptr<join_hash_map_cache> hash_map_cache);

  void execute() override;

 private:
  join_type_type _join_type;
  std::vector<std::shared_ptr<task>> _join_tasks;
  std::shared_ptr<join_hash_map_cache> _hash_map_cache;
};

}  // namespace gqe
