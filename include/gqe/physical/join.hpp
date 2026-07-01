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

#include <gqe/expression/expression.hpp>
#include <gqe/physical/relation.hpp>
#include <gqe/types.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace gqe {

namespace physical {

/**
 * @brief Abstract base class for all physical join relations.
 */
class join_relation_base : public relation {
 public:
  /**
   * @brief Construct a physical join relation.
   *
   * @param[in] left Left table to join.
   * @param[in] right Right table to join.
   * @param[in] subquery_relations Subquery relations that are referenced within the `condition`
   * expression.
   * @param[in] join_type Type of the join.
   * @param[in] condition A boolean expression to define when a left tuple matches with a right
   * tuple.
   * @param[in] projection_indices Column indices to materialize after the join. The rest of columns
   * are discarded.
   * @param[in] unique_keys_pol Whether to enable the unique keys optimization.
   * @param[in] perfect_hashing Whether to use perfect hashing.
   * @param[in] left_filter_condition A boolean expression to filter the left table before the join.
   * @param[in] right_filter_condition A boolean expression to filter the right table before the
   * join.
   * @param[in] use_like_shift_and Whether to use the shift-and algorithm for LIKE middle-pattern
   * matching when evaluating the join condition or filter conditions.
   */
  join_relation_base(std::shared_ptr<relation> left,
                     std::shared_ptr<relation> right,
                     std::vector<std::shared_ptr<relation>> subquery_relations,
                     join_type_type join_type,
                     std::unique_ptr<expression> condition,
                     std::vector<cudf::size_type> projection_indices,
                     gqe::unique_keys_policy unique_keys_pol,
                     bool perfect_hashing,
                     std::unique_ptr<expression> left_filter_condition  = nullptr,
                     std::unique_ptr<expression> right_filter_condition = nullptr,
                     bool use_like_shift_and                            = false)
    : relation({std::move(left), std::move(right)}, std::move(subquery_relations)),
      _join_type(join_type),
      _condition(std::move(condition)),
      _projection_indices(std::move(projection_indices)),
      _unique_keys_policy(unique_keys_pol),
      _perfect_hashing(perfect_hashing),
      _left_filter_condition(std::move(left_filter_condition)),
      _right_filter_condition(std::move(right_filter_condition)),
      _use_like_shift_and(use_like_shift_and)
  {
  }

  /**
   * @brief Return the join type.
   */
  [[nodiscard]] join_type_type join_type() const noexcept { return _join_type; }

  /**
   * @brief Return the join condition.
   *
   * The join condition is a boolean expression to define when a left tuple matches with a right
   * tuple.
   */
  [[nodiscard]] expression* condition() const { return _condition.get(); }

  /**
   * @brief Return the column indices to materialize after the join.
   */
  [[nodiscard]] std::vector<cudf::size_type> projection_indices() const noexcept
  {
    return _projection_indices;
  }

  /*
   * @brief Return a unique_keys_policy indicating whether the unique keys optimization can
   * be enabled with building on the right or left.
   */
  [[nodiscard]] gqe::unique_keys_policy unique_keys_policy() const noexcept
  {
    return _unique_keys_policy;
  }

  /**
   * @brief Return a boolean indicating whether to use perfect hashing.
   */
  [[nodiscard]] bool perfect_hashing() const noexcept { return _perfect_hashing; }

  /**
   * @brief Return the left filter condition.
   *
   * The left filter condition is a boolean expression to filter the left table before the join.
   */
  [[nodiscard]] expression* left_filter_condition() const { return _left_filter_condition.get(); }

  /**
   * @brief Return the right filter condition.
   *
   * The right filter condition is a boolean expression to filter the right table before the join.
   */
  [[nodiscard]] expression* right_filter_condition() const { return _right_filter_condition.get(); }

  /**
   * @brief Return whether to use the shift-and algorithm for LIKE middle-pattern matching when
   * evaluating join expressions.
   */
  [[nodiscard]] bool use_like_shift_and() const noexcept { return _use_like_shift_and; }

  /**
   * @copydoc relation::output_data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> output_data_types() const override;

  /**
   * @brief Print all its members as well as its output data_types
   */
  [[nodiscard]] std::string print() const;

 private:
  join_type_type _join_type;
  // the join condition is a canonicalized boolean expression: NOTs pushed down
  // (e.g. `!(a || b)` rewritten to `!a && !b`) and ORs lifted out.
  std::unique_ptr<expression> _condition;
  std::vector<cudf::size_type> _projection_indices;
  gqe::unique_keys_policy _unique_keys_policy;
  bool _perfect_hashing;
  std::unique_ptr<expression> _left_filter_condition;
  std::unique_ptr<expression> _right_filter_condition;
  bool _use_like_shift_and;
};

/**
 * @brief Indicates whether to broadcast the right relation or the left relation.
 */
enum class broadcast_policy : bool {
  right,  ///< Broadcast the right relation.
  left    ///< Broadcast the left relation. Only supported for an inner join.
};

class broadcast_join_relation : public join_relation_base {
 public:
  /**
   * @brief Construct a physical broadcast join relation.
   *
   * @param[in] left Left table to join.
   * @param[in] right Right table to join.
   * @param[in] subquery_relations Subquery relations that are referenced within the `condition`
   * expression.
   * @param[in] join_type Type of the join.
   * @param[in] condition A boolean expression to define when a left tuple matches with a right
   * tuple.
   * @param[in] projection_indices Column indices to materialize after the join. The rest of columns
   * are discarded.
   * @param[in] policy Whether to broadcast the right relation or the left relation.
   * @param[in] unique_keys_pol Whether to enable the unique keys optimization.
   * @param[in] perfect_hashing Whether to use perfect hashing.
   * @param[in] left_filter_condition A boolean expression to filter the left table before the join.
   * @param[in] right_filter_condition A boolean expression to filter the right table before the
   * join.
   * @param[in] use_hash_map_cache Whether to share one hash table across probe tasks (requires
   * equality-only join, or mark join on semi/anti). Preconditions already validated by the
   * optimizer; if true the executor may skip its fallback check.
   * @param[in] use_mark_join Whether to use a mark-join for left semi/anti (requires numeric join
   * keys). Preconditions already validated by the optimizer.
   * @param[in] use_like_shift_and Whether to use the shift-and algorithm for LIKE middle-pattern
   * matching.
   */
  broadcast_join_relation(std::shared_ptr<relation> left,
                          std::shared_ptr<relation> right,
                          std::vector<std::shared_ptr<relation>> subquery_relations,
                          join_type_type join_type,
                          std::unique_ptr<expression> condition,
                          std::vector<cudf::size_type> projection_indices,
                          broadcast_policy policy,
                          gqe::unique_keys_policy unique_keys_pol = gqe::unique_keys_policy::none,
                          bool perfect_hashing                    = false,
                          std::unique_ptr<expression> left_filter_condition  = nullptr,
                          std::unique_ptr<expression> right_filter_condition = nullptr,
                          bool use_hash_map_cache                            = false,
                          bool use_mark_join                                 = false,
                          bool use_like_shift_and                            = false)
    : join_relation_base(std::move(left),
                         std::move(right),
                         std::move(subquery_relations),
                         join_type,
                         std::move(condition),
                         std::move(projection_indices),
                         unique_keys_pol,
                         perfect_hashing,
                         std::move(left_filter_condition),
                         std::move(right_filter_condition),
                         use_like_shift_and),
      _policy(policy),
      _use_hash_map_cache(use_hash_map_cache),
      _use_mark_join(use_mark_join)
  {
  }

  /**
   * @brief Return a policy indicating whether the join should broadcast the right relation or the
   * left one.
   */
  [[nodiscard]] broadcast_policy policy() const noexcept { return _policy; }

  /**
   * @brief Return whether to use hash map cache (shared hash table across probe tasks).
   */
  [[nodiscard]] bool use_hash_map_cache() const noexcept { return _use_hash_map_cache; }

  /**
   * @brief Return whether to use mark join for left semi/anti joins.
   */
  [[nodiscard]] bool use_mark_join() const noexcept { return _use_mark_join; }

  [[nodiscard]] relation_type type() const noexcept override
  {
    return relation_type::broadcast_join;
  }

  /**
   * @copydoc gqe::physical::relation::accept(relation_visitor&)
   */
  void accept(relation_visitor& visitor) override { visitor.visit(this); }

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

 private:
  broadcast_policy _policy;
  bool _use_hash_map_cache;
  bool _use_mark_join;
};

class shuffle_join_relation : public join_relation_base {
 public:
  /**
   * @brief Construct a physical shuffle join relation (also called repartition join).
   *
   * @param[in] left Left table to join.
   * @param[in] right Right table to join.
   * @param[in] subquery_relations Subquery relations that are referenced within the `condition`
   * expression.
   * @param[in] join_type Type of the join.
   * @param[in] condition A boolean expression to define when a left tuple matches with a right
   * tuple.
   * @param[in] projection_indices Column indices to materialize after the join. The rest of columns
   * are discarded.
   * @param[in] unique_keys_pol Whether to enable the unique keys optimization.
   * @param[in] perfect_hashing Whether to use perfect hashing.
   */
  shuffle_join_relation(std::shared_ptr<relation> left,
                        std::shared_ptr<relation> right,
                        std::vector<std::shared_ptr<relation>> subquery_relations,
                        join_type_type join_type,
                        std::unique_ptr<expression> condition,
                        std::vector<cudf::size_type> projection_indices,
                        gqe::unique_keys_policy unique_keys_pol = gqe::unique_keys_policy::none,
                        bool perfect_hashing                    = false,
                        bool use_like_shift_and                 = false)
    : join_relation_base(std::move(left),
                         std::move(right),
                         std::move(subquery_relations),
                         join_type,
                         std::move(condition),
                         std::move(projection_indices),
                         unique_keys_pol,
                         perfect_hashing,
                         /*left_filter_condition=*/nullptr,
                         /*right_filter_condition=*/nullptr,
                         use_like_shift_and)
  {
  }

  [[nodiscard]] relation_type type() const noexcept override { return relation_type::shuffle_join; }

  /**
   * @copydoc gqe::physical::relation::accept(relation_visitor&)
   */
  void accept(relation_visitor& visitor) override { visitor.visit(this); }

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;
};

}  // namespace physical
}  // namespace gqe
