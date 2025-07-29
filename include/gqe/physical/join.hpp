/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
   */
  join_relation_base(std::shared_ptr<relation> left,
                     std::shared_ptr<relation> right,
                     std::vector<std::shared_ptr<relation>> subquery_relations,
                     join_type_type join_type,
                     std::unique_ptr<expression> condition,
                     std::vector<cudf::size_type> projection_indices)
    : relation({std::move(left), std::move(right)}, std::move(subquery_relations)),
      _join_type(join_type),
      _condition(std::move(condition)),
      _projection_indices(std::move(projection_indices))
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

 private:
  join_type_type _join_type;
  std::unique_ptr<expression> _condition;
  std::vector<cudf::size_type> _projection_indices;
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
   */
  broadcast_join_relation(std::shared_ptr<relation> left,
                          std::shared_ptr<relation> right,
                          std::vector<std::shared_ptr<relation>> subquery_relations,
                          join_type_type join_type,
                          std::unique_ptr<expression> condition,
                          std::vector<cudf::size_type> projection_indices,
                          broadcast_policy policy,
                          gqe::unique_keys_policy unique_keys_pol = gqe::unique_keys_policy::none,
                          bool perfect_hashing                    = false)
    : join_relation_base(std::move(left),
                         std::move(right),
                         std::move(subquery_relations),
                         join_type,
                         std::move(condition),
                         std::move(projection_indices)),
      _policy(policy),
      _unique_keys_policy(unique_keys_pol),
      _perfect_hashing(perfect_hashing)
  {
  }

  /**
   * @brief Return a policy indicating whether the join should broadcast the right relation or the
   * left one.
   */
  [[nodiscard]] broadcast_policy policy() const noexcept { return _policy; }

  /**
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
   * @copydoc gqe::physical::relation::accept(relation_visitor&)
   */
  void accept(relation_visitor& visitor) override { visitor.visit(this); }

 private:
  broadcast_policy _policy;
  gqe::unique_keys_policy _unique_keys_policy;
  bool _perfect_hashing;
};

}  // namespace physical
}  // namespace gqe
