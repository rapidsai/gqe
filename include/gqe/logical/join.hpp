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

#include <gqe/logical/relation.hpp>

namespace gqe {
namespace optimizer {
class join_children_swap;
class optimization_rule;
class projection_pushdown;
}  // namespace optimizer
namespace logical {

class join_relation : public relation {
  friend class gqe::optimizer::join_children_swap;
  friend class gqe::optimizer::projection_pushdown;
  friend class gqe::optimizer::optimization_rule;

 public:
  /**
   * @brief Construct a new join relation object
   *
   * @param left The left input relation
   * @param right The right input relation
   * @param condition The expression to apply to input keys
   * @param join_type Type of join
   * @param projection_indices Column indices to materialize after the join. The rest of columns
   * are discarded.
   */
  join_relation(std::shared_ptr<relation> left,
                std::shared_ptr<relation> right,
                std::vector<std::shared_ptr<relation>> subquery_relations,
                std::unique_ptr<expression> condition,
                join_type_type join_type,
                std::vector<cudf::size_type> projection_indices);

  /**
   * @copydoc relation::type()
   */
  [[nodiscard]] relation_type type() const noexcept override { return relation_type::join; }

  /**
   * @copydoc relation::data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> data_types() const override;

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

  /**
   * @brief Return join type for this relation
   *
   * @return Type of join to perform
   */
  [[nodiscard]] join_type_type join_type() const noexcept { return _join_type; }

  /**
   * @brief Return the join condition for this relation
   *
   * The condition defines when a left key matches a right key
   *
   * @return Join condition
   *
   * @note This function does not share ownership. The caller is responsible for keeping
   * the returned pointer alive.
   */
  [[nodiscard]] expression* condition() const noexcept { return _condition.get(); }

  /**
   * @brief Return the list of projection indices that indicate columns to return
   *
   * @return List of projection indices
   */
  [[nodiscard]] const std::vector<cudf::size_type>& projection_indices() const noexcept
  {
    return _projection_indices;
  }

  /**
   * @copydoc relation::operator==(const relation& other)
   */
  bool operator==(const relation& other) const override;

  /**
   * @brief Return a unique_keys_policy indicating whether the unique keys optimization can
   * be enabled with building on the right, left or either side.
   */
  [[nodiscard]] gqe::unique_keys_policy unique_keys_policy() const noexcept
  {
    return _unique_keys_policy;
  }

  /**
   * @brief Set policy for enabling the unique keys optimization for inner hash join
   *
   * @param policy The policy to be set to
   */
  void set_unique_keys_policy(gqe::unique_keys_policy policy) { _unique_keys_policy = policy; };

 private:
  std::unique_ptr<expression> _condition;
  join_type_type _join_type;
  std::vector<cudf::size_type> _projection_indices;
  gqe::unique_keys_policy _unique_keys_policy = gqe::unique_keys_policy::none;
};

}  // namespace logical
}  // namespace gqe
