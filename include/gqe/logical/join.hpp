/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
