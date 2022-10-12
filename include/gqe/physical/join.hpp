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
   * @param[in] join_type Type of the join.
   * @param[in] condition A boolean expression to define when a left tuple matches with a right
   * tuple.
   * @param[in] projection_indices Column indices to materialize after the join. The rest of columns
   * are discarded.
   */
  join_relation_base(std::shared_ptr<relation> left,
                     std::shared_ptr<relation> right,
                     join_type_type join_type,
                     std::unique_ptr<expression> condition,
                     std::vector<cudf::size_type> projection_indices)
    : relation({std::move(left), std::move(right)}),
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
 * @brief Join the input tables by broadcasting the right table.
 */
class broadcast_join_relation : public join_relation_base {
  using join_relation_base::join_relation_base;

 public:
  /**
   * @copydoc gqe::physical::relation::accept(relation_visitor&)
   */
  void accept(relation_visitor& visitor) override { visitor.visit(this); }
};

}  // namespace physical
}  // namespace gqe
