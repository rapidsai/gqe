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
namespace logical {

class join_relation : public relation {
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
   * @param compare_nulls Whether NULL keys should match or not. By default, a JOIN in SQL treats
   * NULL keys as not equal, but the set operators like INTERSECT treat NULLs as equal.
   */
  join_relation(std::shared_ptr<relation> left,
                std::shared_ptr<relation> right,
                std::vector<std::shared_ptr<relation>> subquery_relations,
                std::unique_ptr<expression> condition,
                join_type_type join_type,
                std::vector<cudf::size_type> projection_indices,
                cudf::null_equality compare_nulls = cudf::null_equality::UNEQUAL);

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
  [[nodiscard]] std::vector<cudf::size_type> projection_indices() const noexcept
  {
    return _projection_indices;
  }

  /**
   * @brief Return whether NULL keys should match or not.
   */
  [[nodiscard]] cudf::null_equality compare_nulls() const noexcept { return _compare_nulls; }

 private:
  void _init_data_types() const;
  std::unique_ptr<expression> _condition;
  join_type_type _join_type;
  std::vector<cudf::size_type> _projection_indices;
  cudf::null_equality _compare_nulls;
  mutable std::vector<cudf::data_type> _data_types;
};

}  // namespace logical
}  // namespace gqe