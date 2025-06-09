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
class projection_pushdown;
class optimization_rule;
}  // namespace optimizer
namespace logical {

class filter_relation : public relation {
  friend class gqe::optimizer::projection_pushdown;
  friend class gqe::optimizer::optimization_rule;

 public:
  /**
   * @brief Construct a new filter relation object
   *
   * @param input_relation Input relation to apply filter on
   * @param condition Filter expression to apply to the input
   * @param[in] projection_indices Column indices to materialize after the filter.
   */
  filter_relation(std::shared_ptr<relation> input_relation,
                  std::vector<std::shared_ptr<relation>> subquery_relations,
                  std::unique_ptr<expression> condition,
                  std::vector<cudf::size_type> projection_indices);

  /**
   * @copydoc relation::type()
   */
  [[nodiscard]] relation_type type() const noexcept override { return relation_type::filter; }

  /**
   * @copydoc relation::data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> data_types() const override;

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

  /**
   * @brief Return the filter condition for this relation
   *
   * The condition defines which rows to return
   *
   * @return Filter condition
   *
   * @note This function does not share ownership. The caller is responsible for keeping
   * the returned pointer alive.
   */
  [[nodiscard]] expression* condition() const noexcept { return _condition.get(); }

  /**
   * @brief Return the column indices to materialize after the filter.
   */
  [[nodiscard]] const std::vector<cudf::size_type>& projection_indices() const noexcept
  {
    return _projection_indices;
  }

  /**
   * @copydoc relation::operator==(const relation& other)
   */
  bool operator==(const relation& other) const override;

 private:
  std::unique_ptr<expression> _condition;
  std::vector<cudf::size_type> _projection_indices;
};

}  // namespace logical
}  // namespace gqe
