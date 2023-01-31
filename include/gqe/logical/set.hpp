/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

class set_relation : public relation {
 public:
  enum set_operator_type {
    // Suppose a row R appears exactly M times in the first query and exactly N times in the second
    // query.
    set_union,  ///< Union and drop duplicates. R appears 1 time in the result if either M>0 or N>0.
    set_union_all,  ///< Union but not drop duplicates. R appears (M + N) times in the result.
    set_intersect,  ///< Intersect and drop duplicates. R appears 1 time in the result if M>0 and
                    ///< N>0.
    set_minus       ///< Minus and drop duplicates. R appears 1 time in the result if M>0 and N=0.
  };

  /**
   * @brief Construct a new set operation.
   *
   * @param[in] lhs Query on the left-hand side of the set operator.
   * @param[in] rhs Query on the right-hand side of the set operator.
   * @param[in] op Set operator type.
   */
  set_relation(std::shared_ptr<relation> lhs, std::shared_ptr<relation> rhs, set_operator_type op);

  /**
   * @brief Return the set operator.
   */
  [[nodiscard]] set_operator_type set_operator() const noexcept { return _op; }

  /**
   * @copydoc relation::type()
   */
  [[nodiscard]] relation_type type() const noexcept override { return relation_type::set; }

  /**
   * @copydoc relation::data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> data_types() const override;

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

 private:
  set_operator_type _op;
};

}  // namespace logical
}  // namespace gqe
