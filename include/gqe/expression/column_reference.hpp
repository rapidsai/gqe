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

#include <cudf/types.hpp>

namespace gqe {

/**
 * @brief An expression referencing a column of the associated table.
 */
class column_reference_expression : public expression {
 public:
  /**
   * @brief Construct a column reference expression.
   *
   * @param[in] column_idx Column index of the referenced column (zero-based).
   */
  column_reference_expression(cudf::size_type column_idx) : expression({}), _column_idx(column_idx)
  {
  }

  /**
   * @copydoc gqe::expression::type()
   */
  [[nodiscard]] expression_type type() const noexcept override
  {
    return expression_type::column_reference;
  }

  /**
   * @copydoc gqe::expression::data_type(std::vector<cudf::data_type> const&)
   */
  [[nodiscard]] cudf::data_type data_type(
    std::vector<cudf::data_type> const& column_types) const override
  {
    return column_types[_column_idx];
  }

  /**
   * @brief Return the column index.
   */
  [[nodiscard]] cudf::size_type column_idx() const noexcept { return _column_idx; }

  /**
   * @copydoc gqe::expression::to_string()
   */
  [[nodiscard]] std::string to_string() const noexcept override
  {
    return "column_reference(" + std::to_string(_column_idx) + ")";
  }

  /**
   * @copydoc gqe::expression::clone()
   */
  [[nodiscard]] std::unique_ptr<expression> clone() const override
  {
    return std::make_unique<column_reference_expression>(*this);
  }

  /**
   * @copydoc gqe::expression::accept()
   */
  void accept(expression_visitor& visitor) const override { visitor.visit(this); }

 private:
  cudf::size_type _column_idx;
};

}  // namespace gqe
