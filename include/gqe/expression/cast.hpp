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

#include <gqe/expression/expression.hpp>

#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

namespace gqe {

/**
 * @brief An expression to cast from one data type to another.
 */
class cast_expression : public expression {
 public:
  /**
   * @brief Construct a cast expression.
   *
   * @param[in] input Expression to cast
   * @param[in] out_type Data type to cast to
   */
  cast_expression(std::shared_ptr<expression> input, cudf::data_type out_type)
    : expression({std::move(input)}), _out_type(out_type)
  {
  }

  /**
   * @copydoc gqe::expression::type()
   */
  [[nodiscard]] expression_type type() const noexcept override { return expression_type::cast; }

  /**
   * @copydoc gqe::expression::data_type(std::vector<cudf::data_type> const&)
   */
  [[nodiscard]] cudf::data_type data_type(
    std::vector<cudf::data_type> const& column_types) const override
  {
    return _out_type;
  }

  /**
   * @copydoc gqe::expression::to_string()
   */
  [[nodiscard]] std::string to_string() const noexcept override
  {
    auto child_exprs = children();
    assert(child_exprs.size() == 1);
    return "cast(" + child_exprs[0]->to_string() + " as " + cudf::type_to_name(_out_type) + ")";
  }

  /**
   * @copydoc gqe::expression::clone()
   */
  [[nodiscard]] std::unique_ptr<expression> clone() const override
  {
    return std::make_unique<cast_expression>(*this);
  }

  /**
   * @copydoc expression::operator==(const relation& other)
   */
  bool operator==(const expression& other) const override;

  /**
   * @copydoc gqe::expression::accept()
   */
  void accept(expression_visitor& visitor) const override { visitor.visit(this); }

  /**
   * @brief Return the data type to cast to
   */
  [[nodiscard]] cudf::data_type out_type() const noexcept { return _out_type; }

 private:
  cudf::data_type _out_type;
};

}  // namespace gqe
