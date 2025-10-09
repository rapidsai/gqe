/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/physical/relation.hpp>

#include <cassert>
#include <vector>

namespace gqe {
namespace physical {

/**
 * @brief Physical relation for shuffling the input relation.
 * This relation is used to shuffle the input relation based on the shuffle columns.
 * check more details in partition_task.
 */
class shuffle_relation : public relation {
 public:
  /**
   * @brief Construct a physical shuffle relation.
   *
   * @param[in] input Input relation to be shuffled.
   * @param[in] subquery_relations Subquery relations that are referenced within the `shuffle_cols`
   * expression.
   * @param[in] shuffle_cols Columns to shuffle the input relation.
   */
  shuffle_relation(std::shared_ptr<relation> input,
                   std::vector<std::shared_ptr<relation>> subquery_relations,
                   std::vector<std::unique_ptr<expression>> shuffle_cols)
    : relation({std::move(input)}, std::move(subquery_relations)),
      _shuffle_cols(std::move(shuffle_cols))
  {
    // Check that the shuffle columns are column reference expressions for debugging
    for ([[maybe_unused]] auto& shuffle_col : _shuffle_cols) {
      // at the moment, we only support column reference expressions for shuffle columns
      assert(shuffle_col->type() == expression::expression_type::column_reference);
    }
  }

  /**
   * @copydoc relation::accept(relation_visitor&)
   */
  void accept(relation_visitor& visitor) override { visitor.visit(this); }

  /**
   * @brief Return the columns to shuffle the input relation.
   */
  [[nodiscard]] std::vector<expression*> shuffle_cols_unsafe() const noexcept
  {
    return utility::to_raw_ptrs(_shuffle_cols);
  }

  /**
   * @copydoc relation::output_data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> output_data_types() const override;

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

 private:
  std::vector<std::unique_ptr<expression>> _shuffle_cols;
};

}  // namespace physical
}  // namespace gqe
