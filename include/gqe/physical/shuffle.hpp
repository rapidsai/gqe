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
