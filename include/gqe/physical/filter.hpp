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

#include <gqe/expression/expression.hpp>
#include <gqe/physical/relation.hpp>

#include <memory>

namespace gqe {
namespace physical {

/**
 * @brief Physical relation for getting a subset of rows based on conditions.
 */
class filter_relation : public relation {
 public:
  /**
   * @brief Construct a physical filter relation.
   *
   * @param[in] input Input table to be filtered.
   * @param[in] subquery_relations Subquery relations that are referenced within the `condition`
   * expression.
   * @param[in] condition A boolean expression evaluated on `input` to represent the filter
   * condition.
   * @param[in] projection_indices Column indices to materialize after the filter.
   */
  filter_relation(std::shared_ptr<relation> input,
                  std::vector<std::shared_ptr<relation>> subquery_relations,
                  std::unique_ptr<expression> condition,
                  std::vector<cudf::size_type> projection_indices)
    : relation({std::move(input)}, std::move(subquery_relations)),
      _condition(std::move(condition)),
      _projection_indices(std::move(projection_indices))
  {
  }

  /**
   * @brief Return the filter condition.
   */
  expression* condition_unsafe() { return _condition.get(); }

  /**
   * @copydoc gqe::physical::relation::accept(relation_visitor&)
   */
  void accept(relation_visitor& visitor) override { visitor.visit(this); }

  /**
   * @brief Return the column indices to materialize after the filter.
   */
  [[nodiscard]] const std::vector<cudf::size_type>& projection_indices() const noexcept
  {
    return _projection_indices;
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
  std::unique_ptr<expression> _condition;
  std::vector<cudf::size_type> _projection_indices;
};

}  // namespace physical
}  // namespace gqe
