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

#include <gqe/catalog.hpp>
#include <gqe/optimizer/logical_optimization.hpp>

namespace gqe {
namespace optimizer {

/**
 * @brief Fix column references in partial filters so that they refer to the correct columns in the
 * base table schema.
 *
 * The partial filters of the read relation of Substrait plans contain column references that refer
 * to the full base table schema. However, when trying to process large data sets, we only load the
 * required columns into memory. This makes these column references stored in the partial filters
 * invalid.
 *
 * This optimization rule reconstructs the correct column references using the following algorithm,
 * based on the information contained in the projected columns of the read relation and the column
 * references of the filter expression.
 *
 * 1. Stores the filter condition of the filter relation. The filter relation should always exist if
 *    a partial filter is used in the read relation.
 * 2. Updates the column references so that they refer to the base table schema.
 * 3. Sets the updated filter condition as the partial filter of the read relation
 * 4. Restores the filter condition of the filter relation stored in step 1.
 *
 * It should be safe to always apply this optimization rule, regardless of whether all columns or
 * only a subset of the columns of a table are loaded. That is because the read relation, the
 * filter relation, and the catalog contain all the required information. This rule brings the
 * partial filter of the read relation into a state that is consistent with the other information.
 */
class fix_partial_filter_column_references : public optimization_rule {
 public:
  explicit fix_partial_filter_column_references(catalog const* cat)
    : optimization_rule(cat, optimization_rule::transform_direction::DOWN)
  {
  }

  std::shared_ptr<logical::relation> try_optimize(
    std::shared_ptr<logical::relation> logical_relation, bool& rule_applied) const override;

  [[nodiscard]] logical_optimization_rule_type type() const noexcept override
  {
    return logical_optimization_rule_type::fix_partial_filter_column_references;
  }

 private:
  void set_expression(gqe::logical::relation* relation,
                      std::unique_ptr<gqe::expression> new_expression) const;

  std::unique_ptr<gqe::expression> replace_column_references(
    const std::vector<std::string>& base_table_columns,
    const std::vector<std::string>& projected_columns,
    gqe::logical::filter_relation* filter_relation) const;
};

}  // namespace optimizer
}  // namespace gqe
