/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <gqe/optimizer/rules/join_extraction_utils.hpp>

#include <gqe/expression/column_reference.hpp>
#include <gqe/logical/project.hpp>

#include <algorithm>

namespace gqe::optimizer::join_extraction_utils {

namespace {

using expr_type = gqe::expression::expression_type;

}  // namespace

void collect_column_indices(gqe::expression const* expr, std::vector<cudf::size_type>& indices)
{
  if (expr->type() == expr_type::column_reference) {
    indices.push_back(static_cast<gqe::column_reference_expression const*>(expr)->column_idx());
    return;
  }
  for (auto const* child : expr->children()) {
    collect_column_indices(child, indices);
  }
}

operand_side classify_operand(gqe::expression const* expr, cudf::size_type num_left)
{
  std::vector<cudf::size_type> indices;
  collect_column_indices(expr, indices);
  if (indices.empty()) return operand_side::invalid;
  if (std::all_of(indices.begin(), indices.end(), [&](auto i) { return i < num_left; })) {
    return operand_side::left;
  }
  if (std::all_of(indices.begin(), indices.end(), [&](auto i) { return i >= num_left; })) {
    return operand_side::right;
  }
  return operand_side::invalid;
}

expression_modifier_functor make_column_reference_shift_modifier(cudf::size_type shift)
{
  return [shift](gqe::expression* e,
                 std::vector<cudf::data_type> const&) -> std::unique_ptr<gqe::expression> {
    if (e->type() == gqe::expression::expression_type::column_reference) {
      auto idx = static_cast<gqe::column_reference_expression*>(e)->column_idx();
      return std::make_unique<gqe::column_reference_expression>(idx - shift);
    }
    return nullptr;
  };
}

std::shared_ptr<gqe::logical::relation> build_side_project(
  std::shared_ptr<gqe::logical::relation> input,
  cudf::size_type num_orig,
  std::vector<gqe::expression*> const& exprs,
  cudf::size_type shift,
  rewrite_with_shift_fn const& rewrite_with_shift)
{
  std::vector<std::unique_ptr<gqe::expression>> outputs;
  outputs.reserve(static_cast<std::size_t>(num_orig) + exprs.size());
  for (cudf::size_type i = 0; i < num_orig; ++i) {
    outputs.push_back(std::make_unique<gqe::column_reference_expression>(i));
  }
  for (auto const* expr : exprs) {
    auto cloned = expr->clone();
    if (shift != 0) {
      if (auto replaced = rewrite_with_shift(cloned.get(), shift)) { cloned = std::move(replaced); }
    }
    outputs.push_back(std::move(cloned));
  }
  return std::make_shared<gqe::logical::project_relation>(
    std::move(input), std::vector<std::shared_ptr<gqe::logical::relation>>{}, std::move(outputs));
}

std::vector<cudf::size_type> remap_projection_for_left_insertion(
  std::vector<cudf::size_type> const& projection_indices,
  cudf::size_type num_left,
  cudf::size_type n_left_added)
{
  if (n_left_added == 0) return projection_indices;
  std::vector<cudf::size_type> new_projection;
  new_projection.reserve(projection_indices.size());
  for (auto idx : projection_indices) {
    new_projection.push_back(idx < num_left ? idx : idx + n_left_added);
  }
  return new_projection;
}

expression_modifier_functor make_operand_remapper(
  std::unordered_map<gqe::expression const*, cudf::size_type> const& operand_target)
{
  return [operand_target](gqe::expression* e,
                          std::vector<cudf::data_type> const&) -> std::unique_ptr<gqe::expression> {
    if (auto it = operand_target.find(e); it != operand_target.end()) {
      return std::make_unique<gqe::column_reference_expression>(it->second);
    }
    return nullptr;
  };
}

}  // namespace gqe::optimizer::join_extraction_utils
