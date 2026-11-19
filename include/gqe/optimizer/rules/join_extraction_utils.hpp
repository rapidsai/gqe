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

#pragma once

#include <gqe/expression/expression.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/optimizer/logical_optimization.hpp>

#include <cudf/types.hpp>

#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gqe::optimizer::join_extraction_utils {

enum class operand_side {
  left,
  right,
  invalid,
};

void collect_column_indices(gqe::expression const* expr, std::vector<cudf::size_type>& indices);
[[nodiscard]] operand_side classify_operand(gqe::expression const* expr, cudf::size_type num_left);

using expression_modifier_functor = std::function<std::unique_ptr<gqe::expression>(
  gqe::expression*, std::vector<cudf::data_type> const&)>;

using rewrite_with_shift_fn =
  std::function<std::unique_ptr<gqe::expression>(gqe::expression*, cudf::size_type)>;
[[nodiscard]] expression_modifier_functor make_column_reference_shift_modifier(
  cudf::size_type shift);

std::shared_ptr<gqe::logical::relation> build_side_project(
  std::shared_ptr<gqe::logical::relation> input,
  cudf::size_type num_orig,
  std::vector<gqe::expression*> const& exprs,
  cudf::size_type shift,
  rewrite_with_shift_fn const& rewrite_with_shift);
[[nodiscard]] std::vector<cudf::size_type> remap_projection_for_left_insertion(
  std::vector<cudf::size_type> const& projection_indices,
  cudf::size_type num_left,
  cudf::size_type n_left_added);
[[nodiscard]] expression_modifier_functor make_operand_remapper(
  std::unordered_map<gqe::expression const*, cudf::size_type> const& operand_target);

}  // namespace gqe::optimizer::join_extraction_utils
