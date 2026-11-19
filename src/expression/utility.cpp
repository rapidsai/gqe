/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/expression/utility.hpp>

#include <gqe/expression/column_reference.hpp>

#include <algorithm>
#include <format>
#include <stdexcept>

namespace gqe {
namespace {

// Accumulate into a shared result so references are deduplicated across every expression root.
void collect_column_references_impl(expression const& current, std::vector<cudf::size_type>& result)
{
  if (current.type() == expression::expression_type::column_reference) {
    auto const index = static_cast<column_reference_expression const&>(current).column_idx();
    if (std::ranges::find(result, index) == result.end()) { result.push_back(index); }
    return;
  }

  for (auto const* child : current.children()) {
    if (child == nullptr) {
      throw std::invalid_argument("collect_column_references: expression contains a null child");
    }
    collect_column_references_impl(*child, result);
  }
}

}  // namespace

std::vector<cudf::size_type> collect_column_references(
  std::span<expression const* const> expressions)
{
  std::vector<cudf::size_type> result;
  for (auto const* root : expressions) {
    if (root == nullptr) {
      throw std::invalid_argument("collect_column_references: root expression must not be null");
    }
    collect_column_references_impl(*root, result);
  }
  return result;
}

void validate_column_references(std::span<cudf::size_type const> column_indices,
                                cudf::size_type logical_column_count)
{
  if (logical_column_count < 0) {
    throw std::out_of_range(
      "validate_column_references: logical column count must be non-negative");
  }
  for (auto const index : column_indices) {
    if (index < 0 || index >= logical_column_count) {
      throw std::out_of_range(std::format(
        "validate_column_references: logical column index {} is out of range for {} columns",
        index,
        logical_column_count));
    }
  }
}

}  // namespace gqe
