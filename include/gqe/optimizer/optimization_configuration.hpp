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

#include <gqe/utility/logger.hpp>

#include <algorithm>
#include <vector>

namespace gqe {
namespace optimizer {
enum class logical_optimization_rule_type {
  not_not_rewrite,
  join_children_swap,
  uniqueness_propagation,
  projection_pushdown,
  string_to_int_literal,
  join_unique_keys,
  fix_partial_filter_column_references,
  column_name_assignment,
  num_rules
};

class optimization_configuration {
 public:
  /**
   * @brief Construct a new optimization configuration object
   *
   * @note If the set of on and off rules overlap, the off rules take precedence
   *
   * @param on_rules List of rules to enable
   * @param off_rules List of rules to disable
   */
  optimization_configuration(std::vector<logical_optimization_rule_type> on_rules  = {},
                             std::vector<logical_optimization_rule_type> off_rules = {});

  /**
   * @brief Return the list of enabled rules
   */
  std::vector<logical_optimization_rule_type> on_rules() { return _on_rules; }

 private:
  std::vector<logical_optimization_rule_type> _on_rules;
};
}  // namespace optimizer
}  // namespace gqe
