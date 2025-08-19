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
