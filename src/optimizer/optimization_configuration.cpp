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

#include <gqe/optimizer/optimization_configuration.hpp>

#include <vector>

namespace gqe {
namespace optimizer {
optimization_configuration::optimization_configuration(
  std::vector<logical_optimization_rule_type> on_rules,
  std::vector<logical_optimization_rule_type> off_rules)
{
  // Enable and diable rules based on user-specified parameters
  // Make sure there is no rule overlap
  std::vector<logical_optimization_rule_type> overlap;
  std::set_intersection(on_rules.begin(),
                        on_rules.end(),
                        off_rules.begin(),
                        off_rules.end(),
                        std::back_inserter(overlap));
  if (overlap.empty()) {
    for (auto on_rule : on_rules) {
      _on_rules.push_back(on_rule);
    }
  } else {
    // If there are some overlap in on and off rules, the off rules take precedence
    GQE_LOG_WARN(
      "There are overlap in on and off optimization rules. The off-rules take precedence.");
    for (auto on_rule : on_rules) {
      if (std::find(overlap.begin(), overlap.end(), on_rule) != overlap.end())
        _on_rules.push_back(on_rule);
    }
  }
}
}  // namespace optimizer
}  // namespace gqe
