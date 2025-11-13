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
