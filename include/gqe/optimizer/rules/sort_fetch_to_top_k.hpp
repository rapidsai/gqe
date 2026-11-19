/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/optimizer/logical_optimization.hpp>

namespace gqe::optimizer {

/**
 * @brief Rewrite `fetch(sort(input))` into a single logical top-K relation.
 *
 * The physical top-K operator is single-key, so sorts with zero or multiple keys retain the
 * existing sort-plus-fetch plan.
 */
class sort_fetch_to_top_k : public optimization_rule {
 public:
  explicit sort_fetch_to_top_k(catalog const* cat) : optimization_rule(cat) {}

  [[nodiscard]] std::shared_ptr<logical::relation> apply(std::shared_ptr<logical::relation> root,
                                                         bool& rule_applied) const override;

  [[nodiscard]] logical_optimization_rule_type type() const noexcept override
  {
    return logical_optimization_rule_type::sort_fetch_to_top_k;
  }

 private:
  class apply_visitor;

  [[nodiscard]] static std::shared_ptr<logical::relation> rewrite(logical::fetch_relation* fetch);
};

}  // namespace gqe::optimizer
