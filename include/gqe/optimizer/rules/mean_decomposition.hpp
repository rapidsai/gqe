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

#pragma once

#include <gqe/optimizer/logical_optimization.hpp>

namespace gqe {
namespace optimizer {

/**
 * @brief Rewrite each `MEAN(x)` measure of an `aggregate_relation` into `SUM(x)` and
 *        `COUNT_VALID(x)`, and wrap the aggregate in a `project_relation` that emits
 *        `SUM(x) / COUNT_VALID(x)` at the original `MEAN` position.
 *
 * # Motivation
 *
 * Expressing `MEAN` as the composition of two simpler aggregations frees downstream aggregate
 * implementations from having to handle `MEAN` natively, and enables two-stage (apply-concat-apply)
 * execution: each partition computes partial `SUM` and `COUNT` independently and the second stage
 * combines per-partition partials, where naive "average of averages" would be incorrect. The
 * decomposition is exact under SQL semantics (`avg(x) = sum(x) / count(x)` where `count(x)`
 * excludes nulls, hence `COUNT_VALID`).
 *
 * The rewrite is structure-preserving and idempotent: the rebuilt aggregate has no `MEAN` measures,
 * so a second pass finds nothing to do. The downstream project reproduces the aggregate's original
 * output schema `[keys..., measures...]`, so consumers above are unaffected (a `MEAN` measure and
 * the `SUM / COUNT` division both yield `FLOAT64`).
 */
class mean_decomposition : public optimization_rule {
 public:
  mean_decomposition(catalog const* cat) : optimization_rule(cat) {}

  [[nodiscard]] std::shared_ptr<logical::relation> apply(std::shared_ptr<logical::relation> root,
                                                         bool& rule_applied) const override;

  [[nodiscard]] logical_optimization_rule_type type() const noexcept override
  {
    return logical_optimization_rule_type::mean_decomposition;
  }

 private:
  class apply_visitor;

  /**
   * @brief Build a `project(divide) -> aggregate(SUM, COUNT_VALID)` replacement for `agg`.
   *
   * Expands each `MEAN(x)` measure into adjacent `SUM(x)`, `COUNT_VALID(x)` measures on a freshly
   * built aggregate, then wraps it in a project that divides the two at the original `MEAN` slot
   * and passes the keys and non-`MEAN` measures through unchanged.
   *
   * @param agg Aggregate relation to decompose (must contain at least one `MEAN` measure)
   * @return The wrapping `project_relation`
   */
  [[nodiscard]] std::shared_ptr<logical::relation> decompose(
    logical::aggregate_relation* agg) const;
};

}  // namespace optimizer
}  // namespace gqe
