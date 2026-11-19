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

#include <gqe/optimizer/rules/mean_decomposition.hpp>

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/project.hpp>

#include <cudf/aggregation.hpp>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace gqe::optimizer {

namespace {

/// @brief `true` if `agg` has at least one `MEAN` measure.
[[nodiscard]] bool has_mean(gqe::logical::aggregate_relation const* agg)
{
  for (auto const& [kind, arg] : agg->measures_unsafe()) {
    if (kind == cudf::aggregation::MEAN) return true;
  }
  return false;
}

}  // namespace

class mean_decomposition::apply_visitor : public gqe::logical::relation_visitor {
 public:
  apply_visitor(mean_decomposition const& rule, bool& rule_applied)
    : _rule{rule}, _rule_applied{rule_applied}
  {
  }

  void visit_relation(gqe::logical::relation* rel) override
  {
    visit_children(rel);  // post-order: recurse into children first

    // Wrap any aggregate child that still computes a MEAN. The root is handled by `apply`.
    auto children = rel->children_safe();
    for (std::size_t i = 0; i < children.size(); ++i) {
      auto& child = children[i];
      if (child->type() != gqe::logical::relation::relation_type::aggregate) continue;
      auto* agg = static_cast<gqe::logical::aggregate_relation*>(child.get());
      if (!has_mean(agg)) continue;
      mean_decomposition::replace_child_at(rel, i, _rule.decompose(agg));
      _rule_applied = true;
    }
  }

 private:
  mean_decomposition const& _rule;
  bool& _rule_applied;
};

std::shared_ptr<gqe::logical::relation> mean_decomposition::decompose(
  gqe::logical::aggregate_relation* agg) const
{
  auto const keys     = agg->keys_unsafe();      // std::vector<expression*>
  auto const measures = agg->measures_unsafe();  // std::vector<(Kind, expression*)>
  auto const num_keys = static_cast<cudf::size_type>(keys.size());

  // Expand each `MEAN(x)` into back-to-back `SUM(x)`, `COUNT_VALID(x)` measures.
  // `new_measure_start` records, for each original measure, the index of the first measure it
  // produces in the new list.
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> new_measures;
  std::vector<cudf::size_type> new_measure_start;
  new_measure_start.reserve(measures.size());
  for (auto const& [kind, arg] : measures) {
    new_measure_start.push_back(static_cast<cudf::size_type>(new_measures.size()));
    if (kind == cudf::aggregation::MEAN) {
      new_measures.emplace_back(cudf::aggregation::SUM, arg->clone());
      new_measures.emplace_back(cudf::aggregation::COUNT_VALID, arg->clone());
    } else {
      new_measures.emplace_back(kind, arg->clone());
    }
  }

  std::vector<std::unique_ptr<gqe::expression>> new_keys;
  new_keys.reserve(keys.size());
  for (auto const* key : keys) {
    new_keys.push_back(key->clone());
  }

  // The rebuilt aggregate keeps the original input and subqueries (the cloned keys/measures still
  // reference the subqueries by index); the project below only references the aggregate's output.
  auto new_aggregate = std::make_shared<gqe::logical::aggregate_relation>(
    agg->children_safe()[0], agg->subqueries_safe(), std::move(new_keys), std::move(new_measures));

  // Build the downstream project reproducing the aggregate's original `[keys..., measures...]`
  // output schema: keys and non-MEAN measures pass through as column references; each MEAN slot
  // becomes `column_reference(sum) / column_reference(count)`.
  std::vector<std::unique_ptr<gqe::expression>> output_exprs;
  output_exprs.reserve(static_cast<std::size_t>(num_keys) + measures.size());
  for (cudf::size_type i = 0; i < num_keys; ++i) {
    output_exprs.push_back(std::make_unique<gqe::column_reference_expression>(i));
  }
  for (std::size_t j = 0; j < measures.size(); ++j) {
    auto const slot = num_keys + new_measure_start[j];
    if (measures[j].first == cudf::aggregation::MEAN) {
      auto sum_ref   = std::make_shared<gqe::column_reference_expression>(slot);
      auto count_ref = std::make_shared<gqe::column_reference_expression>(slot + 1);
      output_exprs.push_back(
        std::make_unique<gqe::divide_expression>(std::move(sum_ref), std::move(count_ref)));
    } else {
      output_exprs.push_back(std::make_unique<gqe::column_reference_expression>(slot));
    }
  }

  return std::make_shared<gqe::logical::project_relation>(
    std::move(new_aggregate),
    std::vector<std::shared_ptr<gqe::logical::relation>>{},
    std::move(output_exprs));
}

std::shared_ptr<gqe::logical::relation> mean_decomposition::apply(
  std::shared_ptr<logical::relation> root, bool& rule_applied) const
{
  apply_visitor visitor{*this, rule_applied};
  root->accept(visitor);

  // Handle the root separately — the visitor only inspects children, not the root itself.
  if (root->type() == logical::relation::relation_type::aggregate) {
    auto* agg = static_cast<logical::aggregate_relation*>(root.get());
    if (has_mean(agg)) {
      rule_applied = true;
      return decompose(agg);
    }
  }
  return root;
}

}  // namespace gqe::optimizer
