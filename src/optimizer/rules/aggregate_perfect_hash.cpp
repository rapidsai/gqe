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

#include <gqe/optimizer/rules/aggregate_perfect_hash.hpp>

#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/logical/aggregate.hpp>
#include <gqe/optimizer/relation_properties.hpp>

#include <cudf/types.hpp>

#include <algorithm>
#include <cassert>
#include <memory>
#include <unordered_set>
#include <vector>

namespace gqe::optimizer {

class aggregate_perfect_hash::apply_visitor : public gqe::logical::relation_visitor {
 public:
  apply_visitor(bool& rule_applied) : _rule_applied{rule_applied} {}

  void visit(gqe::logical::aggregate_relation* aggregate) override
  {
    visit_children(aggregate);  // post-order: children before parent

    auto const key_exprs = aggregate->keys_unsafe();

    bool is_perfect_hashable = false;

    // Pure reductions (no group-by keys) always produce a single output row; perfect hashing
    // via libperfect::unique_indices is only applicable to grouped aggregations.
    if (!key_exprs.empty()) {
      auto const children = aggregate->children_unsafe();
      assert(children.size() == 1);
      auto const& input = children[0];

      // Collect the column indices of group-by keys that are direct column references.
      // Non-column-reference keys (e.g. `GROUP BY a + b`) are not tracked in the input's
      // unique-key sets, so they simply do not contribute toward covering a unique key. Collecting
      // a subset of the group-by keys this way is sufficient to decide uniqueness; see below.
      std::unordered_set<cudf::size_type> group_key_cols;
      for (auto const* expr : key_exprs) {
        if (expr->type() == gqe::expression::expression_type::column_reference) {
          // TODO: Not sure if this check is sufficient, tracked by issue #385.
          group_key_cols.insert(
            static_cast<gqe::column_reference_expression const*>(expr)->column_idx());
        }
      }

      // Perfect hashing is safe when the group-by columns cover at least one unique key-set of
      // the input relation.
      //
      // Why covering a single unique key-set is sufficient:
      //   A superset of a unique key is itself unique. If {a} is unique in the input, then
      //   GROUP BY a, b, c cannot produce duplicate (a, b, c) tuples either — rows already
      //   differ on `a`. Additional grouping columns can only split groups, never merge them.
      //
      // Note: `unique_indices` in libperfect does not require a bounded value range — if the
      // key range is too large for direct-address hashing, HashTable falls back to regular
      // open-addressing transparently. The logical flag is therefore a pure performance gate,
      // not a correctness requirement.
      auto const keys_unique =
        input->relation_traits().properties().covers_unique_key(group_key_cols);

      // All group-by key types must be fixed-width for libperfect::unique_indices. Every key
      // (including non-column-reference expressions) is materialized as a hash-table key, so check
      // all of them. Evaluate each key expression's type against the input schema rather than
      // relying on the output column ordering of `data_types()`.
      if (keys_unique) {
        auto const input_types = input->data_types();
        is_perfect_hashable =
          std::all_of(key_exprs.begin(), key_exprs.end(), [&](auto const* expr) {
            return cudf::is_fixed_width(expr->data_type(input_types));
          });
      }
    }

    if (is_perfect_hashable != aggregate->is_perfect_hashable()) {
      aggregate->set_perfect_hashable(is_perfect_hashable);
      _rule_applied = true;
    }
  }

 private:
  bool& _rule_applied;
};

std::shared_ptr<gqe::logical::relation> aggregate_perfect_hash::apply(
  std::shared_ptr<logical::relation> root, bool& rule_applied) const
{
  apply_visitor visitor{rule_applied};
  root->accept(visitor);
  return root;
}

}  // namespace gqe::optimizer
