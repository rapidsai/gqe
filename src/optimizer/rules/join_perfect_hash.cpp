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

#include <gqe/optimizer/rules/join_perfect_hash.hpp>

#include "../utility.hpp"

#include <gqe/logical/join.hpp>
#include <gqe/types.hpp>

#include <cudf/types.hpp>

#include <memory>

namespace gqe::optimizer {

class join_perfect_hash::apply_visitor : public gqe::logical::relation_visitor {
 public:
  apply_visitor(bool& rule_applied) : _rule_applied{rule_applied} {}

  void visit(gqe::logical::join_relation* join) override
  {
    visit_children(join);  // post-order: recurse first

    // Only apply rule for inner join.
    if (join->join_type() != join_type_type::inner) return;

    auto children = join->children_unsafe();
    // Only apply rule for equijoin predicate.
    auto const keys =
      utility::try_extract_equijoin_keys(join->condition(), children[0]->num_columns());
    if (!keys.has_value() || keys->null_equality != cudf::null_equality::UNEQUAL) { return; }

    bool const left_key_unique =
      utility::keys_cover_unique_key(children[0]->relation_traits().properties(), keys->left);
    bool const right_key_unique =
      utility::keys_cover_unique_key(children[1]->relation_traits().properties(), keys->right);

    gqe::perfect_hash_policy new_policy = gqe::perfect_hash_policy::none;
    if (left_key_unique && right_key_unique) {
      new_policy = gqe::perfect_hash_policy::either;
    } else if (left_key_unique) {
      new_policy = gqe::perfect_hash_policy::left;
    } else if (right_key_unique) {
      new_policy = gqe::perfect_hash_policy::right;
    }

    if (new_policy != gqe::perfect_hash_policy::none &&
        join->perfect_hash_policy() == gqe::perfect_hash_policy::none) {
      _rule_applied = true;
    }
    join->set_perfect_hash_policy(new_policy);
  }

 private:
  bool& _rule_applied;
};

std::shared_ptr<gqe::logical::relation> join_perfect_hash::apply(
  std::shared_ptr<logical::relation> root, bool& rule_applied) const
{
  apply_visitor visitor{rule_applied};
  root->accept(visitor);
  return root;
}

}  // namespace gqe::optimizer
