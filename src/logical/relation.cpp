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

#include <gqe/logical/relation.hpp>

namespace gqe {
namespace logical {

relation::relation(std::vector<std::shared_ptr<relation>> children,
                   std::vector<std::shared_ptr<relation>> subqueries)
  : _children(std::move(children)), _subqueries(std::move(subqueries))
{
  _relation_traits = std::make_unique<optimizer::relation_traits>();
}

[[nodiscard]] cudf::size_type relation::num_columns() const { return data_types().size(); }

[[nodiscard]] std::vector<std::shared_ptr<relation>> relation::children_safe() const noexcept
{
  std::vector<std::shared_ptr<relation>> children_to_return;
  children_to_return.reserve(_children.size());

  for (auto& child : _children) {
    children_to_return.push_back(child);
  }

  return children_to_return;
}

[[nodiscard]] std::vector<relation*> relation::children_unsafe() const noexcept
{
  return gqe::utility::to_raw_ptrs(_children);
}

[[nodiscard]] std::size_t relation::children_size() const noexcept { return _children.size(); }

[[nodiscard]] std::vector<relation*> relation::subqueries_unsafe() const noexcept
{
  return gqe::utility::to_raw_ptrs(_subqueries);
}

[[nodiscard]] std::size_t relation::subqueries_size() const noexcept { return _subqueries.size(); }

[[nodiscard]] optimizer::relation_traits const& relation::relation_traits() const noexcept
{
  return *_relation_traits;
}

void relation::set_relation_traits(std::unique_ptr<optimizer::relation_traits> traits)
{
  _relation_traits = std::move(traits);
}

[[nodiscard]] bool relation::compare_relation_members(const relation& other) const
{
  // Compare children
  if (!gqe::utility::compare_pointer_vectors(children_unsafe(), other.children_unsafe())) {
    return false;
  }
  // Compare subquery_relations
  if (!gqe::utility::compare_pointer_vectors(subqueries_unsafe(), other.subqueries_unsafe())) {
    return false;
  }
  // Compare relation traits
  return relation_traits() == other.relation_traits();
}

}  // namespace logical
}  // namespace gqe
