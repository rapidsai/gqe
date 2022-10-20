/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/physical/join.hpp>
#include <gqe/physical/project.hpp>
#include <gqe/physical/read.hpp>

namespace gqe {

std::shared_ptr<physical::relation> physical_plan_builder::build(
  logical::relation const* logical_relation)
{
  // Search the cache to see whether the physical relation has already been generated
  auto cache_iter = _cache.find(logical_relation);
  if (cache_iter != _cache.end()) return std::shared_ptr<physical::relation>(cache_iter->second);

  // Recursively transform the children logical relations into physical relations
  auto const children_logical = logical_relation->children_unsafe();

  std::vector<std::shared_ptr<physical::relation>> children_physical;
  children_physical.reserve(children_logical.size());

  for (auto const child_logical : children_logical)
    children_physical.push_back(build(child_logical));

  // Transform the current logical relation depending on the relation type
  switch (logical_relation->type()) {
    case logical::relation::relation_type::read: {
      auto const logical_read_relation =
        dynamic_cast<logical::read_relation const*>(logical_relation);
      return std::make_shared<physical::read_relation>(logical_read_relation->table_name(),
                                                       logical_read_relation->column_names());
    }
    case logical::relation::relation_type::join: {
      assert(children_physical.size() == 2);
      auto const logical_join_relation =
        dynamic_cast<logical::join_relation const*>(logical_relation);
      return std::make_shared<physical::broadcast_join_relation>(
        std::move(children_physical[0]),
        std::move(children_physical[1]),
        logical_join_relation->join_type(),
        logical_join_relation->condition()->clone(),
        logical_join_relation->projection_indices());
    }
    case logical::relation::relation_type::project: {
      assert(children_physical.size() == 1);
      auto const logical_project_relation =
        dynamic_cast<logical::project_relation const*>(logical_relation);

      std::vector<std::unique_ptr<expression>> exprs;
      for (auto const& expr : logical_project_relation->output_expressions_unsafe())
        exprs.push_back(expr->clone());

      return std::make_shared<physical::project_relation>(std::move(children_physical[0]),
                                                          std::move(exprs));
    }
    default:
      throw std::logic_error("Cannot convert the logical relation to physical: " +
                             logical_relation->to_string());
  }
}

}  // namespace gqe
