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
#include <gqe/physical/aggregate.hpp>
#include <gqe/physical/fetch.hpp>
#include <gqe/physical/filter.hpp>
#include <gqe/physical/join.hpp>
#include <gqe/physical/project.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/physical/sort.hpp>

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
  std::shared_ptr<physical::relation> out_physical_relation;

  switch (logical_relation->type()) {
    case logical::relation::relation_type::read: {
      auto const logical_read_relation =
        dynamic_cast<logical::read_relation const*>(logical_relation);
      out_physical_relation = std::make_shared<physical::read_relation>(
        logical_read_relation->table_name(), logical_read_relation->column_names());
      break;
    }
    case logical::relation::relation_type::join: {
      assert(children_physical.size() == 2);
      auto const logical_join_relation =
        dynamic_cast<logical::join_relation const*>(logical_relation);
      out_physical_relation = std::make_shared<physical::broadcast_join_relation>(
        std::move(children_physical[0]),
        std::move(children_physical[1]),
        logical_join_relation->join_type(),
        logical_join_relation->condition()->clone(),
        logical_join_relation->projection_indices());
      break;
    }
    case logical::relation::relation_type::project: {
      assert(children_physical.size() == 1);
      auto const logical_project_relation =
        dynamic_cast<logical::project_relation const*>(logical_relation);

      std::vector<std::unique_ptr<expression>> exprs;
      for (auto const& expr : logical_project_relation->output_expressions_unsafe())
        exprs.push_back(expr->clone());

      out_physical_relation = std::make_shared<physical::project_relation>(
        std::move(children_physical[0]), std::move(exprs));
      break;
    }
    case logical::relation::relation_type::fetch: {
      assert(children_physical.size() == 1);
      auto const logical_fetch_relation =
        dynamic_cast<logical::fetch_relation const*>(logical_relation);

      out_physical_relation =
        std::make_shared<physical::fetch_relation>(std::move(children_physical[0]),
                                                   logical_fetch_relation->offset(),
                                                   logical_fetch_relation->count());
      break;
    }
    case logical::relation::relation_type::filter: {
      assert(children_physical.size() == 1);
      auto const logical_filter_relation =
        dynamic_cast<logical::filter_relation const*>(logical_relation);

      out_physical_relation = std::make_shared<physical::filter_relation>(
        std::move(children_physical[0]), logical_filter_relation->condition()->clone());
      break;
    }
    case logical::relation::relation_type::sort: {
      assert(children_physical.size() == 1);
      auto const logical_sort_relation =
        dynamic_cast<logical::sort_relation const*>(logical_relation);

      auto const in_keys = logical_sort_relation->expressions_unsafe();
      std::vector<std::unique_ptr<expression>> keys;
      keys.reserve(in_keys.size());
      for (auto const& key : in_keys)
        keys.push_back(key->clone());

      out_physical_relation = std::make_shared<physical::concatenate_sort_relation>(
        std::move(children_physical[0]),
        std::move(keys),
        logical_sort_relation->column_orders(),
        logical_sort_relation->null_orders());
      break;
    }
    case logical::relation::relation_type::aggregate: {
      assert(children_physical.size() == 1);
      auto const logical_aggregate_relation =
        dynamic_cast<logical::aggregate_relation const*>(logical_relation);

      auto const in_keys = logical_aggregate_relation->keys_unsafe();
      std::vector<std::unique_ptr<expression>> keys;
      keys.reserve(in_keys.size());
      for (auto const& key : in_keys)
        keys.push_back(key->clone());

      auto const in_values = logical_aggregate_relation->measures_unsafe();
      std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<expression>>> values;
      values.reserve(in_values.size());
      for (auto const& [kind, expr] : in_values)
        values.emplace_back(kind, expr->clone());

      out_physical_relation = std::make_shared<physical::concatenate_aggregate_relation>(
        std::move(children_physical[0]), std::move(keys), std::move(values));
      break;
    }
    default:
      throw std::logic_error("Cannot convert the logical relation to physical: " +
                             logical_relation->to_string());
  }

  _cache[logical_relation] = out_physical_relation;

  return out_physical_relation;
}

}  // namespace gqe
