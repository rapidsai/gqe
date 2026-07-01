/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/optimizer/physical_transformation.hpp>

#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/fetch.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/set.hpp>
#include <gqe/logical/sort.hpp>
#include <gqe/logical/user_defined.hpp>
#include <gqe/logical/window.hpp>
#include <gqe/logical/write.hpp>
#include <gqe/physical/aggregate.hpp>
#include <gqe/physical/fetch.hpp>
#include <gqe/physical/filter.hpp>
#include <gqe/physical/project.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/physical/set.hpp>
#include <gqe/physical/sort.hpp>
#include <gqe/physical/user_defined.hpp>
#include <gqe/physical/write.hpp>

namespace gqe {

std::shared_ptr<physical::relation> physical_plan_builder::build(
  logical::relation const* logical_relation)
{
  // Search the cache to see whether the physical relation has already been generated
  auto cache_iter = _cache.find(logical_relation);
  if (cache_iter != _cache.end()) return std::shared_ptr<physical::relation>(cache_iter->second);

  // Recursively transform the children logical relations into physical relations
  auto const children_logical   = logical_relation->children_unsafe();
  auto const subqueries_logical = logical_relation->subqueries_unsafe();

  std::vector<std::shared_ptr<physical::relation>> children_physical;
  std::vector<std::shared_ptr<physical::relation>> subqueries_physical;
  children_physical.reserve(children_logical.size());
  subqueries_physical.reserve(subqueries_logical.size());

  for (auto const child_logical : children_logical)
    children_physical.push_back(build(child_logical));

  for (auto const subquery_logical : subqueries_logical)
    subqueries_physical.push_back(build(subquery_logical));

  // Transform the current logical relation depending on the relation type
  std::shared_ptr<physical::relation> out_physical_relation;

  switch (logical_relation->type()) {
    case logical::relation::relation_type::read: {
      auto const logical_read_relation =
        dynamic_cast<logical::read_relation const*>(logical_relation);
      auto const partial_filter_ptr = logical_read_relation->partial_filter_unsafe();

      {
        auto const& tname     = logical_read_relation->table_name();
        out_physical_relation = std::make_shared<physical::read_relation>(
          std::move(subqueries_physical),
          logical_read_relation->column_names(),
          tname,
          partial_filter_ptr ? partial_filter_ptr->clone() : nullptr,
          logical_read_relation->data_types());
      }
      break;
    }
    case logical::relation::relation_type::write: {
      assert(children_physical.size() == 1);
      auto& input_physical = children_physical[0];

      auto const logical_insert_relation =
        dynamic_cast<logical::write_relation const*>(logical_relation);

      {
        auto const& tname     = logical_insert_relation->table_name();
        out_physical_relation = std::make_shared<physical::write_relation>(
          std::move(input_physical), logical_insert_relation->column_names(), tname);
        break;
      }
    }
    case logical::relation::relation_type::join: {
      out_physical_relation =
        plan_join(static_cast<logical::join_relation const&>(*logical_relation),
                  children_logical,
                  std::move(children_physical),
                  std::move(subqueries_physical));
      break;
    }
    case logical::relation::relation_type::project: {
      assert(children_physical.size() == 1);
      auto const logical_project_relation =
        dynamic_cast<logical::project_relation const*>(logical_relation);

      std::vector<std::unique_ptr<expression>> exprs;
      for (auto const& expr : logical_project_relation->output_expressions_unsafe())
        exprs.push_back(expr->clone());

      bool const use_like_shift_and = _params && _params->filter_use_like_shift_and;
      out_physical_relation =
        std::make_shared<physical::project_relation>(std::move(children_physical[0]),
                                                     std::move(subqueries_physical),
                                                     std::move(exprs),
                                                     use_like_shift_and);
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

      bool const use_like_shift_and = _params && _params->filter_use_like_shift_and;

      out_physical_relation =
        std::make_shared<physical::filter_relation>(std::move(children_physical[0]),
                                                    std::move(subqueries_physical),
                                                    logical_filter_relation->condition()->clone(),
                                                    logical_filter_relation->projection_indices(),
                                                    use_like_shift_and);
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

      bool const use_like_shift_and = _params && _params->filter_use_like_shift_and;

      out_physical_relation = std::make_shared<physical::concatenate_sort_relation>(
        std::move(children_physical[0]),
        std::move(subqueries_physical),
        std::move(keys),
        logical_sort_relation->column_orders(),
        logical_sort_relation->null_orders(),
        use_like_shift_and);
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

      bool const use_like_shift_and  = _params && _params->filter_use_like_shift_and;
      bool const use_perfect_hashing = _params && _params->aggregation_use_perfect_hash &&
                                       logical_aggregate_relation->is_perfect_hashable();

      out_physical_relation =
        std::make_shared<physical::concatenate_aggregate_relation>(std::move(children_physical[0]),
                                                                   std::move(subqueries_physical),
                                                                   std::move(keys),
                                                                   std::move(values),
                                                                   nullptr,
                                                                   use_perfect_hashing,
                                                                   use_like_shift_and);
      break;
    }
    case logical::relation::relation_type::window: {
      out_physical_relation =
        plan_window(static_cast<logical::window_relation const&>(*logical_relation),
                    children_logical,
                    std::move(children_physical),
                    std::move(subqueries_physical));
      break;
    }
    case logical::relation::relation_type::set: {
      assert(children_physical.size() == 2);
      auto const logical_set_relation =
        dynamic_cast<logical::set_relation const*>(logical_relation);

      switch (logical_set_relation->set_operator()) {
        case logical::set_relation::set_union_all: {
          out_physical_relation = std::make_shared<physical::union_all_relation>(
            std::move(children_physical[0]), std::move(children_physical[1]));
          break;
        }
        default: throw std::logic_error("Unsupported set operator");
      }
      break;
    }
    case logical::relation::relation_type::user_defined: {
      if (subqueries_physical.size())
        throw std::logic_error("Subqueries are not supported in user-defined relations");

      auto const logical_user_defined_relation =
        dynamic_cast<logical::user_defined_relation const*>(logical_relation);

      out_physical_relation = std::make_shared<physical::user_defined_relation>(
        std::move(children_physical),
        logical_user_defined_relation->task_functor(),
        logical_user_defined_relation->last_child_break_pipeline());
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
