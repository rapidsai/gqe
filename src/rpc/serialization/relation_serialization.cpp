/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/physical/aggregate.hpp>
#include <gqe/physical/fetch.hpp>
#include <gqe/physical/filter.hpp>
#include <gqe/physical/gen_ident_col.hpp>
#include <gqe/physical/join.hpp>
#include <gqe/physical/project.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/physical/set.hpp>
#include <gqe/physical/shuffle.hpp>
#include <gqe/physical/sort.hpp>
#include <gqe/physical/window.hpp>
#include <gqe/physical/write.hpp>
#include <gqe/rpc/serialization/data_type.hpp>
#include <gqe/rpc/serialization/expression.hpp>
#include <gqe/rpc/serialization/physical_plan.hpp>

#include <stdexcept>

namespace gqe::rpc {

namespace {

class serialize_visitor : public physical::relation_visitor {
 public:
  proto::PhysicalRelation result;

  void visit(physical::read_relation* rel) override
  {
    auto* msg = result.mutable_read();
    msg->set_table_name(rel->table_name());
    for (auto const& name : rel->column_names())
      msg->add_column_names(name);
    for (auto const& dt : rel->output_data_types())
      *msg->add_data_types() = serialize_data_type(dt);
    if (rel->partial_filter_unsafe()) {
      *msg->mutable_partial_filter() = serialize_expression(rel->partial_filter_unsafe());
    }
    for (auto const* sub : rel->subqueries_unsafe()) {
      *msg->add_subquery_relations() = serialize_physical_plan(sub);
    }
  }

  void visit(physical::write_relation* rel) override
  {
    auto* msg = result.mutable_write();
    msg->set_table_name(rel->table_name());
    for (auto const& name : rel->column_names())
      msg->add_column_names(name);
    auto children         = rel->children_unsafe();
    *msg->mutable_child() = serialize_physical_plan(children[0]);
  }

  void visit(physical::broadcast_join_relation* rel) override
  {
    auto* msg             = result.mutable_broadcast_join();
    auto children         = rel->children_unsafe();
    *msg->mutable_left()  = serialize_physical_plan(children[0]);
    *msg->mutable_right() = serialize_physical_plan(children[1]);
    for (auto const* sub : rel->subqueries_unsafe())
      *msg->add_subquery_relations() = serialize_physical_plan(sub);
    msg->set_join_type(static_cast<int32_t>(rel->join_type()));
    if (rel->condition()) { *msg->mutable_condition() = serialize_expression(rel->condition()); }
    for (auto idx : rel->projection_indices())
      msg->add_projection_indices(idx);
    msg->set_broadcast_policy(static_cast<int32_t>(rel->policy()));
    msg->set_unique_keys_policy(static_cast<int32_t>(rel->unique_keys_policy()));
    msg->set_perfect_hashing(rel->perfect_hashing());
    if (rel->left_filter_condition()) {
      *msg->mutable_left_filter_condition() = serialize_expression(rel->left_filter_condition());
    }
    if (rel->right_filter_condition()) {
      *msg->mutable_right_filter_condition() = serialize_expression(rel->right_filter_condition());
    }
    msg->set_use_hash_map_cache(rel->use_hash_map_cache());
    msg->set_use_mark_join(rel->use_mark_join());
    msg->set_use_like_shift_and(rel->use_like_shift_and());
  }

  void visit(physical::shuffle_join_relation* rel) override
  {
    auto* msg             = result.mutable_shuffle_join();
    auto children         = rel->children_unsafe();
    *msg->mutable_left()  = serialize_physical_plan(children[0]);
    *msg->mutable_right() = serialize_physical_plan(children[1]);
    for (auto const* sub : rel->subqueries_unsafe())
      *msg->add_subquery_relations() = serialize_physical_plan(sub);
    msg->set_join_type(static_cast<int32_t>(rel->join_type()));
    if (rel->condition()) { *msg->mutable_condition() = serialize_expression(rel->condition()); }
    for (auto idx : rel->projection_indices())
      msg->add_projection_indices(idx);
    msg->set_unique_keys_policy(static_cast<int32_t>(rel->unique_keys_policy()));
    msg->set_perfect_hashing(rel->perfect_hashing());
    msg->set_use_like_shift_and(rel->use_like_shift_and());
  }

  void visit(physical::project_relation* rel) override
  {
    auto* msg             = result.mutable_project();
    auto children         = rel->children_unsafe();
    *msg->mutable_child() = serialize_physical_plan(children[0]);
    for (auto const* sub : rel->subqueries_unsafe())
      *msg->add_subquery_relations() = serialize_physical_plan(sub);
    for (auto const* e : rel->output_expressions_unsafe())
      *msg->add_output_expressions() = serialize_expression(e);
    msg->set_use_like_shift_and(rel->use_like_shift_and());
  }

  void visit(physical::filter_relation* rel) override
  {
    auto* msg             = result.mutable_filter();
    auto children         = rel->children_unsafe();
    *msg->mutable_child() = serialize_physical_plan(children[0]);
    for (auto const* sub : rel->subqueries_unsafe())
      *msg->add_subquery_relations() = serialize_physical_plan(sub);
    if (rel->condition_unsafe()) {
      *msg->mutable_condition() = serialize_expression(rel->condition_unsafe());
    }
    for (auto idx : rel->projection_indices())
      msg->add_projection_indices(idx);
    msg->set_use_like_shift_and(rel->use_like_shift_and());
  }

  void visit(physical::concatenate_sort_relation* rel) override
  {
    auto* msg             = result.mutable_concatenate_sort();
    auto children         = rel->children_unsafe();
    *msg->mutable_child() = serialize_physical_plan(children[0]);
    for (auto const* sub : rel->subqueries_unsafe())
      *msg->add_subquery_relations() = serialize_physical_plan(sub);
    for (auto const* e : rel->keys_unsafe())
      *msg->add_keys() = serialize_expression(e);
    for (auto order : rel->column_orders())
      msg->add_column_orders(static_cast<int32_t>(order));
    for (auto null_order : rel->null_precedences())
      msg->add_null_precedences(static_cast<int32_t>(null_order));
    msg->set_use_like_shift_and(rel->use_like_shift_and());
  }

  void visit(physical::concatenate_aggregate_relation* rel) override
  {
    auto* msg             = result.mutable_concatenate_aggregate();
    auto children         = rel->children_unsafe();
    *msg->mutable_child() = serialize_physical_plan(children[0]);
    for (auto const* sub : rel->subqueries_unsafe())
      *msg->add_subquery_relations() = serialize_physical_plan(sub);
    for (auto const* e : rel->keys_unsafe())
      *msg->add_keys() = serialize_expression(e);
    for (auto const& [kind, expr] : rel->values_unsafe()) {
      auto* val = msg->add_values();
      val->set_aggregation_kind(static_cast<int32_t>(kind));
      *val->mutable_expression() = serialize_expression(expr);
    }
    if (rel->condition_unsafe()) {
      *msg->mutable_condition() = serialize_expression(rel->condition_unsafe());
    }
    msg->set_perfect_hashing(rel->perfect_hashing());
    msg->set_use_like_shift_and(rel->use_like_shift_and());
  }

  void visit(physical::fetch_relation* rel) override
  {
    auto* msg             = result.mutable_fetch();
    auto children         = rel->children_unsafe();
    *msg->mutable_child() = serialize_physical_plan(children[0]);
    msg->set_offset(rel->offset());
    msg->set_count(rel->count());
  }

  void visit(physical::union_all_relation* rel) override
  {
    auto* msg             = result.mutable_union_all();
    auto children         = rel->children_unsafe();
    *msg->mutable_left()  = serialize_physical_plan(children[0]);
    *msg->mutable_right() = serialize_physical_plan(children[1]);
  }

  void visit(physical::window_relation* rel) override
  {
    auto* msg             = result.mutable_window();
    auto children         = rel->children_unsafe();
    *msg->mutable_child() = serialize_physical_plan(children[0]);
    for (auto const* sub : rel->subqueries_unsafe())
      *msg->add_subquery_relations() = serialize_physical_plan(sub);
    msg->set_aggregation_kind(static_cast<int32_t>(rel->aggr_func()));
    for (auto const* e : rel->ident_cols_unsafe())
      *msg->add_ident_cols() = serialize_expression(e);
    for (auto const* e : rel->arguments_unsafe())
      *msg->add_arguments() = serialize_expression(e);
    for (auto const* e : rel->partition_by_unsafe())
      *msg->add_partition_by() = serialize_expression(e);
    for (auto const* e : rel->order_by_unsafe())
      *msg->add_order_by() = serialize_expression(e);
    for (auto order : rel->order_dirs())
      msg->add_order_dirs(static_cast<int32_t>(order));

    auto serialize_bound = [](window_frame_bound::type const& bound) {
      proto::WindowFrameBound pb;
      if (std::holds_alternative<window_frame_bound::unbounded>(bound)) {
        pb.set_unbounded(true);
      } else {
        pb.set_unbounded(false);
        pb.set_offset(std::get<window_frame_bound::bounded>(bound).get_bound());
      }
      return pb;
    };
    *msg->mutable_lower_bound() = serialize_bound(rel->window_lower_bound());
    *msg->mutable_upper_bound() = serialize_bound(rel->window_upper_bound());
    msg->set_use_like_shift_and(rel->use_like_shift_and());
  }

  void visit(physical::shuffle_relation* rel) override
  {
    auto* msg             = result.mutable_shuffle();
    auto children         = rel->children_unsafe();
    *msg->mutable_child() = serialize_physical_plan(children[0]);
    for (auto const* sub : rel->subqueries_unsafe())
      *msg->add_subquery_relations() = serialize_physical_plan(sub);
    for (auto const* e : rel->shuffle_cols_unsafe())
      *msg->add_shuffle_cols() = serialize_expression(e);
  }

  void visit(physical::gen_ident_col_relation* rel) override
  {
    auto* msg             = result.mutable_gen_ident_col();
    auto children         = rel->children_unsafe();
    *msg->mutable_child() = serialize_physical_plan(children[0]);
  }

  void visit(physical::user_defined_relation*) override
  {
    throw std::logic_error("user_defined_relation cannot be serialized");
  }
};

}  // namespace

proto::PhysicalRelation serialize_physical_plan(physical::relation const* plan)
{
  serialize_visitor visitor;
  const_cast<physical::relation*>(plan)->accept(visitor);
  return visitor.result;
}

}  // namespace gqe::rpc
