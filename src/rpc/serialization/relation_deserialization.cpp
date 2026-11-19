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
#include <gqe/physical/bounded_hash_aggregate.hpp>
#include <gqe/physical/fetch.hpp>
#include <gqe/physical/filter.hpp>
#include <gqe/physical/gen_ident_col.hpp>
#include <gqe/physical/join.hpp>
#include <gqe/physical/project.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/physical/set.hpp>
#include <gqe/physical/shuffle.hpp>
#include <gqe/physical/sort.hpp>
#include <gqe/physical/top_k.hpp>
#include <gqe/physical/window.hpp>
#include <gqe/physical/write.hpp>
#include <gqe/rpc/serialization/data_type.hpp>
#include <gqe/rpc/serialization/expression.hpp>
#include <gqe/rpc/serialization/physical_plan.hpp>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace gqe::rpc {

namespace {

std::vector<std::unique_ptr<expression>> deserialize_expressions(
  google::protobuf::RepeatedPtrField<proto::Expression> const& exprs)
{
  std::vector<std::unique_ptr<expression>> out;
  out.reserve(exprs.size());
  for (auto const& e : exprs)
    out.push_back(deserialize_expression(e));
  return out;
}

window_frame_bound::type deserialize_bound(proto::WindowFrameBound const& pb)
{
  if (pb.unbounded()) return window_frame_bound::unbounded{};
  return window_frame_bound::bounded{pb.offset()};
}

/**
 * @brief Rebuilds one plan, resolving `ReferenceRel`s against the relations
 * already built for it.
 *
 * Holds the ids for a single plan, so one instance cannot be reused across
 * plans.
 */
class plan_deserializer {
 public:
  physical::identified_plan deserialize(proto::PhysicalRelation const& pb)
  {
    auto root = deserialize_relation(pb);
    physical::plan_node_ids node_ids;
    node_ids.reserve(_nodes.size());
    for (auto const& [node_id, relation] : _nodes) {
      node_ids.emplace(relation.get(), node_id);
    }
    return physical::identified_plan{std::move(root), std::move(node_ids)};
  }

 private:
  std::shared_ptr<physical::relation> deserialize_relation(proto::PhysicalRelation const& pb)
  {
    if (pb.relation_case() == proto::PhysicalRelation::kReferenceRel) {
      auto const it = _nodes.find(pb.reference_rel().node_id());
      if (it == _nodes.end()) {
        throw std::logic_error("ReferenceRel refers to an unknown node_id: " +
                               std::to_string(pb.reference_rel().node_id()));
      }
      return it->second;
    }

    if (pb.node_id() <= 0) {
      throw std::logic_error("PhysicalRelation node_id must be positive: " +
                             std::to_string(pb.node_id()));
    }

    auto relation        = deserialize_relation_contents(pb);
    auto const insertion = _nodes.emplace(pb.node_id(), relation);
    if (!insertion.second) {
      throw std::logic_error("PhysicalRelation has duplicate node_id: " +
                             std::to_string(pb.node_id()));
    }
    return relation;
  }

  std::vector<std::shared_ptr<physical::relation>> _deserialize_subqueries(
    google::protobuf::RepeatedPtrField<proto::PhysicalRelation> const& subs)
  {
    std::vector<std::shared_ptr<physical::relation>> out;
    out.reserve(subs.size());
    for (auto const& sub : subs)
      out.push_back(deserialize_relation(sub));
    return out;
  }

  std::shared_ptr<physical::relation> deserialize_relation_contents(
    proto::PhysicalRelation const& pb)
  {
    switch (pb.relation_case()) {
      case proto::PhysicalRelation::kRead: {
        auto const& msg = pb.read();

        std::vector<std::string> column_names(msg.column_names().begin(), msg.column_names().end());
        std::vector<cudf::data_type> data_types;
        for (auto const& dt : msg.data_types())
          data_types.push_back(deserialize_data_type(dt));

        std::unique_ptr<expression> partial_filter;
        if (msg.has_partial_filter()) {
          partial_filter = deserialize_expression(msg.partial_filter());
        }

        auto subqueries = _deserialize_subqueries(msg.subquery_relations());

        return std::make_shared<physical::read_relation>(std::move(subqueries),
                                                         std::move(column_names),
                                                         msg.table_name(),
                                                         std::move(partial_filter),
                                                         std::move(data_types));
      }
      case proto::PhysicalRelation::kWrite: {
        auto const& msg = pb.write();
        auto child      = deserialize_relation(msg.child());

        std::vector<std::string> column_names(msg.column_names().begin(), msg.column_names().end());

        return std::make_shared<physical::write_relation>(
          std::move(child), std::move(column_names), msg.table_name());
      }
      case proto::PhysicalRelation::kBroadcastJoin: {
        auto const& msg = pb.broadcast_join();
        auto left       = deserialize_relation(msg.left());
        auto right      = deserialize_relation(msg.right());
        auto subqueries = _deserialize_subqueries(msg.subquery_relations());

        std::unique_ptr<expression> condition;
        if (msg.has_condition()) condition = deserialize_expression(msg.condition());

        std::vector<cudf::size_type> proj_indices(msg.projection_indices().begin(),
                                                  msg.projection_indices().end());

        std::unique_ptr<expression> left_filter;
        if (msg.has_left_filter_condition())
          left_filter = deserialize_expression(msg.left_filter_condition());

        std::unique_ptr<expression> right_filter;
        if (msg.has_right_filter_condition())
          right_filter = deserialize_expression(msg.right_filter_condition());

        return std::make_shared<physical::broadcast_join_relation>(
          std::move(left),
          std::move(right),
          std::move(subqueries),
          static_cast<join_type_type>(msg.join_type()),
          std::move(condition),
          std::move(proj_indices),
          static_cast<physical::broadcast_policy>(msg.broadcast_policy()),
          static_cast<unique_keys_policy>(msg.unique_keys_policy()),
          msg.perfect_hashing(),
          std::move(left_filter),
          std::move(right_filter),
          msg.use_hash_map_cache(),
          msg.use_mark_join(),
          msg.use_like_shift_and());
      }
      case proto::PhysicalRelation::kShuffleJoin: {
        auto const& msg = pb.shuffle_join();
        auto left       = deserialize_relation(msg.left());
        auto right      = deserialize_relation(msg.right());
        auto subqueries = _deserialize_subqueries(msg.subquery_relations());

        std::unique_ptr<expression> condition;
        if (msg.has_condition()) condition = deserialize_expression(msg.condition());

        std::vector<cudf::size_type> proj_indices(msg.projection_indices().begin(),
                                                  msg.projection_indices().end());

        return std::make_shared<physical::shuffle_join_relation>(
          std::move(left),
          std::move(right),
          std::move(subqueries),
          static_cast<join_type_type>(msg.join_type()),
          std::move(condition),
          std::move(proj_indices),
          static_cast<unique_keys_policy>(msg.unique_keys_policy()),
          msg.perfect_hashing(),
          msg.use_like_shift_and());
      }
      case proto::PhysicalRelation::kProject: {
        auto const& msg = pb.project();
        auto child      = deserialize_relation(msg.child());
        auto subqueries = _deserialize_subqueries(msg.subquery_relations());
        auto exprs      = deserialize_expressions(msg.output_expressions());

        return std::make_shared<physical::project_relation>(
          std::move(child), std::move(subqueries), std::move(exprs), msg.use_like_shift_and());
      }
      case proto::PhysicalRelation::kFilter: {
        auto const& msg = pb.filter();
        auto child      = deserialize_relation(msg.child());
        auto subqueries = _deserialize_subqueries(msg.subquery_relations());

        std::unique_ptr<expression> condition;
        if (msg.has_condition()) condition = deserialize_expression(msg.condition());

        std::vector<cudf::size_type> proj_indices(msg.projection_indices().begin(),
                                                  msg.projection_indices().end());

        return std::make_shared<physical::filter_relation>(std::move(child),
                                                           std::move(subqueries),
                                                           std::move(condition),
                                                           std::move(proj_indices),
                                                           msg.use_like_shift_and());
      }
      case proto::PhysicalRelation::kConcatenateSort: {
        auto const& msg = pb.concatenate_sort();
        auto child      = deserialize_relation(msg.child());
        auto subqueries = _deserialize_subqueries(msg.subquery_relations());
        auto keys       = deserialize_expressions(msg.keys());

        std::vector<cudf::order> orders;
        for (auto o : msg.column_orders())
          orders.push_back(static_cast<cudf::order>(o));

        std::vector<cudf::null_order> null_orders;
        for (auto n : msg.null_precedences())
          null_orders.push_back(static_cast<cudf::null_order>(n));

        return std::make_shared<physical::concatenate_sort_relation>(std::move(child),
                                                                     std::move(subqueries),
                                                                     std::move(keys),
                                                                     std::move(orders),
                                                                     std::move(null_orders),
                                                                     msg.use_like_shift_and());
      }
      case proto::PhysicalRelation::kTopK: {
        auto const& msg = pb.top_k();
        auto child      = deserialize_relation(msg.child());

        return std::make_shared<physical::top_k_relation>(
          std::move(child),
          msg.key_index(),
          static_cast<cudf::order>(msg.column_order()),
          static_cast<cudf::null_order>(msg.null_precedence()),
          msg.offset(),
          msg.count());
      }
      case proto::PhysicalRelation::kConcatenateAggregate: {
        auto const& msg = pb.concatenate_aggregate();
        auto child      = deserialize_relation(msg.child());
        auto subqueries = _deserialize_subqueries(msg.subquery_relations());
        auto keys       = deserialize_expressions(msg.keys());

        std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<expression>>> values;
        for (auto const& v : msg.values()) {
          values.emplace_back(static_cast<cudf::aggregation::Kind>(v.aggregation_kind()),
                              deserialize_expression(v.expression()));
        }

        std::unique_ptr<expression> condition;
        if (msg.has_condition()) condition = deserialize_expression(msg.condition());

        return std::make_shared<physical::concatenate_aggregate_relation>(std::move(child),
                                                                          std::move(subqueries),
                                                                          std::move(keys),
                                                                          std::move(values),
                                                                          std::move(condition),
                                                                          msg.perfect_hashing(),
                                                                          msg.use_like_shift_and());
      }
      case proto::PhysicalRelation::kBoundedHashAggregate: {
        auto const& msg = pb.bounded_hash_aggregate();
        auto child      = deserialize_relation(msg.child());

        std::vector<cudf::size_type> key_indices(msg.key_indices().begin(),
                                                 msg.key_indices().end());

        std::vector<physical::bounded_hash_aggregate_relation::value_type> values;
        values.reserve(msg.values_size());
        for (auto const& value : msg.values()) {
          values.emplace_back(
            static_cast<cudf::aggregation::Kind>(value.aggregation_kind()),
            value.has_value_index() ? std::optional<cudf::size_type>{value.value_index()}
                                    : std::nullopt,
            value.value_not_null() ? nullability::not_null : nullability::nullable);
        }

        auto const desired_load_factor = msg.has_desired_load_factor()
                                           ? std::optional<double>{msg.desired_load_factor()}
                                           : std::nullopt;

        return std::make_shared<physical::bounded_hash_aggregate_relation>(std::move(child),
                                                                           std::move(key_indices),
                                                                           std::move(values),
                                                                           msg.group_count_bound(),
                                                                           desired_load_factor);
      }
      case proto::PhysicalRelation::kFetch: {
        auto const& msg = pb.fetch();
        auto child      = deserialize_relation(msg.child());
        return std::make_shared<physical::fetch_relation>(
          std::move(child), msg.offset(), msg.count());
      }
      case proto::PhysicalRelation::kUnionAll: {
        auto const& msg = pb.union_all();
        auto left       = deserialize_relation(msg.left());
        auto right      = deserialize_relation(msg.right());
        return std::make_shared<physical::union_all_relation>(std::move(left), std::move(right));
      }
      case proto::PhysicalRelation::kWindow: {
        auto const& msg = pb.window();
        auto child      = deserialize_relation(msg.child());
        auto subqueries = _deserialize_subqueries(msg.subquery_relations());

        auto ident_cols   = deserialize_expressions(msg.ident_cols());
        auto arguments    = deserialize_expressions(msg.arguments());
        auto partition_by = deserialize_expressions(msg.partition_by());
        auto order_by     = deserialize_expressions(msg.order_by());

        std::vector<cudf::order> order_dirs;
        for (auto o : msg.order_dirs())
          order_dirs.push_back(static_cast<cudf::order>(o));

        auto lower = msg.has_lower_bound()
                       ? deserialize_bound(msg.lower_bound())
                       : window_frame_bound::type{window_frame_bound::unbounded{}};
        auto upper = msg.has_upper_bound()
                       ? deserialize_bound(msg.upper_bound())
                       : window_frame_bound::type{window_frame_bound::unbounded{}};

        return std::make_shared<physical::window_relation>(
          std::move(child),
          std::move(subqueries),
          static_cast<cudf::aggregation::Kind>(msg.aggregation_kind()),
          std::move(ident_cols),
          std::move(arguments),
          std::move(partition_by),
          std::move(order_by),
          std::move(order_dirs),
          lower,
          upper,
          msg.use_like_shift_and());
      }
      case proto::PhysicalRelation::kShuffle: {
        auto const& msg   = pb.shuffle();
        auto child        = deserialize_relation(msg.child());
        auto subqueries   = _deserialize_subqueries(msg.subquery_relations());
        auto shuffle_cols = deserialize_expressions(msg.shuffle_cols());

        return std::make_shared<physical::shuffle_relation>(
          std::move(child), std::move(subqueries), std::move(shuffle_cols));
      }
      case proto::PhysicalRelation::kGenIdentCol: {
        auto child = deserialize_relation(pb.gen_ident_col().child());
        return std::make_shared<physical::gen_ident_col_relation>(std::move(child));
      }
      case proto::PhysicalRelation::RELATION_NOT_SET:
        throw std::logic_error("PhysicalRelation has no relation set");
      default:
        throw std::logic_error("Unknown PhysicalRelation type: " +
                               std::to_string(pb.relation_case()));
    }
  }

  std::unordered_map<std::int32_t, std::shared_ptr<physical::relation>> _nodes;
};

}  // namespace

physical::identified_plan deserialize_physical_plan(proto::PhysicalRelation const& pb)
{
  return plan_deserializer{}.deserialize(pb);
}

}  // namespace gqe::rpc
