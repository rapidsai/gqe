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

#include <gqe/qep/physical_qep_transformation.hpp>

#include "../executor/join_predicate.hpp"
#include "../optimizer/utility.hpp"
#include "operator/join_output.hpp"
#include "operator/semi_anti_join.hpp"

#include <gqe/aggregation.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/unique_key_inner_join.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/expression/subquery.hpp>
#include <gqe/expression/utility.hpp>
#include <gqe/physical/aggregate.hpp>
#include <gqe/physical/bounded_hash_aggregate.hpp>
#include <gqe/physical/fetch.hpp>
#include <gqe/physical/filter.hpp>
#include <gqe/physical/join.hpp>
#include <gqe/physical/project.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/physical/sort.hpp>
#include <gqe/physical/top_k.hpp>
#include <gqe/physical/write.hpp>
#include <gqe/qep/operator/bounded_hash_aggregate.hpp>
#include <gqe/qep/operator/conditional_inner_join.hpp>
#include <gqe/qep/operator/conditional_semi_anti_join.hpp>
#include <gqe/qep/operator/equality_semi_anti_join.hpp>
#include <gqe/qep/operator/fetch.hpp>
#include <gqe/qep/operator/filter.hpp>
#include <gqe/qep/operator/filter_join_indices.hpp>
#include <gqe/qep/operator/gather.hpp>
#include <gqe/qep/operator/hash_aggregate.hpp>
#include <gqe/qep/operator/hash_join.hpp>
#include <gqe/qep/operator/join_materialization.hpp>
#include <gqe/qep/operator/mark_join.hpp>
#include <gqe/qep/operator/merge_sort.hpp>
#include <gqe/qep/operator/mixed_semi_anti_join.hpp>
#include <gqe/qep/operator/perfect_hash_aggregate.hpp>
#include <gqe/qep/operator/perfect_hash_join.hpp>
#include <gqe/qep/operator/project.hpp>
#include <gqe/qep/operator/reduce_aggregate.hpp>
#include <gqe/qep/operator/top_k.hpp>
#include <gqe/qep/operator/unique_key_inner_join.hpp>
#include <gqe/storage/readable_view.hpp>
#include <gqe/storage/table.hpp>
#include <gqe/storage/table_provider.hpp>
#include <gqe/storage/writeable_view.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <algorithm>
#include <cassert>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gqe {
namespace qep {

namespace {

/**
 * @brief Derive the combine stage's aggregate inputs from the partial aggregate specification.
 *
 * The partial task emits `[keys..., partial aggregates...]`, so combine value `i` reads column
 * `num_keys + i`.
 */
[[nodiscard]] std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>>
derive_combine_value_indices(
  std::span<std::pair<cudf::aggregation::Kind, cudf::size_type> const> value_indices,
  cudf::size_type num_keys)
{
  std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>> combine_value_indices;
  combine_value_indices.reserve(value_indices.size());
  for (std::size_t aggregate_idx = 0; aggregate_idx < value_indices.size(); ++aggregate_idx) {
    combine_value_indices.emplace_back(
      get_second_aggregation_kind(value_indices[aggregate_idx].first),
      num_keys + static_cast<cudf::size_type>(aggregate_idx));
  }
  return combine_value_indices;
}

/**
 * @brief Derive the schema emitted by the first aggregation pass.
 */
[[nodiscard]] std::vector<cudf::data_type> derive_partial_output_types(
  std::span<cudf::data_type const> input_types,
  std::span<cudf::size_type const> key_indices,
  std::span<std::pair<cudf::aggregation::Kind, cudf::size_type> const> value_indices)
{
  std::vector<cudf::data_type> partial_types;
  partial_types.reserve(key_indices.size() + value_indices.size());
  for (auto const index : key_indices) {
    GQE_EXPECTS(index >= 0 && static_cast<std::size_t>(index) < input_types.size(),
                "physical_qep_transformation: aggregate key index is out of range");
    auto const input_type = input_types[index];
    GQE_EXPECTS(input_type.id() == cudf::type_id::EMPTY || !cudf::is_nested(input_type),
                "physical_qep_transformation: nested aggregate key types are not supported");
    partial_types.push_back(input_type);
  }
  for (auto const& [kind, index] : value_indices) {
    GQE_EXPECTS(index >= 0 && static_cast<std::size_t>(index) < input_types.size(),
                "physical_qep_transformation: aggregate value index is out of range");
    auto const input_type = input_types[index];
    GQE_EXPECTS(input_type.id() == cudf::type_id::EMPTY || !cudf::is_nested(input_type),
                "physical_qep_transformation: nested aggregate value types are not supported");
    partial_types.push_back(gqe::detail::compute_target_type(input_type, kind));
  }
  return partial_types;
}

/**
 * @brief The terminal QEP task lowered from a single physical relation.
 *
 * Records the terminal so the parent's lowering can wire it as a predecessor and the output
 * column types needed by any project inserted above it. Representation normalization is confined
 * to explicit regular-table boundaries and is not inferred while lowering logical operators.
 */
struct lowering_result {
  task* terminal = nullptr;                        ///< Terminal QEP task of the lowered relation.
  std::vector<cudf::data_type> output_data_types;  ///< Output column types in emission order.
};

[[nodiscard]] mixed_join_candidate_layout make_mixed_join_candidate_layout(
  expression const& residual,
  std::span<cudf::size_type const> final_projection_indices,
  cudf::size_type left_column_count,
  cudf::size_type total_column_count,
  cudf::size_type left_anchor_index,
  cudf::size_type right_anchor_index)
{
  std::vector<cudf::size_type> left_conditional_columns;
  std::vector<cudf::size_type> right_conditional_columns;
  expression const* residual_roots[] = {&residual};
  for (auto const index : collect_column_references(residual_roots)) {
    GQE_EXPECTS(index >= 0 && index < total_column_count,
                "physical_qep_transformation: residual column index is out of range");
    auto& columns =
      index < left_column_count ? left_conditional_columns : right_conditional_columns;
    columns.push_back(index);
  }

  // cuDF needs a non-empty table on each side to retain its row count and candidate positions.
  // Keep one equality key as an internal anchor when the residual does not reference that side.
  if (left_conditional_columns.empty()) { left_conditional_columns.push_back(left_anchor_index); }
  if (right_conditional_columns.empty()) {
    right_conditional_columns.push_back(right_anchor_index);
  }

  mixed_join_candidate_layout layout;
  layout.logical_candidate_indices.reserve(left_conditional_columns.size() +
                                           right_conditional_columns.size() +
                                           final_projection_indices.size());
  layout.logical_candidate_indices.insert(layout.logical_candidate_indices.end(),
                                          left_conditional_columns.begin(),
                                          left_conditional_columns.end());
  layout.logical_candidate_indices.insert(layout.logical_candidate_indices.end(),
                                          right_conditional_columns.begin(),
                                          right_conditional_columns.end());
  for (auto const index : final_projection_indices) {
    if (std::ranges::find(layout.logical_candidate_indices, index) ==
        layout.logical_candidate_indices.end()) {
      layout.logical_candidate_indices.push_back(index);
    }
  }

  for (std::size_t candidate_index = 0; candidate_index < left_conditional_columns.size();
       ++candidate_index) {
    layout.left_conditional_indices.push_back(static_cast<cudf::size_type>(candidate_index));
  }
  for (std::size_t index = 0; index < right_conditional_columns.size(); ++index) {
    layout.right_conditional_indices.push_back(
      static_cast<cudf::size_type>(left_conditional_columns.size() + index));
  }
  for (auto const index : final_projection_indices) {
    auto const candidate_index = std::ranges::find(layout.logical_candidate_indices, index);
    GQE_EXPECTS(candidate_index != layout.logical_candidate_indices.end(),
                "physical_qep_transformation: final projection is missing from candidate layout");
    layout.output_indices.push_back(static_cast<cudf::size_type>(
      std::distance(layout.logical_candidate_indices.begin(), candidate_index)));
  }

  return layout;
}

/**
 * @brief `true` when `indices` is `[0, 1, ..., input_column_count - 1]`.
 */
[[nodiscard]] bool is_identity_projection(std::span<cudf::size_type const> indices,
                                          std::size_t input_column_count)
{
  assert(indices.size() <= std::numeric_limits<cudf::size_type>::max());

  if (indices.size() != input_column_count) return false;
  for (cudf::size_type i = 0; i < static_cast<cudf::size_type>(indices.size()); ++i) {
    if (indices[i] != i) return false;
  }
  return true;
}

/**
 * @brief `true` when `exprs` is `[col_ref(0), col_ref(1), ..., col_ref(input_column_count - 1)]`.
 */
[[nodiscard]] bool is_identity_projection(std::span<expression* const> exprs,
                                          std::size_t input_column_count)
{
  assert(exprs.size() <= std::numeric_limits<cudf::size_type>::max());

  if (exprs.size() != input_column_count) return false;
  for (cudf::size_type i = 0; i < static_cast<cudf::size_type>(exprs.size()); ++i) {
    if (exprs[i] == nullptr || exprs[i]->type() != expression::expression_type::column_reference) {
      return false;
    }
    if (static_cast<column_reference_expression const*>(exprs[i])->column_idx() != i) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Throw a uniformly-shaped error indicating that a physical relation kind is not yet
 *        supported in QEP lowering.
 */
[[noreturn]] void throw_unsupported(std::string_view kind)
{
  throw std::logic_error(std::format(
    "physical_qep_transformation: physical relation kind not yet supported in QEP lowering: {}",
    kind));
}

/**
 * @brief Internal visitor that lowers a physical relation tree to QEP tasks.
 *
 * The visitor works as follows:
 *
 * 1. `lower(rel)` probes the cache. If hit, then it returns the cached entry. On a miss, it
 * invokes the visitor and caches the published result.
 * 2. Each `visit(...)` overload lowers the relation passed to it and publishes its terminal in
 * `_result` — the return channel from the void-returning `accept`.
 *
 * `lower` is the only cache writer. The cache deduplicates a relation pointer encountered
 * twice, so each physical relation is lowered exactly once. Every recursive descent into
 * children goes through `lower` (not raw `accept`), so visits never need to consult the cache
 * themselves.
 */
class qep_lowering_visitor : public physical::relation_visitor {
 public:
  /**
   * @brief Construct a lowering visitor.
   *
   * @param[in] provider Table provider used to resolve `read_relation::table_name`. Must
   *   outlive the visitor.
   * @param[in] qep_builder The QEP builder collecting tasks and edges. Must outlive the
   *   visitor.
   * @param[in] output_data_types Output types to populate for every lowered task. Must outlive the
   *   visitor.
   * @param[in] parameters Query optimization settings. Must outlive the visitor.
   * @param[in] physical_relations Physical relation provenance to populate for every lowered task.
   *   Must outlive the visitor.
   * @param[in] stable_keys Stable task keys to populate for every lowered task. Must outlive the
   *   visitor.
   * @param[in] relation_node_ids Node IDs for the physical relations lowered by the visitor. Must
   *   outlive the visitor.
   */
  qep_lowering_visitor(storage::table_provider const* provider,
                       optimization_parameters const& parameters,
                       query_execution_plan_builder* qep_builder,
                       task_output_data_types* output_data_types,
                       task_physical_relations* physical_relations,
                       task_stable_keys* stable_keys,
                       physical::plan_node_ids const& relation_node_ids)
    : _table_provider(provider),
      _parameters(parameters),
      _qep_builder(qep_builder),
      _output_data_types(output_data_types),
      _physical_relations(physical_relations),
      _stable_keys(stable_keys),
      _relation_node_ids(relation_node_ids)
  {
  }

  /**
   * @brief Lower `rel`, caching the result by relation pointer to share subplans.
   *
   * @param[in] rel The physical relation to lower.
   *
   * @return The terminal QEP task and its output column types.
   */
  lowering_result lower(physical::relation* rel)
  {
    if (auto it = _cache.find(rel); it != _cache.end()) { return it->second; }

    _result = {};
    rel->accept(*this);
    GQE_EXPECTS(_result.terminal != nullptr,
                "physical_qep_transformation: visit(...) did not publish a lowering result");

    _cache[rel] = _result;
    return _result;
  }

  /**
   * @brief Lower the complete physical root and normalize a terminal Top-K result.
   *
   * A terminal `top_k_task` is already a pipeline breaker, so pipeline normalization does not add
   * a terminal writer bridge after it. Add the sole explicit root gather here while leaving a
   * non-terminal Top-K indirect for facade-aware successors.
   *
   * With `use_late_materialization` disabled, every Top-K output is already gathered at its
   * relation, so the root ends in a `gather_task` and this function adds nothing.
   *
   * @param[in] root Complete physical-plan root.
   */
  void lower_root(physical::relation* root)
  {
    auto const result = lower(root);
    if (dynamic_cast<top_k_task*>(result.terminal) == nullptr) { return; }

    gather_for_ordinary_boundary(result, root, lowering_role::materialize_output);
  }

  void visit(physical::read_relation* rel) override
  {
    auto const table_name = rel->table_name();
    auto tbl              = _table_provider->get_table(table_name);
    if (!tbl || !tbl->is_readable()) {
      throw std::logic_error(std::format(
        "physical_qep_transformation: read_relation for table `{}` has no readable storage",
        table_name));
    }

    // `partial_filter` is a predicate-pushdown hint. Semantically, exact filtering remains the
    // downstream `filter_relation`'s responsibility.
    //
    // Lower the partial filter as follows:
    //
    //  - Decode an in-predicate subquery on a single column reference into a partition-column
    //  pruner.
    //  - Add the subquery as a predecessor edge.
    //  - Pass-through the filter verbatim for the read task to use as it sees fit.
    std::optional<cudf::size_type> partition_column_idx;
    lowering_result haystack_result;
    auto* partial_filter = rel->partial_filter_unsafe();
    if (partial_filter != nullptr &&
        partial_filter->type() == expression::expression_type::subquery) {
      auto const* sq = static_cast<subquery_expression const*>(partial_filter);
      if (sq->subquery_type() == subquery_expression::subquery_type_type::in_predicate) {
        auto const needles = sq->children();
        GQE_EXPECTS(
          needles.size() == 1,
          "physical_qep_transformation: in-predicate partial filter must have exactly one needle "
          "expression");
        GQE_EXPECTS(needles[0]->type() == expression::expression_type::column_reference,
                    "physical_qep_transformation: in-predicate partial filter needle must be a "
                    "column reference");
        auto const* col_ref  = static_cast<column_reference_expression const*>(needles[0]);
        partition_column_idx = col_ref->column_idx();

        auto const subqueries = rel->subqueries_unsafe();
        GQE_EXPECTS(
          sq->relation_index() < subqueries.size(),
          "physical_qep_transformation: in-predicate subquery relation_index out of range");
        haystack_result = lower(subqueries[sq->relation_index()]);
      }
    }

    auto const output_data_types = rel->output_data_types();
    auto reader                  = tbl->readable_view()->get_read_pipeline(
      _parameters.source_reader,
      table_name,
      _table_provider,
      rel->column_names(),
      output_data_types,
      partition_column_idx,
      partial_filter != nullptr ? partial_filter->clone() : nullptr);
    // References are opaque source coordinates. The resolver is the first task to emit columns.
    auto const source_types = reader.resolve ? std::span<cudf::data_type const>{}
                                             : std::span<cudf::data_type const>{output_data_types};
    auto* read_ptr = add_task(std::move(reader.source), rel, source_types, lowering_role::read);

    // Reads have no input dataflow by default. Pruned reads receive their haystack as the
    // single predecessor edge.
    if (partition_column_idx) {
      // The read task consumes a materialized haystack via `qep::to_table_view`.
      auto const haystack = gather_for_ordinary_boundary(std::move(haystack_result), rel);
      _qep_builder->add_successor(haystack.terminal, read_ptr);
    }

    task* terminal = read_ptr;
    if (reader.resolve) {
      terminal =
        add_task(std::move(reader.resolve), rel, output_data_types, lowering_role::read_resolve);
      _qep_builder->add_successor(read_ptr, terminal);
    }
    _result = {terminal, output_data_types};
  }

  void visit(physical::filter_relation* rel) override
  {
    auto* condition = rel->condition_unsafe();
    GQE_EXPECTS(!contains_subquery(condition),
                "physical_qep_transformation: filter conditions containing subquery_expression "
                "are not supported (unnest scalar subqueries into joins upstream)");

    auto const children = rel->children_unsafe();
    GQE_EXPECTS(children.size() == 1,
                "physical_qep_transformation: filter_relation must have exactly one child");
    auto const child = lower(children[0]);

    auto const filter_result = lower_into_filter(child, condition->clone(), rel);

    // Insert a project task to compute non-identity expressions. The `filter_task` does not apply
    // `projection_indices`. As the project passes the filter's valid mask through, the result stays
    // masked unless `materialize_if_early` gathers it.
    auto const& projection_indices = rel->projection_indices();
    auto const child_column_count  = children[0]->output_data_types().size();
    auto projected =
      is_identity_projection(projection_indices, child_column_count)
        ? filter_result
        : lower_into_project(filter_result, projection_indices, rel->output_data_types(), rel);
    _result = materialize_if_early(std::move(projected), rel);
  }

  void visit(physical::project_relation* rel) override
  {
    auto const children = rel->children_unsafe();
    GQE_EXPECTS(children.size() == 1,
                "physical_qep_transformation: project_relation must have exactly one child");
    auto const child = lower(children[0]);

    // Elide identity projections.
    //
    // Besides being no-ops, projections are pipelined operators and thus cannot be applied to
    // materialized edges (e.g. re-emitting an aggregate's output).
    auto const output_expressions = rel->output_expressions_unsafe();
    if (is_identity_projection(output_expressions, children[0]->output_data_types().size())) {
      _result = child;
      return;
    }

    _result = lower_into_project(
      child, clone_expressions(output_expressions), rel->output_data_types(), rel);
  }

  void visit(physical::concatenate_aggregate_relation* rel) override
  {
    auto const children = rel->children_unsafe();
    GQE_EXPECTS(
      children.size() == 1,
      "physical_qep_transformation: concatenate_aggregate_relation must have exactly one child");
    auto child = lower(children[0]);

    // `aggregate_relation_base::condition` is technical debt inherited from the task graph design.
    // It's a pre-aggregation WHERE-style filter applied to the *input* (see
    // `physical/aggregate.hpp`).
    //
    // The QEP unnests the filter as a
    // `filter_task` ahead of the aggregate. `reduce_aggregate_task` consumes the resulting masked
    // logical relation natively, so no gather is inserted between them unless late
    // materialization is disabled.
    auto* condition = rel->condition_unsafe();
    if (condition != nullptr) {
      GQE_EXPECTS(!contains_subquery(condition),
                  "physical_qep_transformation: aggregate input conditions containing "
                  "subquery_expression are not supported (unnest scalar subqueries into joins "
                  "upstream)");
      child = materialize_if_early(
        lower_into_filter(child, condition->clone(), rel), rel, lowering_role::materialize_input);
    }

    // Lower scalar aggregate.
    if (rel->keys_unsafe().empty()) {
      // Flatten each aggregation function expression into a (function kind, column index) pair.
      //
      // The `complex_expression_extraction_into_project` optimizer rule simplifies complex
      // expressions such that we see only a bare column reference here (except `COUNT_ALL`, which
      // reads no column).
      auto const values = rel->values_unsafe();
      std::vector<std::pair<cudf::aggregation::Kind, std::optional<cudf::size_type>>> value_indices;
      value_indices.reserve(values.size());
      for (auto const& [kind, expr] : values) {
        if (kind == cudf::aggregation::COUNT_ALL) {
          value_indices.emplace_back(kind, std::nullopt);
        } else {
          GQE_EXPECTS(
            expr != nullptr && expr->type() == expression::expression_type::column_reference,
            "physical_qep_transformation: aggregate value expressions must be bare column "
            "references "
            "(run the `complex_expression_extraction_into_project` optimizer rule to materialize "
            "computed measures upstream)");
          value_indices.emplace_back(
            kind, static_cast<column_reference_expression const*>(expr)->column_idx());
        }
      }

      auto const output_data_types = rel->output_data_types();
      auto reduce_task =
        std::make_unique<reduce_aggregate_task>(std::move(value_indices), output_data_types);
      auto* reduce_ptr =
        add_task(std::move(reduce_task), rel, output_data_types, lowering_role::aggregate_reduce);
      _qep_builder->add_successor(child.terminal, reduce_ptr);

      _result = {reduce_ptr, output_data_types};
    } else {
      // The default optimizer materializes complex aggregate expressions upstream. Grouped
      // aggregate tasks therefore reference the remaining bare column references directly.
      auto const keys = rel->keys_unsafe();
      std::vector<cudf::size_type> key_indices;
      key_indices.reserve(keys.size());
      for (auto const* key : keys) {
        GQE_EXPECTS(
          key != nullptr && key->type() == expression::expression_type::column_reference,
          "physical_qep_transformation: aggregate key expressions must be bare column references "
          "(run the `complex_expression_extraction_into_project` optimizer rule to materialize "
          "computed grouping keys upstream)");
        key_indices.push_back(static_cast<column_reference_expression const*>(key)->column_idx());
      }

      auto const values = rel->values_unsafe();
      std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>> value_indices;
      value_indices.reserve(values.size());
      for (auto const& [kind, value] : values) {
        GQE_EXPECTS(
          value != nullptr && value->type() == expression::expression_type::column_reference,
          "physical_qep_transformation: aggregate value expressions must be bare column "
          "references "
          "(run the `complex_expression_extraction_into_project` optimizer rule to materialize "
          "computed measures upstream)");
        value_indices.emplace_back(
          kind, static_cast<column_reference_expression const*>(value)->column_idx());
      }

      auto const num_keys        = static_cast<cudf::size_type>(key_indices.size());
      auto combine_value_indices = derive_combine_value_indices(value_indices, num_keys);
      std::vector<cudf::size_type> combine_key_indices(key_indices.size());
      for (cudf::size_type key_idx = 0; key_idx < num_keys; ++key_idx) {
        combine_key_indices[key_idx] = key_idx;
      }

      auto const partial_output_types =
        derive_partial_output_types(child.output_data_types, key_indices, value_indices);
      auto const output_data_types = rel->output_data_types();

      task* partial_ptr = nullptr;
      task* combine_ptr = nullptr;
      bool const use_perfect_hashing =
        _parameters.aggregation_use_perfect_hash && rel->perfect_hashing();
      if (use_perfect_hashing) {
        auto partial_task = std::make_unique<perfect_hash_partial_aggregate_task>(
          key_indices, value_indices, partial_output_types);
        partial_ptr = add_task(
          std::move(partial_task), rel, partial_output_types, lowering_role::aggregate_partial);

        auto combine_task = std::make_unique<perfect_hash_aggregate_task>(
          std::move(combine_key_indices), std::move(combine_value_indices), output_data_types);
        combine_ptr =
          add_task(std::move(combine_task), rel, output_data_types, lowering_role::aggregate_final);
      } else {
        auto partial_task =
          std::make_unique<hash_partial_aggregate_task>(key_indices, value_indices);
        partial_ptr = add_task(
          std::move(partial_task), rel, partial_output_types, lowering_role::aggregate_partial);

        auto combine_task = std::make_unique<hash_aggregate_task>(
          std::move(combine_key_indices), std::move(combine_value_indices), output_data_types);
        combine_ptr =
          add_task(std::move(combine_task), rel, output_data_types, lowering_role::aggregate_final);
      }

      _qep_builder->add_successor(child.terminal, partial_ptr);
      _qep_builder->add_successor(partial_ptr, combine_ptr);
      _result = {combine_ptr, output_data_types};
    }
  }

  void visit(physical::bounded_hash_aggregate_relation* rel) override
  {
    auto const children = rel->children_unsafe();
    GQE_EXPECTS(
      children.size() == 1,
      "physical_qep_transformation: bounded_hash_aggregate_relation must have exactly one child");

    // The build and scan tasks support one grouping key and one aggregate value, so the
    // lowering rejects wider shapes.
    GQE_EXPECTS(rel->key_indices().size() == 1,
                "physical_qep_transformation: the bounded hash aggregate lowering supports "
                "exactly one grouping key (composite keys are not lowered yet)");
    GQE_EXPECTS(rel->values().size() == 1,
                "physical_qep_transformation: the bounded hash aggregate lowering supports "
                "exactly one aggregate value (multiple aggregates are not lowered yet)");

    auto const child = lower(children[0]);

    // The relation asserts the bounded lowering, so an unsupported type combination propagates
    // the operator's construction error instead of falling back.
    auto const& [kind, value_index, value_nullability] = rel->values().front();
    bounded_hash_aggregation const aggregation{
      kind, value_index, /*argument_index=*/{}, value_nullability};
    auto build_task =
      rel->desired_load_factor().has_value()
        ? std::make_unique<bounded_hash_aggregate_build_task>(rel->key_indices().front(),
                                                              aggregation,
                                                              rel->output_data_types(),
                                                              rel->group_count_bound(),
                                                              *rel->desired_load_factor())
        : std::make_unique<bounded_hash_aggregate_build_task>(rel->key_indices().front(),
                                                              aggregation,
                                                              rel->output_data_types(),
                                                              rel->group_count_bound());

    // The build task's output is the map state, not a relational chunk shape, so it registers
    // no output column types (like the perfect-hash join build). Its scan successor pipelines
    // the grouped result back out as the terminal task.
    auto* build_ptr = add_task(std::move(build_task),
                               rel,
                               std::span<cudf::data_type const>{},
                               lowering_role::aggregate_build);
    _qep_builder->add_successor(child.terminal, build_ptr);

    auto scan_task = std::make_unique<bounded_hash_aggregate_scan_task>(
      bounded_hash_aggregate_scan_task::single_output);
    auto scan_output_data_types = rel->output_data_types();
    auto* scan_ptr =
      add_task(std::move(scan_task), rel, scan_output_data_types, lowering_role::aggregate_scan);
    _qep_builder->add_successor(build_ptr, scan_ptr);

    _result = {scan_ptr, std::move(scan_output_data_types)};
  }

  void visit(physical::write_relation* rel) override
  {
    auto const table_name = rel->table_name();
    auto tbl              = _table_provider->get_table(table_name);
    if (!tbl || !tbl->is_writeable()) {
      throw std::logic_error(std::format(
        "physical_qep_transformation: write_relation for table `{}` has no writeable storage",
        table_name));
    }

    auto const children = rel->children_unsafe();
    GQE_EXPECTS(children.size() == 1,
                "physical_qep_transformation: write_relation must have exactly one child");
    auto const child = lower(children[0]);

    // Storage writers consume only ordinary tables. The gather is idempotent for already-normal
    // input and removes any mask or indirection without representation analysis in lowering.
    auto const upstream = gather_for_ordinary_boundary(child, rel);

    auto const output_data_types = rel->output_data_types();
    auto fold      = tbl->writeable_view()->get_fold_task(rel->column_names(), output_data_types);
    auto* fold_ptr = add_task(std::move(fold), rel, output_data_types, lowering_role::write);
    _qep_builder->add_successor(upstream.terminal, fold_ptr);

    _result = {fold_ptr, output_data_types};
  }

  void visit(physical::top_k_relation* rel) override
  {
    auto const children = rel->children_unsafe();
    GQE_EXPECTS(children.size() == 1,
                "physical_qep_transformation: top_k_relation must have exactly one child");
    auto const child = lower(children[0]);

    auto const output_data_types = rel->output_data_types();
    auto top_k                   = std::make_unique<top_k_task>(rel->key_index(),
                                              rel->column_order(),
                                              rel->null_precedence(),
                                              static_cast<cudf::size_type>(rel->offset()),
                                              static_cast<cudf::size_type>(rel->count()),
                                              output_data_types);
    auto* top_k_ptr = add_task(std::move(top_k), rel, output_data_types, lowering_role::top_k);
    _qep_builder->add_successor(child.terminal, top_k_ptr);

    // See `lower_root` for the terminal Top-K boundary.
    _result = materialize_if_early({top_k_ptr, output_data_types}, rel);
  }

  void visit(physical::broadcast_join_relation* rel) override
  {
    auto joined = lower_broadcast_join(rel);
    // Early materialization gathers the whole output, which already releases the join's inputs.
    _result = _parameters.use_late_materialization
                ? append_join_materialization(std::move(joined), rel)
                : materialize_if_early(std::move(joined), rel);
  }
  void visit(physical::shuffle_join_relation*) override { throw_unsupported("shuffle_join"); }

  /**
   * @brief Append a `join_materialization_task` after a late-materialized broadcast join.
   *
   * The task follows residual filtering and precedes `lower`'s caching of the relation, so every
   * consumer of a shared join reads one result. It is skipped when the threshold is zero and for
   * count-only outputs.
   *
   * @param[in] upstream The lowered join output.
   * @param[in] rel The broadcast join relation.
   *
   * @return `upstream`, or the join materialization task's result.
   */
  lowering_result append_join_materialization(lowering_result upstream,
                                              physical::broadcast_join_relation* rel)
  {
    if (_parameters.join_materialization_threshold == 0.0 || upstream.output_data_types.empty()) {
      return upstream;
    }
    auto* materialize_ptr = add_task(
      std::make_unique<join_materialization_task>(_parameters.join_materialization_threshold),
      rel,
      upstream.output_data_types,
      lowering_role::join_materialize);
    _qep_builder->add_successor(upstream.terminal, materialize_ptr);
    return {materialize_ptr, std::move(upstream.output_data_types)};
  }

  /**
   * @brief Lower a broadcast join of any join family.
   *
   * Every join family ends here, so `visit` applies the early-materialization boundary once for
   * all of them.
   *
   * @param[in] rel The broadcast join relation to lower.
   *
   * @return The join output, ending at its probe or residual-filter task.
   */
  lowering_result lower_broadcast_join(physical::broadcast_join_relation* rel)
  {
    auto const is_semi_anti = rel->join_type() == join_type_type::left_semi ||
                              rel->join_type() == join_type_type::left_anti;
    bool const is_mark_join = is_semi_anti && rel->policy() == physical::broadcast_policy::left &&
                              _parameters.join_use_mark_join;
    bool const is_perfect_hashing_enabled =
      _parameters.join_use_perfect_hash && rel->perfect_hashing();
    auto const must_build_logical_right =
      rel->join_type() == join_type_type::left || (is_semi_anti && !is_mark_join);
    GQE_EXPECTS(rel->join_type() == join_type_type::inner ||
                  rel->join_type() == join_type_type::left || is_semi_anti,
                "physical_qep_transformation: broadcast QEP lowering supports only inner, left, "
                "left-semi, and left-anti joins");
    GQE_EXPECTS(!must_build_logical_right || rel->policy() == physical::broadcast_policy::right,
                "physical_qep_transformation: left joins and non-mark left-semi/left-anti joins "
                "must build their logical right input");
    GQE_EXPECTS(
      !is_mark_join ||
        (rel->policy() == physical::broadcast_policy::left && !is_perfect_hashing_enabled),
      "physical_qep_transformation: mark joins must build the left input without perfect hashing");

    auto const children = rel->children_unsafe();
    GQE_EXPECTS(children.size() == 2,
                "physical_qep_transformation: broadcast_join_relation must have exactly two "
                "children");
    GQE_EXPECTS(rel->condition() != nullptr,
                "physical_qep_transformation: broadcast join condition must not be null");
    GQE_EXPECTS(!contains_subquery(rel->condition()),
                "physical_qep_transformation: join conditions containing subquery_expression "
                "are not supported (unnest scalar subqueries into joins upstream)");
    GQE_EXPECTS(rel->left_filter_condition() == nullptr && rel->right_filter_condition() == nullptr,
                "physical_qep_transformation: broadcast join input filters are not supported in "
                "QEP lowering");

    auto const left_column_count_size  = children[0]->output_data_types().size();
    auto const right_column_count_size = children[1]->output_data_types().size();
    auto const max_column_count =
      static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max());
    GQE_EXPECTS(left_column_count_size <= max_column_count &&
                  right_column_count_size <= max_column_count - left_column_count_size,
                "physical_qep_transformation: join input column count exceeds cudf::size_type");
    auto const left_column_count = static_cast<cudf::size_type>(left_column_count_size);

    auto condition_analysis =
      optimizer::utility::analyze_join_condition(rel->condition(), left_column_count);
    GQE_EXPECTS(condition_analysis.has_value(),
                "physical_qep_transformation: join equality keys use mixed null policies");
    bool const is_mixed_join =
      !condition_analysis->keys.left.empty() && condition_analysis->residual != nullptr;

    if (is_semi_anti && !is_mark_join && condition_analysis->residual != nullptr) {
      GQE_EXPECTS(rel->policy() == physical::broadcast_policy::right,
                  "physical_qep_transformation: semi/anti requires broadcast-right");
      auto const right_column_count = static_cast<cudf::size_type>(right_column_count_size);
      auto const& keys              = condition_analysis->keys;
      auto const& left_types        = children[0]->output_data_types();
      auto const& right_types       = children[1]->output_data_types();
      GQE_EXPECTS(keys.left.size() == keys.right.size(), "semi/anti key counts differ");
      for (std::size_t i = 0; i < keys.left.size(); ++i) {
        GQE_EXPECTS(keys.left[i] >= 0 && keys.left[i] < left_column_count && keys.right[i] >= 0 &&
                      keys.right[i] < right_column_count,
                    "semi/anti key index out of range");
        auto const left_type = left_types[keys.left[i]];
        GQE_EXPECTS(left_type == right_types[keys.right[i]] &&
                      detail::is_supported_semi_anti_column(left_type),
                    "semi/anti key types differ or are unsupported");
      }
      GQE_EXPECTS(gqe::detail::is_join_predicate_supported(*condition_analysis->residual),
                  "semi/anti residual is not supported by the cuDF AST");
      auto operands = gqe::detail::try_collect_join_predicate_columns(
        *condition_analysis->residual, left_column_count, right_column_count);
      GQE_EXPECTS(operands.has_value(), "semi/anti residual column out of range");
      auto& [left_operands, right_operands] = *operands;
      if (!is_mixed_join) {
        GQE_EXPECTS(!left_operands.empty() && !right_operands.empty(),
                    "conditional semi/anti predicate must reference both inputs");
        GQE_EXPECTS(left_column_count > 0 && right_column_count > 0,
                    "conditional semi/anti requires nonempty schemas");
      }
      for (auto index : left_operands) {
        GQE_EXPECTS(detail::is_supported_semi_anti_column(left_types[index]),
                    "semi/anti left operand type unsupported");
      }
      for (auto index : right_operands) {
        GQE_EXPECTS(detail::is_supported_semi_anti_column(right_types[index]),
                    "semi/anti right operand type unsupported");
      }
      auto const projection = rel->projection_indices();
      GQE_EXPECTS(
        std::ranges::all_of(projection,
                            [&](auto index) {
                              return index >= 0 && index < left_column_count &&
                                     detail::is_supported_semi_anti_column(left_types[index]);
                            }),
        "semi/anti projects supported logical-left columns only");
      auto output_types = rel->output_data_types();
      std::unique_ptr<task> build_task;
      std::unique_ptr<task> probe_task;
      if (is_mixed_join) {
        build_task = std::make_unique<mixed_semi_anti_join_build_task>(
          right_column_count, keys.right, std::move(right_operands), keys.null_equality);
        probe_task =
          std::make_unique<mixed_semi_anti_join_probe_task>(left_column_count,
                                                            right_column_count,
                                                            keys.left,
                                                            std::move(condition_analysis->residual),
                                                            rel->join_type(),
                                                            projection,
                                                            output_types);
      } else {
        build_task = std::make_unique<conditional_semi_anti_join_build_task>(
          right_column_count, std::move(right_operands));
        probe_task = std::make_unique<conditional_semi_anti_join_probe_task>(
          left_column_count,
          right_column_count,
          std::move(condition_analysis->residual),
          rel->join_type(),
          projection,
          output_types);
      }
      auto const build_input = lower(children[1]);
      auto const probe_input = lower(children[0]);
      auto* build_ptr        = add_task(
        std::move(build_task), rel, std::span<cudf::data_type const>{}, lowering_role::join_build);
      auto* probe_ptr =
        add_task(std::move(probe_task), rel, output_types, lowering_role::join_probe);
      _qep_builder->add_successor(build_input.terminal, build_ptr);
      _qep_builder->add_successor(build_ptr, probe_ptr);
      _qep_builder->add_successor(probe_input.terminal, probe_ptr);
      return {probe_ptr, std::move(output_types)};
    }

    if (condition_analysis->keys.left.empty()) {
      GQE_EXPECTS(
        rel->join_type() == join_type_type::inner && condition_analysis->residual != nullptr,
        "keyless broadcast joins require an inner predicate");
      bool const is_build_left      = rel->policy() == physical::broadcast_policy::left;
      auto const right_column_count = static_cast<cudf::size_type>(right_column_count_size);
      auto operands                 = gqe::detail::try_collect_join_predicate_columns(
        *condition_analysis->residual, left_column_count, right_column_count);
      GQE_EXPECTS(operands.has_value(), "conditional inner predicate column out of range");
      auto& [left_operands, right_operands] = *operands;
      GQE_EXPECTS(!left_operands.empty() && !right_operands.empty(),
                  "conditional inner predicate must reference both inputs");
      for (auto index : left_operands) {
        GQE_EXPECTS(detail::is_supported_semi_anti_column(children[0]->output_data_types()[index]),
                    "conditional inner left operand type unsupported");
      }
      for (auto index : right_operands) {
        GQE_EXPECTS(detail::is_supported_semi_anti_column(children[1]->output_data_types()[index]),
                    "conditional inner right operand type unsupported");
      }
      std::vector<cudf::size_type> build_projection;
      for (auto index : rel->projection_indices()) {
        if (is_build_left && index < left_column_count) { build_projection.push_back(index); }
        if (!is_build_left && index >= left_column_count) {
          build_projection.push_back(index - left_column_count);
        }
      }
      auto build_task = std::make_unique<conditional_inner_join_build_task>(
        is_build_left ? left_column_count : right_column_count,
        is_build_left ? std::move(left_operands) : std::move(right_operands),
        std::move(build_projection));
      auto output_types = rel->output_data_types();
      auto probe_task =
        std::make_unique<conditional_inner_join_probe_task>(left_column_count,
                                                            right_column_count,
                                                            std::move(condition_analysis->residual),
                                                            is_build_left,
                                                            rel->projection_indices(),
                                                            output_types);
      auto const build_input = lower(children[is_build_left ? 0 : 1]);
      auto const probe_input = lower(children[is_build_left ? 1 : 0]);
      auto* build_ptr        = add_task(
        std::move(build_task), rel, std::span<cudf::data_type const>{}, lowering_role::join_build);
      auto* probe_ptr =
        add_task(std::move(probe_task), rel, output_types, lowering_role::join_probe);
      _qep_builder->add_successor(build_input.terminal, build_ptr);
      _qep_builder->add_successor(build_ptr, probe_ptr);
      _qep_builder->add_successor(probe_input.terminal, probe_ptr);
      return {probe_ptr, std::move(output_types)};
    }

    bool const materialized_is_right = rel->policy() == physical::broadcast_policy::right;
    auto const unique_keys_policy    = rel->unique_keys_policy();
    bool const use_perfect_hashing =
      is_perfect_hashing_enabled && unique_keys_policy != gqe::unique_keys_policy::none &&
      (!is_mixed_join || (rel->join_type() == join_type_type::inner &&
                          condition_analysis->keys.null_equality == cudf::null_equality::UNEQUAL));

    bool const perfect_hash_build_is_right =
      unique_keys_policy == gqe::unique_keys_policy::right ||
      (unique_keys_policy == gqe::unique_keys_policy::either && materialized_is_right);
    auto const perfect_hash_build_input = perfect_hash_build_is_right == materialized_is_right
                                            ? perfect_hash_join_build_input::materialized
                                            : perfect_hash_join_build_input::streaming;

    optimizer::utility::equijoin_keys keys;
    if (use_perfect_hashing) {
      GQE_EXPECTS(rel->join_type() == join_type_type::inner,
                  "physical_qep_transformation: perfect hashing supports only inner joins");
      GQE_EXPECTS(condition_analysis->keys.null_equality == cudf::null_equality::UNEQUAL,
                  "physical_qep_transformation: perfect-hashing join condition must be a "
                  "conjunction containing ordinary column equalities between the two inputs");
      keys = std::move(condition_analysis->keys);
    } else {
      GQE_EXPECTS(!condition_analysis->keys.left.empty(),
                  "physical_qep_transformation: hash join condition must contain at least one "
                  "cross-input column equality");
      keys = std::move(condition_analysis->keys);
    }
    GQE_EXPECTS(!keys.left.empty(),
                "physical_qep_transformation: broadcast join must have at least one key");
    GQE_EXPECTS(keys.left.size() == keys.right.size(),
                "physical_qep_transformation: broadcast join key counts differ");
    auto const& left_data_types  = children[0]->output_data_types();
    auto const& right_data_types = children[1]->output_data_types();
    for (std::size_t i = 0; i < keys.left.size(); ++i) {
      GQE_EXPECTS(keys.left[i] >= 0 && keys.left[i] < left_column_count && keys.right[i] >= 0 &&
                    keys.right[i] < static_cast<cudf::size_type>(right_column_count_size),
                  "physical_qep_transformation: broadcast join key index is out of range");
      auto const left_type  = left_data_types[static_cast<std::size_t>(keys.left[i])];
      auto const right_type = right_data_types[static_cast<std::size_t>(keys.right[i])];
      if (is_semi_anti) {
        GQE_EXPECTS(
          !cudf::is_nested(left_type) && !cudf::is_nested(right_type),
          "physical_qep_transformation: nested semi/anti join key types are not supported");
      }
      GQE_EXPECTS(left_type == right_type,
                  "physical_qep_transformation: broadcast join key types differ");
      if (is_mark_join) {
        GQE_EXPECTS(cudf::is_numeric(left_type),
                    "physical_qep_transformation: mark join keys must be numeric");
      }
      if (use_perfect_hashing) {
        GQE_EXPECTS(is_supported_perfect_hash_join_key_type(left_type.id()),
                    "physical_qep_transformation: perfect-hashing join key type is unsupported");
      }
    }

    auto const right_column_count       = static_cast<cudf::size_type>(right_column_count_size);
    auto const total_column_count       = left_column_count + right_column_count;
    auto const final_projection_indices = rel->projection_indices();
    auto const projection_column_count  = is_semi_anti ? left_column_count : total_column_count;
    GQE_EXPECTS(std::ranges::all_of(
                  final_projection_indices,
                  [&](auto index) { return index >= 0 && index < projection_column_count; }),
                "physical_qep_transformation: join projection index is out of range");

    if (is_mark_join) {
      if (condition_analysis->residual != nullptr) {
        GQE_EXPECTS(
          gqe::detail::is_join_predicate_supported(*condition_analysis->residual),
          "physical_qep_transformation: mark join residual is not supported by the cuDF AST");
      }
      GQE_EXPECTS(
        std::ranges::all_of(final_projection_indices,
                            [&](auto index) { return index >= 0 && index < left_column_count; }),
        "physical_qep_transformation: mark joins can project only left columns");
      auto const build_input = lower(children[0]);
      auto const probe_input = lower(children[1]);
      std::vector<cudf::size_type> build_predicate_columns;
      if (condition_analysis->residual) {
        std::vector<expression const*> const roots{condition_analysis->residual.get()};
        for (auto index : collect_column_references(roots)) {
          GQE_EXPECTS(index >= 0 && index < total_column_count,
                      "mark join residual column index out of range");
          if (index < left_column_count) { build_predicate_columns.push_back(index); }
        }
      }
      auto output_types = rel->output_data_types();
      auto* build =
        add_task(std::make_unique<mark_join_build_task>(
                   std::move(keys.left), keys.null_equality, std::move(build_predicate_columns)),
                 rel,
                 std::span<cudf::data_type const>{},
                 lowering_role::join_build);
      auto* probe = add_task(std::make_unique<mark_join_probe_task>(
                               std::move(keys.right), std::move(condition_analysis->residual)),
                             rel,
                             std::span<cudf::data_type const>{},
                             lowering_role::join_probe);
      _qep_builder->add_successor(build_input.terminal, build);
      _qep_builder->add_successor(build, probe);
      _qep_builder->add_successor(probe_input.terminal, probe);
      auto* complete = add_task(std::make_unique<mark_join_complete_task>(),
                                rel,
                                std::span<cudf::data_type const>{},
                                lowering_role::join_complete);
      _qep_builder->add_successor(probe, complete);
      auto* scan = add_task(std::make_unique<mark_join_scan_task>(
                              rel->join_type(), final_projection_indices, output_types),
                            rel,
                            output_types,
                            lowering_role::join_scan);
      _qep_builder->add_successor(build, scan);
      _qep_builder->add_successor(complete, scan);
      return {scan, std::move(output_types)};
    }

    std::optional<mixed_join_candidate_layout> mixed_layout;
    if (is_mixed_join) {
      GQE_EXPECTS(gqe::detail::is_join_predicate_supported(*condition_analysis->residual),
                  "mixed join residual is not supported by the cuDF AST");
      mixed_layout.emplace(
        make_mixed_join_candidate_layout(*condition_analysis->residual,
                                         final_projection_indices,
                                         left_column_count,
                                         total_column_count,
                                         keys.left.front(),
                                         left_column_count + keys.right.front()));
    }
    auto execution_projection_indices =
      is_mixed_join ? mixed_layout->logical_candidate_indices : final_projection_indices;
    if (materialized_is_right && !is_semi_anti) {
      optimizer::utility::swap_projection_indices_inplace(
        execution_projection_indices, left_column_count, right_column_count);
    }
    auto output_data_types = rel->output_data_types();
    std::vector<cudf::data_type> probe_output_data_types;
    if (is_mixed_join) {
      probe_output_data_types.reserve(mixed_layout->logical_candidate_indices.size());
      for (auto const index : mixed_layout->logical_candidate_indices) {
        probe_output_data_types.push_back(
          index < left_column_count
            ? left_data_types[static_cast<std::size_t>(index)]
            : right_data_types[static_cast<std::size_t>(index - left_column_count)]);
      }
    } else {
      probe_output_data_types = output_data_types;
    }

    std::size_t const materialized_child_index = materialized_is_right ? 1 : 0;
    std::size_t const streaming_child_index    = materialized_is_right ? 0 : 1;
    auto const build_logical_column_count =
      materialized_is_right ? right_column_count : left_column_count;
    auto const probe_logical_column_count =
      materialized_is_right ? left_column_count : right_column_count;
    auto build_projected_payload_indices =
      is_semi_anti
        ? std::vector<cudf::size_type>{}
        : detail::join_materialized_payload_indices(
            execution_projection_indices, build_logical_column_count, probe_logical_column_count);
    auto materialized_key_indices =
      materialized_is_right ? std::move(keys.right) : std::move(keys.left);
    auto streaming_key_indices =
      materialized_is_right ? std::move(keys.left) : std::move(keys.right);
    auto const materialized_input = lower(children[materialized_child_index]);
    auto streaming_input          = lower(children[streaming_child_index]);
    if (is_mixed_join && rel->join_type() == join_type_type::left) {
      // The residual filter requires one source per side and original logical-left row identity.
      // The build input is already materialized. Normalize inherited probe sources before joining.
      streaming_input = gather_for_ordinary_boundary(std::move(streaming_input), rel);
    }

    auto const& build_types = materialized_is_right ? right_data_types : left_data_types;
    auto const are_key_types_supported =
      std::ranges::all_of(materialized_key_indices, [&](auto index) {
        return gqe::unique_key_join_supported(build_types[static_cast<std::size_t>(index)]);
      });
    auto const use_unique_keys =
      _parameters.join_use_unique_keys && are_key_types_supported &&
      rel->join_type() == join_type_type::inner &&
      rel->unique_keys_policy() ==
        (materialized_is_right ? gqe::unique_keys_policy::right : gqe::unique_keys_policy::left);

    std::unique_ptr<fold_task> build_task;
    if (is_semi_anti) {
      build_task = std::make_unique<equality_semi_anti_join_build_task>(
        std::move(materialized_key_indices), keys.null_equality);
    } else if (use_perfect_hashing) {
      build_task =
        std::make_unique<perfect_hash_join_build_task>(build_logical_column_count,
                                                       std::move(materialized_key_indices),
                                                       std::move(build_projected_payload_indices));
    } else if (use_unique_keys) {
      std::vector<cudf::data_type> build_key_types;
      for (auto index : materialized_key_indices) {
        build_key_types.push_back(build_types[static_cast<std::size_t>(index)]);
      }
      build_task = std::make_unique<unique_key_inner_join_build_task>(
        build_logical_column_count,
        std::move(materialized_key_indices),
        std::move(build_projected_payload_indices),
        keys.null_equality,
        (_parameters.join_use_hash_map_cache || rel->use_hash_map_cache()),
        std::move(build_key_types));
    } else {
      build_task = std::make_unique<hash_join_build_task>(
        build_logical_column_count,
        std::move(materialized_key_indices),
        std::move(build_projected_payload_indices),
        keys.null_equality,
        (_parameters.join_use_hash_map_cache || rel->use_hash_map_cache()));
    }
    auto* build_ptr = add_task(
      std::move(build_task), rel, std::span<cudf::data_type const>{}, lowering_role::join_build);
    _qep_builder->add_successor(materialized_input.terminal, build_ptr);

    std::unique_ptr<stateful_transform_task> probe_task;
    if (is_semi_anti) {
      probe_task = std::make_unique<equality_semi_anti_join_probe_task>(
        std::move(streaming_key_indices),
        rel->join_type(),
        std::move(execution_projection_indices),
        output_data_types);
    } else if (use_perfect_hashing) {
      probe_task =
        std::make_unique<perfect_hash_join_probe_task>(std::move(streaming_key_indices),
                                                       perfect_hash_build_input,
                                                       std::move(execution_projection_indices),
                                                       probe_output_data_types);
    } else if (use_unique_keys) {
      probe_task =
        std::make_unique<unique_key_inner_join_probe_task>(std::move(streaming_key_indices),
                                                           std::move(execution_projection_indices),
                                                           probe_output_data_types);
    } else {
      probe_task = std::make_unique<hash_join_probe_task>(std::move(streaming_key_indices),
                                                          rel->join_type(),
                                                          std::move(execution_projection_indices),
                                                          probe_output_data_types);
    }
    auto* probe_ptr =
      add_task(std::move(probe_task), rel, probe_output_data_types, lowering_role::join_probe);
    // The breaker predecessor supplies `initialize`; the streaming predecessor supplies `next`.
    _qep_builder->add_successor(build_ptr, probe_ptr);
    _qep_builder->add_successor(streaming_input.terminal, probe_ptr);

    task* gather_input = probe_ptr;
    if (is_mixed_join) {
      auto filter = std::make_unique<filter_join_indices_task>(
        rel->join_type(),
        std::move(condition_analysis->residual),
        left_column_count,
        static_cast<cudf::size_type>(right_column_count_size),
        std::move(*mixed_layout),
        probe_output_data_types,
        output_data_types);
      auto* filter_ptr = add_task(std::move(filter), rel, output_data_types, lowering_role::filter);
      _qep_builder->add_successor(probe_ptr, filter_ptr);
      gather_input = filter_ptr;
    }

    return {gather_input, std::move(output_data_types)};
  }

  /** @brief Lower a concatenate sort with an ordinary-table input boundary. */
  void visit(physical::concatenate_sort_relation* rel) override
  {
    auto const children = rel->children_unsafe();
    GQE_EXPECTS(children.size() == 1,
                "physical_qep_transformation: concatenate_sort_relation must have exactly one "
                "child");
    auto const keys       = rel->keys_unsafe();
    auto column_orders    = rel->column_orders();
    auto null_precedences = rel->null_precedences();
    GQE_EXPECTS(!keys.empty(),
                "physical_qep_transformation: concatenate_sort_relation must have at least one "
                "key",
                std::invalid_argument);
    GQE_EXPECTS(std::ranges::none_of(keys, [](expression const* key) { return key == nullptr; }),
                "physical_qep_transformation: concatenate_sort_relation contains a null key",
                std::invalid_argument);
    GQE_EXPECTS(
      std::ranges::all_of(keys,
                          [](expression const* key) {
                            return key->type() == expression::expression_type::column_reference;
                          }),
      "physical_qep_transformation: concatenate_sort_relation keys must be bare column "
      "references (materialize computed sort keys in the optimizer before physical QEP lowering)");
    GQE_EXPECTS(keys.size() == column_orders.size(),
                "physical_qep_transformation: concatenate_sort_relation keys and column orders "
                "must have equal lengths",
                std::invalid_argument);
    GQE_EXPECTS(
      keys.size() == null_precedences.size(),
      "physical_qep_transformation: concatenate_sort_relation keys and null precedences must "
      "have equal lengths",
      std::invalid_argument);

    auto const max_column_count =
      static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max());
    auto output_data_types        = children[0]->output_data_types();
    auto const input_column_count = output_data_types.size();
    GQE_EXPECTS(input_column_count <= max_column_count,
                "physical_qep_transformation: concatenate_sort_relation input has too many "
                "columns",
                std::invalid_argument);

    std::vector<cudf::size_type> key_column_indices;
    key_column_indices.reserve(keys.size());
    for (auto const* key : keys) {
      auto const column_index = static_cast<column_reference_expression const*>(key)->column_idx();
      GQE_EXPECTS(column_index >= 0 && static_cast<std::size_t>(column_index) < input_column_count,
                  "physical_qep_transformation: concatenate_sort_relation key column index is "
                  "out of range",
                  std::invalid_argument);
      key_column_indices.push_back(column_index);
    }

    // Merge-sort accepts only ordinary-table input.
    auto upstream = gather_for_ordinary_boundary(lower(children[0]), rel);

    auto sort      = std::make_unique<merge_sort_task>(std::move(key_column_indices),
                                                  std::move(column_orders),
                                                  std::move(null_precedences),
                                                  output_data_types);
    auto* sort_ptr = add_task(std::move(sort), rel, output_data_types, lowering_role::sort);
    _qep_builder->add_successor(upstream.terminal, sort_ptr);
    _result = {sort_ptr, std::move(output_data_types)};
  }
  void visit(physical::fetch_relation* rel) override
  {
    auto const children = rel->children_unsafe();
    GQE_EXPECTS(children.size() == 1,
                "physical_qep_transformation: fetch_relation must have exactly one child");
    auto const child = lower(children[0]);

    auto const output_data_types = rel->output_data_types();
    // No producer validates the bounds, so validate the offset before any arithmetic on it. The
    // Substrait parser encodes an unset count as -1, meaning ALL rows after the offset. Translate
    // any negative count to that sentinel and clamp a too-large count to the largest expressible
    // window, so a valid frontend LIMIT can never overflow `offset + count`.
    GQE_EXPECTS(rel->offset() >= 0,
                "physical_qep_transformation: fetch offset must be non-negative");
    auto const max_count = std::numeric_limits<int64_t>::max() - rel->offset();
    auto const count     = rel->count() < 0 ? max_count : std::min(rel->count(), max_count);
    auto fetch           = std::make_unique<fetch_task>(rel->offset(), count);
    auto* fetch_ptr      = add_task(std::move(fetch), rel, output_data_types, lowering_role::fetch);
    _qep_builder->add_successor(child.terminal, fetch_ptr);

    _result = {fetch_ptr, output_data_types};
  }
  void visit(physical::union_all_relation*) override { throw_unsupported("union_all"); }
  void visit(physical::user_defined_relation*) override { throw_unsupported("user_defined"); }
  void visit(physical::window_relation*) override { throw_unsupported("window"); }
  void visit(physical::gen_ident_col_relation*) override { throw_unsupported("gen_ident_col"); }
  void visit(physical::shuffle_relation*) override { throw_unsupported("shuffle"); }

 private:
  /**
   * @brief Register a task and all physical-lowering metadata associated with it.
   *
   * Every task created by this visitor must be registered through this wrapper so output types and
   * physical-relation provenance, and stable keys remain complete and keyed to the same task
   * identity as the QEP.
   *
   * @tparam Task Concrete type derived from `qep::task`.
   * @param[in] new_task Task whose ownership transfers to the QEP builder.
   * @param[in] physical_relation Physical relation responsible for the task.
   * @param[in] output_data_types Output column types in emission order.
   * @param[in] role Semantic role of the task within the physical relation's lowering.
   *
   * @return Non-owning pointer to the registered task.
   */
  template <typename Task>
  Task* add_task(std::unique_ptr<Task> new_task,
                 physical::relation const* physical_relation,
                 std::span<cudf::data_type const> output_data_types,
                 lowering_role role)
  {
    static_assert(std::is_base_of_v<task, Task>,
                  "qep_lowering_visitor::add_task requires Task to be derived from qep::task");
    GQE_EXPECTS(role != lowering_role::invalid,
                "physical_qep_transformation: task lowering role must not be invalid",
                std::logic_error);
    auto const node_id = _relation_node_ids.find(physical_relation);
    GQE_EXPECTS(node_id != _relation_node_ids.end(),
                "physical_qep_transformation: physical relation has no plan node ID",
                std::logic_error);
    stable_task_key stable_key{.relation_id = node_id->second, .role = role};

    auto const inserted = _assigned_stable_keys.insert(stable_key).second;
    GQE_EXPECTS(
      inserted, "physical_qep_transformation: duplicate stable task key", std::logic_error);

    auto* task_ptr = new_task.get();
    _qep_builder->add_task(std::move(new_task));
    _output_data_types->emplace(
      task_ptr, std::vector<cudf::data_type>(output_data_types.begin(), output_data_types.end()));
    _physical_relations->emplace(task_ptr, physical_relation);
    _stable_keys->emplace(task_ptr, stable_key);
    return task_ptr;
  }

  /**
   * @brief Append a `gather_task` at a boundary requiring an ordinary table.
   *
   * Used for materialized reader haystacks, storage-writer and merge-sort inputs, terminal Top-K
   * output, and the boundaries `materialize_if_early` adds. The operation is deliberately
   * unconditional because lowering does not track logical-column representations. `gather_task` is
   * zero-copy for ordinary and count-only input.
   *
   * @param[in] upstream Lowered logical-relation input.
   * @param[in] physical_relation Physical relation responsible for an inserted gather.
   *
   * @param[in] role Whether the boundary materializes relation input or output.
   *
   * @return An ordinary-table result ending at the inserted gather.
   */
  lowering_result gather_for_ordinary_boundary(
    lowering_result upstream,
    physical::relation const* physical_relation,
    lowering_role role = lowering_role::materialize_input)
  {
    auto* gather_ptr = add_task(
      std::make_unique<gather_task>(), physical_relation, upstream.output_data_types, role);
    _qep_builder->add_successor(upstream.terminal, gather_ptr);
    return {gather_ptr, std::move(upstream.output_data_types)};
  }

  /**
   * @brief Gather a masked or indirect result into an ordinary table when late materialization is
   * off.
   *
   * With `use_late_materialization` disabled, lowering calls this at every point that can
   * produce a valid mask or indirect columns:
   *
   * - a filter relation's output, after its projection, so dropped columns are not gathered;
   * - an aggregate's input condition;
   * - every Top-K output;
   * - every broadcast-join exit.
   *
   * The gather is zero-copy for already-ordinary input. A consumer that requires an ordinary table
   * still adds its own boundary gather, which is then zero-copy.
   *
   * @param[in] upstream Lowered result to materialize.
   * @param[in] physical_relation Physical relation responsible for the inserted gather.
   * @param[in] role Whether the gather materializes the relation's input or output.
   *
   * @return `upstream` when late materialization is enabled. Otherwise an ordinary-table result
   *   ending at the inserted gather.
   */
  lowering_result materialize_if_early(lowering_result upstream,
                                       physical::relation const* physical_relation,
                                       lowering_role role = lowering_role::materialize_output)
  {
    if (_parameters.use_late_materialization) { return upstream; }
    return gather_for_ordinary_boundary(std::move(upstream), physical_relation, role);
  }

  /**
   * @brief Append a `filter_task` evaluating `condition` on top of `upstream`.
   *
   * `filter_task` accepts logical relations, so filter chains compose directly without an
   * intervening `gather_task`.
   *
   * @param[in] upstream Lowered filter input, including its terminal task and output metadata.
   * @param[in] condition Owned filter condition expression.
   * @param[in] physical_relation Physical relation responsible for the filter.
   *
   * @return The terminal `filter_task` and its output types. Its output retains the logical data
   *   states and carries a relation valid mask.
   */
  lowering_result lower_into_filter(lowering_result upstream,
                                    std::unique_ptr<expression> condition,
                                    physical::relation const* physical_relation)
  {
    auto filter      = std::make_unique<filter_task>(std::move(condition));
    auto* filter_ptr = add_task(
      std::move(filter), physical_relation, upstream.output_data_types, lowering_role::filter);
    _qep_builder->add_successor(upstream.terminal, filter_ptr);

    return {filter_ptr, std::move(upstream.output_data_types)};
  }

  /**
   * @brief Append a `project_task` evaluating `exprs` on top of `upstream`.
   *
   * `project_task` accepts logical relations (see `project.hpp`), so no gather is inserted.
   *
   * @param[in] upstream The upstream lowering result feeding the project.
   * @param[in] exprs Owned output expressions for the project task.
   * @param[in] output_data_types Project output column types in emission order.
   * @param[in] physical_relation Physical relation responsible for the project.
   *
   * @return The terminal `project_task` and its output types.
   */
  lowering_result lower_into_project(lowering_result upstream,
                                     std::vector<std::unique_ptr<expression>> exprs,
                                     std::vector<cudf::data_type> output_data_types,
                                     physical::relation const* physical_relation)
  {
    auto project = std::make_unique<project_task>(std::move(exprs));
    auto* project_ptr =
      add_task(std::move(project), physical_relation, output_data_types, lowering_role::project);
    _qep_builder->add_successor(upstream.terminal, project_ptr);

    return {project_ptr, std::move(output_data_types)};
  }

  /**
   * @brief Append a `project_task` that selects a subset of upstream columns by index.
   *
   * Synthesizes a `column_reference_expression` per index and delegates to the
   * expression-based `lower_into_project`. Many physical relations carry an index-based
   * column remap (e.g. `filter_relation::projection_indices`) that maps to this pattern.
   *
   * @param[in] upstream The upstream lowering result feeding the project.
   * @param[in] column_indices Column indices into the upstream's column layout, in output
   *   order.
   * @param[in] output_data_types Project output column types in emission order.
   * @param[in] physical_relation Physical relation responsible for the project.
   *
   * @return The terminal `project_task`; an input valid mask is passed through to the output,
   *   so the output shape matches the input's.
   */
  lowering_result lower_into_project(lowering_result upstream,
                                     std::span<cudf::size_type const> column_indices,
                                     std::vector<cudf::data_type> output_data_types,
                                     physical::relation const* physical_relation)
  {
    std::vector<std::unique_ptr<expression>> exprs;
    exprs.reserve(column_indices.size());

    for (auto idx : column_indices) {
      exprs.push_back(std::make_unique<column_reference_expression>(idx));
    }

    return lower_into_project(
      std::move(upstream), std::move(exprs), std::move(output_data_types), physical_relation);
  }

  storage::table_provider const* _table_provider;  ///< Non-owning. Resolves table names to
                                                   ///< storage tables for `read_relation`.
  optimization_parameters const& _parameters;      ///< Runtime optimization gates.
  query_execution_plan_builder* _qep_builder;      ///< Non-owning. Collects lowered tasks and
                                                   ///< their dataflow edges.
  task_output_data_types* _output_data_types;      ///< Non-owning. Output types per lowered task.
  task_physical_relations* _physical_relations;    ///< Non-owning. Physical origin per task.
  std::unordered_map<physical::relation*, lowering_result>
    _cache;  ///< Subplan deduplication: each `physical::relation*` is lowered at most once;
             ///< repeated occurrences reuse the cached result. Written only by `lower`.
  lowering_result _result;  ///< Result published by the most recent `visit(...)`; consumed by
                            ///< `lower` immediately after `accept` returns.
  task_stable_keys* _stable_keys;  ///< Non-owning. Task to stable task key map.
  std::unordered_set<stable_task_key, stable_task_key_hash>
    _assigned_stable_keys;  ///< Stable keys already assigned by this visitor.
  physical::plan_node_ids const& _relation_node_ids;  ///< Physical relation to node ID lookup.
};

}  // namespace

physical_qep_transformation_result physical_qep_transformation(
  physical::identified_plan const& physical_plan,
  storage::table_provider const* table_provider,
  optimization_parameters const& parameters)
{
  auto root = physical_plan.root().get();
  GQE_EXPECTS(
    root != nullptr, "physical_qep_transformation: root must not be null", std::invalid_argument);
  GQE_EXPECTS(table_provider != nullptr,
              "physical_qep_transformation: table_provider must not be null",
              std::invalid_argument);

  query_execution_plan_builder qep_bld;
  task_output_data_types output_data_types;
  task_stable_keys stable_keys;
  task_physical_relations physical_relations;

  qep_lowering_visitor visitor(table_provider,
                               parameters,
                               &qep_bld,
                               &output_data_types,
                               &physical_relations,
                               &stable_keys,
                               physical_plan.node_ids());
  visitor.lower_root(root);

  auto plan = qep_bld.build();
  GQE_EXPECTS(stable_keys.size() == plan.tasks().size(),
              "physical_qep_transformation: stable task key count differs from QEP task count",
              std::logic_error);
  for (auto* task : plan.tasks()) {
    GQE_EXPECTS(stable_keys.contains(task),
                "physical_qep_transformation: QEP task has no stable task key",
                std::logic_error);
  }
  return physical_qep_transformation_result{.plan               = std::move(plan),
                                            .output_data_types  = std::move(output_data_types),
                                            .physical_relations = std::move(physical_relations),
                                            .stable_keys        = std::move(stable_keys)};
}

}  // namespace qep
}  // namespace gqe
