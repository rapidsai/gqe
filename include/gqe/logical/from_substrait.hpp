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

#pragma once

#include <gqe/catalog.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/logical/relation.hpp>

#include <cudf/types.hpp>

#include <substrait/algebra.pb.h>
#include <substrait/plan.pb.h>

#include <memory>
#include <string>
#include <vector>

namespace gqe {

struct ddl_command {
  enum class operation { create_table, create_or_replace_table, drop_table, drop_table_if_exists };
  operation op;
  std::string table_name;
  std::vector<std::string> column_names;
  std::vector<cudf::data_type> column_types;
  // All UNIQUE / PRIMARY KEY constraints as key-sets (column index lists).
  // Size 1 = single-column, size >= 2 = composite. Indices are into column_names / column_types.
  std::vector<std::vector<std::size_t>> unique_keys;
  // Storage backend for the new table, parsed from the SQL WITH (...) clause.
  // Defaults to boost_shared_memory when no WITH clause is present.
  storage_kind::type storage = storage_kind::boost_shared_memory{};
};

struct write_command {
  std::string table_name;
  std::string file_path;
  std::vector<std::string> column_names;
  std::vector<cudf::data_type> column_types;
};

class substrait_parser {
 public:
  /**
   * @brief Construct a new substrait parser.
   *
   * @param tables_catalog Catalog containing input table metadata.
   */
  substrait_parser(catalog* tables_catalog) : _catalog(tables_catalog) {}

  /**
   * @brief Parse substrait binary file into logical plan
   *
   * @param substrait_file Substrait binary file path
   * @return A `std::vector` of `std::shared_ptr` to `gqe::logical::relation`s corresponding to
   * substrait plan in the input file. Each relation in the vector corresponds to each relation
   * tree associated with this plan
   */
  std::vector<std::shared_ptr<gqe::logical::relation>> from_file(std::string substrait_file);

  /**
   * @brief Parse a Substrait plan from a binary protobuf buffer.
   *
   * @param data Pointer to the serialized protobuf bytes.
   * @param size Number of bytes.
   * @return A `std::vector` of `std::shared_ptr` to `gqe::logical::relation`s corresponding to
   * the substrait plan. Each relation in the vector corresponds to each relation tree associated
   * with this plan.
   */
  std::vector<std::shared_ptr<gqe::logical::relation>> from_binary(const void* data,
                                                                   std::size_t size);

  /**
   * @brief Parse substrait Expression message into gqe::expression.
   *
   * This function calls appropriate `parse_[expression_type]()` by check which `rex_type` field
   * is set.
   *
   * @param expression Substrait expression
   * @param subquery_relations Vector reference to store subquery relations
   * @return The parsed expression
   */
  std::unique_ptr<gqe::expression> parse_expression(
    substrait::Expression const& expression,
    std::vector<std::shared_ptr<gqe::logical::relation>>& subquery_relations) const;

  /**
   * @brief Parse substrait Relation message into gqe::logical::relation
   *
   * @param relation Substrait relation
   * @return The parsed relation
   */
  std::unique_ptr<gqe::logical::relation> parse_relation(substrait::Rel const& relation) const;

  /**
   * @brief Parse a Substrait DdlRel into a ddl_command.
   */
  ddl_command parse_ddl_command(substrait::DdlRel const& ddl) const;

  /**
   * @brief Parse a Substrait WriteRel into a write_command.
   */
  write_command parse_write_command(substrait::WriteRel const& write) const;

 private:
  /**
   * @brief Shared logic for processing a deserialized Substrait Plan.
   *
   * Registers function references and parses relation trees.
   * DDL relations are executed directly (table registered/unregistered in catalog).
   * Write relations produce a read->write logical plan.
   *
   * @param query_plan The deserialized Substrait plan.
   * @return A vector of parsed relation trees (empty for DDL-only plans).
   */
  std::vector<std::shared_ptr<gqe::logical::relation>> from_plan(substrait::Plan& query_plan);

  void add_function_reference(uint32_t reference, std::string function_name);

  std::string get_function_name(uint32_t reference) const;

  /**
   * @brief Parse Substrait Cast expression message into `gqe::cast_expression`
   *
   * @param cast_expression Substrait Cast expression
   * @param subquery_relations Vector reference to store subquery relations
   * @return The parsed cast expression
   */
  std::unique_ptr<gqe::expression> parse_cast_expression(
    substrait::Expression_Cast const& cast_expression,
    std::vector<std::shared_ptr<gqe::logical::relation>>& subquery_relations) const;

  /**
   * @brief Parse Substrait IfThen expression message into `gqe::if_then_else_expression`
   *
   * @param if_then_expression Substrait IfThen expression
   * @param subquery_relations Vector reference to store subquery relations
   * @return The parsed if-then-else expression
   */
  std::unique_ptr<expression> parse_if_then_expression(
    substrait::Expression_IfThen const& if_then_expression,
    std::vector<std::shared_ptr<gqe::logical::relation>>& subquery_relations) const;

  /**
   * @brief Parse Substrait SingularOrList expression message into gqe expression
   *
   * @param singular_or_list Substrait SingularOrList expression
   * @param subquery_relations reference to store subquery relations
   * @return The parsed expression consisting of `logical_or_expression`s and `equal_expression`s
   */
  std::unique_ptr<expression> parse_in_list_expression(
    substrait::Expression_SingularOrList const& singular_or_list,
    std::vector<std::shared_ptr<gqe::logical::relation>>& subquery_relations) const;

  /**
   * @brief Parse Substrait Literal expression message into `gqe::literal_expression`
   *
   * @param literal_expression Substrait literal expression
   * @return The parsed lireral expression
   */
  std::unique_ptr<expression> parse_literal_expression(
    substrait::Expression_Literal const& literal_expression) const;

  /**
   * @brief Parse binary or multi-arg ScalarFunction expression from the specified function name and
   * list of Substrait arguments
   *
   * @param function_name Name of function to parse
   * @param arg_expressions List of argument expressions
   * @param subquery_relations Vector reference to store subquery relations
   * @return std::unique_ptr<gqe::expression>
   */
  std::unique_ptr<gqe::expression> _parse_scalar_function_expression(
    std::string function_name,
    std::vector<substrait::Expression> const& arg_expressions,
    std::vector<std::shared_ptr<gqe::logical::relation>>& subquery_relations) const;

  /**
   * @brief Parse Substrait ScalarFunction expression message into `gqe::scalar_function_expression`
   *
   * @param selection_expression Substrait scalar function expression
   * @param subquery_relations Vector reference to store subquery relations
   * @return The parsed ScalarFunction expression
   */
  std::unique_ptr<gqe::expression> parse_scalar_function_expression(
    substrait::Expression_ScalarFunction const& selection_expression,
    std::vector<std::shared_ptr<gqe::logical::relation>>& subquery_relations) const;

  /**
   * @brief Parse Substrait FieldReference expression message into
   * `gqe::column_reference_expression`
   *
   * @param selection_expression Substrait field reference expression
   * @return The parsed FieldReference (select) expression
   */
  std::unique_ptr<gqe::expression> parse_selection_expression(
    substrait::Expression_FieldReference const& selection_expression) const;

  /**
   * @brief Parse Substrait Subquery expression message into `gqe::subquery_expression`
   *
   * @param subquery_expression Substrait subquery expression
   * @param subquery_relations Vector reference to store subquery relations
   * @return The parsed Subquery expression
   */
  std::unique_ptr<gqe::expression> parse_subquery_expression(
    substrait::Expression_Subquery const& subquery_expression,
    std::vector<std::shared_ptr<gqe::logical::relation>>& subquery_relations) const;

  /**
   * @brief Parse Substrait SortFields into `expressions`, `columns_orders` and `null_precedences`
   *
   * @param sorts List of Substrait SortFields
   * @param expressions Parsed sort expressions
   * @param column_orders Parsed sort directions for all sort fields
   * @param null_precedences Parsed null orders for all sort fields
   * @param subquery_relations Vector reference to store subquery relations
   */
  void parse_sorts(google::protobuf::RepeatedPtrField<substrait::SortField> const& sorts,
                   std::vector<std::unique_ptr<gqe::expression>>& expressions,
                   std::vector<cudf::order>& column_orders,
                   std::vector<cudf::null_order>& null_precedences,
                   std::vector<std::shared_ptr<gqe::logical::relation>>& subquery_relations) const;

  /**
   * @brief Parse Substrait WindowFunction expression into gqe window function relation
   *
   * @param window_function_expression Substrait WindowFunction to parse
   * @param input_relation Input relation for the parsed window function relation
   * @param subquery_relations Vector reference to store subquery relations
   * @return The parsed window function relation
   */
  std::unique_ptr<gqe::logical::relation> parse_window_function_expression(
    substrait::Expression::WindowFunction const& window_function_expression,
    std::unique_ptr<gqe::logical::relation> input_relation,
    std::vector<std::shared_ptr<gqe::logical::relation>>& subquery_relations) const;

  // TODO: Add more expressions

  /**
   * @brief Parse Substrait AggregateFunction message
   *
   * This function turns the Substrait AggregateFunction into a pair of cudf
   * aggregation operator kind and the value expression to apply the function to
   *
   * @param aggregate_function Substrait aggregate function
   * @param subquery_relations Vector reference to store subquery relations
   * @return The parsed aggregate function
   */
  std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>> parse_aggregate_function(
    substrait::AggregateFunction const& aggregate_function,
    std::vector<std::shared_ptr<gqe::logical::relation>>& subquery_relations) const;

  /**
   * @brief Parse Substrait Aggregate relation into gqe::logical::aggregate_relation
   *
   * @param aggregate_relation Substrait aggregate relation
   * @return The parsed aggregate relation
   */
  std::unique_ptr<gqe::logical::relation> parse_aggregate_relation(
    substrait::AggregateRel const& aggregate_relation) const;

  /**
   * @brief Parse Substrait Fetch Relation into gqe::logical::fetch_relation
   *
   * @param fetch_relation Substrait fetch relation
   * @return The parsed fetch relation
   */
  std::unique_ptr<gqe::logical::relation> parse_fetch_relation(
    substrait::FetchRel const& fetch_relation) const;

  /**
   * @brief Parse Substrait Sort Relation into gqe::logical::sort_relation
   *
   * @param sort_relation Substrait sort relation
   * @return The parsed sort relation
   */
  std::unique_ptr<gqe::logical::relation> parse_sort_relation(
    substrait::SortRel const& sort_relation) const;

  /**
   * @brief Parse Substrait Filter Relation into gqe::logical::filter_relation
   *
   * @param filter_relation Substrait filter relation
   * @return The parsed filter relation
   */
  std::unique_ptr<gqe::logical::relation> parse_filter_relation(
    substrait::FilterRel const& filter_relation) const;

  /**
   * @brief Parse Substrait Join Relation into gqe::logical::join_relation
   *
   * @param join_relation Substrait join relation
   * @return The parsed join relation
   */
  std::unique_ptr<gqe::logical::relation> parse_join_relation(
    substrait::JoinRel const& join_relation) const;

  /**
   * @brief Parse Substrait Project Relation into gqe::logical::project_relation
   *
   * @param project_relation Substrait project relation
   * @return The parsed project relation
   */
  std::unique_ptr<gqe::logical::relation> parse_project_relation(
    substrait::ProjectRel const& project_relation) const;

  /**
   * @brief Parse Substrait Read relation
   *        - ReadRel should not contain any input Rel.
   *        - The only currently used field is the `NamedStruct base_schema`
   *
   * @param read_relation Substrait relation of type ReadRel
   * @return The parsed read logical relation
   */
  std::unique_ptr<gqe::logical::relation> parse_read_relation(
    substrait::ReadRel const& read_relation) const;

  /**
   * @brief Parse Substrait Set relation into gqe::logical::set_relation
   *
   * @param set_relation  Substrait set relation
   * @return The parsed set relation
   */
  std::unique_ptr<gqe::logical::relation> parse_set_relation(
    substrait::SetRel const& set_relation) const;

  catalog* _catalog;
  std::unordered_map<uint32_t, std::string> function_reference_to_name;
};
}  // namespace gqe
