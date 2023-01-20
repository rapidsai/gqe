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

#pragma once

#include <gqe/catalog.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/logical/relation.hpp>

#include <substrait/algebra.pb.h>
#include <substrait/plan.pb.h>

#include <vector>

namespace gqe {

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

 private:
  /**
   * @brief Storing function references and their names into `function_reference_to_name` map
   *
   * @param reference Substrait function anchor id
   * @param function_name Substrait function name
   */
  void add_function_reference(uint32_t reference, std::string function_name);

  std::string get_function_name(uint32_t reference) const;

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

  catalog* _catalog;
  std::unordered_map<uint32_t, std::string> function_reference_to_name;
};
}  // namespace gqe
