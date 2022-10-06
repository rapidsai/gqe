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

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/logical/relation.hpp>

#include <substrait/algebra.pb.h>
#include <substrait/plan.pb.h>

#include <string>
#include <vector>

namespace gqe {

class substrait_parser {
 public:
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
   * @return The parsed expression
   */
  std::unique_ptr<gqe::expression> parse_expression(substrait::Expression const& expression) const;

  /**
   * @brief Parse substrait Relation message into gqe::logical::relation
   *
   * @param relation Substrait relation
   * @return The parsed relation
   */
  std::unique_ptr<gqe::logical::relation> parse_relation(substrait::Rel const& relation) const;

  /**
   * @brief Store information about table `table_name` and its columns and types in
   * `input_column_types`
   *
   * @param table_name Name of the table to register
   * @param column_names List of column names in table `table_name`
   * @param column_types List of cudf data types corresponding to each column in `column_names`
   * @param file_paths Path to file containing data to populate table `table_name`
   *
   * @note The column order must match when the plan is generated
   */
  void register_input_table(std::string table_name,
                            std::vector<std::string> const& column_names,
                            std::vector<cudf::data_type> const& column_types,
                            std::vector<std::string> const& file_paths);

 private:
  /**
   * @brief Storing function references and their names into `function_reference_to_name` map
   *
   * @param reference Substrait function anchor id
   * @param function_name Substrait function name
   */
  void add_function_reference(uint32_t reference, std::string function_name);

  /**
   * @brief Parse Substrait Literal expression message into `gqe::literal_expression`
   *
   * @param literal_expression Substrait literal expression
   * @return The parsed lireral expression
   */
  std::unique_ptr<expression> parse_literal_expression(
    substrait::Expression_Literal const& literal_expression) const;

  /**
   * @brief Parse Substrait ScalarFunction expression message into `gqe::scalar_function_expression`
   *
   * @param selection_expression Substrait scalar function expression
   * @return The parsed ScalarFunction expression
   */
  std::unique_ptr<gqe::expression> parse_scalar_function_expression(
    substrait::Expression_ScalarFunction const& selection_expression) const;

  /**
   * @brief Parse Substrait FieldReference expression message into
   * `gqe::column_reference_expression`
   *
   * @param selection_expression Substrait field reference expression
   * @return The parsed FieldReference (select) expression
   */
  std::unique_ptr<gqe::expression> parse_selection_expression(
    substrait::Expression_FieldReference const& selection_expression) const;
  // TODO: Add more expressions

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

  // table_name => [column_name => [column_type]]
  std::unordered_map<std::string, std::unordered_map<std::string, cudf::data_type>>
    input_column_types;
  std::unordered_map<uint32_t, std::string> function_reference_to_name;
};
}  // namespace gqe
