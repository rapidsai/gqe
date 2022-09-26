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

#include <gqe/expression/expression.hpp>
#include <gqe/logical/relation.hpp>
#include <substrait/plan.pb.h>

#include <memory>
#include <string>
#include <unordered_map>
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
   * @brief Parse substrait Expression message into gqe::expression
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
   * `input_column_types`. This is used for validating table and column references.
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
   * @brief Parse Substrait FieldReference expression message into gqe::column_reference_expression
   *
   * @param selection_expression Substrait field reference expression
   * @return The parsed FieldReference (select) expression
   */
  std::unique_ptr<gqe::expression> parse_selection_expression(
    substrait::Expression_FieldReference const& selection_expression) const;
  // TODO: Add more expressions

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
};
}  // namespace gqe