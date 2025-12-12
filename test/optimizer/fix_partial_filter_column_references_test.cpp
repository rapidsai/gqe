/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/json_formatter.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/optimizer/logical_optimization.hpp>
#include <gqe/optimizer/optimization_configuration.hpp>
#include <gtest/gtest.h>

// This fixture contains convenience methods to:
// - Register a table in the catalog
// - Create a read relation with a base table schema and a filter condition on a specific column
// - Create a filter relation with a filter condition on a specific column
// - Print the relevant information of the read relations and filter relations for debugging
// The constructor also creates and configures the optimizer that can be used in the tests.
class FixPartialFilterColumnReferencesTest : public testing::Test {
 protected:
  FixPartialFilterColumnReferencesTest()
  {
    _catalog = std::make_unique<gqe::catalog>();
    std::vector<gqe::optimizer::logical_optimization_rule_type> on_rules = {
      gqe::optimizer::logical_optimization_rule_type::fix_partial_filter_column_references};
    std::vector<gqe::optimizer::logical_optimization_rule_type> off_rules = {};
    _optimization_configuration =
      std::make_unique<gqe::optimizer::optimization_configuration>(on_rules, off_rules);
    optimizer = std::make_unique<gqe::optimizer::logical_optimizer>(
      _optimization_configuration.get(), _catalog.get());
  }

  void register_table(const std::vector<std::string>& columns_names)
  {
    std::vector<gqe::column_traits> column_traits;
    std::transform(columns_names.begin(),
                   columns_names.end(),
                   std::back_inserter(column_traits),
                   [](const std::string& column_name) -> gqe::column_traits {
                     return {column_name, cudf::data_type(cudf::type_id::INT8)};
                   });
    _catalog->register_table(_table_name,
                             column_traits,
                             gqe::storage_kind::system_memory{},
                             gqe::partitioning_schema_kind::none{});
  }

  std::shared_ptr<gqe::logical::relation> create_read_relation(
    const std::vector<std::string>& column_names, cudf::size_type partial_filter_column_index) const
  {
    std::vector<std::shared_ptr<gqe::logical::relation>> subquery_relations;
    std::vector<cudf::data_type> column_types;
    std::transform(
      column_names.begin(),
      column_names.end(),
      std::back_inserter(column_types),
      [](std::string_view column_name) { return cudf::data_type(cudf::type_id::INT8); });
    std::unique_ptr<gqe::expression> partial_filter =
      create_filter_condition(partial_filter_column_index);
    const auto read_relation = std::make_shared<gqe::logical::read_relation>(
      subquery_relations, column_names, column_types, _table_name, std::move(partial_filter));
    return read_relation;
  }

  std::shared_ptr<gqe::logical::relation> create_filter_relation(
    std::shared_ptr<gqe::logical::relation> child,
    cudf::size_type filter_condition_column_index) const
  {
    std::vector<std::shared_ptr<gqe::logical::relation>> subquery_relations;
    std::vector<cudf::size_type> projection_indices = {0};
    std::unique_ptr<gqe::expression> condition =
      create_filter_condition(filter_condition_column_index);
    const auto filter_relation = std::make_shared<gqe::logical::filter_relation>(
      child, subquery_relations, std::move(condition), projection_indices);
    return filter_relation;
  }

  static std::unique_ptr<gqe::expression> create_filter_condition(
    const cudf::size_type column_index)
  {
    constexpr int8_t some_value = 42;
    return std::make_unique<gqe::less_expression>(
      std::make_shared<gqe::column_reference_expression>(column_index),
      std::make_shared<gqe::literal_expression<int8_t>>(some_value));
  }

  void print_relation(const std::string_view prefix,
                      gqe::logical::relation* relation,
                      int indent = 0) const
  {
    auto indent_helper = [&indent]() {
      for (int i = 0; i < indent; ++i)
        std::cout << ' ';
    };
    auto vector_to_string = [](const std::vector<std::string>& elements) {
      std::stringstream ss;
      if (!elements.empty()) { ss << elements.front(); }
      std::for_each(elements.begin() + 1, elements.end(), [&ss](const std::string& element) {
        ss << ", " << element;
      });
      return ss.str();
    };
    switch (relation->type()) {
      case gqe::logical::relation::relation_type::read: {
        const auto read_relation = static_cast<const gqe::logical::read_relation*>(relation);
        indent_helper();
        std::cout << "- " << prefix
                  << ": read: column_names = " << vector_to_string(read_relation->column_names())
                  << "; base table = "
                  << vector_to_string(_catalog->column_names(read_relation->table_name())) << "; "
                  << gqe::expression_json_formatter::to_json(
                       *read_relation->partial_filter_unsafe())
                  << std::endl;
      } break;
      case gqe::logical::relation::relation_type::filter: {
        const auto filter_relation = static_cast<const gqe::logical::filter_relation*>(relation);
        indent_helper();
        std::cout << prefix << ": filter: "
                  << gqe::expression_json_formatter::to_json(*filter_relation->condition())
                  << std::endl;
      } break;
      default: std::cout << prefix << ": other relation" << std::endl;
    }
    for (const auto child : relation->children_unsafe()) {
      print_relation(prefix, child, indent + 2);
    }
  }

 private:
  const std::string _table_name = "test_table";
  std::unique_ptr<gqe::catalog> _catalog;
  std::unique_ptr<gqe::optimizer::optimization_configuration> _optimization_configuration;

 protected:
  std::unique_ptr<gqe::optimizer::logical_optimizer> optimizer;
};

// The list of projected columns of the read relation is the same as the list of registered columns
// of the table. Therefore, the of the read relation partial filter should use the same column
// references as the filter relation condition.
TEST_F(FixPartialFilterColumnReferencesTest, projectedColumnsMatchBaseTableColumns)
{
  // The table is registered as a table with columns a, b, c.
  std::vector<std::string> columns = {"a", "b", "c"};
  register_table(columns);

  // The read relation projects the columns a, b, c and has a partial filter that references an
  // unknown column index, i.e., 10.
  cudf::size_type unknown_column_index = 10;
  auto input_read_relation             = create_read_relation(columns, unknown_column_index);

  // The filter relation condition references column 1 (= column b) from the columns projected by
  // the read relation
  cudf::size_type filter_condition_column_index = 1;
  auto input_filter_relation =
    create_filter_relation(input_read_relation, filter_condition_column_index);

  // Expected plan: the column reference in the read relation partial filter references the same
  // column as the filter relation condition.
  auto expected_read_relation = create_read_relation(columns, filter_condition_column_index);
  auto expected_filter_relation =
    create_filter_relation(expected_read_relation, filter_condition_column_index);

  // Run the optimizer
  auto optimized_relation = optimizer->optimize(input_filter_relation);
  print_relation("input", input_filter_relation.get());
  print_relation("expected", expected_filter_relation.get());
  print_relation("actual", optimized_relation.get());
  EXPECT_EQ(*optimized_relation, *expected_filter_relation);
}

// The list of projected columns of the read relation is a permutation of the list of registered
// columns of the table. The read relation partial filter and the filter relation condition should
// reference the same column. However, filter relation condition references the column based on the
// order of the list of projected columns by the read relation, whereas the read relation partial
// filter references the column based on the base table.
TEST_F(FixPartialFilterColumnReferencesTest, projectedColumnsAreAPermutationOfBaseTableColumns)
{
  // The table is registered a table with columns a, b, c.
  std::vector<std::string> base_table_columns = {"a", "b", "c"};
  register_table(base_table_columns);

  // The read relation projects the columns in a different order b, c, a and has a partial filter
  // that references an unknown column index, i.e., 10.
  std::vector<std::string> read_projected_columns = {"c", "a", "b"};
  cudf::size_type unknown_column_index            = 10;
  auto input_read_relation = create_read_relation(read_projected_columns, unknown_column_index);

  // The filter relation condition references column 2 (= column b) from the columns projected by
  // the read relation
  cudf::size_type filter_condition_column_index = 2;
  auto input_filter_relation =
    create_filter_relation(input_read_relation, filter_condition_column_index);

  // Expected plan: the column reference in the read relation partial filter now references column 1
  // (= column b) from the base table schema.
  cudf::size_type replaced_partial_filter_column_index = 1;
  auto expected_read_relation =
    create_read_relation(read_projected_columns, replaced_partial_filter_column_index);
  auto expected_filter_relation =
    create_filter_relation(expected_read_relation, filter_condition_column_index);

  // Run the optimizer
  auto optimized_relation = optimizer->optimize(input_filter_relation);
  print_relation("input", input_filter_relation.get());
  print_relation("expected", expected_filter_relation.get());
  print_relation("actual", optimized_relation.get());
  EXPECT_EQ(*optimized_relation, *expected_filter_relation);
}

// The base table columns and the columns projected by the read relation differ, i.e., the table is
// not projected, i.e.,  The partial filter is unchanged.
TEST_F(FixPartialFilterColumnReferencesTest, unprojectedTable)
{
  // The table is registered as a *unprojected* table with columns a, b, c.
  std::vector<std::string> base_table_columns = {"a", "b", "c"};
  register_table(base_table_columns);

  // The read relation projects a subset of the base table columns and has a partial filter
  // that references column 1 (= column b) from the base table schema.
  std::vector<std::string> read_projected_columns = {"b", "c"};
  cudf::size_type partial_filter_column_index     = 1;
  auto input_read_relation =
    create_read_relation(read_projected_columns, partial_filter_column_index);

  // The filter relation condition references column 0 (= column b) from the columns projected by
  // the read relation. This is the same column as in the partial filter because column a was
  // projected away.
  cudf::size_type filter_condition_column_index = 0;
  auto input_filter_relation =
    create_filter_relation(input_read_relation, filter_condition_column_index);

  // Expected plan: no changes, same plan as above
  auto expected_read_relation =
    create_read_relation(read_projected_columns, partial_filter_column_index);
  auto expected_filter_relation =
    create_filter_relation(expected_read_relation, filter_condition_column_index);

  // Run the optimizer
  auto optimized_relation = optimizer->optimize(input_filter_relation);
  print_relation("input", input_filter_relation.get());
  print_relation("expected", expected_filter_relation.get());
  print_relation("actual", optimized_relation.get());
  EXPECT_EQ(*optimized_relation, *expected_filter_relation);
}

// The list of projected columns of the read relation is a permutation of the list of registered
// columns of the table. There are two filter relations immediately after the read relation. The
// read relation column references should be replaced with the filter relation that is the direct
// parent of the read relation.
TEST_F(FixPartialFilterColumnReferencesTest, useImmediateParentFilterToReplaceColumnReferences)
{
  // The table is registered as a projected table with columns a, b, c.
  std::vector<std::string> base_table_columns = {"a", "b", "c"};
  register_table(base_table_columns);

  // The read relation projects the columns in a different order b, c, a and has a partial filter
  // that references an unknown column index, i.e., 10.
  std::vector<std::string> read_projected_columns = {"c", "a", "b"};
  cudf::size_type unknown_column_index            = 10;
  auto input_read_relation = create_read_relation(read_projected_columns, unknown_column_index);

  // The filter relation condition references column 2 (= column b) from the columns projected by
  // the read relation
  cudf::size_type filter_condition_column_index_1 = 2;
  auto input_filter_relation_1 =
    create_filter_relation(input_read_relation, filter_condition_column_index_1);

  // The second filter relation references column 0.
  cudf::size_type filter_condition_column_index_2 = 0;
  auto input_filter_relation_2 =
    create_filter_relation(input_filter_relation_1, filter_condition_column_index_2);

  // Expected plan: the column reference in the read relation partial filter now references column 1
  // (= column b) from the base table schema.
  cudf::size_type replaced_partial_filter_column_index = 1;
  auto expected_read_relation =
    create_read_relation(read_projected_columns, replaced_partial_filter_column_index);
  auto expected_filter_relation_1 =
    create_filter_relation(expected_read_relation, filter_condition_column_index_1);
  auto expected_filter_relation_2 =
    create_filter_relation(expected_filter_relation_1, filter_condition_column_index_2);

  // Run the optimizer
  auto optimized_relation = optimizer->optimize(input_filter_relation_2);
  print_relation("input", input_filter_relation_2.get());
  print_relation("expected", expected_filter_relation_2.get());
  print_relation("actual", optimized_relation.get());
  EXPECT_EQ(*optimized_relation, *expected_filter_relation_2);
}

// The filter relation is not an immediate parent of the read relation. Do nothing.
TEST_F(FixPartialFilterColumnReferencesTest, interveningRelation)
{
  // Helper to create a project relation
  auto create_project_relation = [](std::shared_ptr<gqe::logical::relation> child) {
    std::vector<std::shared_ptr<gqe::logical::relation>> subquery_relations;
    std::vector<std::unique_ptr<gqe::expression>> output_expressions;
    // Need at least one projected column here, otherwise filter_relation::data_types() segfaults
    output_expressions.push_back(create_filter_condition(0));
    auto project_relation = std::make_shared<gqe::logical::project_relation>(
      child, subquery_relations, std::move(output_expressions));
    return project_relation;
  };

  // The table is registered as a projected table with columns a, b, c.
  std::vector<std::string> base_table_columns = {"a", "b", "c"};
  register_table(base_table_columns);

  // The read relation projects the columns in a different order b, c, a and has a partial filter
  // that references an unknown column index, i.e., 10.
  std::vector<std::string> read_projected_columns = {"c", "a", "b"};
  cudf::size_type unknown_column_index            = 10;
  auto input_read_relation = create_read_relation(read_projected_columns, unknown_column_index);

  // An project relation is between the read relation and the filter
  auto input_project_relation = create_project_relation(input_read_relation);

  // The filter relation condition references column 0 (= column c) from the columns projected by
  // the project relation.
  cudf::size_type filter_condition_column_index = 2;
  auto input_filter_relation =
    create_filter_relation(input_project_relation, filter_condition_column_index);

  // Expected plan: unchanged
  auto expected_read_relation = create_read_relation(read_projected_columns, unknown_column_index);
  auto expected_project_relation = create_project_relation(expected_read_relation);
  auto expected_filter_relation =
    create_filter_relation(expected_project_relation, filter_condition_column_index);

  // Run the optimizer
  auto optimized_relation = optimizer->optimize(input_filter_relation);
  print_relation("input", input_filter_relation.get());
  print_relation("expected", expected_filter_relation.get());
  print_relation("actual", optimized_relation.get());
  EXPECT_EQ(*optimized_relation, *expected_filter_relation);
}
