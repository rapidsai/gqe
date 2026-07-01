/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/optimizer/rules/uniqueness_propagation.hpp>

#include "../utility.hpp"

#include <gqe/catalog.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/fetch.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/relation.hpp>
#include <gqe/logical/set.hpp>
#include <gqe/logical/sort.hpp>
#include <gqe/logical/window.hpp>
#include <gqe/logical/write.hpp>
#include <gqe/optimizer/logical_optimization.hpp>
#include <gqe/optimizer/relation_properties.hpp>
#include <gqe/optimizer/relation_traits.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/base_fixture.hpp>
#include <gtest/gtest.h>

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <cassert>
#include <memory>
#include <vector>

class UniquenessPropagationTest : public gqe::test::BaseFixture {
 protected:
  void initialize_optimizer()
  {
    gqe::optimizer::optimization_configuration rule_config(
      {gqe::optimizer::logical_optimization_rule_type::uniqueness_propagation}, {});
    catalog.register_table("test_table1",
                           {{"t1_c1_unique", cudf::data_type(cudf::type_id::INT64)},
                            {"t1_c2", cudf::data_type(cudf::type_id::INT64)},
                            {"t1_c3", cudf::data_type(cudf::type_id::INT32)},
                            {"t1_c4_unique", cudf::data_type(cudf::type_id::INT64)}},
                           gqe::storage_kind::parquet_file{},
                           gqe::partitioning_schema_kind::automatic{},
                           {{"t1_c1_unique"}, {"t1_c4_unique"}});

    catalog.register_table("test_table2",
                           {{"t2_c1_unique", cudf::data_type(cudf::type_id::INT64)},
                            {"t2_c2_unique", cudf::data_type(cudf::type_id::INT32)}},
                           gqe::storage_kind::parquet_file{},
                           gqe::partitioning_schema_kind::automatic{},
                           {{"t2_c1_unique"}, {"t2_c2_unique"}});

    catalog.register_table("test_table_composite",
                           {{"tc_c1", cudf::data_type(cudf::type_id::INT64)},
                            {"tc_c2", cudf::data_type(cudf::type_id::INT64)},
                            {"tc_c3", cudf::data_type(cudf::type_id::INT64)}},
                           gqe::storage_kind::parquet_file{},
                           gqe::partitioning_schema_kind::automatic{},
                           {{"tc_c1", "tc_c2"}});  // COMPOSITE PRIMARY KEY (tc_c1, tc_c2)

    optimizer = std::make_unique<gqe::optimizer::logical_optimizer>(&rule_config, &catalog);
  }

  std::vector<std::shared_ptr<gqe::logical::relation>> empty_relations() { return {}; }

  std::unique_ptr<gqe::logical::relation> construct_read_one_unique(bool optimized)
  {
    // Hand coded logical plan for testing
    std::vector<std::string> column_names = {"t1_c1_unique", "t1_c3"};
    auto column_types                     = {cudf::data_type(cudf::type_id::INT64),
                                             cudf::data_type(cudf::type_id::INT32)};
    std::vector<std::shared_ptr<gqe::logical::relation>> subquery_relations;

    auto read_rel = std::make_unique<gqe::logical::read_relation>(
      subquery_relations, column_names, column_types, "test_table1", nullptr);

    if (optimized) {
      // Optimized plan should have uniqueness info in the plan
      _add_column_uniqueness(read_rel.get(), {0});  // only column `t1_c1_unique` is unique
    }
    return read_rel;
  }

  std::unique_ptr<gqe::logical::relation> construct_read_all_unique(bool optimized)
  {
    // Hand coded logical plan for testing
    std::vector<std::string> column_names = {"t2_c1_unique", "t2_c2_unique"};
    auto column_types                     = {cudf::data_type(cudf::type_id::INT64),
                                             cudf::data_type(cudf::type_id::INT32)};
    std::vector<std::shared_ptr<gqe::logical::relation>> subquery_relations;

    auto read_rel = std::make_unique<gqe::logical::read_relation>(
      subquery_relations, column_names, column_types, "test_table2", nullptr);

    if (optimized) {
      // Optimized plan should have uniqueness info in the plan
      // Both columns are unique
      _add_column_uniqueness(read_rel.get(), {0, 1});
    }
    return read_rel;
  }

  std::unique_ptr<gqe::logical::relation> construct_plan_agg(bool optimized)
  {
    auto read_rel = construct_read_one_unique(optimized);
    std::vector<std::unique_ptr<gqe::expression>> keys;
    std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> measures;

    keys.push_back(std::make_unique<gqe::column_reference_expression>(0));
    auto agg_rel = std::make_unique<gqe::logical::aggregate_relation>(
      std::move(read_rel), empty_relations(), std::move(keys), std::move(measures));

    if (optimized) {
      // Optimized plan should have uniqueness info in the plan
      _add_column_uniqueness(agg_rel.get(), {0});  // Only the key column in
                                                   // unique
    }

    return agg_rel;
  }

  std::unique_ptr<gqe::logical::relation> construct_plan_join_unique_RHS(bool optimized)
  {
    auto read_rel_0 = construct_read_one_unique(optimized);
    auto read_rel_1 = construct_read_all_unique(optimized);

    auto col_1 = std::make_shared<gqe::column_reference_expression>(1);  // not unique
    auto col_2 = std::make_shared<gqe::column_reference_expression>(2);  // unique if optimized
    auto cond  = std::make_unique<gqe::equal_expression>(col_1, col_2);  // unique RHS if optimized
    std::vector<cudf::size_type> projection_indices = {0, 1, 2, 3};

    auto join_rel = std::make_unique<gqe::logical::join_relation>(std::move(read_rel_0),
                                                                  std::move(read_rel_1),
                                                                  empty_relations(),
                                                                  std::move(cond),
                                                                  gqe::join_type_type::inner,
                                                                  projection_indices);

    if (optimized) {
      // Optimized plan should have uniqueness info in the plan
      // LHS columns get to keep their uniqueness
      _add_column_uniqueness(join_rel.get(), {0});
    }

    return join_rel;
  }

  std::unique_ptr<gqe::logical::relation> construct_plan_join_compound_condition(bool optimized)
  {
    auto read_rel_0 = construct_read_one_unique(optimized);
    auto read_rel_1 = construct_read_all_unique(optimized);

    auto col_0  = std::make_shared<gqe::column_reference_expression>(0);  // unique if optimized
    auto col_1  = std::make_shared<gqe::column_reference_expression>(1);  // not unique
    auto col_2  = std::make_shared<gqe::column_reference_expression>(2);  // unique if optimized
    auto col_3  = std::make_shared<gqe::column_reference_expression>(3);  // unique if optimized
    auto cond_0 = std::make_shared<gqe::equal_expression>(col_1, col_2);  // unique RHS if optimized
    auto cond_1 = std::make_shared<gqe::equal_expression>(col_1, col_3);  // unique RHS if optimized
    auto cond_0_and_1 = std::make_shared<gqe::logical_and_expression>(cond_0, cond_1);
    auto cond_2 =
      std::make_shared<gqe::equal_expression>(col_0, col_2);  // unique LHS & RHS if optimized
    auto cond_0_and_1_or_2 = std::make_unique<gqe::logical_or_expression>(cond_0_and_1, cond_2);
    std::vector<cudf::size_type> projection_indices = {0, 1, 2, 3};

    auto join_rel = std::make_unique<gqe::logical::join_relation>(std::move(read_rel_0),
                                                                  std::move(read_rel_1),
                                                                  empty_relations(),
                                                                  std::move(cond_0_and_1_or_2),
                                                                  gqe::join_type_type::inner,
                                                                  projection_indices);

    if (optimized) {
      // Optimized plan should have uniqueness info in the plan
      // LHS columns get to keep their uniqueness
      _add_column_uniqueness(join_rel.get(), {0});
    }

    return join_rel;
  }

  std::unique_ptr<gqe::logical::relation> construct_plan_join_unique_LHS_RHS(bool optimized)
  {
    auto read_rel_0 = construct_read_one_unique(optimized);
    auto read_rel_1 = construct_read_all_unique(optimized);

    auto col_0 = std::make_shared<gqe::column_reference_expression>(0);  // unique if optimized
    auto col_2 = std::make_shared<gqe::column_reference_expression>(2);  // unique if optimized
    auto cond =
      std::make_unique<gqe::equal_expression>(col_0, col_2);  // unique LHS & RHS if optimized
    std::vector<cudf::size_type> projection_indices = {0, 1, 2, 3};

    auto join_rel = std::make_unique<gqe::logical::join_relation>(std::move(read_rel_0),
                                                                  std::move(read_rel_1),
                                                                  empty_relations(),
                                                                  std::move(cond),
                                                                  gqe::join_type_type::inner,
                                                                  projection_indices);

    if (optimized) {
      // Optimized plan should have uniqueness info in the plan
      // All columns get to keep their uniqueness
      _add_column_uniqueness(join_rel.get(), {0, 2, 3});
    }

    return join_rel;
  }

  std::unique_ptr<gqe::logical::relation> construct_plan_fetch(bool optimized)
  {
    auto read_rel_0 = construct_read_one_unique(optimized);  // uniqueness: true, false
    auto fetch_rel  = std::make_unique<gqe::logical::fetch_relation>(std::move(read_rel_0), 0, 10);

    if (optimized) {
      // Optimized plan should have uniqueness info in the plan
      // Only column 0 is unique
      _add_column_uniqueness(fetch_rel.get(), {0});
    }

    return fetch_rel;
  }

  std::unique_ptr<gqe::logical::relation> construct_plan_project(bool optimized)
  {
    auto read_rel_0 = construct_read_one_unique(optimized);  // uniqueness: true, false
    std::vector<std::unique_ptr<gqe::expression>> output_expressions;
    // Switch column order in the output
    output_expressions.push_back(std::make_unique<gqe::column_reference_expression>(1));
    output_expressions.push_back(std::make_unique<gqe::column_reference_expression>(0));
    auto project_rel = std::make_unique<gqe::logical::project_relation>(
      std::move(read_rel_0), empty_relations(), std::move(output_expressions));

    if (optimized) {
      // Optimized plan should have uniqueness info in the plan
      // Column 1 is unique due to column order switch
      _add_column_uniqueness(project_rel.get(), {1});
    }

    return project_rel;
  }

  std::unique_ptr<gqe::logical::relation> construct_plan_set(
    bool optimized, gqe::logical::set_relation::set_operator_type set_op)
  {
    auto read_rel_0 = construct_read_one_unique(optimized);  // uniqueness: true, false
    auto read_rel_1 = construct_read_all_unique(optimized);  // uniqueness: true, true

    auto set_rel = std::make_unique<gqe::logical::set_relation>(
      std::move(read_rel_0), std::move(read_rel_1), set_op);

    if (optimized) {
      if (set_op == gqe::logical::set_relation::set_intersect ||
          set_op == gqe::logical::set_relation::set_minus) {
        // INTERSECT/MINUS: every left unique key propagates; left only has {0}.
        _add_column_uniqueness(set_rel.get(), {0});
      } else if (set_op == gqe::logical::set_relation::set_union) {
        // UNION DISTINCT: only the full-row composite is guaranteed unique. A key that
        // appears on both inputs is not safe to propagate because the value sets may
        // overlap across the two inputs (e.g. both have id=1 with different other cols).
        _set_unique_keys(set_rel.get(), {{0, 1}});
      }
    }

    return set_rel;
  }

  std::unique_ptr<gqe::logical::relation> construct_plan_sort(bool optimized)
  {
    auto read_rel_0 = construct_read_one_unique(optimized);  // uniqueness: true, false
    std::vector<std::unique_ptr<gqe::expression>> sort_exprs;
    sort_exprs.push_back(std::make_unique<gqe::column_reference_expression>(1));

    auto sort_rel = std::make_unique<gqe::logical::sort_relation>(
      std::move(read_rel_0),
      empty_relations(),
      std::vector<cudf::order>({cudf::order::ASCENDING}),
      std::vector<cudf::null_order>({cudf::null_order::BEFORE}),
      std::move(sort_exprs));

    if (optimized) {
      // Optimized plan should have uniqueness info in the plan
      // Column 1 is unique due to column order switch
      _add_column_uniqueness(sort_rel.get(), {0});
    }

    return sort_rel;
  }

  std::unique_ptr<gqe::logical::relation> construct_plan_window(bool optimized)
  {
    auto read_rel_0 = construct_read_one_unique(optimized);  // uniqueness: true, false
    std::vector<std::unique_ptr<gqe::expression>> arguments;
    std::vector<std::unique_ptr<gqe::expression>> order_by;
    std::vector<std::unique_ptr<gqe::expression>> partition_by;

    arguments.push_back(std::make_unique<gqe::column_reference_expression>(0));
    order_by.push_back(std::make_unique<gqe::column_reference_expression>(0));
    partition_by.push_back(std::make_unique<gqe::column_reference_expression>(1));

    auto window_rel = std::make_unique<gqe::logical::window_relation>(
      std::move(read_rel_0),
      empty_relations(),
      cudf::aggregation::Kind::RANK,
      std::move(arguments),
      std::move(order_by),
      std::move(partition_by),
      std::vector<cudf::order>({cudf::order::ASCENDING}),
      gqe::window_frame_bound::unbounded(),
      gqe::window_frame_bound::unbounded());

    if (optimized) {
      // Optimized plan should have uniqueness info in the plan
      // Passthrough uniqueness from input
      _add_column_uniqueness(window_rel.get(), {0});
    }

    return window_rel;
  }

  std::unique_ptr<gqe::logical::relation> construct_plan_write(bool optimized)
  {
    auto read_rel_0 = construct_read_one_unique(optimized);  // uniqueness: true, false

    std::vector<std::string> col_names = {"t1_c1_unique", "t1_c3"};
    auto col_types = {cudf::data_type(cudf::type_id::INT64), cudf::data_type(cudf::type_id::INT32)};
    auto write_rel = std::make_unique<gqe::logical::write_relation>(
      std::move(read_rel_0), std::move(col_names), std::move(col_types), "test_table1");

    if (optimized) { _add_column_uniqueness(write_rel.get(), {0}); }

    return write_rel;
  }

  std::unique_ptr<gqe::logical::relation> construct_plan_filter(bool optimized)
  {
    auto read_rel_0 = construct_read_all_unique(optimized);  // uniqueness: true, false
    // project subset of columns
    std::vector<cudf::size_type> filter_projection_indices = {1};
    auto filter_rel = std::make_unique<gqe::logical::filter_relation>(
      std::move(read_rel_0),
      empty_relations(),
      std::make_unique<gqe::literal_expression<bool>>(true),
      std::move(filter_projection_indices));

    if (optimized) { _add_column_uniqueness(filter_rel.get(), {0}); }

    return filter_rel;
  }

  std::unique_ptr<gqe::logical::relation> construct_three_level_plan(bool optimized)
  {
    // fetch(join(filter(read0), read1))
    auto read_rel_0 = construct_read_one_unique(optimized);
    auto read_rel_1 = construct_read_all_unique(optimized);

    // project all columns
    std::vector<cudf::size_type> filter_projection_indices(read_rel_0->num_columns());
    std::iota(filter_projection_indices.begin(), filter_projection_indices.end(), 0);

    auto filter_rel = std::make_unique<gqe::logical::filter_relation>(
      std::move(read_rel_0),
      empty_relations(),
      std::make_unique<gqe::literal_expression<bool>>(true),
      std::move(filter_projection_indices));
    if (optimized) _add_column_uniqueness(filter_rel.get(), {0});

    auto col_0 = std::make_shared<gqe::column_reference_expression>(0);
    auto col_2 = std::make_shared<gqe::column_reference_expression>(2);
    auto cond  = std::make_unique<gqe::equal_expression>(col_0, col_2);
    std::vector<cudf::size_type> projection_indices = {0, 1, 2, 3};

    auto join_rel = std::make_unique<gqe::logical::join_relation>(std::move(filter_rel),
                                                                  std::move(read_rel_1),
                                                                  empty_relations(),
                                                                  std::move(cond),
                                                                  gqe::join_type_type::inner,
                                                                  projection_indices);
    if (optimized) _add_column_uniqueness(join_rel.get(), {0, 2, 3});

    auto fetch_rel = std::make_unique<gqe::logical::fetch_relation>(std::move(join_rel), 0, 10);
    if (optimized) _add_column_uniqueness(fetch_rel.get(), {0, 2, 3});

    return fetch_rel;
  }

  // Read all three columns of test_table_composite (composite PK {tc_c1, tc_c2} → cols {0,1}).
  std::unique_ptr<gqe::logical::relation> construct_read_composite_pk(bool optimized)
  {
    std::vector<std::string> column_names = {"tc_c1", "tc_c2", "tc_c3"};
    auto column_types                     = {cudf::data_type(cudf::type_id::INT64),
                                             cudf::data_type(cudf::type_id::INT64),
                                             cudf::data_type(cudf::type_id::INT64)};
    auto read_rel                         = std::make_unique<gqe::logical::read_relation>(
      empty_relations(), column_names, column_types, "test_table_composite", nullptr);
    if (optimized) { _set_unique_keys(read_rel.get(), {{0, 1}}); }
    return read_rel;
  }

  // Inner join: left = composite-PK table, right = test_table2 (singleton key {t2_c1_unique}).
  // Join condition: tc_c3 (left col 2) == t2_c1_unique (right col 0, globally col 3).
  // t2_c1_unique is singleton-unique on the right → propagate_left = true.
  // The left's composite key {0, 1} should survive projection onto the join output.
  std::unique_ptr<gqe::logical::relation> construct_plan_join_composite_lhs(bool optimized)
  {
    auto read_rel_0 = construct_read_composite_pk(optimized);  // cols 0,1,2
    auto read_rel_1 = construct_read_all_unique(optimized);    // cols 0,1 → global 3,4

    auto col_2 = std::make_shared<gqe::column_reference_expression>(2);  // tc_c3
    auto col_3 = std::make_shared<gqe::column_reference_expression>(3);  // t2_c1_unique (unique)
    auto cond  = std::make_unique<gqe::equal_expression>(col_2, col_3);
    std::vector<cudf::size_type> projection_indices = {0, 1, 2, 3, 4};

    auto join_rel = std::make_unique<gqe::logical::join_relation>(std::move(read_rel_0),
                                                                  std::move(read_rel_1),
                                                                  empty_relations(),
                                                                  std::move(cond),
                                                                  gqe::join_type_type::inner,
                                                                  projection_indices);
    if (optimized) { _set_unique_keys(join_rel.get(), {{0, 1}}); }
    return join_rel;
  }

  // Aggregate over composite-PK table: 3 group-by keys → full-tuple composite UK {0,1,2}.
  // The input only has composite key {0,1}, so no singleton promotions occur.
  std::unique_ptr<gqe::logical::relation> construct_plan_composite_agg_full_tuple(bool optimized)
  {
    auto read_rel = construct_read_composite_pk(optimized);
    std::vector<std::unique_ptr<gqe::expression>> keys;
    std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> measures;
    keys.push_back(std::make_unique<gqe::column_reference_expression>(0));
    keys.push_back(std::make_unique<gqe::column_reference_expression>(1));
    keys.push_back(std::make_unique<gqe::column_reference_expression>(2));
    auto agg_rel = std::make_unique<gqe::logical::aggregate_relation>(
      std::move(read_rel), empty_relations(), std::move(keys), std::move(measures));
    if (optimized) { _set_unique_keys(agg_rel.get(), {{0, 1, 2}}); }
    return agg_rel;
  }

  // Aggregate over test_table1 (singleton UKs at cols 0 and 1 in read_one_unique output):
  // group by {col_ref(0), col_ref(1)} → full-tuple {0,1} + singleton promotion for col 0
  // (references singleton-unique input col 0). Expected keys: {{0,1}, {0}}.
  std::unique_ptr<gqe::logical::relation> construct_plan_agg_with_singleton_promotion(
    bool optimized)
  {
    auto read_rel = construct_read_one_unique(optimized);  // cols: t1_c1_unique(0), t1_c3(1)
    std::vector<std::unique_ptr<gqe::expression>> keys;
    std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> measures;
    keys.push_back(std::make_unique<gqe::column_reference_expression>(0));
    keys.push_back(std::make_unique<gqe::column_reference_expression>(1));
    auto agg_rel = std::make_unique<gqe::logical::aggregate_relation>(
      std::move(read_rel), empty_relations(), std::move(keys), std::move(measures));
    if (optimized) {
      // Full tuple key {0,1}; col 0 references singleton-unique input col → also emit {0}.
      _set_unique_keys(agg_rel.get(), {{0, 1}, {0}});
    }
    return agg_rel;
  }

  // Filter over composite-PK table that projects all three columns.
  // The composite key {0,1} is remapped identity → survives.
  std::unique_ptr<gqe::logical::relation> construct_plan_composite_filter(bool optimized)
  {
    auto read_rel                     = construct_read_composite_pk(optimized);
    std::vector<cudf::size_type> proj = {0, 1, 2};
    auto filter_rel                   = std::make_unique<gqe::logical::filter_relation>(
      std::move(read_rel),
      empty_relations(),
      std::make_unique<gqe::literal_expression<bool>>(true),
      std::move(proj));
    if (optimized) { _set_unique_keys(filter_rel.get(), {{0, 1}}); }
    return filter_rel;
  }

  // Project that reorders columns: {col_ref(1), col_ref(0), col_ref(2)}.
  // Key {0,1} remaps: 0→1, 1→0 → remapped {0,1} after sort.
  std::unique_ptr<gqe::logical::relation> construct_plan_composite_project_reorder(bool optimized)
  {
    auto read_rel = construct_read_composite_pk(optimized);
    std::vector<std::unique_ptr<gqe::expression>> exprs;
    exprs.push_back(std::make_unique<gqe::column_reference_expression>(1));
    exprs.push_back(std::make_unique<gqe::column_reference_expression>(0));
    exprs.push_back(std::make_unique<gqe::column_reference_expression>(2));
    auto proj_rel = std::make_unique<gqe::logical::project_relation>(
      std::move(read_rel), empty_relations(), std::move(exprs));
    if (optimized) { _set_unique_keys(proj_rel.get(), {{0, 1}}); }
    return proj_rel;
  }

  // Project that drops col 1 (tc_c2): {col_ref(0), col_ref(2)}.
  // Key {0,1} loses col 1 → entire key dropped. Expected: no unique keys.
  std::unique_ptr<gqe::logical::relation> construct_plan_composite_project_drop(bool optimized)
  {
    auto read_rel = construct_read_composite_pk(optimized);
    std::vector<std::unique_ptr<gqe::expression>> exprs;
    exprs.push_back(std::make_unique<gqe::column_reference_expression>(0));
    exprs.push_back(std::make_unique<gqe::column_reference_expression>(2));
    auto proj_rel = std::make_unique<gqe::logical::project_relation>(
      std::move(read_rel), empty_relations(), std::move(exprs));
    // optimized: no unique keys (composite key component dropped)
    return proj_rel;
  }

  // UNION DISTINCT of two composite-PK reads.
  // Only the full-row composite {0,1,2} is safe to propagate.
  std::unique_ptr<gqe::logical::relation> construct_plan_composite_union(bool optimized)
  {
    auto lhs     = construct_read_composite_pk(optimized);
    auto rhs     = construct_read_composite_pk(optimized);
    auto set_rel = std::make_unique<gqe::logical::set_relation>(
      std::move(lhs), std::move(rhs), gqe::logical::set_relation::set_union);
    if (optimized) { _set_unique_keys(set_rel.get(), {{0, 1, 2}}); }
    return set_rel;
  }

  // INTERSECT of two composite-PK reads → left keys propagate verbatim: {0,1}.
  std::unique_ptr<gqe::logical::relation> construct_plan_composite_intersect(bool optimized)
  {
    auto lhs     = construct_read_composite_pk(optimized);
    auto rhs     = construct_read_composite_pk(optimized);
    auto set_rel = std::make_unique<gqe::logical::set_relation>(
      std::move(lhs), std::move(rhs), gqe::logical::set_relation::set_intersect);
    if (optimized) { _set_unique_keys(set_rel.get(), {{0, 1}}); }
    return set_rel;
  }

  // MINUS of two composite-PK reads → left keys propagate verbatim: {0,1}.
  std::unique_ptr<gqe::logical::relation> construct_plan_composite_minus(bool optimized)
  {
    auto lhs     = construct_read_composite_pk(optimized);
    auto rhs     = construct_read_composite_pk(optimized);
    auto set_rel = std::make_unique<gqe::logical::set_relation>(
      std::move(lhs), std::move(rhs), gqe::logical::set_relation::set_minus);
    if (optimized) { _set_unique_keys(set_rel.get(), {{0, 1}}); }
    return set_rel;
  }

  // Sort over composite-PK table: key {0,1} passes through unchanged.
  std::unique_ptr<gqe::logical::relation> construct_plan_composite_sort(bool optimized)
  {
    auto read_rel = construct_read_composite_pk(optimized);
    std::vector<std::unique_ptr<gqe::expression>> sort_exprs;
    sort_exprs.push_back(std::make_unique<gqe::column_reference_expression>(2));
    auto sort_rel = std::make_unique<gqe::logical::sort_relation>(
      std::move(read_rel),
      empty_relations(),
      std::vector<cudf::order>({cudf::order::ASCENDING}),
      std::vector<cudf::null_order>({cudf::null_order::BEFORE}),
      std::move(sort_exprs));
    if (optimized) { _set_unique_keys(sort_rel.get(), {{0, 1}}); }
    return sort_rel;
  }

  // Window over composite-PK table: key {0,1} passes through unchanged.
  std::unique_ptr<gqe::logical::relation> construct_plan_composite_window(bool optimized)
  {
    auto read_rel = construct_read_composite_pk(optimized);
    std::vector<std::unique_ptr<gqe::expression>> arguments, order_by, partition_by;
    arguments.push_back(std::make_unique<gqe::column_reference_expression>(2));
    order_by.push_back(std::make_unique<gqe::column_reference_expression>(2));
    partition_by.push_back(std::make_unique<gqe::column_reference_expression>(0));
    auto window_rel = std::make_unique<gqe::logical::window_relation>(
      std::move(read_rel),
      empty_relations(),
      cudf::aggregation::Kind::RANK,
      std::move(arguments),
      std::move(order_by),
      std::move(partition_by),
      std::vector<cudf::order>({cudf::order::ASCENDING}),
      gqe::window_frame_bound::unbounded(),
      gqe::window_frame_bound::unbounded());
    if (optimized) { _set_unique_keys(window_rel.get(), {{0, 1}}); }
    return window_rel;
  }

  // Fetch over composite-PK table: key {0,1} passes through unchanged.
  std::unique_ptr<gqe::logical::relation> construct_plan_composite_fetch(bool optimized)
  {
    auto read_rel  = construct_read_composite_pk(optimized);
    auto fetch_rel = std::make_unique<gqe::logical::fetch_relation>(std::move(read_rel), 0, 10);
    if (optimized) { _set_unique_keys(fetch_rel.get(), {{0, 1}}); }
    return fetch_rel;
  }

  // Left-semi join: left = composite-PK (cols 0,1,2 with key {0,1}),
  // right = test_table2 (cols 3,4). Only left cols in output.
  // propagate_left = true for left_semi → composite key {0,1} survives projection {0,1,2}.
  std::unique_ptr<gqe::logical::relation> construct_plan_composite_left_semi(bool optimized)
  {
    auto read_rel_0 = construct_read_composite_pk(optimized);  // cols 0,1,2
    auto read_rel_1 = construct_read_all_unique(optimized);    // cols 3,4 globally

    auto col_2 = std::make_shared<gqe::column_reference_expression>(2);
    auto col_3 = std::make_shared<gqe::column_reference_expression>(3);
    auto cond  = std::make_unique<gqe::equal_expression>(col_2, col_3);
    std::vector<cudf::size_type> projection_indices = {0, 1, 2};  // left cols only

    auto join_rel = std::make_unique<gqe::logical::join_relation>(std::move(read_rel_0),
                                                                  std::move(read_rel_1),
                                                                  empty_relations(),
                                                                  std::move(cond),
                                                                  gqe::join_type_type::left_semi,
                                                                  projection_indices);
    if (optimized) { _set_unique_keys(join_rel.get(), {{0, 1}}); }
    return join_rel;
  }

  // Inner join: left = composite-PK table (cols 0,1,2), right = composite-PK table (cols 3,4,5).
  // Join condition: col_0 == col_3 AND col_1 == col_4 (both sides have only composite PK {0,1}).
  // No singleton unique key exists on either side, so check_join_condition_for_propagation returns
  // (false, false). The composite-coverage block must fire: left PK {0,1} is fully covered by the
  // equijoin pairs → propagate_right; right PK {0,1} (global {3,4}) is also covered →
  // propagate_left. Both composite keys should appear on the output.
  std::unique_ptr<gqe::logical::relation> construct_plan_join_composite_driven(bool optimized)
  {
    auto read_rel_0 = construct_read_composite_pk(optimized);  // cols 0,1,2
    auto read_rel_1 = construct_read_composite_pk(optimized);  // cols 0,1,2 → global 3,4,5

    auto col_0    = std::make_shared<gqe::column_reference_expression>(0);
    auto col_1    = std::make_shared<gqe::column_reference_expression>(1);
    auto col_3    = std::make_shared<gqe::column_reference_expression>(3);
    auto col_4    = std::make_shared<gqe::column_reference_expression>(4);
    auto eq_01_34 = std::make_unique<gqe::logical_and_expression>(
      std::make_shared<gqe::equal_expression>(col_0, col_3),
      std::make_shared<gqe::equal_expression>(col_1, col_4));
    std::vector<cudf::size_type> projection_indices = {0, 1, 2, 3, 4, 5};

    auto join_rel = std::make_unique<gqe::logical::join_relation>(std::move(read_rel_0),
                                                                  std::move(read_rel_1),
                                                                  empty_relations(),
                                                                  std::move(eq_01_34),
                                                                  gqe::join_type_type::inner,
                                                                  projection_indices);
    if (optimized) { _set_unique_keys(join_rel.get(), {{0, 1}, {3, 4}}); }
    return join_rel;
  }

  gqe::catalog catalog{get_task_manager_ctx()};
  std::unique_ptr<gqe::optimizer::logical_optimizer> optimizer;
  std::shared_ptr<gqe::logical::relation> test_plan;
  std::unique_ptr<gqe::logical::relation> ref_plan;

  void _add_column_uniqueness(gqe::logical::relation* rel, std::vector<cudf::size_type> col_indices)
  {
    gqe::optimizer::relation_properties props;
    for (auto idx : col_indices) {
      props.add_unique_key({idx});
    }
    auto traits = std::make_unique<gqe::optimizer::relation_traits>(props);
    rel->set_relation_traits(std::move(traits));
  }

  void _set_unique_keys(gqe::logical::relation* rel, std::vector<std::vector<cudf::size_type>> keys)
  {
    gqe::optimizer::relation_properties props;
    for (auto const& key : keys) {
      props.add_unique_key(key);
    }
    auto traits = std::make_unique<gqe::optimizer::relation_traits>(props);
    rel->set_relation_traits(std::move(traits));
  }
};

TEST_F(UniquenessPropagationTest, SimpleRead)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans

  test_plan = construct_read_one_unique(false);
  ref_plan  = construct_read_one_unique(true);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, SimpleAgg)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_agg(false);
  ref_plan  = construct_plan_agg(true);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, SimpleJoinUniqueRHS)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_join_unique_RHS(false);
  ref_plan  = construct_plan_join_unique_RHS(true);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, SimpleJoinUniqueLhsRhs)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_join_unique_LHS_RHS(false);
  ref_plan  = construct_plan_join_unique_LHS_RHS(true);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, CompoundConditionJoin)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_join_compound_condition(false);
  ref_plan  = construct_plan_join_compound_condition(true);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, SimpleFetch)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_fetch(false);
  ref_plan  = construct_plan_fetch(true);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, SimpleFilter)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_filter(false);
  ref_plan  = construct_plan_filter(true);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, SimpleProject)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_project(false);
  ref_plan  = construct_plan_project(true);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, SimpleIntersect)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_set(false, gqe::logical::set_relation::set_intersect);
  ref_plan  = construct_plan_set(true, gqe::logical::set_relation::set_intersect);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, SimpleMinus)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_set(false, gqe::logical::set_relation::set_minus);
  ref_plan  = construct_plan_set(true, gqe::logical::set_relation::set_minus);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, SimpleUnion)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_set(false, gqe::logical::set_relation::set_union);
  ref_plan  = construct_plan_set(true, gqe::logical::set_relation::set_union);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, SimpleUnionAll)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_set(false, gqe::logical::set_relation::set_union_all);
  ref_plan  = construct_plan_set(true, gqe::logical::set_relation::set_union_all);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, SimpleSort)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_sort(false);
  ref_plan  = construct_plan_sort(true);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, SimpleWindow)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_window(false);
  ref_plan  = construct_plan_window(true);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, SimpleWrite)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_write(false);
  ref_plan  = construct_plan_write(true);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, ComplexPlan)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_three_level_plan(false);
  ref_plan  = construct_three_level_plan(true);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, CompositeRead)
{
  initialize_optimizer();

  test_plan = construct_read_composite_pk(false);
  ref_plan  = construct_read_composite_pk(true);

  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, CompositeKeyThroughJoin)
{
  initialize_optimizer();

  test_plan = construct_plan_join_composite_lhs(false);
  ref_plan  = construct_plan_join_composite_lhs(true);

  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, CompositeDrivenJoinPropagation)
{
  // Neither side has a singleton UK; propagation is driven purely by composite PKs.
  initialize_optimizer();

  test_plan = construct_plan_join_composite_driven(false);
  ref_plan  = construct_plan_join_composite_driven(true);

  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, CompositeAggregateGroupByAllColumnsEmitsFullTupleKey)
{
  // GROUP BY all 3 columns of composite-PK table → full-tuple composite UK {0,1,2}.
  // Input only has composite {0,1}, so no singleton promotion occurs.
  initialize_optimizer();

  test_plan = construct_plan_composite_agg_full_tuple(false);
  ref_plan  = construct_plan_composite_agg_full_tuple(true);

  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest,
       CompositeAggregateGroupByReferencesSingletonUniquePromotesSingleton)
{
  // GROUP BY {singleton-unique col, non-unique col}: emits full-tuple {0,1} AND singleton {0}.
  initialize_optimizer();

  test_plan = construct_plan_agg_with_singleton_promotion(false);
  ref_plan  = construct_plan_agg_with_singleton_promotion(true);

  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, CompositeFilterCompositeKeyPropagates)
{
  initialize_optimizer();

  test_plan = construct_plan_composite_filter(false);
  ref_plan  = construct_plan_composite_filter(true);

  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, CompositeProjectReorderedColumnsRemapped)
{
  // Swap col 0 and col 1: composite key {0,1} remaps to {1,0} → sorted {0,1}.
  initialize_optimizer();

  test_plan = construct_plan_composite_project_reorder(false);
  ref_plan  = construct_plan_composite_project_reorder(true);

  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, CompositeProjectComponentDroppedKeyRemoved)
{
  // Projecting away one component of the composite key removes the entire key.
  initialize_optimizer();

  test_plan = construct_plan_composite_project_drop(false);
  ref_plan  = construct_plan_composite_project_drop(true);

  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, CompositeUnionDistinctEmitsOnlyFullTuple)
{
  initialize_optimizer();

  test_plan = construct_plan_composite_union(false);
  ref_plan  = construct_plan_composite_union(true);

  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, CompositeIntersectLeftCompositeKeyPropagates)
{
  initialize_optimizer();

  test_plan = construct_plan_composite_intersect(false);
  ref_plan  = construct_plan_composite_intersect(true);

  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, CompositeMinusLeftCompositeKeyPropagates)
{
  initialize_optimizer();

  test_plan = construct_plan_composite_minus(false);
  ref_plan  = construct_plan_composite_minus(true);

  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, CompositeSortCompositeKeyPropagates)
{
  initialize_optimizer();

  test_plan = construct_plan_composite_sort(false);
  ref_plan  = construct_plan_composite_sort(true);

  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, CompositeWindowCompositeKeyPropagates)
{
  initialize_optimizer();

  test_plan = construct_plan_composite_window(false);
  ref_plan  = construct_plan_composite_window(true);

  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, CompositeFetchCompositeKeyPropagates)
{
  initialize_optimizer();

  test_plan = construct_plan_composite_fetch(false);
  ref_plan  = construct_plan_composite_fetch(true);

  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, CompositeLeftSemiJoinCompositeLeftKeyPropagates)
{
  // Left-semi always propagates left uniqueness; composite key {0,1} should survive.
  initialize_optimizer();

  test_plan = construct_plan_composite_left_semi(false);
  ref_plan  = construct_plan_composite_left_semi(true);

  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  EXPECT_EQ(*ref_plan, *optimized_plan);
}

// UNION DISTINCT must not propagate a key that is unique on each input
// individually but may have overlapping values across the two sides.
TEST_F(UniquenessPropagationTest, UnionDistinctSharedSingletonKeyNotPropagated)
{
  initialize_optimizer();

  // Both sides read from test_table1 projecting {t1_c1_unique, t1_c3} — col 0 is unique
  // on each side. But the union output can have duplicate col-0 values when both sides
  // happen to contain the same id paired with different col-1 values.
  auto make_read = [&](bool optimized) {
    std::vector<std::string> col_names     = {"t1_c1_unique", "t1_c3"};
    std::vector<cudf::data_type> col_types = {cudf::data_type(cudf::type_id::INT64),
                                              cudf::data_type(cudf::type_id::INT32)};
    auto rel                               = std::make_unique<gqe::logical::read_relation>(
      empty_relations(), col_names, col_types, "test_table1", nullptr);
    if (optimized) _add_column_uniqueness(rel.get(), {0});
    return rel;
  };

  // test: no traits attached to the set_relation
  auto lhs_t = make_read(false);
  auto rhs_t = make_read(false);
  test_plan  = std::make_unique<gqe::logical::set_relation>(
    std::move(lhs_t), std::move(rhs_t), gqe::logical::set_relation::set_union);

  // reference: only the full-row composite {0,1} — NOT {0}
  auto lhs_r   = make_read(true);
  auto rhs_r   = make_read(true);
  auto ref_set = std::make_unique<gqe::logical::set_relation>(
    std::move(lhs_r), std::move(rhs_r), gqe::logical::set_relation::set_union);
  _set_unique_keys(ref_set.get(), {{0, 1}});
  ref_plan = std::move(ref_set);

  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  EXPECT_EQ(*ref_plan, *optimized_plan);
}

// Projecting the same unique column twice should mark both output positions unique.
TEST_F(UniquenessPropagationTest, ProjectDuplicateColumnRefBothPositionsUnique)
{
  initialize_optimizer();

  // SELECT t1_c1_unique, t1_c1_unique FROM test_table1 — col 0 of the read is unique.
  // Both output positions 0 and 1 reference the same unique input column, so both must be
  // marked as singleton unique keys on the project output.
  auto make_read = [&](bool optimized) {
    std::vector<std::string> col_names     = {"t1_c1_unique"};
    std::vector<cudf::data_type> col_types = {cudf::data_type(cudf::type_id::INT64)};
    auto rel                               = std::make_unique<gqe::logical::read_relation>(
      empty_relations(), col_names, col_types, "test_table1", nullptr);
    if (optimized) _add_column_uniqueness(rel.get(), {0});
    return rel;
  };

  auto make_proj = [&](bool optimized) {
    auto read = make_read(optimized);
    std::vector<std::unique_ptr<gqe::expression>> exprs;
    exprs.push_back(std::make_unique<gqe::column_reference_expression>(0));
    exprs.push_back(std::make_unique<gqe::column_reference_expression>(0));
    auto proj = std::make_unique<gqe::logical::project_relation>(
      std::move(read), empty_relations(), std::move(exprs));
    if (optimized) _set_unique_keys(proj.get(), {{0}, {1}});
    return proj;
  };

  test_plan = make_proj(false);
  ref_plan  = make_proj(true);

  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  EXPECT_EQ(*ref_plan, *optimized_plan);
}

// Composite key with a duplicated component should expand to all valid remaps.
TEST_F(UniquenessPropagationTest, ProjectCompositeKeyDuplicateRefsExpandToCombinations)
{
  initialize_optimizer();

  // SELECT tc_c1, tc_c1, tc_c2 FROM test_table_composite
  // Composite PK is {tc_c1, tc_c2} = input cols {0, 1}.
  // Output: col_ref(0), col_ref(0), col_ref(1) → input_to_outputs = {0→[0,1], 1→[2]}.
  // Cartesian product of [0,1] × [2] = (0,2), (1,2) → emit keys {0,2} and {1,2}.
  auto make_read = [&](bool optimized) {
    std::vector<std::string> col_names     = {"tc_c1", "tc_c2", "tc_c3"};
    std::vector<cudf::data_type> col_types = {cudf::data_type(cudf::type_id::INT64),
                                              cudf::data_type(cudf::type_id::INT64),
                                              cudf::data_type(cudf::type_id::INT64)};
    auto rel                               = std::make_unique<gqe::logical::read_relation>(
      empty_relations(), col_names, col_types, "test_table_composite", nullptr);
    if (optimized) _set_unique_keys(rel.get(), {{0, 1}});
    return rel;
  };

  auto make_proj = [&](bool optimized) {
    auto read = make_read(optimized);
    std::vector<std::unique_ptr<gqe::expression>> exprs;
    exprs.push_back(std::make_unique<gqe::column_reference_expression>(0));  // tc_c1
    exprs.push_back(std::make_unique<gqe::column_reference_expression>(0));  // tc_c1 again
    exprs.push_back(std::make_unique<gqe::column_reference_expression>(1));  // tc_c2
    auto proj = std::make_unique<gqe::logical::project_relation>(
      std::move(read), empty_relations(), std::move(exprs));
    if (optimized) _set_unique_keys(proj.get(), {{0, 2}, {1, 2}});
    return proj;
  };

  test_plan = make_proj(false);
  ref_plan  = make_proj(true);

  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  EXPECT_EQ(*ref_plan, *optimized_plan);
}

// (filter path): filter_relation with duplicate projection_indices must mark
// both output positions unique when the source column is unique.
TEST_F(UniquenessPropagationTest, FilterDuplicateProjectionIndexBothPositionsUnique)
{
  initialize_optimizer();

  // filter_projection_indices = {0, 0}: input col 0 (t1_c1_unique) maps to both output slots.
  // Expected: both output positions 0 and 1 are singleton-unique.
  auto make_filter = [&](bool optimized) {
    auto read = construct_read_one_unique(optimized);  // col 0 = t1_c1_unique (unique)
    std::vector<cudf::size_type> proj_idx = {0, 0};
    auto filter                           = std::make_unique<gqe::logical::filter_relation>(
      std::move(read),
      empty_relations(),
      std::make_unique<gqe::literal_expression<bool>>(true),
      std::move(proj_idx));
    if (optimized) _add_column_uniqueness(filter.get(), {0, 1});
    return filter;
  };

  test_plan = make_filter(false);
  ref_plan  = make_filter(true);

  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  EXPECT_EQ(*ref_plan, *optimized_plan);
}

// (filter path, composite key): filter with duplicate projection_indices on a
// composite-key table must expand to all valid remaps via cartesian product.
TEST_F(UniquenessPropagationTest, FilterCompositeKeyDuplicateProjectionIndexExpandsToCombinations)
{
  initialize_optimizer();

  // test_table_composite has composite PK {tc_c1, tc_c2} = input cols {0, 1}.
  // filter_projection_indices = {0, 0, 1}:
  //   input_to_outputs = {0 → [0, 1], 1 → [2]}
  //   cartesian product of [0,1] × [2] = (0,2), (1,2) → emit keys {0,2} and {1,2}.
  auto make_filter = [&](bool optimized) {
    std::vector<std::string> col_names     = {"tc_c1", "tc_c2", "tc_c3"};
    std::vector<cudf::data_type> col_types = {cudf::data_type(cudf::type_id::INT64),
                                              cudf::data_type(cudf::type_id::INT64),
                                              cudf::data_type(cudf::type_id::INT64)};
    auto read                              = std::make_unique<gqe::logical::read_relation>(
      empty_relations(), col_names, col_types, "test_table_composite", nullptr);
    if (optimized) _set_unique_keys(read.get(), {{0, 1}});

    std::vector<cudf::size_type> proj_idx = {0, 0, 1};
    auto filter                           = std::make_unique<gqe::logical::filter_relation>(
      std::move(read),
      empty_relations(),
      std::make_unique<gqe::literal_expression<bool>>(true),
      std::move(proj_idx));
    if (optimized) _set_unique_keys(filter.get(), {{0, 2}, {1, 2}});
    return filter;
  };

  test_plan = make_filter(false);
  ref_plan  = make_filter(true);

  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  EXPECT_EQ(*ref_plan, *optimized_plan);
}

// (join path): join_relation with duplicate projection_indices must propagate
// uniqueness to all output positions that reference the same unique input column.
TEST_F(UniquenessPropagationTest, JoinDuplicateProjectionIndexPropagatesMultipleUniqueKeys)
{
  initialize_optimizer();

  // Left:  test_table1  {t1_c1_unique (col 0, unique), t1_c3 (col 1)}
  // Right: test_table2  {t2_c1_unique (col 0/global 2, unique), t2_c2_unique (col 1/global 3)}
  // Join condition: col 0 (left unique) == col 2 (right unique) → propagate_left AND right.
  // projection_indices = {0, 0, 2, 3}:
  //   in_to_outs = {0→[0,1], 2→[2], 3→[3]}
  //   left key  {0} → key_global {0} → emit {0} and {1}
  //   right key {0} → key_global {2} → emit {2}
  //   right key {1} → key_global {3} → emit {3}
  // Expected output keys: singletons {0}, {1}, {2}, {3}.
  auto make_join = [&](bool optimized) {
    auto left  = construct_read_one_unique(optimized);
    auto right = construct_read_all_unique(optimized);

    auto col_0 = std::make_shared<gqe::column_reference_expression>(0);  // left unique
    auto col_2 = std::make_shared<gqe::column_reference_expression>(2);  // right unique (global)
    auto cond  = std::make_unique<gqe::equal_expression>(col_0, col_2);

    std::vector<cudf::size_type> proj_idx = {0, 0, 2, 3};
    auto join = std::make_unique<gqe::logical::join_relation>(std::move(left),
                                                              std::move(right),
                                                              empty_relations(),
                                                              std::move(cond),
                                                              gqe::join_type_type::inner,
                                                              proj_idx);
    if (optimized) _add_column_uniqueness(join.get(), {0, 1, 2, 3});
    return join;
  };

  test_plan = make_join(false);
  ref_plan  = make_join(true);

  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  EXPECT_EQ(*ref_plan, *optimized_plan);
}
