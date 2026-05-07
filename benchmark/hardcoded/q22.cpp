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

#include "../utility.hpp"

#include <gqe/catalog.hpp>
#include <gqe/context_reference.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/fetch.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/set.hpp>
#include <gqe/logical/sort.hpp>
#include <gqe/optimizer/logical_optimization.hpp>
#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/query_context.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/helpers.hpp>

#include <cudf/io/parquet.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

using col_prop = gqe::column_traits::column_property;

constexpr int64_t DMS = 1200;

void print_usage()
{
  std::cout << "Run TPC-DS Q22 benchmark with hardcoded logical plan" << std::endl
            << "./q22 <path-to-dataset>" << std::endl;
}

std::shared_ptr<gqe::logical::read_relation> read_table(
  std::string table_name,
  std::vector<std::string> column_names,
  gqe::catalog const* tpcds_catalog,
  std::shared_ptr<gqe::logical::project_relation> partial_filter_haystack = nullptr,
  std::unique_ptr<gqe::expression> partial_filter                         = nullptr)
{
  std::vector<cudf::data_type> column_types;
  column_types.reserve(column_names.size());
  for (auto const& column_name : column_names)
    column_types.push_back(tpcds_catalog->column_type(table_name, column_name));
  return std::make_shared<gqe::logical::read_relation>(
    partial_filter
      ? std::vector<std::shared_ptr<gqe::logical::relation>>{std::move(partial_filter_haystack)}
      : std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::move(column_names),
    std::move(column_types),
    std::move(table_name),
    std::move(partial_filter));  // partial_filter
}

int main(int argc, char* argv[])
{
  // Parse the command line arguments to get the dataset location
  if (argc != 2) {
    print_usage();
    return EXIT_FAILURE;
  }
  std::string const dataset_location(argv[1]);

  // Configure the memory pool
  // FIXME: For multi-GPU, we need to construct a memory pool for each device
  auto const pool_size = gqe::benchmark::get_memory_pool_size();
  rmm::mr::cuda_memory_resource cuda_mr;
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{
    &cuda_mr, pool_size, pool_size};
  rmm::mr::set_current_device_resource(&pool_mr);

  // Register the input tables
  gqe::task_manager_context task_manager_ctx;
  gqe::catalog tpcds_catalog{&task_manager_ctx};
  tpcds_catalog.register_table("inventory",
                               {{"inv_date_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"inv_item_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"inv_quantity_on_hand", cudf::data_type(cudf::type_id::INT64)}},
                               gqe::storage_kind::parquet_file{
                                 gqe::utility::get_parquet_files(dataset_location + "/inventory")},
                               gqe::partitioning_schema_kind::automatic{});
  tpcds_catalog.register_table(
    "date_dim",
    {{"d_date_sk", cudf::data_type(cudf::type_id::INT64), {col_prop::unique}},
     {"d_month_seq", cudf::data_type(cudf::type_id::INT64)}},
    gqe::storage_kind::parquet_file{
      gqe::utility::get_parquet_files(dataset_location + "/date_dim")},
    gqe::partitioning_schema_kind::automatic{});
  tpcds_catalog.register_table(
    "item",
    {{"i_item_sk", cudf::data_type(cudf::type_id::INT64), {col_prop::unique}},
     {"i_product_name", cudf::data_type(cudf::type_id::STRING)},
     {"i_brand", cudf::data_type(cudf::type_id::STRING)},
     {"i_class", cudf::data_type(cudf::type_id::STRING)},
     {"i_category", cudf::data_type(cudf::type_id::STRING)}},
    gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(dataset_location + "/item")},
    gqe::partitioning_schema_kind::automatic{});

  // Hand-code the logical plan
  std::shared_ptr<gqe::logical::relation> date_dim_table =
    read_table("date_dim", {"d_date_sk", "d_month_seq"}, &tpcds_catalog);
  date_dim_table = std::make_shared<gqe::logical::filter_relation>(
    std::move(date_dim_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::make_unique<gqe::logical_and_expression>(
      std::make_shared<gqe::greater_equal_expression>(
        std::make_shared<gqe::column_reference_expression>(1),
        std::make_shared<gqe::literal_expression<int64_t>>(DMS)),
      std::make_shared<gqe::less_equal_expression>(
        std::make_shared<gqe::column_reference_expression>(1),
        std::make_shared<gqe::literal_expression<int64_t>>(DMS + 11))),
    std::vector<cudf::size_type>({0, 1}));

  // Join the inventory table with the `date_dim_table`
  // After this operation, `inventory_table` contains columns
  // ["inv_item_sk", "inv_quantity_on_hand"]
  std::vector<std::unique_ptr<gqe::expression>> col_0_exprs;
  col_0_exprs.push_back(std::make_unique<gqe::column_reference_expression>(0));

  auto partial_filter_haystack = std::make_shared<gqe::logical::project_relation>(
    date_dim_table,
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery relations
    std::move(col_0_exprs));

  auto partial_filter = std::make_unique<gqe::in_predicate_expression>(
    std::vector<std::shared_ptr<gqe::expression>>{
      std::make_shared<gqe::column_reference_expression>(0)},  // inv_date_sk
    0);

  std::shared_ptr<gqe::logical::relation> inventory_table =
    read_table("inventory",
               {"inv_date_sk", "inv_item_sk", "inv_quantity_on_hand"},
               &tpcds_catalog,
               std::move(partial_filter_haystack),
               std::move(partial_filter));

  inventory_table = std::make_shared<gqe::logical::join_relation>(
    std::move(inventory_table),
    std::move(date_dim_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                            std::make_shared<gqe::column_reference_expression>(3)),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>({1, 2}));

  // Join the `inventory_table` with the item table
  // After this operation, `inventory_table` contains columns
  // ["i_product_name", "i_brand", "i_class", "i_category", "inv_quantity_on_hand"]
  auto item_table = read_table(
    "item", {"i_item_sk", "i_product_name", "i_brand", "i_class", "i_category"}, &tpcds_catalog);
  inventory_table = std::make_shared<gqe::logical::join_relation>(
    std::move(inventory_table),
    std::move(item_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                            std::make_shared<gqe::column_reference_expression>(2)),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>({3, 4, 5, 6, 1}));

  // Groupby on (i_product_name, i_brand, i_class, i_category)
  // After this operation, `groupby_0` contains columns
  // ["i_product_name", "i_brand", "i_class", "i_category", avg(inv_quantity_on_hand)]
  std::vector<std::unique_ptr<gqe::expression>> groupby0_keys;
  groupby0_keys.push_back(std::make_unique<gqe::column_reference_expression>(0));
  groupby0_keys.push_back(std::make_unique<gqe::column_reference_expression>(1));
  groupby0_keys.push_back(std::make_unique<gqe::column_reference_expression>(2));
  groupby0_keys.push_back(std::make_unique<gqe::column_reference_expression>(3));

  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>>
    groupby0_measures;
  groupby0_measures.emplace_back(cudf::aggregation::MEAN,
                                 std::make_unique<gqe::column_reference_expression>(4));

  auto groupby0 = std::make_shared<gqe::logical::aggregate_relation>(
    inventory_table,
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::move(groupby0_keys),
    std::move(groupby0_measures));

  // Groupby on (i_product_name, i_brand, i_class, NULL)
  std::vector<std::unique_ptr<gqe::expression>> groupby1_keys;
  groupby1_keys.push_back(std::make_unique<gqe::column_reference_expression>(0));
  groupby1_keys.push_back(std::make_unique<gqe::column_reference_expression>(1));
  groupby1_keys.push_back(std::make_unique<gqe::column_reference_expression>(2));
  groupby1_keys.push_back(std::make_unique<gqe::literal_expression<std::string>>("", true));

  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>>
    groupby1_measures;
  groupby1_measures.emplace_back(cudf::aggregation::MEAN,
                                 std::make_unique<gqe::column_reference_expression>(4));

  auto groupby1 = std::make_shared<gqe::logical::aggregate_relation>(
    inventory_table,
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::move(groupby1_keys),
    std::move(groupby1_measures));

  // Groupby on (i_product_name, i_brand, NULL, NULL)
  std::vector<std::unique_ptr<gqe::expression>> groupby2_keys;
  groupby2_keys.push_back(std::make_unique<gqe::column_reference_expression>(0));
  groupby2_keys.push_back(std::make_unique<gqe::column_reference_expression>(1));
  groupby2_keys.push_back(std::make_unique<gqe::literal_expression<std::string>>("", true));
  groupby2_keys.push_back(std::make_unique<gqe::literal_expression<std::string>>("", true));

  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>>
    groupby2_measures;
  groupby2_measures.emplace_back(cudf::aggregation::MEAN,
                                 std::make_unique<gqe::column_reference_expression>(4));

  auto groupby2 = std::make_shared<gqe::logical::aggregate_relation>(
    inventory_table,
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::move(groupby2_keys),
    std::move(groupby2_measures));

  // Groupby on (i_product_name, NULL, NULL, NULL)
  std::vector<std::unique_ptr<gqe::expression>> groupby3_keys;
  groupby3_keys.push_back(std::make_unique<gqe::column_reference_expression>(0));
  groupby3_keys.push_back(std::make_unique<gqe::literal_expression<std::string>>("", true));
  groupby3_keys.push_back(std::make_unique<gqe::literal_expression<std::string>>("", true));
  groupby3_keys.push_back(std::make_unique<gqe::literal_expression<std::string>>("", true));

  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>>
    groupby3_measures;
  groupby3_measures.emplace_back(cudf::aggregation::MEAN,
                                 std::make_unique<gqe::column_reference_expression>(4));

  auto groupby3 = std::make_shared<gqe::logical::aggregate_relation>(
    inventory_table,
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::move(groupby3_keys),
    std::move(groupby3_measures));

  // Groupby on (NULL, NULL, NULL, NULL)
  std::vector<std::unique_ptr<gqe::expression>> groupby4_keys;
  groupby4_keys.push_back(std::make_unique<gqe::literal_expression<std::string>>("", true));
  groupby4_keys.push_back(std::make_unique<gqe::literal_expression<std::string>>("", true));
  groupby4_keys.push_back(std::make_unique<gqe::literal_expression<std::string>>("", true));
  groupby4_keys.push_back(std::make_unique<gqe::literal_expression<std::string>>("", true));

  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>>
    groupby4_measures;
  groupby4_measures.emplace_back(cudf::aggregation::MEAN,
                                 std::make_unique<gqe::column_reference_expression>(4));

  auto groupby4 = std::make_shared<gqe::logical::aggregate_relation>(
    inventory_table,
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::move(groupby4_keys),
    std::move(groupby4_measures));

  // Union the groupbys together
  // After this operation, `groupby_union` contains columns
  // ["i_product_name", "i_brand", "i_class", "i_category", avg(inv_quantity_on_hand)]
  auto groupby_union = std::make_shared<gqe::logical::set_relation>(
    std::make_shared<gqe::logical::set_relation>(
      std::make_shared<gqe::logical::set_relation>(
        std::make_shared<gqe::logical::set_relation>(
          std::move(groupby0), std::move(groupby1), gqe::logical::set_relation::set_union_all),
        std::move(groupby2),
        gqe::logical::set_relation::set_union_all),
      std::move(groupby3),
      gqe::logical::set_relation::set_union_all),
    std::move(groupby4),
    gqe::logical::set_relation::set_union_all);

  // order by qoh, i_product_name, i_brand, i_class, i_category
  std::vector<std::unique_ptr<gqe::expression>> sort_exprs;
  sort_exprs.push_back(std::make_unique<gqe::column_reference_expression>(4));
  sort_exprs.push_back(std::make_unique<gqe::column_reference_expression>(0));
  sort_exprs.push_back(std::make_unique<gqe::column_reference_expression>(1));
  sort_exprs.push_back(std::make_unique<gqe::column_reference_expression>(2));
  sort_exprs.push_back(std::make_unique<gqe::column_reference_expression>(3));

  std::shared_ptr<gqe::logical::relation> sorted = std::make_shared<gqe::logical::sort_relation>(
    std::move(groupby_union),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::vector<cudf::order>({cudf::order::ASCENDING,
                              cudf::order::ASCENDING,
                              cudf::order::ASCENDING,
                              cudf::order::ASCENDING,
                              cudf::order::ASCENDING}),
    std::vector<cudf::null_order>({cudf::null_order::BEFORE,
                                   cudf::null_order::BEFORE,
                                   cudf::null_order::BEFORE,
                                   cudf::null_order::BEFORE,
                                   cudf::null_order::BEFORE}),
    std::move(sort_exprs));

  // limit 100
  sorted = std::make_shared<gqe::logical::fetch_relation>(std::move(sorted), 0, 100);

  // Generate the task graph and execute
  auto logical_plan_handcoded = std::move(sorted);

  gqe::optimizer::optimization_configuration logical_rule_config(
    {gqe::optimizer::logical_optimization_rule_type::uniqueness_propagation,
     gqe::optimizer::logical_optimization_rule_type::join_unique_keys},
    {});
  gqe::optimizer::logical_optimizer optimizer(&logical_rule_config, &tpcds_catalog);
  auto logical_plan = optimizer.optimize(logical_plan_handcoded);

  gqe::physical_plan_builder plan_builder(&tpcds_catalog);
  auto physical_plan = plan_builder.build(logical_plan.get());

  gqe::query_context query_ctx(gqe::optimization_parameters{});
  gqe::context_reference ctx_ref{&task_manager_ctx, &query_ctx};

  gqe::task_graph_builder graph_builder(ctx_ref, &tpcds_catalog);
  auto task_graph = graph_builder.build(physical_plan.get());

  auto start_time = std::chrono::high_resolution_clock::now();
  gqe::execute_task_graph_single_gpu(ctx_ref, task_graph.get());
  auto stop_time = std::chrono::high_resolution_clock::now();
  auto duration  = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time);
  std::cout << "Execution time: " << duration.count() << " ms." << std::endl;

  // Output the result to disk
  assert(task_graph->root_tasks.size() == 1);
  auto destination = cudf::io::sink_info("output.parquet");
  auto options     = cudf::io::parquet_writer_options::builder(
    destination, task_graph->root_tasks[0]->result().value());
  cudf::io::write_parquet(options);

  // Output performance information to disk
  std::ofstream out;
  out.open("bandwidth.json");
  out << query_ctx.disk_timer.to_string();
  out << query_ctx.h2d_timer.to_string();
  out << query_ctx.decomp_timer.to_string();
  out << query_ctx.decode_timer.to_string();

  return 0;
}
