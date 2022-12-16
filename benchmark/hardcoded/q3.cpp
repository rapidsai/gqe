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

#include "../utility.hpp"

#include <gqe/catalog.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/optimizer/physical_transformation.hpp>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cudf/io/parquet.hpp>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

void print_usage()
{
  std::cout << "Run TPC-DS Q3 benchmark with hardcoded logical plan" << std::endl
            << "./q3 <path-to-dataset>" << std::endl;
}

std::shared_ptr<gqe::logical::read_relation> read_table(std::string table_name,
                                                        std::vector<std::string> column_names,
                                                        gqe::catalog const* tpcds_catalog)
{
  std::vector<cudf::data_type> column_types;
  column_types.reserve(column_names.size());
  for (auto const& column_name : column_names)
    column_types.push_back(tpcds_catalog->column_type(table_name, column_name));

  return std::make_shared<gqe::logical::read_relation>(
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::move(column_names),
    std::move(column_types),
    std::move(table_name),
    nullptr);  // partial_filter
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
  rmm::mr::cuda_memory_resource cuda_mr;
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{&cuda_mr};
  rmm::mr::set_current_device_resource(&pool_mr);

  // Register the input tables
  gqe::catalog tpcds_catalog;
  tpcds_catalog.register_table("store_sales",
                               {{"ss_item_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"ss_sold_date_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"ss_ext_sales_price", cudf::data_type(cudf::type_id::FLOAT64)}},
                               gqe::benchmark::get_file_paths(dataset_location + "/store_sales"),
                               gqe::file_format_type::parquet);
  tpcds_catalog.register_table("date_dim",
                               {{"d_date_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"d_year", cudf::data_type(cudf::type_id::INT64)},
                                {"d_moy", cudf::data_type(cudf::type_id::INT64)}},
                               gqe::benchmark::get_file_paths(dataset_location + "/date_dim"),
                               gqe::file_format_type::parquet);
  tpcds_catalog.register_table("item",
                               {{"i_item_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"i_brand_id", cudf::data_type(cudf::type_id::INT64)},
                                {"i_brand", cudf::data_type(cudf::type_id::STRING)},
                                {"i_manufact_id", cudf::data_type(cudf::type_id::INT64)}},
                               gqe::benchmark::get_file_paths(dataset_location + "/item"),
                               gqe::file_format_type::parquet);

  // Hand-code the logical plan
  std::shared_ptr<gqe::logical::relation> date_dim_table =
    read_table("date_dim", {"d_date_sk", "d_year", "d_moy"}, &tpcds_catalog);
  date_dim_table = std::make_shared<gqe::logical::filter_relation>(
    std::move(date_dim_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::make_unique<gqe::equal_expression>(
      std::make_shared<gqe::column_reference_expression>(2),
      std::make_shared<gqe::literal_expression<int64_t>>(11)));

  std::shared_ptr<gqe::logical::relation> item_table =
    read_table("item", {"i_item_sk", "i_brand_id", "i_brand", "i_manufact_id"}, &tpcds_catalog);
  item_table = std::make_shared<gqe::logical::filter_relation>(
    std::move(item_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::make_unique<gqe::equal_expression>(
      std::make_shared<gqe::column_reference_expression>(3),
      std::make_shared<gqe::literal_expression<int64_t>>(128)));

  std::shared_ptr<gqe::logical::relation> store_sales_table = read_table(
    "store_sales", {"ss_item_sk", "ss_sold_date_sk", "ss_ext_sales_price"}, &tpcds_catalog);

  // After this operation, store_sales_table contains columns
  // ["ss_item_sk", "ss_ext_sales_price", "d_year"]
  store_sales_table = std::make_shared<gqe::logical::join_relation>(
    std::move(store_sales_table),
    std::move(date_dim_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::make_unique<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(1),
                                            std::make_shared<gqe::column_reference_expression>(3)),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>({0, 2, 4}));

  // After this operation, store_sales_table contains columns
  // ["ss_ext_sales_price", "d_year", "i_brand_id", "i_brand"]
  store_sales_table = std::make_shared<gqe::logical::join_relation>(
    std::move(store_sales_table),
    std::move(item_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::make_unique<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                            std::make_shared<gqe::column_reference_expression>(3)),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>({1, 2, 4, 5}));

  // Groupby on d_year, i_brand, i_brand_id
  // After this operation, store_sales_table contains columns
  // ["d_year", "i_brand_id", "i_brand", SUM(ss_ext_sales_price)]
  std::vector<std::unique_ptr<gqe::expression>> groupby_keys;
  groupby_keys.push_back(std::make_unique<gqe::column_reference_expression>(1));
  groupby_keys.push_back(std::make_unique<gqe::column_reference_expression>(2));
  groupby_keys.push_back(std::make_unique<gqe::column_reference_expression>(3));

  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> groupby_values;
  groupby_values.push_back(
    std::make_pair(cudf::aggregation::SUM, std::make_unique<gqe::column_reference_expression>(0)));

  store_sales_table = std::make_shared<gqe::logical::aggregate_relation>(
    std::move(store_sales_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::move(groupby_keys),
    std::move(groupby_values));

  // Sort on d_year, SUM(ss_ext_sales_price) desc, brand_id
  // After this operation, store_sales_table contains columns
  // ["d_year", "i_brand_id", "i_brand", SUM(ss_ext_sales_price)]
  std::vector<std::unique_ptr<gqe::expression>> sort_exprs;
  sort_exprs.push_back(std::make_unique<gqe::column_reference_expression>(0));
  sort_exprs.push_back(std::make_unique<gqe::column_reference_expression>(3));
  sort_exprs.push_back(std::make_unique<gqe::column_reference_expression>(1));

  store_sales_table = std::make_shared<gqe::logical::sort_relation>(
    std::move(store_sales_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::vector<cudf::order>(
      {cudf::order::ASCENDING, cudf::order::DESCENDING, cudf::order::ASCENDING}),
    std::vector<cudf::null_order>(
      {cudf::null_order::BEFORE, cudf::null_order::BEFORE, cudf::null_order::BEFORE}),
    std::move(sort_exprs));

  // LIMIT 100
  store_sales_table =
    std::make_shared<gqe::logical::fetch_relation>(std::move(store_sales_table), 0, 100);

  auto logical_plan = std::move(store_sales_table);

  gqe::physical_plan_builder plan_builder(&tpcds_catalog);
  auto physical_plan = plan_builder.build(logical_plan.get());

  gqe::task_graph_builder graph_builder(&tpcds_catalog);
  auto task_graph = graph_builder.build(physical_plan.get());

  gqe::execute_task_graph_single_gpu(task_graph.get());

  // Output the result to disk
  assert(task_graph->root_tasks.size() == 1);
  auto destination = cudf::io::sink_info("output.parquet");
  auto options     = cudf::io::parquet_writer_options::builder(
    destination, task_graph->root_tasks[0]->result().value());
  cudf::io::write_parquet(options);

  return 0;
}
