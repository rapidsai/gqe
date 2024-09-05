/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/expression/subquery.hpp>
#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/fetch.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/sort.hpp>
#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/query_context.hpp>
#include <gqe/utility/helpers.hpp>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cudf/io/parquet.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

constexpr int64_t YEAR  = 2001;  // Default to the qualification substitution parameters
constexpr int64_t MONTH = 1;

void print_usage()
{
  std::cout << "Run TPC-DS Q6 benchmark with hardcoded logical plan" << std::endl
            << "./q6 <path-to-dataset>" << std::endl;
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
  gqe::catalog tpcds_catalog;
  tpcds_catalog.register_table("store_sales",
                               {{"ss_item_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"ss_sold_date_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"ss_customer_sk", cudf::data_type(cudf::type_id::INT64)}},
                               gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(
                                 dataset_location + "/store_sales")},
                               gqe::partitioning_schema_kind::automatic{});
  tpcds_catalog.register_table("date_dim",
                               {{"d_date_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"d_year", cudf::data_type(cudf::type_id::INT64)},
                                {"d_moy", cudf::data_type(cudf::type_id::INT64)},
                                {"d_month_seq", cudf::data_type(cudf::type_id::INT64)}},
                               gqe::storage_kind::parquet_file{
                                 gqe::utility::get_parquet_files(dataset_location + "/date_dim")},
                               gqe::partitioning_schema_kind::automatic{});
  tpcds_catalog.register_table(
    "item",
    {{"i_item_sk", cudf::data_type(cudf::type_id::INT64)},
     {"i_current_price", cudf::data_type(cudf::type_id::FLOAT64)},
     {"i_category", cudf::data_type(cudf::type_id::STRING)}},
    gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(dataset_location + "/item")},
    gqe::partitioning_schema_kind::automatic{});
  tpcds_catalog.register_table("customer",
                               {{"c_customer_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"c_current_addr_sk", cudf::data_type(cudf::type_id::INT64)}},
                               gqe::storage_kind::parquet_file{
                                 gqe::utility::get_parquet_files(dataset_location + "/customer")},
                               gqe::partitioning_schema_kind::automatic{});
  tpcds_catalog.register_table("customer_address",
                               {{"ca_address_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"ca_state", cudf::data_type(cudf::type_id::STRING)}},
                               gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(
                                 dataset_location + "/customer_address")},
                               gqe::partitioning_schema_kind::automatic{});

  // Hand-code the logical plan
  std::shared_ptr<gqe::logical::relation> date_dim_table =
    read_table("date_dim", {"d_date_sk", "d_year", "d_moy", "d_month_seq"}, &tpcds_catalog);

  auto filtered_date_dim_table = std::make_shared<gqe::logical::filter_relation>(
    date_dim_table,
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::logical_and_expression>(
      std::make_shared<gqe::equal_expression>(
        std::make_shared<gqe::column_reference_expression>(1),
        std::make_shared<gqe::literal_expression<int64_t>>(YEAR)),
      std::make_shared<gqe::equal_expression>(
        std::make_shared<gqe::column_reference_expression>(2),
        std::make_shared<gqe::literal_expression<int64_t>>(MONTH))),
    std::vector<cudf::size_type>({0, 1, 2, 3}));

  // Filter based on d.d_month_seq = (SELECT distinct (d_month_seq) FROM date_dim WHERE d_year =
  // 2001 AND d_moy = 1)
  // After this operation, `date_dim_table` contains columns ["d_date_sk"]
  date_dim_table = std::make_shared<gqe::logical::join_relation>(
    std::move(date_dim_table),
    std::move(filtered_date_dim_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(3),
                                            std::make_shared<gqe::column_reference_expression>(7)),
    gqe::join_type_type::left_semi,
    std::vector<cudf::size_type>({0}));

  // Aggregate the "item" table
  // After this operation, `agg_item_table` contains columns ["i_category", "i_current_price"]
  std::shared_ptr<gqe::logical::relation> item_table =
    read_table("item", {"i_item_sk", "i_current_price", "i_category"}, &tpcds_catalog);

  std::vector<std::unique_ptr<gqe::expression>> item_agg_keys;
  item_agg_keys.push_back(std::make_unique<gqe::column_reference_expression>(2));

  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>>
    item_agg_measures;
  item_agg_measures.emplace_back(cudf::aggregation::MEAN,
                                 std::make_unique<gqe::column_reference_expression>(1));

  auto agg_item_table = std::make_shared<gqe::logical::aggregate_relation>(
    item_table,
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::move(item_agg_keys),
    std::move(item_agg_measures));

  // Convert subquery (SELECT avg(j.i_current_price) FROM item j WHERE j.i_category = i.i_category)
  // into a join
  // After this operation, `item_table` contains columns
  // ["i_item_sk", "i_current_price", "i_current_price"]
  item_table = std::make_shared<gqe::logical::join_relation>(
    std::move(item_table),
    std::move(agg_item_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(2),
                                            std::make_shared<gqe::column_reference_expression>(3)),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>({0, 1, 4}));

  // Filter i.i_current_price > 1.2 * ...
  // After this operation, `item_table` contains columns
  // ["i_item_sk", "i_current_price", "i_current_price"]
  item_table = std::make_shared<gqe::logical::filter_relation>(
    std::move(item_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::greater_expression>(
      std::make_shared<gqe::column_reference_expression>(1),
      std::make_shared<gqe::multiply_expression>(
        std::make_shared<gqe::column_reference_expression>(2),
        std::make_shared<gqe::literal_expression<double>>(1.2))),
    std::vector<cudf::size_type>({0, 1, 2}));

  // Join "store_sales" table with `date_dim_table`
  // After this operation, `store_sales_table` contains columns ["ss_item_sk", "ss_customer_sk"]
  std::vector<std::unique_ptr<gqe::expression>> project_exprs;
  project_exprs.push_back(std::make_unique<gqe::column_reference_expression>(0));

  auto partial_filter_haystack = std::make_shared<gqe::logical::project_relation>(
    date_dim_table,
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery relations
    std::move(project_exprs));

  auto partial_filter = std::make_unique<gqe::in_predicate_expression>(
    std::vector<std::shared_ptr<gqe::expression>>{
      std::make_shared<gqe::column_reference_expression>(0)},  // ss_sold_date_sk
    0);

  std::shared_ptr<gqe::logical::relation> store_sales_table =
    read_table("store_sales",
               {"ss_sold_date_sk", "ss_item_sk", "ss_customer_sk"},
               &tpcds_catalog,
               std::move(partial_filter_haystack),
               std::move(partial_filter));

  store_sales_table = std::make_shared<gqe::logical::join_relation>(
    std::move(store_sales_table),
    std::move(date_dim_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                            std::make_shared<gqe::column_reference_expression>(3)),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>({1, 2}));

  // Join `store_sales_table` with `item_table`
  // After this operation, `store_sales_table` contains columns ["ss_customer_sk"]
  store_sales_table = std::make_shared<gqe::logical::join_relation>(
    std::move(store_sales_table),
    std::move(item_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                            std::make_shared<gqe::column_reference_expression>(2)),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>({1}));

  // Join `store_sales_table` with the "customer" table
  // After this operation, `store_sales_table` contains columns ["c_current_addr_sk"]
  auto customer_table =
    read_table("customer", {"c_customer_sk", "c_current_addr_sk"}, &tpcds_catalog);

  store_sales_table = std::make_shared<gqe::logical::join_relation>(
    std::move(store_sales_table),
    std::move(customer_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                            std::make_shared<gqe::column_reference_expression>(1)),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>({2}));

  // Join `store_sales_table` with the "customer_address" table
  // After this operation, `store_sales_table` contains columns ["ca_state"]
  auto customer_address_table =
    read_table("customer_address", {"ca_address_sk", "ca_state"}, &tpcds_catalog);

  store_sales_table = std::make_shared<gqe::logical::join_relation>(
    std::move(store_sales_table),
    std::move(customer_address_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                            std::make_shared<gqe::column_reference_expression>(1)),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>({2}));

  // GROUP BY a.ca_state
  // After this operation, `store_sales_table` contains columns ["ca_state", count(*)]
  std::vector<std::unique_ptr<gqe::expression>> state_agg_keys;
  state_agg_keys.push_back(std::make_unique<gqe::column_reference_expression>(0));

  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>>
    state_agg_measures;
  state_agg_measures.emplace_back(cudf::aggregation::COUNT_ALL,
                                  std::make_unique<gqe::column_reference_expression>(0));

  store_sales_table = std::make_shared<gqe::logical::aggregate_relation>(
    std::move(store_sales_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::move(state_agg_keys),
    std::move(state_agg_measures));

  // Filter cnt >= 10
  // After this operation, `store_sales_table` contains columns ["ca_state", count(*)]
  store_sales_table = std::make_shared<gqe::logical::filter_relation>(
    std::move(store_sales_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::greater_equal_expression>(
      std::make_shared<gqe::column_reference_expression>(1),
      std::make_shared<gqe::literal_expression<int64_t>>(10)),
    std::vector<cudf::size_type>({0, 1}));

  // ORDER BY cnt
  // After this operation, `store_sales_table` contains columns ["ca_state", count(*)]
  std::vector<std::unique_ptr<gqe::expression>> sort_exprs;
  sort_exprs.push_back(std::make_unique<gqe::column_reference_expression>(1));

  store_sales_table = std::make_shared<gqe::logical::sort_relation>(
    std::move(store_sales_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::vector<cudf::order>({cudf::order::ASCENDING}),
    std::vector<cudf::null_order>({cudf::null_order::BEFORE}),
    std::move(sort_exprs));

  // LIMIT 100
  store_sales_table =
    std::make_shared<gqe::logical::fetch_relation>(std::move(store_sales_table), 0, 100);

  // Execution
  auto logical_plan = std::move(store_sales_table);

  gqe::physical_plan_builder plan_builder(&tpcds_catalog);
  auto physical_plan = plan_builder.build(logical_plan.get());

  gqe::query_context qctx(gqe::optimization_parameters{});

  gqe::task_graph_builder graph_builder(&qctx, &tpcds_catalog);
  auto task_graph = graph_builder.build(physical_plan.get());

  gqe::utility::time_function(gqe::execute_task_graph_single_gpu, &qctx, task_graph.get());

  // Output the result to disk
  assert(task_graph->root_tasks.size() == 1);
  auto destination = cudf::io::sink_info("output.parquet");
  auto options     = cudf::io::parquet_writer_options::builder(
    destination, task_graph->root_tasks[0]->result().value());
  cudf::io::write_parquet(options);

  // Output performance information to disk
  std::ofstream out;
  out.open("bandwidth.json");
  out << qctx.disk_timer.to_string();
  out << qctx.h2d_timer.to_string();
  out << qctx.decomp_timer.to_string();
  out << qctx.decode_timer.to_string();

  return 0;
}
