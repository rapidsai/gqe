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
#include <gqe/executor/query_context.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/utility/helpers.hpp>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cudf/io/parquet.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

constexpr auto MS_1    = "M";
constexpr auto ES_1    = "4 yr Degree";
constexpr auto MS_2    = "D";
constexpr auto ES_2    = "2 yr Degree";
constexpr auto MS_3    = "S";
constexpr auto ES_3    = "College";
constexpr auto STATE_1 = "CO";
constexpr auto STATE_2 = "OH";
constexpr auto STATE_3 = "TX";
constexpr auto STATE_4 = "OR";
constexpr auto STATE_5 = "MN";
constexpr auto STATE_6 = "KY";
constexpr auto STATE_7 = "VA";
constexpr auto STATE_8 = "CA";
constexpr auto STATE_9 = "MS";
constexpr int64_t YEAR = 2000;

void print_usage()
{
  std::cout << "Run TPC-DS Q48 benchmark with a hardcoded logical plan" << std::endl
            << "./q48 <path-to-dataset>" << std::endl;
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
    partial_filter ? std::vector<std::shared_ptr<gqe::logical::relation>>{partial_filter_haystack}
                   : std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::move(column_names),
    std::move(column_types),
    std::move(table_name),
    std::move(partial_filter));
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

  tpcds_catalog.register_table("date_dim",
                               {{"d_date_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"d_year", cudf::data_type(cudf::type_id::INT64)}},
                               gqe::storage_kind::parquet_file{
                                 gqe::utility::get_parquet_files(dataset_location + "/date_dim")},
                               gqe::partitioning_schema_kind::automatic{});

  tpcds_catalog.register_table("store_sales",
                               {{"ss_sold_date_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"ss_cdemo_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"ss_addr_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"ss_sales_price", cudf::data_type(cudf::type_id::FLOAT64)},
                                {"ss_net_profit", cudf::data_type(cudf::type_id::FLOAT64)},
                                {"ss_quantity", cudf::data_type(cudf::type_id::INT64)},
                                {"ss_store_sk", cudf::data_type(cudf::type_id::INT64)}},
                               gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(
                                 dataset_location + "/store_sales")},
                               gqe::partitioning_schema_kind::automatic{});

  tpcds_catalog.register_table(
    "store",
    {{"s_store_sk", cudf::data_type(cudf::type_id::INT64)}},
    gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(dataset_location + "/store")},
    gqe::partitioning_schema_kind::automatic{});

  tpcds_catalog.register_table("customer_demographics",
                               {{"cd_demo_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"cd_marital_status", cudf::data_type(cudf::type_id::STRING)},
                                {"cd_education_status", cudf::data_type(cudf::type_id::STRING)}},
                               gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(
                                 dataset_location + "/customer_demographics")},
                               gqe::partitioning_schema_kind::automatic{});

  tpcds_catalog.register_table("customer_address",
                               {{"ca_address_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"ca_country", cudf::data_type(cudf::type_id::STRING)},
                                {"ca_state", cudf::data_type(cudf::type_id::STRING)}},
                               gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(
                                 dataset_location + "/customer_address")},
                               gqe::partitioning_schema_kind::automatic{});

  // Hand-code the logical plan
  std::shared_ptr<gqe::logical::relation> date_dim_table =
    read_table("date_dim", {"d_date_sk", "d_year"}, &tpcds_catalog);
  date_dim_table = std::make_shared<gqe::logical::filter_relation>(
    std::move(date_dim_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::make_unique<gqe::equal_expression>(
      std::make_shared<gqe::column_reference_expression>(1),
      std::make_shared<gqe::literal_expression<int64_t>>(YEAR)));

  // Predicate pushdown via partial filter
  std::vector<std::unique_ptr<gqe::expression>> col_0_exprs;
  col_0_exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(0));

  auto const partial_filter_haystack = std::make_shared<gqe::logical::project_relation>(
    date_dim_table,
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery relations
    std::move(col_0_exprs));

  auto partial_filter = std::make_unique<gqe::in_predicate_expression>(
    std::vector<std::shared_ptr<gqe::expression>>{
      std::make_shared<gqe::column_reference_expression>(0)},  // ss_sold_date_sk
    0);

  std::shared_ptr<gqe::logical::relation> store_sales_table =
    read_table("store_sales",
               {"ss_sold_date_sk",
                "ss_cdemo_sk",
                "ss_addr_sk",
                "ss_sales_price",
                "ss_net_profit",
                "ss_quantity",
                "ss_store_sk"},
               &tpcds_catalog,
               std::move(partial_filter_haystack),
               std::move(partial_filter));

  // Join `store_sales_table` with the store table
  // After this operation, `store_sales_table` contains columns
  // ["ss_sold_date_sk", "ss_cdemo_sk", "ss_addr_sk", "ss_sales_price", "ss_net_profit",
  // "ss_quantity"]
  auto store_table = read_table("store", {"s_store_sk"}, &tpcds_catalog);

  // FIXME: We can simply include non-NULL "ss_store_sk" instead of a join
  store_sales_table = std::make_shared<gqe::logical::join_relation>(
    std::move(store_sales_table),
    std::move(store_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(6),
                                            std::make_shared<gqe::column_reference_expression>(7)),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>({0, 1, 2, 3, 4, 5}));

  // Join `store_sales_table` with the `customer_demographics` table
  // After this operation, `store_sales_table` contains columns
  // ["ss_addr_sk", "ss_sales_price", "ss_net_profit", "ss_quantity", "cd_marital_status",
  // "cd_education_status"]
  auto customer_demographics_table =
    read_table("customer_demographics",
               {"cd_demo_sk", "cd_marital_status", "cd_education_status"},
               &tpcds_catalog);

  store_sales_table = std::make_shared<gqe::logical::join_relation>(
    std::move(store_sales_table),
    std::move(customer_demographics_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(1),
                                            std::make_shared<gqe::column_reference_expression>(6)),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>({2, 3, 4, 5, 7, 8}));

  // Filter the `store_sales_table` with customer demographics
  // After this operation, `store_sales_table` contains columns
  // ["ss_addr_sk", "ss_sales_price", "ss_net_profit", "ss_quantity", "cd_marital_status",
  // "cd_education_status"]
  auto filter_customer_demographics_expr = [](std::string cd_marital_status,
                                              std::string cd_education_status,
                                              double ss_sales_price_low,
                                              double ss_sales_price_high) {
    auto cond_0 = std::make_shared<gqe::equal_expression>(
      std::make_shared<gqe::column_reference_expression>(4),
      std::make_shared<gqe::literal_expression<std::string>>(cd_marital_status));
    auto cond_1 = std::make_shared<gqe::equal_expression>(
      std::make_shared<gqe::column_reference_expression>(5),
      std::make_shared<gqe::literal_expression<std::string>>(cd_education_status));
    auto cond_2 = std::make_shared<gqe::greater_equal_expression>(
      std::make_shared<gqe::column_reference_expression>(1),
      std::make_shared<gqe::literal_expression<double>>(ss_sales_price_low));
    auto cond_3 = std::make_shared<gqe::less_equal_expression>(
      std::make_shared<gqe::column_reference_expression>(1),
      std::make_shared<gqe::literal_expression<double>>(ss_sales_price_high));

    return std::make_shared<gqe::logical_and_expression>(
      std::make_shared<gqe::logical_and_expression>(
        std::make_shared<gqe::logical_and_expression>(std::move(cond_0), std::move(cond_1)),
        std::move(cond_2)),
      std::move(cond_3));
  };

  store_sales_table = std::make_shared<gqe::logical::filter_relation>(
    std::move(store_sales_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::logical_or_expression>(
      std::make_shared<gqe::logical_or_expression>(
        filter_customer_demographics_expr(MS_1, ES_1, 100.0, 150.0),
        filter_customer_demographics_expr(MS_2, ES_2, 50.0, 100.0)),
      filter_customer_demographics_expr(MS_3, ES_3, 150.0, 200.0)));

  // Filter customer_address table with predicate ca_country = 'United States'
  // After this operation, `customer_address_table` contains columns
  // ["ca_address_sk", "ca_country", "ca_state"]
  std::shared_ptr<gqe::logical::relation> customer_address_table =
    read_table("customer_address", {"ca_address_sk", "ca_country", "ca_state"}, &tpcds_catalog);

  customer_address_table = std::make_shared<gqe::logical::filter_relation>(
    std::move(customer_address_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::equal_expression>(
      std::make_shared<gqe::column_reference_expression>(1),
      std::make_shared<gqe::literal_expression<std::string>>("United States")));

  // Join `store_sales_table` with `customer_address_table`
  // After this operation, `store_sales_table` contains columns
  // ["ss_net_profit", "ss_quantity", "ca_state"]
  store_sales_table = std::make_shared<gqe::logical::join_relation>(
    std::move(store_sales_table),
    std::move(customer_address_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                            std::make_shared<gqe::column_reference_expression>(6)),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>({2, 3, 8}));

  // Filter `store_sales_table` with customer address
  // After this operation, `store_sales_table` contains columns
  // ["ss_net_profit", "ss_quantity", "ca_state"]
  auto filter_customer_address_expr = [](std::string ca_state_0,
                                         std::string ca_state_1,
                                         std::string ca_state_2,
                                         double ss_net_profit_low,
                                         double ss_net_profit_high) {
    auto ca_state_expr_0 = std::make_shared<gqe::equal_expression>(
      std::make_shared<gqe::column_reference_expression>(2),
      std::make_shared<gqe::literal_expression<std::string>>(ca_state_0));
    auto ca_state_expr_1 = std::make_shared<gqe::equal_expression>(
      std::make_shared<gqe::column_reference_expression>(2),
      std::make_shared<gqe::literal_expression<std::string>>(ca_state_1));
    auto ca_state_expr_2 = std::make_shared<gqe::equal_expression>(
      std::make_shared<gqe::column_reference_expression>(2),
      std::make_shared<gqe::literal_expression<std::string>>(ca_state_2));

    auto ca_state_expr = std::make_shared<gqe::logical_or_expression>(
      std::make_shared<gqe::logical_or_expression>(std::move(ca_state_expr_0),
                                                   std::move(ca_state_expr_1)),
      std::move(ca_state_expr_2));

    auto cond_0 = std::make_shared<gqe::greater_equal_expression>(
      std::make_shared<gqe::column_reference_expression>(0),
      std::make_shared<gqe::literal_expression<double>>(ss_net_profit_low));
    auto cond_1 = std::make_shared<gqe::less_equal_expression>(
      std::make_shared<gqe::column_reference_expression>(0),
      std::make_shared<gqe::literal_expression<double>>(ss_net_profit_high));

    return std::make_shared<gqe::logical_and_expression>(
      std::make_shared<gqe::logical_and_expression>(std::move(cond_0), std::move(cond_1)),
      std::move(ca_state_expr));
  };

  store_sales_table = std::make_shared<gqe::logical::filter_relation>(
    std::move(store_sales_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::logical_or_expression>(
      std::make_shared<gqe::logical_or_expression>(
        filter_customer_address_expr(STATE_1, STATE_2, STATE_3, 0, 2000),
        filter_customer_address_expr(STATE_4, STATE_5, STATE_6, 150, 3000)),
      filter_customer_address_expr(STATE_7, STATE_8, STATE_9, 50, 25000)));

  // Calculate sum (ss_quantity)
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> measures;
  measures.emplace_back(cudf::aggregation::SUM,
                        std::make_unique<gqe::column_reference_expression>(1));

  store_sales_table = std::make_shared<gqe::logical::aggregate_relation>(
    std::move(store_sales_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::vector<std::unique_ptr<gqe::expression>>(),
    std::move(measures));

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

  return 0;
}
