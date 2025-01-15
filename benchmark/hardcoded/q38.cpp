/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
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
#include <gqe/context_reference.hpp>
#include <gqe/executor/optimization_parameters.hpp>
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
#include <gqe/query_context.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/utility/helpers.hpp>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

constexpr int64_t DMS = 1200;  // Default to the qualification substitution parameters

void print_usage()
{
  std::cout << "Run TPC-DS Q38 benchmark with a hardcoded logical plan" << std::endl
            << "./q38 <path-to-dataset>" << std::endl;
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

// This helper function implements the following part of the SQL query for one of the store_sales,
// catalog_sales or web_sales table.
//
//    select distinct c_last_name, c_first_name, d_date
//    from store_sales, date_dim, customer
//          where store_sales.ss_sold_date_sk = date_dim.d_date_sk
//      and store_sales.ss_customer_sk = customer.c_customer_sk
//      and d_month_seq between [DMS] and [DMS] + 11
//
// The output table contains columns ["c_last_name", "c_first_name", "d_date"]
std::shared_ptr<gqe::logical::relation> process_sales_table(
  std::string table_name,
  std::vector<std::string> column_names,
  std::shared_ptr<gqe::logical::relation> const& date_dim_table,
  std::shared_ptr<gqe::logical::relation> const& customer_table,
  gqe::catalog const* tpcds_catalog)
{
  assert(column_names.size() == 2);

  // predicate pushdown via partial filter
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

  // The exact column name varies based on different sales table. We use "store_sales" table here as
  // an example.
  // After this operation, sales_table contains columns
  // ["ss_sold_date_sk", "ss_customer_sk"]
  std::shared_ptr<gqe::logical::relation> sales_table =
    read_table(table_name,
               column_names,
               tpcds_catalog,
               std::move(partial_filter_haystack),
               std::move(partial_filter));

  // std::shared_ptr<gqe::logical::relation> sales_table =
  //  read_table(table_name, column_names, tpcds_catalog);

  // After this operation, sales_table contains columns
  // ["ss_customer_sk", "d_date"]
  sales_table = std::make_shared<gqe::logical::join_relation>(
    std::move(sales_table),
    date_dim_table,
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                            std::make_shared<gqe::column_reference_expression>(2)),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>({1, 4}));

  // After this operation, sales_table contains columns
  // ["c_last_name", "c_first_name", "d_date"]
  sales_table = std::make_shared<gqe::logical::join_relation>(
    std::move(sales_table),
    customer_table,
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                            std::make_shared<gqe::column_reference_expression>(2)),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>({3, 4, 1}));

  // Use aggregate relation to drop duplicates
  // After this operation, sales_table contains columns
  // ["c_last_name", "c_first_name", "d_date"]
  std::vector<std::unique_ptr<gqe::expression>> agg_keys;
  agg_keys.push_back(std::make_unique<gqe::column_reference_expression>(0));
  agg_keys.push_back(std::make_unique<gqe::column_reference_expression>(1));
  agg_keys.push_back(std::make_unique<gqe::column_reference_expression>(2));

  sales_table = std::make_shared<gqe::logical::aggregate_relation>(
    std::move(sales_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::move(agg_keys),
    std::vector<std::pair<cudf::aggregation::Kind,
                          std::unique_ptr<gqe::expression>>>()  // values are empty
  );

  return sales_table;
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
                               {{"ss_sold_date_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"ss_customer_sk", cudf::data_type(cudf::type_id::INT64)}},
                               gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(
                                 dataset_location + "/store_sales")},
                               gqe::partitioning_schema_kind::automatic{});

  tpcds_catalog.register_table("catalog_sales",
                               {{"cs_sold_date_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"cs_bill_customer_sk", cudf::data_type(cudf::type_id::INT64)}},
                               gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(
                                 dataset_location + "/catalog_sales")},
                               gqe::partitioning_schema_kind::automatic{});

  tpcds_catalog.register_table("web_sales",
                               {{"ws_sold_date_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"ws_bill_customer_sk", cudf::data_type(cudf::type_id::INT64)}},
                               gqe::storage_kind::parquet_file{
                                 gqe::utility::get_parquet_files(dataset_location + "/web_sales")},
                               gqe::partitioning_schema_kind::automatic{});

  tpcds_catalog.register_table("date_dim",
                               {{"d_date_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"d_month_seq", cudf::data_type(cudf::type_id::INT64)},
                                {"d_date", cudf::data_type(cudf::type_id::TIMESTAMP_DAYS)}},
                               gqe::storage_kind::parquet_file{
                                 gqe::utility::get_parquet_files(dataset_location + "/date_dim")},
                               gqe::partitioning_schema_kind::automatic{});

  tpcds_catalog.register_table("customer",
                               {{"c_customer_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"c_last_name", cudf::data_type(cudf::type_id::STRING)},
                                {"c_first_name", cudf::data_type(cudf::type_id::STRING)}},
                               gqe::storage_kind::parquet_file{
                                 gqe::utility::get_parquet_files(dataset_location + "/customer")},
                               gqe::partitioning_schema_kind::automatic{});

  // Hand-code the logical plan

  // After this operation, date_dim_table contains columns
  // ["d_date_sk", "d_month_seq", "d_date"]
  std::shared_ptr<gqe::logical::relation> date_dim_table =
    read_table("date_dim", {"d_date_sk", "d_month_seq", "d_date"}, &tpcds_catalog);
  date_dim_table = std::make_shared<gqe::logical::filter_relation>(
    std::move(date_dim_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::logical_and_expression>(
      std::make_shared<gqe::greater_equal_expression>(
        std::make_shared<gqe::column_reference_expression>(1),
        std::make_shared<gqe::literal_expression<int64_t>>(DMS)),
      std::make_shared<gqe::less_equal_expression>(
        std::make_shared<gqe::column_reference_expression>(1),
        std::make_shared<gqe::literal_expression<int64_t>>(DMS + 11))),
    std::vector<cudf::size_type>({0, 1, 2}));

  // After this operation, customer_table contains columns
  // ["c_customer_sk", "c_last_name", "c_first_name"]
  std::shared_ptr<gqe::logical::relation> customer_table =
    read_table("customer", {"c_customer_sk", "c_last_name", "c_first_name"}, &tpcds_catalog);

  auto store_sales_table = process_sales_table("store_sales",
                                               {"ss_sold_date_sk", "ss_customer_sk"},
                                               date_dim_table,
                                               customer_table,
                                               &tpcds_catalog);

  auto catalog_sales_table = process_sales_table("catalog_sales",
                                                 {"cs_sold_date_sk", "cs_bill_customer_sk"},
                                                 date_dim_table,
                                                 customer_table,
                                                 &tpcds_catalog);

  auto web_sales_table = process_sales_table("web_sales",
                                             {"ws_sold_date_sk", "ws_bill_customer_sk"},
                                             date_dim_table,
                                             customer_table,
                                             &tpcds_catalog);

  // Since sales tables have already been de-duplicated, we can use inner joins to implement
  // intersections. Note that SQL considers NULL keys to be equal in set operations.
  auto intersect_condition = std::make_unique<gqe::logical_and_expression>(
    std::make_shared<gqe::logical_and_expression>(
      std::make_shared<gqe::nulls_equal_expression>(
        std::make_shared<gqe::column_reference_expression>(0),
        std::make_shared<gqe::column_reference_expression>(3)),
      std::make_shared<gqe::nulls_equal_expression>(
        std::make_shared<gqe::column_reference_expression>(1),
        std::make_shared<gqe::column_reference_expression>(4))),
    std::make_shared<gqe::nulls_equal_expression>(
      std::make_shared<gqe::column_reference_expression>(2),
      std::make_shared<gqe::column_reference_expression>(5)));

  auto catalog_web_intersection = std::make_shared<gqe::logical::join_relation>(
    std::move(catalog_sales_table),
    std::move(web_sales_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    intersect_condition->clone(),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>({0, 1, 2}));

  auto logical_plan = std::make_shared<gqe::logical::join_relation>(
    std::move(store_sales_table),
    std::move(catalog_web_intersection),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    intersect_condition->clone(),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>({0, 1, 2}));

  gqe::physical_plan_builder plan_builder(&tpcds_catalog);
  auto physical_plan = plan_builder.build(logical_plan.get());

  gqe::task_manager_context dbctx{};
  gqe::query_context qctx(gqe::optimization_parameters{});
  gqe::context_reference ctx_ref{&dbctx, &qctx};

  gqe::task_graph_builder graph_builder(ctx_ref, &tpcds_catalog);
  auto task_graph = graph_builder.build(physical_plan.get());

  gqe::utility::time_function(gqe::execute_task_graph_single_gpu, ctx_ref, task_graph.get());

  assert(task_graph->root_tasks.size() == 1);
  std::cout << "Result: " << task_graph->root_tasks[0]->result().value().num_rows() << std::endl;

  // Output performance information to disk
  std::ofstream out;
  out.open("bandwidth.json");
  out << qctx.disk_timer.to_string();
  out << qctx.h2d_timer.to_string();
  out << qctx.decomp_timer.to_string();
  out << qctx.decode_timer.to_string();

  return 0;
}
