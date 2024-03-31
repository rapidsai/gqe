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
#include <gqe/expression/if_then_else.hpp>
#include <gqe/expression/literal.hpp>
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
#include <memory>
#include <string>

/* TPC-DS Q43
select s_store_name, s_store_id,
       sum(case when (d_day_name='Sunday') then ss_sales_price else null end) sun_sales,
       sum(case when (d_day_name='Monday') then ss_sales_price else null end) mon_sales,
       sum(case when (d_day_name='Tuesday') then ss_sales_price else  null end) tue_sales,
       sum(case when (d_day_name='Wednesday') then ss_sales_price else null end) wed_sales,
       sum(case when (d_day_name='Thursday') then ss_sales_price else null end) thu_sales,
       sum(case when (d_day_name='Friday') then ss_sales_price else null end) fri_sales,
       sum(case when (d_day_name='Saturday') then ss_sales_price else null end) sat_sales
  from date_dim, store_sales, store
  where d_date_sk = ss_sold_date_sk and
        s_store_sk = ss_store_sk and
        s_gmt_offset = -5 and
        d_year = 2000
  group by s_store_name, s_store_id
  order by s_store_name, s_store_id,
                         sun_sales,mon_sales,tue_sales,wed_sales,thu_sales,fri_sales,sat_sales
  limit 100
*/

void print_usage()
{
  std::cout << "Run TPC-DS Q43 benchmark with hardcoded logical plan" << std::endl
            << "./q43 <path-to-dataset>" << std::endl;
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
  tpcds_catalog.register_table("date_dim",
                               {{"d_date_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"d_day_name", cudf::data_type(cudf::type_id::STRING)},
                                {"d_year", cudf::data_type(cudf::type_id::INT64)}},
                               gqe::storage_kind::parquet_file{
                                 gqe::utility::get_parquet_files(dataset_location + "/date_dim")},
                               gqe::partitioning_schema_kind::automatic{});
  tpcds_catalog.register_table("store_sales",
                               {{"ss_sold_date_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"ss_store_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"ss_sales_price", cudf::data_type(cudf::type_id::FLOAT64)}},
                               gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(
                                 dataset_location + "/store_sales")},
                               gqe::partitioning_schema_kind::automatic{});
  tpcds_catalog.register_table(
    "store",
    {{"s_store_sk", cudf::data_type(cudf::type_id::INT64)},
     {"s_store_id", cudf::data_type(cudf::type_id::STRING)},
     {"s_store_name", cudf::data_type(cudf::type_id::STRING)},
     {"s_gmt_offset", cudf::data_type(cudf::type_id::FLOAT64)}},
    gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(dataset_location + "/store")},
    gqe::partitioning_schema_kind::automatic{});

  // Hand-code the logical plan
  std::shared_ptr<gqe::logical::relation> date_dim_table =
    read_table("date_dim", {"d_date_sk", "d_day_name", "d_year"}, &tpcds_catalog);
  date_dim_table = std::make_shared<gqe::logical::filter_relation>(
    std::move(date_dim_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::equal_expression>(
      std::make_shared<gqe::column_reference_expression>(2),
      std::make_shared<gqe::literal_expression<int64_t>>(2000)));  // filter by d_year = 2000

  std::shared_ptr<gqe::logical::relation> store_table = read_table(
    "store", {"s_store_sk", "s_store_id", "s_store_name", "s_gmt_offset"}, &tpcds_catalog);
  store_table = std::make_shared<gqe::logical::filter_relation>(
    std::move(store_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::equal_expression>(
      std::make_shared<gqe::column_reference_expression>(3),
      std::make_shared<gqe::literal_expression<double>>(-5)));  // filter by s_gmt_offset = -5

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

  std::shared_ptr<gqe::logical::relation> store_sales_table =
    read_table("store_sales",
               {"ss_sold_date_sk", "ss_store_sk", "ss_sales_price"},
               &tpcds_catalog,
               std::move(partial_filter_haystack),
               std::move(partial_filter));

  // Join `store_sales_table` with `date_dim_table`
  // After this operation, `store_sales_table` contains columns ["ss_store_sk", "ss_sales_price",
  // "d_day_name"]
  store_sales_table = std::make_shared<gqe::logical::join_relation>(
    std::move(store_sales_table),
    std::move(date_dim_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                            std::make_shared<gqe::column_reference_expression>(3)),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>({1, 2, 4}));

  // Join `store_sales_table` with `store_table`
  // After this operation, `store_sales_table` contains columns ["s_store_name", "s_store_id",
  // "d_day_name", "ss_sales_price"]
  store_sales_table = std::make_shared<gqe::logical::join_relation>(
    std::move(store_sales_table),
    std::move(store_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::make_unique<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                            std::make_shared<gqe::column_reference_expression>(3)),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>({5, 4, 2, 1}));

  std::vector<std::unique_ptr<gqe::expression>> agg_keys;
  agg_keys.push_back(std::make_unique<gqe::column_reference_expression>(0));
  agg_keys.push_back(std::make_unique<gqe::column_reference_expression>(1));

  auto day_expr         = std::make_shared<gqe::column_reference_expression>(2);
  auto sales_price_expr = std::make_shared<gqe::column_reference_expression>(3);
  auto null_expr        = std::make_shared<gqe::literal_expression<double>>(0, true);

  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> agg_measures;
  for (auto const day :
       {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"}) {
    agg_measures.emplace_back(
      cudf::aggregation::SUM,
      std::make_unique<gqe::if_then_else_expression>(
        std::make_shared<gqe::equal_expression>(
          day_expr, std::make_shared<gqe::literal_expression<std::string>>(day)),
        sales_price_expr,
        null_expr));
  }

  // GROUP BY "s_store_name", "s_store_id"
  store_sales_table = std::make_shared<gqe::logical::aggregate_relation>(
    std::move(store_sales_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::move(agg_keys),
    std::move(agg_measures));

  std::vector<std::unique_ptr<gqe::expression>> sort_exprs;
  for (cudf::size_type col = 0; col < 9; ++col) {
    sort_exprs.push_back(std::make_unique<gqe::column_reference_expression>(col));
  }

  // ORDER BY all columns
  store_sales_table = std::make_shared<gqe::logical::sort_relation>(
    std::move(store_sales_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::vector<cudf::order>(9, cudf::order::ASCENDING),
    std::vector<cudf::null_order>(9, cudf::null_order::BEFORE),
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

  return 0;
}
