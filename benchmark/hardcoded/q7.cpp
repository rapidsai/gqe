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
#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/fetch.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/sort.hpp>
#include <gqe/optimizer/physical_transformation.hpp>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cudf/io/parquet.hpp>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

/* TPC-DS Q7:
SELECT i_item_id,
       avg(ss_quantity) agg1,
       avg(ss_list_price) agg2,
       avg(ss_coupon_amt) agg3,
       avg(ss_sales_price) agg4
FROM store_sales, customer_demographics, date_dim, item, promotion
WHERE ss_sold_date_sk = d_date_sk AND
      ss_item_sk = i_item_sk AND
      ss_cdemo_sk = cd_demo_sk AND
      ss_promo_sk = p_promo_sk AND
      cd_gender = 'M' AND
      cd_marital_status = 'S' AND
      cd_education_status = 'College' AND
      (p_channel_email = 'N' or p_channel_event = 'N') AND
      d_year = 2000
GROUP BY i_item_id
ORDER BY i_item_id LIMIT 100
*/

void print_usage()
{
  std::cout << "Run TPC-DS Q7 benchmark with hardcoded logical plan" << std::endl
            << "./q7 <path-to-dataset>" << std::endl;
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
  rmm::mr::cuda_memory_resource cuda_mr;
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{&cuda_mr};
  rmm::mr::set_current_device_resource(&pool_mr);

  // Register the input tables
  gqe::catalog tpcds_catalog;
  tpcds_catalog.register_table("store_sales",
                               {{"ss_quantity", cudf::data_type(cudf::type_id::INT64)},
                                {"ss_list_price", cudf::data_type(cudf::type_id::FLOAT64)},
                                {"ss_coupon_amt", cudf::data_type(cudf::type_id::FLOAT64)},
                                {"ss_sales_price", cudf::data_type(cudf::type_id::FLOAT64)},
                                {"ss_sold_date_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"ss_item_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"ss_cdemo_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"ss_promo_sk", cudf::data_type(cudf::type_id::INT64)}},
                               gqe::benchmark::get_file_paths(dataset_location + "/store_sales"),
                               gqe::file_format_type::parquet);
  tpcds_catalog.register_table(
    "customer_demographics",
    {{"cd_demo_sk", cudf::data_type(cudf::type_id::INT64)},
     {"cd_gender", cudf::data_type(cudf::type_id::STRING)},
     {"cd_marital_status", cudf::data_type(cudf::type_id::STRING)},
     {"cd_education_status", cudf::data_type(cudf::type_id::STRING)}},
    gqe::benchmark::get_file_paths(dataset_location + "/customer_demographics"),
    gqe::file_format_type::parquet);
  tpcds_catalog.register_table("date_dim",
                               {{"d_date_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"d_year", cudf::data_type(cudf::type_id::INT64)}},
                               gqe::benchmark::get_file_paths(dataset_location + "/date_dim"),
                               gqe::file_format_type::parquet);
  tpcds_catalog.register_table("item",
                               {{"i_item_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"i_item_id", cudf::data_type(cudf::type_id::STRING)}},
                               gqe::benchmark::get_file_paths(dataset_location + "/item"),
                               gqe::file_format_type::parquet);
  tpcds_catalog.register_table("promotion",
                               {{"p_promo_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"p_channel_email", cudf::data_type(cudf::type_id::STRING)},
                                {"p_channel_event", cudf::data_type(cudf::type_id::STRING)}},
                               gqe::benchmark::get_file_paths(dataset_location + "/promotion"),
                               gqe::file_format_type::parquet);

  std::shared_ptr<gqe::logical::relation> date_dim_table =
    read_table("date_dim", {"d_date_sk", "d_year"}, &tpcds_catalog);
  date_dim_table = std::make_shared<gqe::logical::filter_relation>(
    std::move(date_dim_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::make_unique<gqe::equal_expression>(
      std::make_shared<gqe::column_reference_expression>(1),
      std::make_shared<gqe::literal_expression<int64_t>>(2000)));

  // predicate pushdown via partial filter
  std::vector<std::unique_ptr<gqe::expression>> col_0_exprs;
  col_0_exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(0));

  auto const partial_filter_haystack = std::make_shared<gqe::logical::project_relation>(
    date_dim_table,
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery relations
    std::move(col_0_exprs));

  auto partial_filter = std::make_unique<gqe::in_predicate_expression>(
    std::vector<std::shared_ptr<gqe::expression>>{
      std::make_shared<gqe::column_reference_expression>(4)},  // ss_sold_date_sk
    0);

  // Hand-code the logical plan
  std::shared_ptr<gqe::logical::relation> store_sales_table =
    read_table("store_sales",
               {"ss_quantity",
                "ss_list_price",
                "ss_coupon_amt",
                "ss_sales_price",
                "ss_sold_date_sk",
                "ss_item_sk",
                "ss_cdemo_sk",
                "ss_promo_sk"},
               &tpcds_catalog,
               std::move(partial_filter_haystack),
               std::move(partial_filter));

  std::shared_ptr<gqe::logical::relation> customer_demographics_table =
    read_table("customer_demographics",
               {"cd_demo_sk", "cd_gender", "cd_marital_status", "cd_education_status"},
               &tpcds_catalog);
  customer_demographics_table = std::make_shared<gqe::logical::filter_relation>(
    std::move(customer_demographics_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::make_unique<gqe::logical_and_expression>(
      std::make_shared<gqe::equal_expression>(
        std::make_shared<gqe::column_reference_expression>(1),
        std::make_shared<gqe::literal_expression<std::string>>("M")),
      std::make_shared<gqe::logical_and_expression>(
        std::make_shared<gqe::equal_expression>(
          std::make_shared<gqe::column_reference_expression>(2),
          std::make_shared<gqe::literal_expression<std::string>>("S")),
        std::make_shared<gqe::equal_expression>(
          std::make_shared<gqe::column_reference_expression>(3),
          std::make_shared<gqe::literal_expression<std::string>>("College")))));

  std::shared_ptr<gqe::logical::relation> item_table =
    read_table("item", {"i_item_sk", "i_item_id"}, &tpcds_catalog);

  std::shared_ptr<gqe::logical::relation> promotion_table =
    read_table("promotion", {"p_promo_sk", "p_channel_email", "p_channel_event"}, &tpcds_catalog);
  promotion_table = std::make_shared<gqe::logical::filter_relation>(
    std::move(promotion_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::make_unique<gqe::logical_or_expression>(
      std::make_shared<gqe::equal_expression>(
        std::make_shared<gqe::column_reference_expression>(1),
        std::make_shared<gqe::literal_expression<std::string>>("N")),
      std::make_shared<gqe::equal_expression>(
        std::make_shared<gqe::column_reference_expression>(2),
        std::make_shared<gqe::literal_expression<std::string>>("N"))));

  // Join store_sales with date_dim on condition "ss_sold_date_sk = d_date_sk"
  // After this operation, store_sales_table contains columns
  // ["ss_quantity", "ss_list_price", "ss_coupon_amt", "ss_sales_price", "ss_item_sk",
  // "ss_cdemo_sk", "ss_promo_sk"]
  store_sales_table = std::make_shared<gqe::logical::join_relation>(
    std::move(store_sales_table),
    std::move(date_dim_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::make_unique<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(4),
                                            std::make_shared<gqe::column_reference_expression>(8)),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>({0, 1, 2, 3, 5, 6, 7}));

  // Join store_sales with customer_demographics on condition "ss_cdemo_sk = cd_demo_sk"
  // After this operation, store_sales_table contains columns
  // ["ss_quantity", "ss_list_price", "ss_coupon_amt", "ss_sales_price", "ss_item_sk",
  // "ss_promo_sk"]
  store_sales_table = std::make_shared<gqe::logical::join_relation>(
    std::move(store_sales_table),
    std::move(customer_demographics_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::make_unique<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(5),
                                            std::make_shared<gqe::column_reference_expression>(7)),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>({0, 1, 2, 3, 4, 6}));

  // Join store_sales with promotion on condition "ss_promo_sk = p_promo_sk"
  // After this operation, store_sales_table contains columns
  // ["ss_quantity", "ss_list_price", "ss_coupon_amt", "ss_sales_price", "ss_item_sk"]
  store_sales_table = std::make_shared<gqe::logical::join_relation>(
    std::move(store_sales_table),
    std::move(promotion_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::make_unique<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(5),
                                            std::make_shared<gqe::column_reference_expression>(6)),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>({0, 1, 2, 3, 4}));

  // Join store_sales with item on condition "ss_item_sk = i_item_sk"
  // After this operation, store_sales_table contains columns
  // ["ss_quantity", "ss_list_price", "ss_coupon_amt", "ss_sales_price", "i_item_id"]
  store_sales_table = std::make_shared<gqe::logical::join_relation>(
    std::move(store_sales_table),
    std::move(item_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::make_unique<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(4),
                                            std::make_shared<gqe::column_reference_expression>(5)),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>({0, 1, 2, 3, 6}));

  // Groupby on i_item_id
  // After this operation, store_sales_table contains columns
  // ["i_item_id", AVG(ss_quantity)", "AVG(ss_list_price)", "AVG(ss_coupon_amt)",
  // "AVG(ss_sales_price)"]
  std::vector<std::unique_ptr<gqe::expression>> groupby_keys;
  groupby_keys.push_back(std::make_unique<gqe::column_reference_expression>(4));

  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> groupby_values;
  groupby_values.emplace_back(
    std::make_pair(cudf::aggregation::MEAN, std::make_unique<gqe::column_reference_expression>(0)));
  groupby_values.emplace_back(
    std::make_pair(cudf::aggregation::MEAN, std::make_unique<gqe::column_reference_expression>(1)));
  groupby_values.emplace_back(
    std::make_pair(cudf::aggregation::MEAN, std::make_unique<gqe::column_reference_expression>(2)));
  groupby_values.emplace_back(
    std::make_pair(cudf::aggregation::MEAN, std::make_unique<gqe::column_reference_expression>(3)));

  store_sales_table = std::make_shared<gqe::logical::aggregate_relation>(
    std::move(store_sales_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::move(groupby_keys),
    std::move(groupby_values));

  // Sort on i_item_id
  // After this operation, store_sales_table contains columns
  // ["i_item_id", "AVG(ss_quantity)", "AVG(ss_list_price)", "AVG(ss_coupon_amt)",
  // "AVG(ss_sales_price)"]
  std::vector<std::unique_ptr<gqe::expression>> sort_exprs;
  sort_exprs.push_back(std::make_unique<gqe::column_reference_expression>(0));

  store_sales_table = std::make_shared<gqe::logical::sort_relation>(
    std::move(store_sales_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::vector<cudf::order>({cudf::order::ASCENDING}),
    std::vector<cudf::null_order>({cudf::null_order::BEFORE}),
    std::move(sort_exprs));

  // LIMIT 100
  store_sales_table =
    std::make_shared<gqe::logical::fetch_relation>(std::move(store_sales_table), 0, 100);

  auto logical_plan = std::move(store_sales_table);

  gqe::physical_plan_builder plan_builder(&tpcds_catalog);
  auto physical_plan = plan_builder.build(logical_plan.get());

  gqe::task_graph_builder graph_builder(&tpcds_catalog);
  auto task_graph = graph_builder.build(physical_plan.get());

  gqe::benchmark::time_function(gqe::execute_task_graph_single_gpu, task_graph.get());

  // Output the result to disk
  assert(task_graph->root_tasks.size() == 1);
  auto destination = cudf::io::sink_info("output.parquet");
  auto options     = cudf::io::parquet_writer_options::builder(
    destination, task_graph->root_tasks[0]->result().value());
  cudf::io::write_parquet(options);

  return 0;
}
