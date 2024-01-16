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

#include <gqe/catalog.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/query_context.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/write.hpp>
#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/physical/relation.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/helpers.hpp>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>

void print_usage()
{
  std::cout << "Load TPC-DS Parquet files into a GQE table and time the bulk load\n"
            << "./load_tpcds <path-to-parquet-dataset> <path-to-gqe-storage>" << std::endl;
}

int main(int argc, char* argv[])
{
  // Parse the command line arguments to get the dataset location
  if (argc != 3) {
    print_usage();
    return EXIT_FAILURE;
  }
  std::string const dataset_location(argv[1]);
  std::string const storage_location(argv[2]);

  // Configure the memory pool
  rmm::mr::cuda_memory_resource cuda_mr;
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{&cuda_mr};
  rmm::mr::set_current_device_resource(&pool_mr);

  // Register the Parquet files as input tables
  gqe::catalog tpcds_catalog;
  tpcds_catalog.register_table("in_store_sales",
                               {{"ss_sold_time_sk", cudf::data_type(cudf::type_id::INT32)},
                                {"ss_item_sk", cudf::data_type(cudf::type_id::INT32)},
                                {"ss_customer_sk", cudf::data_type(cudf::type_id::INT32)},
                                {"ss_cdemo_sk", cudf::data_type(cudf::type_id::INT32)},
                                {"ss_hdemo_sk", cudf::data_type(cudf::type_id::INT32)},
                                {"ss_addr_sk", cudf::data_type(cudf::type_id::INT32)},
                                {"ss_store_sk", cudf::data_type(cudf::type_id::INT32)},
                                {"ss_promo_sk", cudf::data_type(cudf::type_id::INT32)},
                                {"ss_ticket_number", cudf::data_type(cudf::type_id::INT32)},
                                {"ss_quantity", cudf::data_type(cudf::type_id::INT32)},
                                {"ss_wholesale_cost", cudf::data_type(cudf::type_id::FLOAT64)},
                                {"ss_list_price", cudf::data_type(cudf::type_id::FLOAT64)},
                                {"ss_sales_price", cudf::data_type(cudf::type_id::FLOAT64)},
                                {"ss_ext_discount_amt", cudf::data_type(cudf::type_id::FLOAT64)},
                                {"ss_ext_sales_price", cudf::data_type(cudf::type_id::FLOAT64)},
                                {"ss_ext_wholesale_cost", cudf::data_type(cudf::type_id::FLOAT64)},
                                {"ss_ext_list_price", cudf::data_type(cudf::type_id::FLOAT64)},
                                {"ss_ext_tax", cudf::data_type(cudf::type_id::FLOAT64)},
                                {"ss_coupon_amt", cudf::data_type(cudf::type_id::FLOAT64)},
                                {"ss_net_paid", cudf::data_type(cudf::type_id::FLOAT64)},
                                {"ss_net_paid_inc_tax", cudf::data_type(cudf::type_id::FLOAT64)},
                                {"ss_net_profit", cudf::data_type(cudf::type_id::FLOAT64)},
                                {"ss_sold_date_sk", cudf::data_type(cudf::type_id::FLOAT64)}},
                               gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(
                                 dataset_location + "/store_sales")},
                               gqe::partitioning_schema_kind::automatic{});
  tpcds_catalog.register_table("in_date_dim",
                               {{"d_date_sk", cudf::data_type(cudf::type_id::INT32)},
                                {"d_date_id", cudf::data_type(cudf::type_id::STRING)},
                                {"d_date", cudf::data_type(cudf::type_id::INT32)},
                                {"d_month_seq", cudf::data_type(cudf::type_id::INT32)},
                                {"d_week_seq", cudf::data_type(cudf::type_id::INT32)},
                                {"d_quarter_seq", cudf::data_type(cudf::type_id::INT32)},
                                {"d_year", cudf::data_type(cudf::type_id::INT32)},
                                {"d_dow", cudf::data_type(cudf::type_id::INT32)},
                                {"d_moy", cudf::data_type(cudf::type_id::INT32)},
                                {"d_dom", cudf::data_type(cudf::type_id::INT32)},
                                {"d_qoy", cudf::data_type(cudf::type_id::INT32)},
                                {"d_fy_year", cudf::data_type(cudf::type_id::INT32)},
                                {"d_fy_quarter_seq", cudf::data_type(cudf::type_id::INT32)},
                                {"d_fy_week_seq", cudf::data_type(cudf::type_id::INT32)},
                                {"d_day_name", cudf::data_type(cudf::type_id::STRING)},
                                {"d_quarter_name", cudf::data_type(cudf::type_id::STRING)},
                                {"d_holiday", cudf::data_type(cudf::type_id::STRING)},
                                {"d_weekend", cudf::data_type(cudf::type_id::STRING)},
                                {"d_following_holiday", cudf::data_type(cudf::type_id::STRING)},
                                {"d_first_dom", cudf::data_type(cudf::type_id::INT32)},
                                {"d_last_dom", cudf::data_type(cudf::type_id::INT32)},
                                {"d_same_day_ly", cudf::data_type(cudf::type_id::INT32)},
                                {"d_same_day_lq", cudf::data_type(cudf::type_id::INT32)},
                                {"d_current_day", cudf::data_type(cudf::type_id::STRING)},
                                {"d_current_week", cudf::data_type(cudf::type_id::STRING)},
                                {"d_current_month", cudf::data_type(cudf::type_id::STRING)},
                                {"d_current_quarter", cudf::data_type(cudf::type_id::STRING)},
                                {"d_current_year", cudf::data_type(cudf::type_id::STRING)}},
                               gqe::storage_kind::parquet_file{
                                 gqe::utility::get_parquet_files(dataset_location + "/date_dim")},
                               gqe::partitioning_schema_kind::automatic{});
  tpcds_catalog.register_table(
    "in_item",
    {{"i_item_sk", cudf::data_type(cudf::type_id::INT32)},
     {"i_item_id", cudf::data_type(cudf::type_id::STRING)},
     {"i_rec_start_date", cudf::data_type(cudf::type_id::INT32)},
     {"i_rec_end_date", cudf::data_type(cudf::type_id::INT32)},
     {"i_item_desc", cudf::data_type(cudf::type_id::STRING)},
     {"i_current_price", cudf::data_type(cudf::type_id::FLOAT64)},
     {"i_wholesale_cost", cudf::data_type(cudf::type_id::FLOAT64)},
     {"i_brand_id", cudf::data_type(cudf::type_id::INT32)},
     {"i_brand", cudf::data_type(cudf::type_id::STRING)},
     {"i_class_id", cudf::data_type(cudf::type_id::INT32)},
     {"i_class", cudf::data_type(cudf::type_id::STRING)},
     {"i_category_id", cudf::data_type(cudf::type_id::INT32)},
     {"i_category", cudf::data_type(cudf::type_id::STRING)},
     {"i_manufact_id", cudf::data_type(cudf::type_id::INT32)},
     {"i_manufact", cudf::data_type(cudf::type_id::STRING)},
     {"i_size", cudf::data_type(cudf::type_id::STRING)},
     {"i_formulation", cudf::data_type(cudf::type_id::STRING)},
     {"i_color", cudf::data_type(cudf::type_id::STRING)},
     {"i_units", cudf::data_type(cudf::type_id::STRING)},
     {"i_container", cudf::data_type(cudf::type_id::STRING)},
     {"i_manager_id", cudf::data_type(cudf::type_id::INT32)},
     {"i_product_name", cudf::data_type(cudf::type_id::STRING)}},
    gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(dataset_location + "/item")},
    gqe::partitioning_schema_kind::automatic{});

  // Register the GQE tables
  tpcds_catalog.register_table(
    "store_sales",
    {{"ss_sold_time_sk", cudf::data_type(cudf::type_id::INT32)},
     {"ss_item_sk", cudf::data_type(cudf::type_id::INT32)},
     {"ss_customer_sk", cudf::data_type(cudf::type_id::INT32)},
     {"ss_cdemo_sk", cudf::data_type(cudf::type_id::INT32)},
     {"ss_hdemo_sk", cudf::data_type(cudf::type_id::INT32)},
     {"ss_addr_sk", cudf::data_type(cudf::type_id::INT32)},
     {"ss_store_sk", cudf::data_type(cudf::type_id::INT32)},
     {"ss_promo_sk", cudf::data_type(cudf::type_id::INT32)},
     {"ss_ticket_number", cudf::data_type(cudf::type_id::INT32)},
     {"ss_quantity", cudf::data_type(cudf::type_id::INT32)},
     {"ss_wholesale_cost", cudf::data_type(cudf::type_id::FLOAT64)},
     {"ss_list_price", cudf::data_type(cudf::type_id::FLOAT64)},
     {"ss_sales_price", cudf::data_type(cudf::type_id::FLOAT64)},
     {"ss_ext_discount_amt", cudf::data_type(cudf::type_id::FLOAT64)},
     {"ss_ext_sales_price", cudf::data_type(cudf::type_id::FLOAT64)},
     {"ss_ext_wholesale_cost", cudf::data_type(cudf::type_id::FLOAT64)},
     {"ss_ext_list_price", cudf::data_type(cudf::type_id::FLOAT64)},
     {"ss_ext_tax", cudf::data_type(cudf::type_id::FLOAT64)},
     {"ss_coupon_amt", cudf::data_type(cudf::type_id::FLOAT64)},
     {"ss_net_paid", cudf::data_type(cudf::type_id::FLOAT64)},
     {"ss_net_paid_inc_tax", cudf::data_type(cudf::type_id::FLOAT64)},
     {"ss_net_profit", cudf::data_type(cudf::type_id::FLOAT64)},
     {"ss_sold_date_sk", cudf::data_type(cudf::type_id::FLOAT64)}},
    gqe::storage_kind::parquet_file{{storage_location + "/store_sales.parquet"}},
    gqe::partitioning_schema_kind::none{});
  tpcds_catalog.register_table(
    "date_dim",
    {{"d_date_sk", cudf::data_type(cudf::type_id::INT32)},
     {"d_date_id", cudf::data_type(cudf::type_id::STRING)},
     {"d_date", cudf::data_type(cudf::type_id::INT32)},
     {"d_month_seq", cudf::data_type(cudf::type_id::INT32)},
     {"d_week_seq", cudf::data_type(cudf::type_id::INT32)},
     {"d_quarter_seq", cudf::data_type(cudf::type_id::INT32)},
     {"d_year", cudf::data_type(cudf::type_id::INT32)},
     {"d_dow", cudf::data_type(cudf::type_id::INT32)},
     {"d_moy", cudf::data_type(cudf::type_id::INT32)},
     {"d_dom", cudf::data_type(cudf::type_id::INT32)},
     {"d_qoy", cudf::data_type(cudf::type_id::INT32)},
     {"d_fy_year", cudf::data_type(cudf::type_id::INT32)},
     {"d_fy_quarter_seq", cudf::data_type(cudf::type_id::INT32)},
     {"d_fy_week_seq", cudf::data_type(cudf::type_id::INT32)},
     {"d_day_name", cudf::data_type(cudf::type_id::STRING)},
     {"d_quarter_name", cudf::data_type(cudf::type_id::STRING)},
     {"d_holiday", cudf::data_type(cudf::type_id::STRING)},
     {"d_weekend", cudf::data_type(cudf::type_id::STRING)},
     {"d_following_holiday", cudf::data_type(cudf::type_id::STRING)},
     {"d_first_dom", cudf::data_type(cudf::type_id::INT32)},
     {"d_last_dom", cudf::data_type(cudf::type_id::INT32)},
     {"d_same_day_ly", cudf::data_type(cudf::type_id::INT32)},
     {"d_same_day_lq", cudf::data_type(cudf::type_id::INT32)},
     {"d_current_day", cudf::data_type(cudf::type_id::STRING)},
     {"d_current_week", cudf::data_type(cudf::type_id::STRING)},
     {"d_current_month", cudf::data_type(cudf::type_id::STRING)},
     {"d_current_quarter", cudf::data_type(cudf::type_id::STRING)},
     {"d_current_year", cudf::data_type(cudf::type_id::STRING)}},
    gqe::storage_kind::parquet_file{{storage_location + "/date_dim.parquet"}},
    gqe::partitioning_schema_kind::none{});
  tpcds_catalog.register_table(
    "item",
    {{"i_item_sk", cudf::data_type(cudf::type_id::INT32)},
     {"i_item_id", cudf::data_type(cudf::type_id::STRING)},
     {"i_rec_start_date", cudf::data_type(cudf::type_id::INT32)},
     {"i_rec_end_date", cudf::data_type(cudf::type_id::INT32)},
     {"i_item_desc", cudf::data_type(cudf::type_id::STRING)},
     {"i_current_price", cudf::data_type(cudf::type_id::FLOAT64)},
     {"i_wholesale_cost", cudf::data_type(cudf::type_id::FLOAT64)},
     {"i_brand_id", cudf::data_type(cudf::type_id::INT32)},
     {"i_brand", cudf::data_type(cudf::type_id::STRING)},
     {"i_class_id", cudf::data_type(cudf::type_id::INT32)},
     {"i_class", cudf::data_type(cudf::type_id::STRING)},
     {"i_category_id", cudf::data_type(cudf::type_id::INT32)},
     {"i_category", cudf::data_type(cudf::type_id::STRING)},
     {"i_manufact_id", cudf::data_type(cudf::type_id::INT32)},
     {"i_manufact", cudf::data_type(cudf::type_id::STRING)},
     {"i_size", cudf::data_type(cudf::type_id::STRING)},
     {"i_formulation", cudf::data_type(cudf::type_id::STRING)},
     {"i_color", cudf::data_type(cudf::type_id::STRING)},
     {"i_units", cudf::data_type(cudf::type_id::STRING)},
     {"i_container", cudf::data_type(cudf::type_id::STRING)},
     {"i_manager_id", cudf::data_type(cudf::type_id::INT32)},
     {"i_product_name", cudf::data_type(cudf::type_id::STRING)}},
    gqe::storage_kind::parquet_file{{storage_location + "/item.parquet"}},
    gqe::partitioning_schema_kind::none{});

  // Setup the logical plans
  auto copy_table = [&](std::vector<std::string> column_names,
                        std::string src_table_name,
                        std::string dst_table_name) -> std::shared_ptr<gqe::logical::relation> {
    std::vector<cudf::data_type> column_types;
    column_types.reserve(column_names.size());
    for (auto const& column_name : column_names) {
      column_types.push_back(tpcds_catalog.column_type(src_table_name, column_name));
    }

    auto read_table = std::make_shared<gqe::logical::read_relation>(
      std::vector<std::shared_ptr<gqe::logical::relation>>(),
      column_names,
      column_types,
      src_table_name,
      nullptr);

    return std::make_shared<gqe::logical::write_relation>(std::move(read_table),
                                                          std::move(column_names),
                                                          std::move(column_types),
                                                          std::move(dst_table_name));
  };

  auto copy_store_sales = copy_table(
    {"ss_item_sk", "ss_sold_date_sk", "ss_ext_sales_price"}, "in_store_sales", "store_sales");
  auto copy_date_dim = copy_table({"d_date_sk", "d_year", "d_moy"}, "in_date_dim", "date_dim");
  auto copy_item =
    copy_table({"i_item_sk", "i_brand_id", "i_brand", "i_manufact_id"}, "in_item", "item");

  // Prepare for execution
  std::vector<std::shared_ptr<gqe::logical::relation>> logical_plans = {
    copy_store_sales, copy_date_dim, copy_item};

  gqe::physical_plan_builder plan_builder(&tpcds_catalog);

  gqe::query_context qctx(gqe::optimization_parameters{});

  gqe::task_graph_builder graph_builder(&qctx, &tpcds_catalog);

  std::vector<std::shared_ptr<gqe::physical::relation>> physical_plans;
  physical_plans.reserve(logical_plans.size());
  std::transform(logical_plans.cbegin(),
                 logical_plans.cend(),
                 std::back_inserter(physical_plans),
                 [&](auto const& logical_plan) { return plan_builder.build(logical_plan.get()); });

  std::vector<std::unique_ptr<gqe::task_graph>> task_graphs;
  task_graphs.reserve(logical_plans.size());
  std::transform(
    physical_plans.cbegin(),
    physical_plans.cend(),
    std::back_inserter(task_graphs),
    [&](auto const& phyiscal_plan) { return graph_builder.build(phyiscal_plan.get()); });

  // Execute
  gqe::utility::time_function([&]() {
    std::for_each(task_graphs.begin(), task_graphs.end(), [&](auto const& task_graph) {
      gqe::execute_task_graph_single_gpu(&qctx, task_graph.get());
    });
  });
}
