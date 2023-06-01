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
#include "common.hpp"

#include <gqe/catalog.hpp>
#include <gqe/logical/from_substrait.hpp>

#include <cudf/types.hpp>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv)
{
  if (argc != 2) throw std::runtime_error("Accept exactly 1 positional arguments");
  std::string substrait_file = argv[1];

  std::cout << "Converting Substrait plan into logical plan" << std::endl;

  // Register all available tables
  auto const identifier_type = cudf::data_type(cudf::type_id::INT64);
  auto const integer_type    = cudf::data_type(cudf::type_id::INT64);
  auto const decimal_type    = cudf::data_type(cudf::type_id::FLOAT64);
  auto const string_type     = cudf::data_type(cudf::type_id::STRING);
  auto const date_type       = cudf::data_type(cudf::type_id::TIMESTAMP_DAYS);
  gqe::catalog tpcds_catalog;
  tpcds_catalog.register_table("store_sales",
                               {{"ss_sold_date_sk", identifier_type},
                                {"ss_sold_time_sk", identifier_type},
                                {"ss_item_sk", identifier_type},
                                {"ss_customer_sk", identifier_type},
                                {"ss_cdemo_sk", identifier_type},
                                {"ss_hdemo_sk", identifier_type},
                                {"ss_addr_sk", identifier_type},
                                {"ss_store_sk", identifier_type},
                                {"ss_promo_sk", identifier_type},
                                {"ss_ticket_number", identifier_type},
                                {"ss_quantity", integer_type},
                                {"ss_wholesale_cost", decimal_type},
                                {"ss_list_price", decimal_type},
                                {"ss_sales_price", decimal_type},
                                {"ss_ext_discount_amt", decimal_type},
                                {"ss_ext_sales_price", decimal_type},
                                {"ss_ext_wholesale_cost", decimal_type},
                                {"ss_ext_list_price", decimal_type},
                                {"ss_ext_tax", decimal_type},
                                {"ss_coupon_amt", decimal_type},
                                {"ss_net_paid", decimal_type},
                                {"ss_net_paid_inc_tax", decimal_type},
                                {"ss_net_profit", decimal_type}},
                               gqe::storage_kind::parquet_file{{"/store_sales"}},
                               gqe::partitioning_schema_kind::automatic{});

  tpcds_catalog.register_table("date_dim",
                               {{"d_date_sk", identifier_type},
                                {"d_date_id", string_type},
                                {"d_date", date_type},
                                {"d_month_seq", integer_type},
                                {"d_week_seq", integer_type},
                                {"d_quarter_seq", integer_type},
                                {"d_year", integer_type},
                                {"d_dow", integer_type},
                                {"d_moy", integer_type},
                                {"d_dom", integer_type},
                                {"d_qoy", integer_type},
                                {"d_fy_year", integer_type},
                                {"d_fy_quarter_seq", integer_type},
                                {"d_fy_week_seq", integer_type},
                                {"d_day_name", string_type},
                                {"d_quarter_name", string_type},
                                {"d_holiday", string_type},
                                {"d_weekend", string_type},
                                {"d_following_holiday", string_type},
                                {"d_first_dom", integer_type},
                                {"d_last_dom", integer_type},
                                {"d_same_day_ly", integer_type},
                                {"d_same_day_lq", integer_type},
                                {"d_current_day", string_type},
                                {"d_current_week", string_type},
                                {"d_current_month", string_type},
                                {"d_current_quarter", string_type},
                                {"d_current_year", string_type}},
                               gqe::storage_kind::parquet_file{{"/date_dim"}},
                               gqe::partitioning_schema_kind::automatic{});

  tpcds_catalog.register_table(
    "item",
    {{"i_item_sk", identifier_type},     {"i_item_id", string_type},
     {"i_rec_start_date", date_type},    {"i_rec_end_date", date_type},
     {"i_item_desc", string_type},       {"i_current_price", decimal_type},
     {"i_wholesale_cost", decimal_type}, {"i_brand_id", integer_type},
     {"i_brand", string_type},           {"i_class_id", integer_type},
     {"i_class", string_type},           {"i_category_id", integer_type},
     {"i_category", string_type},        {"i_manufact_id", integer_type},
     {"i_manufact", string_type},        {"i_size", string_type},
     {"i_formulation", string_type},     {"i_color", string_type},
     {"i_units", string_type},           {"i_container", string_type},
     {"i_manager_id", integer_type},     {"i_product_name", string_type}},
    gqe::storage_kind::parquet_file{{"/item"}},
    gqe::partitioning_schema_kind::automatic{});

  tpcds_catalog.register_table("customer_demographics",
                               {{"cd_demo_sk", identifier_type},
                                {"cd_gender", string_type},
                                {"cd_marital_status", string_type},
                                {"cd_education_status", string_type},
                                {"cd_purchase_estimate", integer_type},
                                {"cd_credit_rating", string_type},
                                {"cd_dep_count", integer_type},
                                {"cd_dep_employed_count", integer_type},
                                {"cd_dep_college_count", integer_type}},
                               gqe::storage_kind::parquet_file{{"/customer_demographics"}},
                               gqe::partitioning_schema_kind::automatic{});

  tpcds_catalog.register_table("promotion",
                               {{"p_promo_sk", identifier_type},
                                {"p_promo_id", string_type},
                                {"p_start_date_sk", identifier_type},
                                {"p_end_date_sk", identifier_type},
                                {"p_item_sk", identifier_type},
                                {"p_cost", decimal_type},
                                {"p_response_target", integer_type},
                                {"p_promo_name", string_type},
                                {"p_channel_dmail", string_type},
                                {"p_channel_email", string_type},
                                {"p_channel_catalog", string_type},
                                {"p_channel_tv", string_type},
                                {"p_channel_radio", string_type},
                                {"p_channel_press", string_type},
                                {"p_channel_event", string_type},
                                {"p_channel_demo", string_type},
                                {"p_channel_details", string_type},
                                {"p_purpose", string_type},
                                {"p_discount_active", string_type}},
                               gqe::storage_kind::parquet_file{{"/promotion"}},
                               gqe::partitioning_schema_kind::automatic{});

  tpcds_catalog.register_table("store",
                               {{"s_store_sk", identifier_type},
                                {"s_store_id", string_type},
                                {"s_rec_start_date", date_type},
                                {"s_rec_end_date", date_type},
                                {"s_closed_date_sk", identifier_type},
                                {"s_store_name", string_type},
                                {"s_number_employees", integer_type},
                                {"s_floor_space", integer_type},
                                {"s_hours", string_type},
                                {"s_manager", string_type},
                                {"s_market_id", integer_type},
                                {"s_geography_class", string_type},
                                {"s_market_desc", string_type},
                                {"s_market_manager", string_type},
                                {"s_division_id", integer_type},
                                {"s_division_name", string_type},
                                {"s_company_id", integer_type},
                                {"s_company_name", string_type},
                                {"s_street_number", string_type},
                                {"s_street_name", string_type},
                                {"s_street_type", string_type},
                                {"s_suite_number", string_type},
                                {"s_city", string_type},
                                {"s_county", string_type},
                                {"s_state", string_type},
                                {"s_zip", string_type},
                                {"s_country", string_type},
                                {"s_gmt_offset", decimal_type},
                                {"s_tax_percentage", decimal_type}},
                               gqe::storage_kind::parquet_file{{"/store"}},
                               gqe::partitioning_schema_kind::automatic{});

  tpcds_catalog.register_table("customer_address",
                               {{"ca_address_sk", identifier_type},
                                {"ca_address_id", string_type},
                                {"ca_street_number", string_type},
                                {"ca_street_name", string_type},
                                {"ca_street_type", string_type},
                                {"ca_suite_number", string_type},
                                {"ca_city", string_type},
                                {"ca_county", string_type},
                                {"ca_state", string_type},
                                {"ca_zip", string_type},
                                {"ca_country", string_type},
                                {"ca_gmt_offset", decimal_type},
                                {"ca_location_type", string_type}},
                               gqe::storage_kind::parquet_file{{"/customer_address"}},
                               gqe::partitioning_schema_kind::automatic{});

  tpcds_catalog.register_table("customer",
                               {{"c_customer_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"c_last_name", cudf::data_type(cudf::type_id::STRING)},
                                {"c_first_name", cudf::data_type(cudf::type_id::STRING)},
                                {"c_current_addr_sk", cudf::data_type(cudf::type_id::INT64)}},
                               gqe::storage_kind::parquet_file{{"/customer"}},
                               gqe::partitioning_schema_kind::automatic{});
  tpcds_catalog.register_table("catalog_sales",
                               {{"cs_sold_date_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"cs_bill_customer_sk", cudf::data_type(cudf::type_id::INT64)}},
                               gqe::storage_kind::parquet_file{{"/catalog_sales"}},
                               gqe::partitioning_schema_kind::automatic{});

  tpcds_catalog.register_table("web_sales",
                               {{"ws_sold_date_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"ws_bill_customer_sk", cudf::data_type(cudf::type_id::INT64)}},
                               gqe::storage_kind::parquet_file{{"/web_sales"}},
                               gqe::partitioning_schema_kind::automatic{});

  // Read and parse substrait file
  gqe::substrait_parser parser(&tpcds_catalog);
  std::vector<std::shared_ptr<gqe::logical::relation>> query_plan =
    parser.from_file(substrait_file);
  // Print gqe logical relation in json format
  std::cout << "Visiting plan nodes" << std::endl;
  std::cout << "Relation size = " << query_plan.size() << std::endl;
  std::cout << query_plan[0]->to_string() << std::endl;

  return 0;
}
