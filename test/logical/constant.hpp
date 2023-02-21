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
#include <cudf/types.hpp>

#include <map>
#include <vector>

struct ddl_t {
  std::vector<std::string> column_names;
  std::vector<cudf::data_type> column_types;
  std::vector<std::string> file_paths;
};

// TODO: Add more tables from TPC-DS

// TPD-H DDLs

std::map<std::string, ddl_t> ddls = {
  {"SUPPLIER",
   {{"S_SUPPKEY", "S_NAME", "S_ADDRESS", "S_NATIONKEY", "S_PHONE", "S_ACCTBAL", "S_COMMENT"},
    {cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::STRING),
     cudf::data_type(cudf::type_id::STRING),
     cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::FLOAT64),
     cudf::data_type(cudf::type_id::STRING)},
    // TODO: Build catalog with file paths for executor
    {"dummy_path"}}},
  {"PART",
   {{"P_PARTKEY",
     "P_NAME",
     "P_MFGR",
     "P_BRAND",
     "P_TYPE",
     "P_SIZE",
     "P_CONTAINER",
     "P_RETAILPRICE",
     "P_COMMENT"},
    {cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::STRING),
     cudf::data_type(cudf::type_id::STRING),
     cudf::data_type(cudf::type_id::STRING),
     cudf::data_type(cudf::type_id::STRING),
     cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::STRING),
     cudf::data_type(cudf::type_id::FLOAT64),
     cudf::data_type(cudf::type_id::STRING)},
    {"dummy_path"}}},
  {"PARTSUPP",
   {{"PS_PARTKEY", "PS_SUPPKEY", "PS_AVAILQTY", "PS_SUPPLYCOST", "PS_COMMENT"},
    {cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::FLOAT64),
     cudf::data_type(cudf::type_id::STRING)},
    {"dummy_path"}}},
  // Partial TPC-DS DDLs
  {"date_dim",
   {{"d_date_sk", "d_year", "d_moy", "d_day_name"},
    {cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::STRING)},
    {"dummy_path"}}},
  {"store",
   {{"s_store_sk", "s_store_id", "s_store_name", "s_gmt_offset"},
    {cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::STRING),
     cudf::data_type(cudf::type_id::INT64)},
    {"dummy_path"}}},
  {"customer_demographics",
   {{"cd_demo_sk", "cd_gender", "cd_marital_status", "cd_education_status"},
    {cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::STRING),
     cudf::data_type(cudf::type_id::STRING),
     cudf::data_type(cudf::type_id::STRING)},
    {"dummy_path"}}},
  {"customer_address",
   {{"ca_address_sk", "ca_country", "ca_state"},
    {cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::STRING),
     cudf::data_type(cudf::type_id::STRING)}}},
  {"promotion",
   {{"p_promo_sk", "p_channel_email", "p_channel_event"},
    {cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::STRING),
     cudf::data_type(cudf::type_id::STRING)},
    {"dummy_path"}}},
  {"customer_address",
   {{"ca_address_sk", "ca_country", "ca_state"},
    {cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::STRING),
     cudf::data_type(cudf::type_id::STRING)},
    {"dummy_path"}}},
  {"store_sales",
   {{"ss_sold_date_sk",
     "ss_item_sk",
     "ss_ext_sales_price",
     "ss_list_price",
     "ss_coupon_amt",
     "ss_promo_sk",
     "ss_store_sk",
     "ss_sales_price",
     "ss_cdemo_sk",
     "ss_addr_sk",
     "ss_quantity",
     "ss_net_profit"},
    {cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::FLOAT64),
     cudf::data_type(cudf::type_id::FLOAT64),
     cudf::data_type(cudf::type_id::FLOAT64),
     cudf::data_type(cudf::type_id::FLOAT64),
     cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::FLOAT64),
     cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::FLOAT64)},
    {"dummy_path"}}},
  {"item",
   {{"i_item_sk", "i_item_id", "i_brand_id", "i_brand", "i_manufact_id"},
    {cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::STRING),
     cudf::data_type(cudf::type_id::INT64)}}}};
