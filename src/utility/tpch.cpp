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

#include <gqe/utility/tpch.hpp>

#include <gqe/catalog.hpp>

using col_prop = gqe::column_traits::column_property;

namespace gqe::utility {

namespace tpch {

std::unordered_map<std::string, std::vector<gqe::column_traits>> const& table_definitions(
  bool use_opt_type_for_single_char_col) noexcept
{
  static const std::unordered_map<std::string, std::vector<gqe::column_traits>> definitions = {
    {"part",
     {
       {"p_partkey", identifier_type, {col_prop::unique}},  // primary key
       {"p_name", string_type},
       {"p_mfgr", string_type},
       {"p_brand", string_type},
       {"p_type", string_type},
       {"p_size", integer_type},
       {"p_container", string_type}
       // {"p_retailprice", decimal_type},
       // {"p_comment", string_type}
     }},
    {"supplier",
     {{"s_suppkey", identifier_type, {col_prop::unique}},  // primary key
      {"s_name", string_type},
      {"s_address", string_type},
      {"s_nationkey", identifier_type},
      {"s_phone", string_type},
      {"s_acctbal", decimal_type},
      {"s_comment", string_type}}},
    {"partsupp",
     {
       {"ps_partkey", identifier_type},  // primary key
       {"ps_suppkey", identifier_type},  // primary key
       {"ps_availqty", integer_type},
       {"ps_supplycost", decimal_type},
       // {"ps_comment", string_type}
     }},
    {"customer",
     {{"c_custkey", identifier_type, {col_prop::unique}},  // primary key
      {"c_name", string_type},
      {"c_address", string_type},
      {"c_nationkey", identifier_type},
      {"c_phone", string_type},
      {"c_acctbal", decimal_type},
      {"c_mktsegment", string_type},
      {"c_comment", string_type}}},
    {"orders",
     {{"o_orderkey", identifier_type, {col_prop::unique}},  // primary key
      {"o_custkey", identifier_type},
      {"o_orderstatus", use_opt_type_for_single_char_col ? char_type : string_type},
      {"o_totalprice", decimal_type},
      {"o_orderdate", date_type},
      {"o_orderpriority", string_type},
      // {"o_clerk", string_type},
      {"o_shippriority", integer_type},
      {"o_comment", string_type}}},
    {"lineitem",
     {
       {"l_orderkey", identifier_type},  // primary key
       {"l_partkey", identifier_type},
       {"l_suppkey", identifier_type},
       {"l_linenumber", integer_type},  // primary key
       {"l_quantity", decimal_type},
       {"l_extendedprice", decimal_type},
       {"l_discount", decimal_type},
       {"l_tax", decimal_type},
       {"l_returnflag", use_opt_type_for_single_char_col ? char_type : string_type},
       {"l_linestatus", use_opt_type_for_single_char_col ? char_type : string_type},
       {"l_shipdate", date_type},
       {"l_commitdate", date_type},
       {"l_receiptdate", date_type},
       {"l_shipinstruct", string_type},
       {"l_shipmode", string_type},
       // {"l_comment", string_type}
     }},
    {"nation",
     {
       {"n_nationkey", identifier_type, {col_prop::unique}},  // primary key
       {"n_name", string_type},
       {"n_regionkey", identifier_type},
       // {"n_comment", string_type}
     }},
    {"region",
     {
       {"r_regionkey", identifier_type, {col_prop::unique}},  // primary key
       {"r_name", string_type},
       // {"r_comment", string_type}
     }}};
  return definitions;
}

}  // namespace tpch
}  // namespace gqe::utility
