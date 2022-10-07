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
#include "constant.hpp"

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

  // Get all available table names (from constant.hpp)
  std::vector<std::string> tables;
  for (auto& ddl : ddls) {
    tables.push_back(ddl.first);
  }

  // Register all available tables
  gqe::catalog tables_catalog;

  for (auto table_name : tables) {
    auto ddl = ddls[table_name];

    std::vector<std::pair<std::string, cudf::data_type>> columns;
    columns.reserve(ddl.column_names.size());
    std::transform(ddl.column_names.begin(),
                   ddl.column_names.end(),
                   ddl.column_types.begin(),
                   std::back_inserter(columns),
                   [](std::string const& column_name, cudf::data_type const& column_type) {
                     return std::make_pair(column_name, column_type);
                   });

    tables_catalog.register_table(
      table_name, columns, ddl.file_paths, gqe::file_format_type::parquet);
  }
  // Read and parse substrait file
  gqe::substrait_parser parser(&tables_catalog);
  std::vector<std::shared_ptr<gqe::logical::relation>> query_plan =
    parser.from_file(substrait_file);
  // Print gqe logical relation in json format
  std::cout << "Visiting plan nodes" << std::endl;
  std::cout << "Relation size = " << query_plan.size() << std::endl;
  std::cout << query_plan[0]->to_string() << std::endl;

  return 0;
}
