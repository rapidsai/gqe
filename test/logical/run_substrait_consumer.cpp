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
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/logical/from_substrait.hpp>
#include <gqe/utility/tpcds.hpp>
#include <gqe/utility/tpch.hpp>

#include <cudf/types.hpp>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

void print_usage()
{
  std::cout << "Run GQE Substrait consumer" << std::endl
            << "./run_substrait_consumer <{ds, h}> <path-to-substrait-plan>" << std::endl;
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    print_usage();
    return 0;
  }

  std::string const tpc_type(argv[1]);
  if (tpc_type != "ds" && tpc_type != "h") {
    print_usage();
    return 0;
  }

  std::string substrait_file = argv[2];

  std::cout << "Converting Substrait plan into logical plan" << std::endl;

  gqe::catalog catalog;

  gqe::optimization_parameters opt_params;

  // Register all tables
  auto const& table_definitions =
    (tpc_type == "ds") ? gqe::utility::tpcds::table_definitions(opt_params.use_fixed_point)
                       : gqe::utility::tpch::table_definitions(opt_params.use_fixed_point);
  for (auto const& [name, definition] : table_definitions) {
    catalog.register_table(name,
                           definition,
                           gqe::storage_kind::parquet_file{{"/" + name}},
                           gqe::partitioning_schema_kind::automatic{});
  }

  // Read and parse substrait file
  gqe::substrait_parser parser(&catalog, opt_params);
  std::vector<std::shared_ptr<gqe::logical::relation>> query_plan =
    parser.from_file(substrait_file);
  // Print gqe logical relation in json format
  std::cout << "Visiting plan nodes" << std::endl;
  std::cout << "Relation size = " << query_plan.size() << std::endl;
  std::cout << query_plan[0]->to_string() << std::endl;

  return 0;
}
