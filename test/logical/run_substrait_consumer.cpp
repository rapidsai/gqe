/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "common.hpp"

#include <gqe/catalog.hpp>
#include <gqe/logical/from_substrait.hpp>
#include <gqe/task_manager_context.hpp>
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

  gqe::task_manager_context task_manager_ctx;
  gqe::catalog catalog{&task_manager_ctx};

  // Register all tables
  auto& table_definitions = (tpc_type == "ds") ? gqe::utility::tpcds::table_definitions()
                                               : gqe::utility::tpch::table_definitions();
  for (auto& [name, definition] : table_definitions) {
    catalog.register_table(name,
                           definition,
                           gqe::storage_kind::parquet_file{{"/" + name}},
                           gqe::partitioning_schema_kind::automatic{});
  }

  // Read and parse substrait file
  gqe::substrait_parser parser(&catalog);
  std::vector<std::shared_ptr<gqe::logical::relation>> query_plan =
    parser.from_file(substrait_file);
  // Print gqe logical relation in json format
  std::cout << "Visiting plan nodes" << std::endl;
  std::cout << "Relation size = " << query_plan.size() << std::endl;
  std::cout << query_plan[0]->to_string() << std::endl;

  return 0;
}
