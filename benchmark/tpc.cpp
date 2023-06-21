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

#include <gqe/catalog.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/logical/from_substrait.hpp>
#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/tpcds.hpp>
#include <gqe/utility/tpch.hpp>

#include <cudf/io/parquet.hpp>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cassert>
#include <iostream>
#include <stdexcept>

void print_usage()
{
  std::cout << "Run TPC benchmark" << std::endl
            << "./tpc <{ds, h}> <path-to-substrait-plan> <path-to-dataset>" << std::endl;
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    print_usage();
    return 0;
  }

  std::string const tpc_type(argv[1]);
  if (tpc_type != "ds" && tpc_type != "h") {
    print_usage();
    return 0;
  }

  std::string const substrait_plan_file(argv[2]);
  std::string const dataset_location(argv[3]);

  // Configure the memory pool
  // FIXME: For multi-GPU, we need to construct a memory pool for each device
  rmm::mr::cuda_memory_resource cuda_mr;
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{&cuda_mr};
  rmm::mr::set_current_device_resource(&pool_mr);

  gqe::catalog catalog;

  // register all tables
  auto const& table_definitions = (tpc_type == "ds") ? gqe::utility::tpcds::table_definitions()
                                                     : gqe::utility::tpch::table_definitions();
  for (auto const& [name, definition] : table_definitions) {
    catalog.register_table(name,
                           definition,
                           gqe::storage_kind::parquet_file{
                             gqe::utility::get_parquet_files(dataset_location + "/" + name)},
                           gqe::partitioning_schema_kind::automatic{});
  }

  gqe::substrait_parser parser(&catalog);
  auto logical_plan = parser.from_file(substrait_plan_file);
  assert(logical_plan.size() == 1);

  gqe::physical_plan_builder plan_builder(&catalog);
  auto physical_plan = plan_builder.build(logical_plan[0].get());

  gqe::task_graph_builder graph_builder(&catalog);
  auto task_graph = graph_builder.build(physical_plan.get());

  gqe::utility::time_function(gqe::execute_task_graph_single_gpu, task_graph.get());

  assert(task_graph->root_tasks.size() == 1);
  auto destination = cudf::io::sink_info("output.parquet");
  auto options     = cudf::io::parquet_writer_options::builder(
    destination, task_graph->root_tasks[0]->result().value());
  cudf::io::write_parquet(options);

  return 0;
}
