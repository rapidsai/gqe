/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "utility.hpp"

#include <gqe/optimizer/logical_optimization.hpp>

#include <gqe/catalog.hpp>
#include <gqe/context_reference.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/logical/from_substrait.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/relation.hpp>
#include <gqe/logical/write.hpp>
#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/query_context.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/tpcds.hpp>
#include <gqe/utility/tpch.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <variant>

void print_usage()
{
  std::cout << "Run TPC benchmark" << std::endl
            << "./tpc <{ds, h}> <path-to-substrait> <path-to-dataset> <storage-kind>" << std::endl
            << "<path-to-substrait> can be either a single Substrait file or a directory "
               "containing substrait files"
            << std::endl;
}

gqe::storage_kind::type parse_storage_kind(const std::string& storage_kind_description,
                                           const std::vector<std::string>& file_paths)
{
  std::string normalized_description;
  normalized_description.reserve(storage_kind_description.size());
  std::transform(storage_kind_description.begin(),
                 storage_kind_description.end(),
                 std::back_inserter(normalized_description),
                 [](auto c) { return std::tolower(c); });
  std::map<std::string, gqe::storage_kind::type> const storage_kinds{
    {"system_memory", gqe::storage_kind::system_memory{}},
    {"numa_memory", gqe::storage_kind::numa_memory{gqe::cpu_set(0)}},
    {"pinned_memory", gqe::storage_kind::pinned_memory{}},
    {"device_memory", gqe::storage_kind::device_memory{rmm::cuda_device_id(0)}},
    {"managed_memory", gqe::storage_kind::managed_memory{}},
    {"parquet_file", gqe::storage_kind::parquet_file{file_paths}}};
  return storage_kinds.at(normalized_description);
}

// Parse the number of row groups for the in-memory tables from the environment variable
// `GQE_NUM_ROW_GROUPS`. Default to `8` if the environment variable is not set.
int32_t parse_num_row_groups()
{
  constexpr int32_t default_num_row_groups = 8;

  auto const val_str = std::getenv("GQE_NUM_ROW_GROUPS");
  int32_t value = (val_str != nullptr) ? std::stoi(val_str, nullptr, 10) : default_num_row_groups;

  GQE_LOG_DEBUG("GQE_NUM_ROW_GROUPS = " + std::to_string(value));

  return value;
}

struct tpc_nvtx_domain {
  static constexpr char const* name{"TPC"};
};

using nvtx_scoped_range = nvtx3::scoped_range_in<tpc_nvtx_domain>;

class copy_plan_builder {
 public:
  copy_plan_builder(gqe::context_reference ctx_ref, const gqe::catalog* catalog)
    : _ctx_ref(ctx_ref), _catalog(catalog)
  {
  }

  void add_table(const std::string& src_table_name, const std::string& dst_table_name)
  {
    const auto src_column_names = _catalog->column_names(src_table_name);
    const auto dst_column_names = _catalog->column_names(dst_table_name);
    if (!std::equal(
          src_column_names.cbegin(), src_column_names.cend(), dst_column_names.cbegin())) {
      throw std::logic_error(
        "Failed to copy data to in-memory table because column names do not match.");
    }

    std::vector<cudf::data_type> src_column_types;
    src_column_types.reserve(src_column_names.size());
    std::transform(
      src_column_names.cbegin(),
      src_column_names.cend(),
      std::back_inserter(src_column_types),
      [&](const auto& column_name) { return _catalog->column_type(src_table_name, column_name); });

    std::vector<cudf::data_type> dst_column_types;
    dst_column_types.reserve(dst_column_names.size());
    std::transform(
      dst_column_names.cbegin(),
      dst_column_names.cend(),
      std::back_inserter(dst_column_types),
      [&](const auto& column_name) { return _catalog->column_type(dst_table_name, column_name); });

    if (!std::equal(
          src_column_types.cbegin(), src_column_types.cend(), dst_column_types.cbegin())) {
      throw std::logic_error(
        "Failed to copy data to in-memory table because column types do not match.");
    }

    auto read_table = std::make_shared<gqe::logical::read_relation>(
      std::vector<std::shared_ptr<gqe::logical::relation>>(),
      src_column_names,
      src_column_types,
      src_table_name,
      nullptr);

    _logical_plans.push_back(
      std::make_shared<gqe::logical::write_relation>(std::move(read_table),
                                                     std::move(dst_column_names),
                                                     std::move(dst_column_types),
                                                     std::move(dst_table_name)));
  };

  void execute_copy()
  {
    if (_logical_plans.empty()) { return; }

    nvtx_scoped_range load_data_range("load_data");

    gqe::physical_plan_builder plan_builder(_catalog);
    gqe::task_graph_builder graph_builder(_ctx_ref, _catalog);

    std::vector<std::shared_ptr<gqe::physical::relation>> physical_plans;
    physical_plans.reserve(_logical_plans.size());
    std::transform(
      _logical_plans.cbegin(),
      _logical_plans.cend(),
      std::back_inserter(physical_plans),
      [&](auto const& logical_plan) { return plan_builder.build(logical_plan.get()); });

    std::vector<std::unique_ptr<gqe::task_graph>> task_graphs;
    task_graphs.reserve(_logical_plans.size());
    std::transform(
      physical_plans.cbegin(),
      physical_plans.cend(),
      std::back_inserter(task_graphs),
      [&](auto const& phyiscal_plan) { return graph_builder.build(phyiscal_plan.get()); });

    // Execute the copies
    std::for_each(task_graphs.begin(), task_graphs.end(), [=](auto const& task_graph) {
      gqe::execute_task_graph_single_gpu(_ctx_ref, task_graph.get());
    });

    // Clear the state
    _logical_plans.clear();
  }

 private:
  gqe::context_reference _ctx_ref;
  const gqe::catalog* _catalog;
  std::vector<std::shared_ptr<gqe::logical::relation>> _logical_plans;
};

class query_result_writer {
 public:
  query_result_writer(gqe::context_reference ctx_ref, gqe::catalog* catalog)
    : _ctx_ref(ctx_ref), _catalog(catalog)
  {
  }

  std::shared_ptr<gqe::logical::relation> append_to_plan(
    const std::shared_ptr<gqe::logical::relation> plan,
    const std::string result_table_name,
    const gqe::storage_kind::type result_storage_kind,
    const gqe::partitioning_schema_kind::type result_partitioning_schema)
  {
    const auto data_types = plan->data_types();

    std::vector<std::pair<std::string, cudf::data_type>> column_definitions;
    std::vector<std::string> column_names;
    column_definitions.reserve(data_types.size());
    column_names.reserve(data_types.size());
    for (std::size_t column_idx = 0; column_idx < data_types.size(); ++column_idx) {
      auto name = "column_" + std::to_string(column_idx);
      column_definitions.emplace_back(name, data_types[column_idx]);
      column_names.push_back(std::move(name));
    }

    _catalog->register_table(
      result_table_name, column_definitions, result_storage_kind, result_partitioning_schema);

    _result_table_name  = result_table_name;
    _column_definitions = std::move(column_definitions);

    return std::make_shared<gqe::logical::write_relation>(
      plan, column_names, data_types, result_table_name);
  }

  void execute_parquet_write(std::string const& result_path, std::string const& parquet_table_name)
  {
    _catalog->register_table(parquet_table_name,
                             _column_definitions,
                             gqe::storage_kind::parquet_file{{result_path}},
                             gqe::partitioning_schema_kind::none{});

    copy_plan_builder copy_plan(_ctx_ref, _catalog);
    copy_plan.add_table(_result_table_name, parquet_table_name);
    copy_plan.execute_copy();

    _result_table_name.clear();
    _column_definitions.clear();
  }

 private:
  gqe::context_reference _ctx_ref;
  gqe::catalog* _catalog;
  std::string _result_table_name;
  std::vector<std::pair<std::string, cudf::data_type>> _column_definitions;
};

int main(int argc, char* argv[])
{
  std::string storage_kind_name;

  switch (argc) {
    case 4: storage_kind_name = "parquet_file"; break;
    case 5: storage_kind_name = argv[4]; break;
    default: print_usage(); std::exit(EXIT_FAILURE);
  }

  std::string const tpc_type(argv[1]);
  std::string const substrait_location(argv[2]);
  std::string const dataset_location(argv[3]);

  // Check commandline inputs
  if (tpc_type != "ds" && tpc_type != "h") {
    std::cerr << "Invalid TPC benchmark." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (!std::filesystem::exists(substrait_location)) {
    std::cerr << "Invalid path to Substrait plan." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (!std::filesystem::exists(dataset_location)) {
    std::cerr << "Invalid path to dataset location." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (!std::filesystem::is_directory(dataset_location)) {
    std::cerr << "Data set location is not a directory" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Configure the memory pool
  // FIXME: For multi-GPU, we need to construct a memory pool for each device
  auto const pool_size = gqe::benchmark::get_memory_pool_size();
  rmm::mr::cuda_memory_resource cuda_mr;
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{
    &cuda_mr, pool_size, pool_size};
  rmm::mr::set_current_device_resource(&pool_mr);

  gqe::optimization_parameters read_opms{};
  read_opms.max_num_partitions = parse_num_row_groups();
  gqe::task_manager_context task_manager_ctx{};
  gqe::query_context query_ctx(read_opms);
  gqe::context_reference ctx_ref{&task_manager_ctx, &query_ctx};
  gqe::catalog catalog;

  // register all tables
  auto const& table_definitions = (tpc_type == "ds") ? gqe::utility::tpcds::table_definitions()
                                                     : gqe::utility::tpch::table_definitions();
  copy_plan_builder copy_plan(ctx_ref, &catalog);
  for (auto const& [name, definition] : table_definitions) {
    const auto file_paths   = gqe::utility::get_parquet_files(dataset_location + "/" + name);
    const auto storage_kind = parse_storage_kind(storage_kind_name, file_paths);

    auto register_and_copy = [&, name = std::cref(name), definition = std::cref(definition)]() {
      constexpr auto name_suffix = "_parquet";
      auto parquet_name          = std::string(name).append(name_suffix);

      catalog.register_table(parquet_name,
                             definition,
                             gqe::storage_kind::parquet_file{file_paths},
                             gqe::partitioning_schema_kind::automatic{});
      catalog.register_table(name, definition, storage_kind, gqe::partitioning_schema_kind::none{});

      copy_plan.add_table(parquet_name, name);
    };

    std::visit(gqe::utility::overloaded{
                 [&](const gqe::storage_kind::system_memory) { register_and_copy(); },
                 [&](const gqe::storage_kind::numa_memory) { register_and_copy(); },
                 [&](const gqe::storage_kind::pinned_memory) { register_and_copy(); },
                 [&](const gqe::storage_kind::device_memory) { register_and_copy(); },
                 [&](const gqe::storage_kind::managed_memory) { register_and_copy(); },
                 [&catalog, name = std::cref(name), definition = std::cref(definition)](
                   const gqe::storage_kind::parquet_file& pf) {
                   catalog.register_table(
                     name, definition, pf, gqe::partitioning_schema_kind::automatic{});
                 }},
               storage_kind);
  }
  GQE_LOG_INFO("Copying Parquet files to in-memory tables (if required).");
  copy_plan.execute_copy();

  // If needed, parse the directory to get all substrait plans
  std::vector<std::filesystem::path> substrait_plans;
  if (std::filesystem::is_regular_file(substrait_location)) {
    substrait_plans.emplace_back(substrait_location);
  } else {
    for (auto const& entry : std::filesystem::directory_iterator(substrait_location)) {
      substrait_plans.emplace_back(entry.path());
    }
  }

  for (auto const& substrait_plan_file : substrait_plans) {
    auto const query_identifier = substrait_plan_file.stem().string();

    GQE_LOG_INFO("Building query plan for " + query_identifier);
    gqe::substrait_parser parser(&catalog);
    auto logical_plan = parser.from_file(substrait_plan_file.string());
    assert(logical_plan.size() == 1);

    gqe::optimizer::optimization_configuration logical_rule_config(
      {gqe::optimizer::logical_optimization_rule_type::projection_pushdown}, {});
    gqe::optimizer::logical_optimizer optimizer(&logical_rule_config, &catalog);
    auto opt_logical_plan = optimizer.optimize(logical_plan[0]);

    gqe::optimization_parameters execute_opms{};
    query_ctx.parameters = execute_opms;

    query_result_writer result_writer(ctx_ref, &catalog);
    auto const cached_result_name = "query_result_" + query_identifier;

    auto logical_plan_with_result =
      result_writer.append_to_plan(opt_logical_plan,
                                   cached_result_name,
                                   gqe::storage_kind::device_memory{rmm::cuda_device_id{0}},
                                   gqe::partitioning_schema_kind::none{});

    gqe::physical_plan_builder plan_builder(&catalog);
    auto physical_plan = plan_builder.build(logical_plan_with_result.get());

    gqe::task_graph_builder graph_builder(ctx_ref, &catalog);
    auto task_graph = graph_builder.build(physical_plan.get());

    GQE_LOG_INFO("Starting query execution " + query_identifier);
    {
      nvtx_scoped_range query_range("query_execution_" + query_identifier);
      gqe::utility::time_function(gqe::execute_task_graph_single_gpu, ctx_ref, task_graph.get());
    }

    auto const result_path = "output_" + query_identifier + ".parquet";
    GQE_LOG_INFO("Writing query result to \"" + result_path + "\".");
    result_writer.execute_parquet_write(result_path, "parquet_result_" + query_identifier);
  }

  std::exit(EXIT_SUCCESS);
}
