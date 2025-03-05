/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <cxx_gqe/executor.hpp>

#include <gqe/context_reference.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/task_graph.hpp>

// Include types declared in Rust by the gqe-sys crate.
#include <gqe-sys/src/lib.rs.h>

// Include wrappers of Rust std library types.
#include "rust/cxx.h"

#include <exception>
#include <memory>

namespace cxx_gqe {

cxx_gqe::optimization_parameters new_optimization_parameters()
{
  auto p = gqe::optimization_parameters(false);

  return {p.max_num_workers,
          p.max_num_partitions,
          p.log_level,
          p.join_use_hash_map_cache,
          p.read_zero_copy_enable,
          p.use_customized_io,
          p.io_bounce_buffer_size,
          p.io_auxiliary_threads,
          p.use_opt_type_for_single_char_col,
          p.in_memory_table_compression_format,
          p.io_block_size,
          p.io_engine,
          p.io_pipelining,
          p.io_alignment};
}

gqe::optimization_parameters to_gqe(cxx_gqe::optimization_parameters const& p)
{
  gqe::optimization_parameters n(true);

  n.max_num_workers                    = p.max_num_workers;
  n.max_num_partitions                 = p.max_num_partitions;
  n.log_level                          = std::string(p.log_level);
  n.join_use_hash_map_cache            = p.join_use_hash_map_cache;
  n.read_zero_copy_enable              = p.read_zero_copy_enable;
  n.use_customized_io                  = p.use_customized_io;
  n.io_bounce_buffer_size              = p.io_bounce_buffer_size;
  n.io_auxiliary_threads               = p.io_auxiliary_threads;
  n.use_opt_type_for_single_char_col   = p.use_opt_type_for_single_char_col;
  n.in_memory_table_compression_format = p.in_memory_table_compression_format;
  n.io_block_size                      = p.io_block_size;
  n.io_engine                          = p.io_engine;
  n.io_pipelining                      = p.io_pipelining;
  n.io_alignment                       = p.io_alignment;

  return n;
}

std::unique_ptr<query_context> new_query_context(cxx_gqe::optimization_parameters const& parameters)
{
  auto opms = to_gqe(parameters);
  auto qc   = gqe::query_context(std::move(opms));
  return std::make_unique<query_context>(std::move(qc));
}

std::unique_ptr<task_manager_context> new_task_manager_context()
{
  auto dbc = gqe::task_manager_context();
  return std::make_unique<task_manager_context>(std::move(dbc));
}

std::unique_ptr<task_graph> task_graph_builder::build(
  std::shared_ptr<physical_relation> root_relation)
{
  return _builder.build(root_relation.get());
}

std::unique_ptr<task_graph_builder> new_task_graph_builder(
  task_manager_context& task_manager_context, query_context& query_context, catalog& catalog)
{
  return std::make_unique<task_graph_builder>(std::move(gqe::task_graph_builder(
    gqe::context_reference{&task_manager_context.get(), &query_context.get()}, &catalog.get())));
}

void execute_task_graph_single_gpu(task_manager_context& task_manager_context,
                                   query_context& query_context,
                                   task_graph const& task_graph)
{
  gqe::execute_task_graph_single_gpu(
    gqe::context_reference{&task_manager_context.get(), &query_context.get()}, &task_graph);
}

}  // namespace cxx_gqe
