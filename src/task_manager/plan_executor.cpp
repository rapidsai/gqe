/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/task_manager/plan_executor.hpp>

#include <gqe/context_reference.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/query_context.hpp>
#include <gqe/rpc/serialization/statistics.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/utility/cudf_to_arrow.hpp>
#include <gqe/utility/logger.hpp>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>

#include <format>

namespace gqe::task_manager {

namespace {

/// Serialize result views to Arrow IPC bytes.
std::string serialize_to_arrow_ipc(std::span<cudf::table_view const> result_views)
{
  std::vector<std::string> column_names;
  auto num_cols = result_views[0].num_columns();
  column_names.reserve(num_cols);
  for (cudf::size_type i = 0; i < num_cols; ++i) {
    column_names.push_back(std::format("column_{}", i));
  }

  auto batch  = gqe::utility::cudf_table_to_arrow(result_views, column_names);
  auto sink   = arrow::io::BufferOutputStream::Create().ValueOrDie();
  auto writer = arrow::ipc::MakeStreamWriter(sink, batch->schema()).ValueOrDie();
  auto status = writer->WriteRecordBatch(*batch);
  if (!status.ok()) {
    throw std::runtime_error(std::format("Failed to write Arrow IPC: {}", status.ToString()));
  }
  status = writer->Close();
  if (!status.ok()) {
    throw std::runtime_error(
      std::format("Failed to close Arrow IPC writer: {}", status.ToString()));
  }
  auto buffer = sink->Finish().ValueOrDie();
  return std::string(reinterpret_cast<char const*>(buffer->data()), buffer->size());
}

}  // namespace

plan_executor::plan_executor(task_manager_context* ctx, int rank, bool multi_process)
  : _ctx(ctx), _cache(ctx), _rank(rank), _multi_process(multi_process)
{
}

service::execution_result plan_executor::operator()(std::shared_ptr<physical::relation> plan,
                                                    utility::uuid query_id,
                                                    std::vector<storage::descriptor> descriptors,
                                                    gqe::optimization_parameters opt_params)
{
  GQE_LOG_INFO("Task manager rank {} executing query {}", _rank, query_id);

  // Write plans (COPY) use ALL_TO_ALL so that every rank executes the write
  // task and populates its local in-memory table cache with row group references.
  // Only rank 0 actually writes data to shared memory (guarded by nvshmem_rank_zero()),
  // but all ranks create column_base references pointing to the shared objects.
  // Query plans use ROUND_ROBIN to distribute work across ranks.
  bool const is_write = plan->type() == physical::relation::relation_type::write;

  if (_multi_process) {
    auto* mp_ctx = dynamic_cast<multi_process_task_manager_context*>(_ctx);
    if (mp_ctx) {
      mp_ctx->update_scheduler(is_write ? SCHEDULER_TYPE::ALL_TO_ALL : SCHEDULER_TYPE::ROUND_ROBIN);
    }
  }

  _cache.update(std::move(descriptors));

  if (_multi_process) {
    // FIXME: Multi-GPU task migration doesn't work with zero-copy enabled.
    opt_params.read_zero_copy_enable = false;
  }
  gqe::query_context query_ctx{query_id, opt_params, gqe::optimizer::optimization_configuration{}};
  gqe::context_reference ctx_ref{_ctx, &query_ctx};

  // RAII guard to unregister all tasks for this query on scope exit (success or exception).
  struct query_registration_guard {
    task_migration_service* service;
    utility::uuid query_id;
    ~query_registration_guard()
    {
      if (service) service->unregister_query(query_id);
    }
  };

  task_migration_service* migration_svc = nullptr;
  if (_multi_process) {
    auto* mp_ctx = dynamic_cast<multi_process_task_manager_context*>(_ctx);
    if (mp_ctx) migration_svc = mp_ctx->migration_service.get();
  }
  query_registration_guard guard{migration_svc, query_id};

  gqe::task_graph_builder graph_builder(ctx_ref, &_cache);
  auto task_graph = graph_builder.build(plan.get());

  if (_multi_process) {
    gqe::execute_task_graph_multi_process(ctx_ref, task_graph.get());
  } else {
    gqe::execute_task_graph_single_gpu(ctx_ref, task_graph.get());
  }

  service::execution_result result;

  if (is_write) {
    // Queries that write their result to a table don't return any result rows.

    if (task_graph->write_statistics) {
      // Collect and serialize write statistics from the task graph.
      auto stats             = task_graph->write_statistics->statistics();
      result.has_write_stats = true;
      result.write_stats     = rpc::serialize_table_statistics(stats);
    }
  } else {
    // Collect result rows for non-write queries.
    std::vector<cudf::table_view> result_views;
    for (auto const& root_task : task_graph->root_tasks) {
      if (root_task->result().has_value()) { result_views.push_back(root_task->result().value()); }
    }

    if (!result_views.empty()) { result.arrow_ipc = serialize_to_arrow_ipc(result_views); }
  }

  return result;
}

}  // namespace gqe::task_manager
