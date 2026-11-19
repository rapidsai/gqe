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
#include <gqe/executor/qep_task_graph_transformation.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/qep/physical_qep_transformation.hpp>
#include <gqe/qep/pipeline_normalization.hpp>
#include <gqe/qep/shapes/write_statistics.hpp>
#include <gqe/query_context.hpp>
#include <gqe/rpc/serialization/statistics.hpp>
#include <gqe/storage/in_memory.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/cudf_to_arrow.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>
#include <gqe/utility/multi_process_helpers.hpp>

#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>

#include <algorithm>
#include <cstdint>
#include <format>
#include <stdexcept>

namespace gqe::task_manager {

namespace {

/**
 * @brief One query's counter frame on the query memory statistics adaptor.
 *
 * The constructor pushes a frame of zero counters and the destructor pops it, also when the query
 * throws, so frames never accumulate. RMM folds a popped frame into the one below it. The frame's
 * peak is the query's peak above the bytes allocated when it started. If the query frees
 * allocations that predate it and reuses the space, the peak understates the query's own
 * allocations by the freed bytes.
 *
 * The frames live on one stack shared by all queries on the resource. Concurrent queries therefore
 * read and pop each other's frames, so the peak is attributed to one query only while it runs
 * alone. Pushes and pops stay balanced, so a pop never underflows the stack.
 *
 * A null adaptor, which means tracking is off, makes the frame a no-op.
 */
class query_memory_frame {
 public:
  explicit query_memory_frame(rmm::mr::statistics_resource_adaptor* statistics)
    : _statistics{statistics}
  {
    if (_statistics != nullptr) { _statistics->push_counters(); }
  }

  ~query_memory_frame()
  {
    if (_statistics != nullptr) { _statistics->pop_counters(); }
  }

  query_memory_frame(query_memory_frame const&)            = delete;
  query_memory_frame& operator=(query_memory_frame const&) = delete;
  query_memory_frame(query_memory_frame&&)                 = delete;
  query_memory_frame& operator=(query_memory_frame&&)      = delete;

  /**
   * @brief Return whether a frame was pushed, i.e. whether peak tracking is on.
   */
  [[nodiscard]] bool is_tracking() const noexcept { return _statistics != nullptr; }

  /**
   * @brief Return the peak bytes allocated since the frame was pushed.
   *
   * @pre `is_tracking()`, which the caller checks.
   */
  [[nodiscard]] std::uint64_t peak_bytes() const noexcept
  {
    return static_cast<std::uint64_t>(
      std::max<std::int64_t>(_statistics->get_bytes_counter().peak, 0));
  }

 private:
  rmm::mr::statistics_resource_adaptor* _statistics;  ///< Null when peak tracking is off.
};

/// Serialize result views to Arrow IPC bytes.
std::string serialize_to_arrow_ipc(std::span<cudf::table_view const> result_views,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  std::vector<std::string> column_names;
  auto num_cols = result_views[0].num_columns();
  column_names.reserve(num_cols);
  for (cudf::size_type i = 0; i < num_cols; ++i) {
    column_names.push_back(std::format("column_{}", i));
  }

  auto batch  = gqe::utility::cudf_table_to_arrow(result_views, column_names, stream, mr);
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

/// Read the single record batch out of an Arrow IPC stream buffer.
std::shared_ptr<arrow::RecordBatch> deserialize_arrow_ipc(std::string const& bytes)
{
  auto buffer = arrow::Buffer::FromString(bytes);
  // Throw (rather than ValueOrDie's abort) on a malformed or incompatible payload, so the
  // RefreshStatsTables handler can report success=false instead of terminating the task manager.
  auto reader_result =
    arrow::ipc::RecordBatchStreamReader::Open(std::make_shared<arrow::io::BufferReader>(buffer));
  if (!reader_result.ok()) {
    throw std::runtime_error(
      std::format("Failed to open stats Arrow IPC stream: {}", reader_result.status().ToString()));
  }
  auto reader = std::move(reader_result).ValueOrDie();
  std::shared_ptr<arrow::RecordBatch> batch;
  auto status = reader->ReadNext(&batch);
  if (!status.ok()) {
    throw std::runtime_error(std::format("Failed to read stats Arrow IPC: {}", status.ToString()));
  }
  if (!batch) { throw std::runtime_error("Stats Arrow IPC stream contained no record batch"); }
  return batch;
}

/// Convert an Arrow record batch to a cuDF device table via the Arrow C data interface.
std::unique_ptr<cudf::table> arrow_batch_to_cudf(arrow::RecordBatch const& batch,
                                                 cuda::stream_ref stream)
{
  ArrowSchema c_schema;
  ArrowArray c_array;
  auto status = arrow::ExportRecordBatch(batch, &c_array, &c_schema);
  if (!status.ok()) {
    throw std::runtime_error(std::format("Failed to export record batch: {}", status.ToString()));
  }
  auto table = cudf::from_arrow(&c_schema, &c_array, stream);
  // cudf::from_arrow copies the data; release the C structures we exported.
  if (c_array.release != nullptr) { c_array.release(&c_array); }
  if (c_schema.release != nullptr) { c_schema.release(&c_schema); }
  return table;
}

/** @brief Extract the terminal write statistics emitted by a QEP task graph. */
table_statistics collect_qep_write_statistics(task_graph const& task_graph)
{
  GQE_EXPECTS(task_graph.root_tasks.size() == 1,
              "QEP write task graph must have exactly one root task",
              std::logic_error);

  auto state = task_graph.root_tasks.front()->qep_state_result();
  GQE_EXPECTS(state.has_value(), "QEP write root task did not emit a result", std::logic_error);

  auto statistics = qep::try_write_statistics(*state);
  GQE_EXPECTS(
    statistics.has_value(), "QEP write root task did not emit write statistics", std::logic_error);
  return std::move(*statistics);
}

}  // namespace

plan_executor::plan_executor(task_manager_context* ctx, int rank, bool multi_process)
  : _ctx(ctx), _cache(ctx), _rank(rank), _multi_process(multi_process)
{
}

service::execution_result plan_executor::operator()(physical::identified_plan physical_plan,
                                                    utility::uuid query_id,
                                                    std::vector<storage::descriptor> descriptors,
                                                    gqe::optimization_parameters opt_params)
{
  GQE_LOG_INFO("Task manager rank {} executing query {}", _rank, query_id);
  utility::nvtx_scoped_range execute_range("execute_plan");

  // Write plans (COPY) use ALL_TO_ALL so that every rank executes the write
  // task and populates its local in-memory table cache with row group references.
  // Only rank 0 actually writes data to shared memory (guarded by nvshmem_rank_zero()),
  // but all ranks create column_base references pointing to the shared objects.
  // Query plans use ROUND_ROBIN to distribute work across ranks.
  bool const is_write = physical_plan.root()->type() == physical::relation::relation_type::write;

  if (opt_params.execution_backend == execution_backend_type::qep_task_graph) {
    if (_multi_process) {
      throw std::invalid_argument(
        "QEP task-graph execution does not support multiple task manager processes");
    }
  }

  if (_multi_process) {
    auto* mp_ctx = dynamic_cast<multi_process_task_manager_context*>(_ctx);
    if (mp_ctx) {
      mp_ctx->update_scheduler(is_write ? SCHEDULER_TYPE::ALL_TO_ALL : SCHEDULER_TYPE::ROUND_ROBIN);
    }
  }

  _cache.update(std::move(descriptors));

  if (_multi_process) {
    // FIXME: Multi-GPU task migration doesn't work with zero-copy enabled.
    opt_params.read_use_zero_copy = false;
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

  // Peak tracking is off unless `track_query_memory` was set when the context was created.
  // Declared before the task graph so that the graph's deallocations fall inside the frame.
  query_memory_frame memory_frame{_ctx->query_memory_statistics()};

  // Declared after `query_ctx` so the graph and all QEP-owned state are destroyed before the
  // query-owned streams they may rely on, including during exception unwinding.
  std::unique_ptr<gqe::task_graph> task_graph;
  {
    utility::nvtx_scoped_range build_range("build_task_graph");
    switch (query_ctx.parameters.execution_backend) {
      case execution_backend_type::legacy_task_graph: {
        gqe::task_graph_builder graph_builder(ctx_ref, &_cache);
        task_graph = graph_builder.build(physical_plan.root().get());
        break;
      }
      case execution_backend_type::qep_task_graph: {
        auto lowered =
          qep::physical_qep_transformation(physical_plan, &_cache, query_ctx.parameters);
        auto normalized = qep::pipeline_normalization(
          lowered.plan, lowered.output_data_types, lowered.stable_keys, lowered.physical_relations);
        task_graph = qep_task_graph_transform(ctx_ref, normalized);
        break;
      }
      default: throw std::logic_error("Unsupported execution backend");
    }
  }

  {
    utility::nvtx_scoped_range exec_range("execute_task_graph");
    if (_multi_process) {
      gqe::execute_task_graph_multi_process(ctx_ref, task_graph.get());
    } else {
      gqe::execute_task_graph_single_gpu(ctx_ref, task_graph.get());
    }
  }

  if (memory_frame.is_tracking()) {
    // Profilers match the mark by name on the thread that pushed `execute_plan`. Under a CUPTI
    // NVTX subscriber, each distinct payload grows CUPTI's attribute intern table, and every later
    // GQE-domain NVTX event in the process pays for its size. See `track_query_memory`.
    auto const peak_bytes = memory_frame.peak_bytes();
    utility::nvtx_mark(
      nvtx3::event_attributes{"query_memory_peak_bytes", nvtx3::payload{peak_bytes}});
    GQE_LOG_DEBUG("Task manager rank {} query {} peak query memory {} bytes above its start",
                  _rank,
                  query_id,
                  peak_bytes);
  }

  service::execution_result result;

  {
    utility::nvtx_scoped_range output_range("collect_results");

    // QEP operations such as the Parquet writer can use dedicated query-owned streams. Drain
    // them before copying results to Arrow storage or publishing written storage through the
    // catalog.
    if (query_ctx.parameters.execution_backend == execution_backend_type::qep_task_graph) {
      query_ctx.close_and_synchronize_streams();
    }

    if (is_write) {
      // Queries that write their result to a table don't return any result rows.
      GQE_EXPECTS(
        query_ctx.parameters.execution_backend == execution_backend_type::qep_task_graph ||
          task_graph->write_statistics != nullptr,
        "Legacy write task graph did not provide a statistics manager",
        std::logic_error);
      auto statistics =
        query_ctx.parameters.execution_backend == execution_backend_type::qep_task_graph
          ? collect_qep_write_statistics(*task_graph)
          : task_graph->write_statistics->statistics();
      result.has_write_stats = true;
      result.write_stats     = rpc::serialize_table_statistics(statistics);
    } else {
      // Collect result rows for non-write queries.
      //
      // Both backends synchronize their producing streams before publishing results. Dedicated
      // query-owned streams were also drained above, so serialization can use this thread's
      // cuDF default stream.
      auto const result_stream = cudf::get_default_stream();
      auto const result_mr     = cudf::get_current_device_resource_ref();
      std::vector<cudf::table_view> result_views;
      for (auto const& root_task : task_graph->root_tasks) {
        if (root_task->result().has_value()) {
          result_views.push_back(root_task->result().value());
        }
      }

      if (!result_views.empty()) {
        result.arrow_ipc = serialize_to_arrow_ipc(result_views, result_stream, result_mr);
      }
    }
  }

  return result;
}

void plan_executor::refresh_stats(std::vector<service::stats_table_write> writes,
                                  utility::uuid query_id,
                                  std::vector<storage::descriptor> descriptors,
                                  std::uint64_t epoch)
{
  // The node manager owns the statistics tables but cannot write shared row groups itself (the
  // shared write path requires a task manager's NVSHMEM rank-0 context and CUDA stream). So it
  // projects the current statistics into Arrow batches and ships them here; rank 0 publishes them
  // into the node-created shared segment via shared_in_memory_table::replace_contents(), which
  // advances the table's storage_revision. Every other rank simply reattaches to the new contents
  // on its next read of the table, when the read's descriptor carries the bumped revision (see
  // remote_table_cache::update) — so only rank 0 writes, and the data lives once in shared memory.
  //
  // (void)epoch: the previous implementation mixed `epoch` into per-refresh shared-object names to
  // avoid collisions; replace_contents now frees the prior row group before publishing, so the
  // epoch is no longer needed for naming.
  (void)epoch;
  GQE_LOG_INFO("Task manager rank {} refreshing {} stats table(s) (refresh {})",
               _rank,
               writes.size(),
               query_id);

  // Register the statistics-table descriptors so this rank can resolve them; on non-writer ranks
  // this also ensures a subsequent read reattaches to the contents rank 0 publishes below.
  _cache.update(std::move(descriptors));

  // Only rank 0 writes the shared contents; other ranks reattach via storage_revision on read.
  if (!gqe::utility::multi_process::nvshmem_rank_zero()) { return; }

  auto stream   = cudf::get_default_stream();
  auto const mr = _ctx->get_table_memory_resource(memory_kind::shared_segment{});

  for (auto const& write : writes) {
    auto table      = _cache.get_table(write.table_name);
    auto* in_memory = dynamic_cast<storage::shared_in_memory_table*>(table.get());
    if (in_memory == nullptr) {
      throw std::runtime_error(
        std::format("Stats refresh target '{}' is not a shared in-memory table", write.table_name));
    }

    if (write.arrow_ipc.empty()) {
      // Nothing to materialize (e.g. no source tables): leave the stats table empty.
      in_memory->clear_row_groups(stream, mr);
      continue;
    }

    auto batch      = deserialize_arrow_ipc(write.arrow_ipc);
    auto cudf_table = arrow_batch_to_cudf(*batch, stream);
    // replace_contents reads from this device table; make sure it is fully materialized first.
    stream.sync();
    in_memory->replace_contents(cudf_table->view(), mr, stream);
    stream.sync();
  }
}

}  // namespace gqe::task_manager
